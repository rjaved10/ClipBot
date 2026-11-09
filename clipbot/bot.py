"""Clip bot: X -> watermark/quality check -> headline -> layout -> TikTok, on a loop.

    python bot.py            run forever
    python bot.py --once     one pass (find, make, post at most one)
    python bot.py --dry      make videos but never post
    python bot.py --url https://x.com/.../status/123   make one specific tweet (testing)
"""
import nowin  # noqa: F401  (no console windows for child processes)
import argparse
import difflib
import json
import shutil
import sqlite3
import subprocess
import sys
import time
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path

import clipping
import youtube
import fingerprint
import headline
import layout
import mrbeast
import poster
import twitchsource
import watermark
import xsource

ROOT = Path(__file__).parent
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
WORK = ROOT / "work"
OUT = ROOT / "out"
LOGS = ROOT / "logs"
for d in (WORK, OUT, LOGS):
    d.mkdir(exist_ok=True)


def log(msg):
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S}  {msg}"
    print(line, flush=True)
    with open(LOGS / "bot.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_cfg():
    return json.loads((ROOT / "config.json").read_text(encoding="utf-8"))


def db():
    con = sqlite3.connect(ROOT / "clips.db", timeout=30)
    con.execute("""create table if not exists clips(
        media_key text primary key, tweet_id text, status text, reason text,
        transcript text, video text, description text, likes int, created text, posted text)""")
    con.execute("create table if not exists yt_videos(video_id text primary key, title text, published text, "
                "status text, tried text)")
    have = {r[1] for r in con.execute("pragma table_info(clips)")}
    for col in ("bounty text", "clip_sub text", "clip_tries int",   # clipping.net submissions
                "yt text", "yt_tries int", "yt_at text", "yt_sub text", "yt_sub_tries int",                                # YouTube Shorts upload
                "fp blob"):                                                # audio fingerprint
        if col.split()[0] not in have:
            con.execute(f"alter table clips add column {col}")
    return con


def set_status(con, cand, status, **kw):
    cols = {"tweet_id": cand["tweet_id"], "status": status, "likes": cand.get("likes", 0),
            "created": cand.get("date"), **kw}
    con.execute(f"insert into clips(media_key,{','.join(cols)}) values(?{',?' * len(cols)}) "
                f"on conflict(media_key) do update set {','.join(f'{k}=excluded.{k}' for k in cols)}",
                [cand["media_key"], *cols.values()])
    con.commit()
    if status == "skip":
        # Owner couldn't see what happened to "1 new and good enough" clips: say why each one was dropped.
        log(f"  skipped @{cand.get('author', '?')} ({cand.get('likes', 0)} likes): {kw.get('reason', '')}")


def probe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height:format=duration", "-of", "json", str(path)],
                         capture_output=True, text=True).stdout
    j = json.loads(out)
    s = j["streams"][0]
    return int(s["width"]), int(s["height"]), float(j["format"]["duration"])


def is_repeat(con, transcript):
    """Same story posted again by another account (compares the post text)."""
    if not transcript or len(transcript) < 25:
        return False
    rows = con.execute("select transcript from clips where status in ('ready','posted') "
                       "and transcript is not null order by rowid desc limit 300").fetchall()
    return any(difflib.SequenceMatcher(None, transcript[:300], t[:300]).ratio() > 0.8 for (t,) in rows)


def same_footage(con, fp, statuses=("posted", "ready", "mb_ready"), hours=48):
    """Text of a recent clip with the same footage (another account's post of one moment), or None."""
    if not fp:
        return None
    since = (datetime.now() - timedelta(hours=hours)).isoformat()
    marks = ",".join("?" * len(statuses))
    rows = con.execute(f"select transcript, fp from clips where status in ({marks}) and fp is not null "
                       "and coalesce(posted, created) > ?", [*statuses, since]).fetchall()
    return next((t or "?" for t, other in rows if fingerprint.same(fp, other)), None)


def like_speed(cand):
    """Likes per minute since the post went up: how fast it's blowing up."""
    if not cand.get("date"):
        return 0.0
    mins = (datetime.now(timezone.utc) - datetime.fromisoformat(cand["date"])).total_seconds() / 60
    return cand["likes"] / max(mins, 5)


def is_fast(cfg, cand):
    """Clip pages whose posts are nearly always our kind of clip (@yoxics, @PlaqueReport): owner 2026-11-07,
    "when the moment happens it should instantly post". Their tweets skip the likes wait (they used to sit
    10-30 min until 60 likes); the funny/watermark/footage checks still apply."""
    fast = {h.lower().lstrip("@") for h in cfg.get("fast_accounts", [])}
    return (cand.get("author") or "").lower() in fast


def fresh_enough(cfg, cand):
    if cand.get("watched"):
        return True
    if is_fast(cfg, cand) and cand.get("date"):
        return datetime.now(timezone.utc) - datetime.fromisoformat(cand["date"]) <= timedelta(hours=cfg["max_age_hours"])
    # Per-creator likes bar, from the creators the tweet names (DDG 20, owner 2026-11-08: a 2-minute-old DDG
    # clip with 22 likes was being ignored until it reached the global 60).
    named = headline.creators_named(cfg, cand.get("text") or "")
    floor = min([cfg["creators"][k].get("min_likes", cfg["min_likes"]) for k in named] or [cfg["min_likes"]])
    if cand["likes"] < floor:
        return False
    if cand.get("date"):
        age = datetime.now(timezone.utc) - datetime.fromisoformat(cand["date"])
        if age > timedelta(hours=cfg["max_age_hours"]):
            return False
    # Slow posts (few likes for their age) are clips nobody cared about.
    return like_speed(cand) >= cfg.get("min_likes_per_minute", 0.5)


def _badge_visible(lay, own):
    """True if the stream's own badge box (name, x, y, w, h) stays fully inside what we show."""
    _, x, y, w, h = own
    if lay["mode"] == "reaction":
        return True                     # the bottom half shows the whole frame
    if lay["mode"] != "single":
        return False
    crops = [c for _, _, c in lay["shots"]] if lay.get("shots") else [lay["crop"]]
    return all(cx <= x and x + w <= cx + cw and cy <= y and y + h <= cy + ch for cx, cy, cw, ch in crops)


def twitch_copy(cfg, cand, src, dur, job):
    """The same moment from the named creator's own Twitch recording, or None."""
    if not cfg.get("twitch_source", True) or not cand.get("date"):
        return None
    for k in headline.creators_named(cfg, cand.get("text") or ""):
        channel = cfg["creators"][k].get("twitch")
        if not channel:
            continue
        try:
            out = twitchsource.clean_copy(channel, src, cand["date"], dur, job / "twitch", log)
        except Exception as e:
            log(f"  twitch: failed ({str(e)[:120]})")
            continue
        if out:
            log(f"  using the clean copy from twitch.tv/{channel} instead of the watermarked one")
            return out
    return None


def make(cfg, con, cand):
    """Download, check and render one candidate. Returns True if it is ready to post."""
    job = WORK / cand["media_key"]
    log(f"  checking @{cand.get('author', '?')} ({cand.get('likes', 0)} likes): {' '.join((cand.get('text') or '').split())[:80]}")
    try:
        src = xsource.download(cfg, cand, job)
        if not src:
            return set_status(con, cand, "skip", reason="download failed")
        w, h, dur = probe(src)
        if min(w, h) < 720:
            return set_status(con, cand, "skip", reason=f"low quality {w}x{h}")
        campaign = cand.get("mrbeast")
        lo = max(cfg["min_seconds"], cfg["mrbeast"].get("min_video_seconds", 15)) if campaign else cfg["min_seconds"]
        if not lo <= dur <= cfg["max_seconds"]:
            return set_status(con, cand, "skip", reason=f"length {dur:.0f}s")
        allowed = cfg["allowed_handles"] + (["mrbeast", "beast"] if campaign else [])
        bad = watermark.check(src, allowed)
        if bad and bad != "could not read video" and not campaign:
            # Owner 2026-11-08: a clip page's watermark -> take the same moment from the streamer's own Twitch
            # recording instead. That copy is the streamer's raw feed, so everything on it is their own overlay
            # (segment titles like "SONG CHALLENGE : DUOS" are not a burned-in headline): no watermark check.
            clean = twitch_copy(cfg, cand, src, dur, job)
            if clean:
                src, bad = clean, None
                w, h, dur = probe(src)
                watermark.check(src, allowed)            # fills the OCR cache (Rising Star, badge) only
                if not cfg["min_seconds"] <= dur <= cfg["max_seconds"]:
                    return set_status(con, cand, "skip", reason=f"length {dur:.0f}s (Twitch copy)")
        if bad:
            return set_status(con, cand, "skip", reason=bad)
        if watermark.foreign_bar(src, allowed):
            return set_status(con, cand, "skip", reason="someone else's Kick/Twitch watermark")
        text = headline.clean_tweet(cand["text"])
        if is_repeat(con, text):
            return set_status(con, cand, "skip", reason="same clip already made", transcript=text)
        fp = fingerprint.compute(src)
        if same_footage(con, fp):
            return set_status(con, cand, "skip", reason="same footage already made", transcript=text)
        if campaign:
            return make_mrbeast(cfg, con, cand, src, job, text, w, h, dur, fp)
        info = headline.write(cfg, cand)
        if not info["relevant"]:
            return set_status(con, cand, "skip", reason="not one of our creators / politics / meme post", transcript=text)
        # Funny, or going crazy on X. Slow, unfunny posts (news updates, promos) don't make the cut.
        speed = like_speed(cand)
        viral = speed >= cfg.get("viral_likes_per_minute", 5)
        # Some creators get fewer views (Lacy: median 134 vs DDG 591), so their clips need a higher bar.
        # A clip with several creators uses the easiest bar among them.
        bars = [cfg["creators"].get(k, {}) for k in info["creators"]] or [{}]
        min_funny = min(b.get("min_funny", cfg.get("min_funny", 6)) for b in bars)
        min_speed = min(b.get("min_likes_per_minute", 0) for b in bars)
        # Per-creator X-likes floor, from the creator the clip is ABOUT (owner 2026-11-06: Jason clips under
        # 100 X likes mostly got 50-250 TikTok views; at 100+ they averaged ~550).
        lead = [cfg["creators"].get(k, {}) for k in headline.leads(cfg, info["headline"])] or bars
        floor = min(b.get("min_likes", cfg["min_likes"]) for b in lead)
        fast = is_fast(cfg, cand)
        if cand["likes"] < floor and not viral and not fast:
            return set_status(con, cand, "skip", reason=f"under {floor} X likes for this creator ({cand['likes']})",
                              transcript=text)
        if info.get("kind") == "promo" or not ((info.get("funny", 0) >= min_funny and (speed >= min_speed or fast)) or viral):
            why = f"not funny or viral ({info.get('kind')} {info.get('funny', 0):g}/10, {speed:.1f} likes/min)"
            return set_status(con, cand, "skip", reason=why, transcript=text)
        lay = layout.analyse(src)
        # Owner 2026-11-06 (told me several times): NO reaction split on Max's clips. His Rising Star stream is a
        # room with cameras, no webcam box; people sitting still there kept getting read as a facecam.
        leads = headline.leads(cfg, info["headline"])
        rising = watermark.rising_star(src)     # read before a trim swaps src (the OCR is cached per file)
        no_split = set(leads) & set(cfg.get("never_reaction", [])) or rising
        if lay["mode"] == "reaction" and no_split:
            lay = layout._single(lay["w"], lay["h"], lay["faces"])
            log("no reaction split (Max / Rising Star stream)")
        if lay["mode"] != "reaction":
            # Opens on a screen-share (an X post, a website) with nobody on screen: start where the face shows
            # (owner's call 2026-11-05: the first seconds are the TikTok preview).
            t0 = layout.face_start(src)
            # Never more than 40% of the clip: a wide room shot (Rising Star) with small faces read as "no face"
            # for 28s of a 36s clip and left 7s (2026-11-08). A real screen-share intro is a few seconds.
            if t0 and 2 <= t0 <= 0.4 * dur and dur - t0 >= cfg["min_seconds"]:
                src = layout.trim_start(src, t0, job / "from_face.mp4")
                lay = layout.analyse(src)
                if lay["mode"] == "reaction" and no_split:
                    lay = layout._single(lay["w"], lay["h"], lay["faces"])
                log(f"cut the first {t0:.1f}s (no face on screen)")
        out = OUT / f"{cand['media_key']}.mp4"
        camp_badge = layout.badge_for(headline.leads(cfg, info["headline"]))   # only the clip's own creator
        # Badge: any of our creators named anywhere in the headline ("Cuffem and Lil Yachty hit PlaqueBoyMax's
        # stream" still gets Max's badge, owner 2026-11-06). The campaign stays strict (lead creator only).
        badge = camp_badge or layout.badge_for(headline.creators_named(cfg, info["headline"]))
        # The stream's own name badge (e.g. @plaqueboymax), read on full-size frames of the file we render.
        own = watermark.name_box(src, [p.stem for p in (layout.ASSETS / "badges").glob("*.png")])
        if lay["mode"] == "single" and own:
            # That badge ends up off to one side after the crop: slide the crop so it's centred (no blur/erase).
            lay = layout.center_on(lay, own[1] + own[3] / 2)
        # Owner 2026-11-06: Max and Jason clips ALWAYS show a badge, but never two. Ours is only left off
        # when the stream's own badge for that creator is steady on screen AND inside the final crop.
        if badge and own and own[0] == badge.stem and _badge_visible(lay, own):
            badge = None
            log(f"no badge added: clip already shows @{own[0]}")
        layout.render(src, lay, info["headline"], info["emoji"], out, job, badge=badge)
        desc = headline.description(cfg, info)
        set_status(con, cand, "ready", video=str(out), description=desc, transcript=text,
                   bounty=clipping.bounty_for(cfg, camp_badge, headline.leads(cfg, info["headline"]),
                                              rising=rising), fp=fp)
        log(f"READY [{lay['mode']}] {w}x{h} {dur:.0f}s {info.get('kind')} {info.get('funny', 0):g}/10 {speed:.1f}/min  {desc}")
        return True
    except Exception as e:
        log(f"error on {cand['url']}: {e}")
        (LOGS / "errors.log").open("a", encoding="utf-8").write(traceback.format_exc() + "\n")
        set_status(con, cand, "error", reason=str(e)[:300])
    finally:
        shutil.rmtree(job, ignore_errors=True)


def make_mrbeast(cfg, con, cand, src, job, text, w, h, dur, fp):
    """A clip of MrBeast's new upload: campaign hashtags only, no name badge."""
    info = mrbeast.write(cfg, cand)
    if not info["relevant"]:
        return set_status(con, cand, "skip", reason="not a clip of the new MrBeast video", transcript=text)
    if info["score"] < cfg["mrbeast"].get("min_score", 6):
        return set_status(con, cand, "skip", reason=f"MrBeast clip not good enough ({info['score']:g}/10)",
                          transcript=text)
    lay = layout.analyse(src)
    out = OUT / f"{cand['media_key']}.mp4"
    layout.render(src, lay, info["headline"], info["emoji"], out, job)
    desc = mrbeast.description(cfg, info)
    set_status(con, cand, "mb_ready", video=str(out), description=desc, transcript=text, fp=fp)
    log(f"MRBEAST READY [{lay['mode']}] {w}x{h} {dur:.0f}s {info['score']:g}/10  {desc}")
    return True


def first_tags(cfg):
    """Tags of "always_first" creators (DDG): their clips jump the queue and skip the daily limit."""
    # "always_first_until" = a time-limited boost (PlaqueBoyMax's Rising Star event, Nov 6-8 2026).
    now = datetime.now().isoformat()
    return [t for c in cfg["creators"].values()
            if c.get("always_first") or c.get("always_first_until", "") > now for t in c["tags"]]


def is_first(cfg, desc):
    return any(t in (desc or "").lower() for t in first_tags(cfg))


def can_post(cfg, con, uncapped=False):
    rows = con.execute("select posted, description from clips where status='posted' and posted is not null").fetchall()
    all_times = [datetime.fromisoformat(r[0]) for r in rows]
    # DDG posts (owner's call 2026-11-04) don't count toward the limit, so they never use up other creators' slots.
    times = [datetime.fromisoformat(r[0]) for r in rows if not is_first(cfg, r[1])]
    now = datetime.now()
    if uncapped:
        return not all_times or now - max(all_times) >= timedelta(minutes=cfg["min_minutes_between_posts"])
    # Daily limit resets at midnight, so a late-night run doesn't block the next day.
    # "count_from" in config.json restarts the day's count by hand (posts before it don't count).
    since = datetime.fromisoformat(cfg["count_from"]) if cfg.get("count_from") else datetime.min
    today = len([t for t in times if t.date() == now.date() and t > since])
    if today >= cfg["max_posts_per_day"]:
        return False
    # Spread the day's posts out so the evening (when streamers are live) still has slots: by any time
    # of day only that share of the limit may be used, plus a small head start to catch up on a quiet stretch.
    # Peak hours (7pm-midnight: the best clips AND views) weigh more, so ~10 of 16 slots wait for them.
    lo, hi = cfg.get("peak_hours", [19, 24])
    weight = [cfg.get("peak_weight", 11) if lo <= h < hi else 1 for h in range(24)]
    share = (sum(weight[:now.hour]) + weight[now.hour] * now.minute / 60) / sum(weight)
    if today >= int(cfg["max_posts_per_day"] * share) + cfg.get("pace_head_start", 2):
        return False
    return not all_times or now - max(all_times) >= timedelta(minutes=cfg["min_minutes_between_posts"])


def post_next(cfg, con, dry):
    if dry or cfg["post_mode"] == "none":
        return
    only_first = False
    if not can_post(cfg, con):
        # Out of slots: only a DDG clip may still go up (no daily limit for DDG, 15 min gap still applies).
        ready = [d for (d,) in con.execute("select description from clips where status='ready'")]
        if not (any(is_first(cfg, d) for d in ready) and can_post(cfg, con, uncapped=True)):
            return
        only_first = True
    now = datetime.now(timezone.utc)
    # Clips go stale fast: drop anything older than max_queue_hours instead of posting it late.
    stale = (now - timedelta(hours=cfg.get("max_queue_hours", 3))).isoformat()
    n = con.execute("update clips set status='skip', reason='too old by the time a slot opened' "
                    "where status='ready' and created is not null and created < ?", [stale]).rowcount
    con.commit()                 # always: an open transaction would keep the database locked while idle
    if n:
        log(f"dropped {n} clip(s) that got too old in the queue")
    since = (datetime.now() - timedelta(hours=12)).isoformat()
    posted = [t for (t,) in con.execute("select transcript from clips where status='posted' and posted > ? "
                                        "and transcript is not null", [since])]

    queue = con.execute("select media_key, video, description, transcript, likes, created, tweet_id from clips "
                        "where status='ready'").fetchall()
    if not queue:
        return post_mrbeast(cfg, con)
    # Re-check the likes on X right now: the clip blowing up fastest at this moment goes up.
    fresh = xsource.current_likes(cfg, [r[6] for r in queue])
    for r in queue:
        if r[6] in fresh:
            con.execute("update clips set likes=? where media_key=?", [fresh[r[6]], r[0]])
    con.commit()
    queue = [(*r[:4], fresh.get(r[6], r[4]), r[5]) for r in queue]

    def score(row):
        """Likes per minute since the post went up (current numbers)."""
        likes, created = row[4] or 0, row[5]
        mins = (now - datetime.fromisoformat(created)).total_seconds() / 60 if created else 60
        return likes / max(mins, 5)
    row = None
    # Newest good clip first: of the clips still getting likes fast enough (live numbers), the most
    # recently posted on X goes up. Only if none qualify does the fastest-rising older one go.
    min_speed = cfg.get("min_likes_per_minute", 0.5)
    def priority(r):
        """DDG / PlaqueBoyMax clips get the most views, Lacy's the fewest (TikTok Studio, Nov 2026)."""
        tags = (r[2] or "").lower()
        return max([c.get("priority", 1) for c in cfg["creators"].values() if any(t in tags for t in c["tags"])] or [1])
    # "always_first" creators (DDG, owner's call 2026-11-04) jump the queue even when their likes are slower.
    top = sorted((r for r in queue if is_first(cfg, r[2])), key=lambda r: r[5] or "", reverse=True)
    queue = [] if only_first else [r for r in queue if r not in top]
    good = sorted((r for r in queue if score(r) >= min_speed), key=lambda r: (priority(r), r[5] or ""), reverse=True)
    rest = sorted((r for r in queue if score(r) < min_speed), key=score, reverse=True)
    best = top + good + rest

    def age(r):
        return f"{(now - datetime.fromisoformat(r[5])).total_seconds() / 60:.0f}m old" if r[5] else "?"
    log("queue now: " + " | ".join(f"{age(r)} {score(r):.1f}/min" for r in best[:5]))
    for cand in best:
        fp = con.execute("select fp from clips where media_key=?", [cand[0]]).fetchone()[0]
        twin = same_footage(con, fp, ("posted",))
        if twin:
            log(f"skip {cand[0]}: same footage as an earlier post ({twin[:60]})")
            con.execute("update clips set status='skip', reason='same footage already posted' where media_key=?",
                        [cand[0]])
            con.commit()
            continue
        dup = headline.same_story(cfg, cand[3] or cand[2], posted) if cand[3] else None
        if dup is not None:
            log(f"skip {cand[0]}: same moment as an earlier post ({posted[dup][:60]})")
            con.execute("update clips set status='skip', reason='same moment already posted' where media_key=?",
                        [cand[0]])
            con.commit()
            continue
        row = cand[:3]
        break
    if not row:
        return None if only_first else post_mrbeast(cfg, con)
    _post(cfg, con, *row)


def post_mrbeast(cfg, con):
    """No streamer clip waiting: post the next MrBeast campaign clip (at most one per hour or so)."""
    mb = cfg.get("mrbeast", {})
    if not mb.get("enabled"):
        return
    last = con.execute("select max(posted) from clips where reason='mrbeast' and status='posted'").fetchone()[0]
    if last and datetime.now() - datetime.fromisoformat(last) < timedelta(minutes=mb.get("min_minutes_between", 60)):
        return
    # Newest clip of the video first.
    row = con.execute("select media_key, video, description from clips where status='mb_ready' "
                      "order by created desc limit 1").fetchone()
    if row:
        con.execute("update clips set reason='mrbeast' where media_key=?", [row[0]])
        _post(cfg, con, *row)


def _post(cfg, con, key, video, desc):
    log(f"posting {key}: {desc}")
    try:
        poster.post(Path(video), desc, headless=cfg["browser_headless"], log=log)
        con.execute("update clips set status='posted', posted=? where media_key=?", [datetime.now().isoformat(), key])
    except Exception as e:
        log(f"post failed: {e}")
        con.execute("update clips set status='post_error', reason=? where media_key=?", [str(e)[:300], key])
    con.commit()


_last_mb = 0.0


def one_pass(cfg, con, dry):
    try:
        cands = xsource.discover(cfg, log)
    except Exception as e:
        log(f"discover failed: {e}")
        cands = []
    mb = cfg.get("mrbeast", {})
    global _last_mb
    if mb.get("enabled") and time.time() - _last_mb >= mb.get("check_minutes", 10) * 60:
        # MrBeast campaign: only clips of a video he just uploaded. Checked every 10 minutes, not every
        # pass, so X doesn't slow our searches down.
        _last_mb = time.time()
        try:
            mrbeast.new_uploads(cfg, con, log)
            for video in mrbeast.active(cfg, con):
                cands += [c for c in mrbeast.discover(cfg, video, log) if c["likes"] >= mb.get("min_likes", 20)]
        except Exception as e:
            log(f"MrBeast check failed: {e}")
    known = {r[0] for r in con.execute("select media_key from clips where status != 'error'")}
    # Pages that stamp their own logo on every clip (everythingDDG's "eDDG") are never usable.
    blocked = {h.lower().lstrip("@") for h in cfg.get("blocked_authors", [])}
    cands = [c for c in cands if c.get("author", "").lower() not in blocked]
    new = [c for c in cands if c["media_key"] not in known and (c.get("mrbeast") or fresh_enough(cfg, c))]
    new.sort(key=lambda c: (not c.get("watched"), not is_fast(cfg, c), -like_speed(c)))   # fastest-rising first
    log(f"found {len(cands)} video tweets, {len(new)} new and good enough")

    for cand in new:
        make(cfg, con, cand)
        post_next(cfg, con, dry)   # post as soon as one is ready instead of waiting for the batch
    post_next(cfg, con, dry)
    if not dry:
        try:
            clipping.run(cfg, con, log)
        except Exception as e:
            log(f"clipping.net submit failed: {e}")
        try:
            youtube.run(cfg, con, log)
        except Exception as e:
            log(f"YouTube upload failed: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--url")
    a = ap.parse_args()
    con = db()
    if a.url:
        cfg = load_cfg()
        tid = a.url.rstrip("/").split("/")[-1].split("?")[0]
        cand = {"tweet_id": tid, "media_key": f"{tid}_1", "text": "", "author": a.url.split("/")[3],
                "likes": 0, "date": None, "url": a.url, "watched": True}
        out, _ = xsource._gallery_dl([*xsource._auth_args(cfg), "-j", f"https://x.com/i/status/{tid}"])
        meta = xsource._parse_json(out)
        if meta:
            cand.update({k: v for k, v in meta[0].items() if v})
        make(cfg, con, cand)
        return
    while True:
        cfg = load_cfg()
        one_pass(cfg, con, a.dry)
        if a.once:
            break
        time.sleep(cfg["poll_seconds"])


if __name__ == "__main__":
    main()
