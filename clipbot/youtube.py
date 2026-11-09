"""Upload our posted TikToks to YouTube Shorts too, through YouTube Studio in the bot's Edge profile.

Log in once with `login_youtube.bat` (pick the clips channel). Runs after the TikTok post and the
clipping.net pass, in the same browser profile, one thing at a time. Only clips posted after
youtube.since get uploaded (no backfill of old TikToks).
"""
import re
import time
from datetime import datetime, timedelta
from pathlib import Path

from playwright.sync_api import sync_playwright

import poster

ROOT = Path(__file__).parent
UPLOAD_URL = "https://www.youtube.com/upload"


def _title(desc):
    """Headline + emoji, no hashtags; YouTube allows 100 chars and no angle brackets."""
    t = re.sub(r"\s#\S+", "", desc).strip().replace("<", "").replace(">", "")
    return t if len(t) <= 100 else t[:97].rsplit(" ", 1)[0] + "..."


def _description(desc):
    tags = re.findall(r"#\S+", desc)
    if "#shorts" not in [t.lower() for t in tags]:
        tags.append("#shorts")
    return f"{_title(desc)}\n\n{' '.join(tags)}"


def _fill(page, box, text):
    box.click()
    page.keyboard.press("Control+A")
    page.keyboard.press("Delete")
    page.keyboard.insert_text(text)
    page.wait_for_timeout(500)


def upload(video: Path, desc: str, headless=True, log=print):
    """Upload one Short, public, not made for kids. Returns the video's URL."""
    with sync_playwright() as p:
        ctx = poster._context(p, headless)
        page = ctx.pages[0] if ctx.pages else ctx.new_page()
        try:
            page.goto(UPLOAD_URL, wait_until="domcontentloaded", timeout=60000)
            page.wait_for_timeout(5000)
            if "accounts.google.com" in page.url or "ServiceLogin" in page.url:
                raise RuntimeError("YouTube is logged out: run login_youtube.bat once")
            page.locator("input[type=file]").first.set_input_files(str(video))

            title = page.locator("#title-textarea #textbox")
            title.wait_for(timeout=120_000)
            page.wait_for_timeout(3000)
            _fill(page, title, _title(desc))
            _fill(page, page.locator("#description-textarea #textbox"), _description(desc))
            page.locator("tp-yt-paper-radio-button[name='VIDEO_MADE_FOR_KIDS_NOT_MFK']").click()
            page.wait_for_timeout(800)

            # Details -> Video elements -> Checks -> Visibility. Jump straight to the Visibility step and
            # make sure Public really got ticked: otherwise the last button says "Save" and YouTube keeps
            # the video as a private draft (happened on the first test, 2026-11-06).
            dlg = page.locator("ytcp-uploads-dialog")
            pub = dlg.locator("tp-yt-paper-radio-button[name='PUBLIC']")
            for _ in range(5):
                dlg.locator("#step-badge-3").first.click()
                page.wait_for_timeout(2500)
                if pub.count():
                    pub.first.click()
                    page.wait_for_timeout(1500)
                    if pub.first.get_attribute("aria-checked") == "true":
                        break
            else:
                raise RuntimeError("couldn't select Public")

            url = None
            link = page.locator("ytcp-video-info a, a.ytcp-video-info").first
            try:
                url = link.get_attribute("href", timeout=60_000)
            except Exception:
                pass

            done = dlg.locator("#done-button")
            deadline = time.time() + 900                # Publish stays greyed out while the file uploads
            while time.time() < deadline:
                if done.count() and done.first.is_enabled() and done.first.get_attribute("aria-disabled") != "true":
                    break
                page.wait_for_timeout(3000)
            else:
                raise RuntimeError("YouTube upload never finished")
            if "publish" not in done.first.inner_text().lower():
                raise RuntimeError(f"last button says {done.first.inner_text()!r}, not Publish")
            # Publishing before YouTube's copyright check ends brings a "you may get a strike" popup:
            # wait for "Checks complete" instead of clicking "Publish anyway".
            deadline = time.time() + 1200
            while time.time() < deadline and not dlg.get_by_text(re.compile(r"Checks complete", re.I)).count():
                page.wait_for_timeout(5000)
            checks = dlg.get_by_text(re.compile(r"Checks complete", re.I))
            log(f"YouTube checks: {checks.first.inner_text().strip()[:120] if checks.count() else 'still running after 20 min'}")
            done.first.click()
            page.wait_for_timeout(3000)
            if page.get_by_text("still checking your content").count():
                page.get_by_role("button", name="Go back").first.click()
                raise RuntimeError("YouTube still checking the video; left as a private draft")

            deadline = time.time() + 180
            while time.time() < deadline:
                if page.get_by_text(re.compile(r"^\s*(Video|Short) published\s*$", re.I)).count():
                    break
                page.wait_for_timeout(2000)
            else:
                page.screenshot(path=str(ROOT / "logs" / f"yt_fail_{int(time.time())}.png"))
                raise RuntimeError("never saw YouTube's published confirmation")
            if not url:
                share = page.locator("a[href*='youtube.com/shorts'], a[href*='youtu.be']").first
                url = share.get_attribute("href") if share.count() else "?"
            log(f"YouTube: uploaded {url}")
            return url
        finally:
            ctx.close()


_last_check = 0.0


def _tiktok_views(cfg, rows):
    """TikTok play counts for our posted clips, read from TikTok Studio: {media_key: views}."""
    import clipping
    with sync_playwright() as p:
        ctx = poster._context(p, cfg["browser_headless"])
        page = ctx.pages[0] if ctx.pages else ctx.new_page()
        try:
            items = clipping._studio_items(page)
        finally:
            ctx.close()
    views = {}
    for key, _, desc, posted, _ in rows:
        since = time.mktime(datetime.fromisoformat(posted).timetuple()) - 900
        item = next((i for i in items if clipping._key(i.get("desc")) == clipping._key(desc)
                     and int(i.get("create_time") or 0) >= since and i.get("visibility") == 1), None)
        if item:
            views[key] = int(item.get("play_count") or 0)
    return views


def run(cfg, con, log):
    """Winners only (owner 2026-11-06): a clip goes to YouTube once it beats min_views on TikTok,
    at most max_per_day Shorts a day, biggest first. Clips that never get there are dropped after
    give_up_hours. Spamming every clip would sink the channel's averages."""
    global _last_check
    y = cfg.get("youtube", {})
    if not y.get("enabled") or time.time() - _last_check < y.get("check_minutes", 30) * 60:
        return
    _last_check = time.time()
    now = datetime.now()
    oldest = max(y["since"], (now - timedelta(hours=y.get("give_up_hours", 48) + 24)).isoformat())
    rows = con.execute("select media_key, video, description, posted, yt_tries from clips where status='posted' "
                       "and yt is null and posted > ? and description is not null order by posted",
                       [oldest]).fetchall()
    if not rows:
        return
    views = _tiktok_views(cfg, rows)
    winners = []
    for key, video, desc, posted, tries in rows:
        age = (now - datetime.fromisoformat(posted)).total_seconds() / 3600
        v = views.get(key, 0)
        if v >= y.get("min_views", 500):
            winners.append((v, key, video, desc, tries))
        elif age >= y.get("give_up_hours", 48):
            con.execute("update clips set yt=? where media_key=?", [f"skip: {v} TikTok views", key])
    con.commit()
    done_today = con.execute("select count(*) from clips where yt like 'http%' and yt_at > ?",
                             [(now - timedelta(hours=24)).isoformat()]).fetchone()[0]
    room = y.get("max_per_day", 5) - done_today
    if winners:
        log(f"YouTube: {len(winners)} TikTok winners waiting, {max(room, 0)} uploads left today")
    for v, key, video, desc, tries in sorted(winners, reverse=True)[:max(room, 0)]:
        tries = (tries or 0) + 1
        if not video or not Path(video).exists():
            done = "skip: video file gone"
        else:
            try:
                log(f"YouTube: uploading {key} ({v} TikTok views)")
                done = upload(Path(video), desc, headless=cfg["browser_headless"], log=log)
            except Exception as e:
                log(f"YouTube upload failed for {key}: {str(e)[:200]}")
                done = f"error: {str(e)[:200]}" if tries >= 3 else None
                if "logged out" in str(e):
                    return
        con.execute("update clips set yt=?, yt_tries=?, yt_at=? where media_key=?",
                    [done, tries, datetime.now().isoformat(), key])
        con.commit()


if __name__ == "__main__":
    poster.login("https://studio.youtube.com/", "YouTube (pick the clips channel)")
