"""MrBeast campaign: when MrBeast uploads a new video, find clips of THAT video on X.

Only for brand-new uploads (spotted on his channel's YouTube feed), never random MrBeast posts.
Clips go through the same watermark/quality checks as the streamer clips, get our headline, and
carry only the campaign's hashtags (#mrbeast #mrbeastpartner). They wait in the 'mb_ready' queue
and go up when no streamer clip is waiting.
"""
import json
import re
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone

import requests

import headline
import xsource

FEED = "https://www.youtube.com/feeds/videos.xml?channel_id={}"
NS = {"a": "http://www.w3.org/2005/Atom", "yt": "http://www.youtube.com/xml/schemas/2015"}

# Title words too common to find clips of one video with ("Beat LeBron, Win $100,000" -> lebron).
COMMON = set("""a an the and or but of to in on at for with vs by from i my me you your we our he his she her they
them it its this that these those is are was be been am do did does done can will would should could have has had
not no yes all any every each last first one two three four five ten 100 1 vs beat win won wins winner lose survive
survived days day hours hour minutes years year world worlds biggest best worst most every people person
million billion dollars dollar money cash prize free extreme challenge challenges vs new old just only who what
which how why when where if then than vs mrbeast mr beast""".split())

CLIP_PROMPT = """MrBeast uploaded a YouTube video titled "{title}" on {date}.
Here is an X post with a video, by @{author}:
"{tweet}"

Answer:
- relevant: true only if this post's video is a clip FROM that MrBeast video (not a different MrBeast
  video, not a reaction channel, not an edit/meme, not an ad or giveaway promo).
- adult: true if it is sexual content.
- score 0-10: how much a random TikTok viewer would want to watch it.
- headline: reword the post into a short TikTok headline (max 14 words) saying what happens. Stay true
  to the post; don't exaggerate or invent anything (campaign rule: no misrepresenting the video or MrBeast).
- emoji: one emoji that fits.

Reply with JSON only: {{"relevant": true/false, "adult": true/false, "score": 0-10, "headline": "...", "emoji": "..."}}"""


def new_uploads(cfg, con, log):
    """Note any new MrBeast upload (not Shorts) so we start looking for its clips on X."""
    mb = cfg["mrbeast"]
    r = requests.get(FEED.format(mb["channel_id"]), timeout=20)
    if r.status_code != 200 or not r.text.lstrip().startswith("<?xml"):
        return                      # YouTube's feed sometimes hands back an error page: try next time
    root = ET.fromstring(r.text)
    known = {r[0] for r in con.execute("select video_id from yt_videos")}
    since = datetime.now(timezone.utc) - timedelta(hours=mb.get("max_video_age_hours", 48))
    for e in root.findall("a:entry", NS):
        vid = e.find("yt:videoId", NS).text
        pub = datetime.fromisoformat(e.find("a:published", NS).text)
        link = e.find("a:link", NS).get("href", "")
        if vid in known or pub < since or "/shorts/" in link:
            continue
        title = e.find("a:title", NS).text
        words = [w.lower() for w in re.findall(r"[A-Za-z][A-Za-z']{2,}", title) if w.lower() not in COMMON][:3]
        con.execute("insert into yt_videos(video_id, title, published, status, tried) values(?,?,?,?,?)",
                    [vid, title, pub.isoformat(), json.dumps(words), datetime.now(timezone.utc).isoformat()])
        con.commit()
        log(f"new MrBeast upload: {title} - looking for its clips on X ({', '.join(words) or 'title words'})")


def active(cfg, con):
    """MrBeast uploads still new enough to clip."""
    since = (datetime.now(timezone.utc) - timedelta(hours=cfg["mrbeast"].get("max_video_age_hours", 48))).isoformat()
    rows = con.execute("select video_id, title, published, status from yt_videos where published > ?", [since])
    return [{"id": r[0], "title": r[1], "published": r[2], "keywords": json.loads(r[3] or "[]")} for r in rows]


def _query(video):
    # No stand-out title word: search all MrBeast video posts; the AI check keeps only this video's clips.
    terms = f" ({' OR '.join(video['keywords'])})" if video["keywords"] else ""
    return (f'(mrbeast OR "mr beast"){terms} lang:en filter:native_video -filter:retweets '
            f"-filter:replies since:{video['published'][:10]}")


def discover(cfg, video, log):
    """Video posts on X about this upload, posted after it went up."""
    auth = xsource._auth_args(cfg)
    q = urllib.parse.quote(_query(video))
    rng = f"1-{cfg['search_results_per_poll']}"
    found = {}
    for url in (f"https://x.com/search?f=live&q={q}",):     # newest only: X slows down if we search too much
        out, err = xsource._gallery_dl([*auth, "-j", "--range", rng, url])
        items = xsource._parse_json(out)
        if not items and err.strip():
            log(f"gallery-dl: {err.strip().splitlines()[-1][:200]}")
        for c in items:
            if c.get("date") and c["date"] < video["published"]:
                continue
            c["mrbeast"] = video
            found.setdefault(c["media_key"], c)
    return list(found.values())


def write(cfg, cand):
    """Is this X post a clip of the new upload? If so, our headline for it."""
    video = cand["mrbeast"]
    tweet = headline.clean_tweet(cand["text"])
    no = {"relevant": False, "headline": "", "emoji": "", "score": 0}
    if headline.ADULT.search(cand["text"]) or headline.FOREIGN.search(cand["text"]):
        return no
    raw = headline._ask(cfg, CLIP_PROMPT.format(title=video["title"], date=video["published"][:10],
                                                author=cand["author"], tweet=tweet))
    data = json.loads(raw[raw.find("{"): raw.rfind("}") + 1])
    text = headline.censor(re.sub(r"[#@]\w+", "", data.get("headline", "")).strip().rstrip("."))
    try:
        score = float(data.get("score") or 0)
    except (TypeError, ValueError):
        score = 0
    return {"relevant": bool(data.get("relevant")) and not data.get("adult") and bool(text),
            "headline": text, "emoji": (data.get("emoji") or "😳").strip()[:1], "score": score}


def description(cfg, info):
    text = info["headline"][0].upper() + info["headline"][1:]
    return f"{text} {info['emoji']} {' '.join(cfg['mrbeast']['hashtags'])}"
