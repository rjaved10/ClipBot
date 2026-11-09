"""Find and download video tweets from X with gallery-dl (needs your X login cookies)."""
import time
import nowin  # noqa: F401  (no console windows for child processes)
import json
import subprocess
import sys
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent


def _auth_args(cfg):
    if cfg.get("x_cookies_browser"):
        return ["--cookies-from-browser", cfg["x_cookies_browser"]]
    path = ROOT / cfg["x_cookies_file"]
    if not path.exists():
        raise RuntimeError(f"X cookies missing: put your cookies.txt at {path} (see README)")
    return ["--cookies", str(path)]


def _gallery_dl(args, timeout=180):
    cmd = [sys.executable, "-m", "gallery_dl", *args]
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)
    return res.stdout, res.stderr


def search_query(cfg):
    # "search" overrides names for creators whose name is an everyday word on X (dub).
    names = [n for c in cfg["creators"].values() for n in c.get("search", c["names"])]
    # X search treats "DDG’s" / "DDG's" as a different word from "DDG": posts like "her ex DDG’s song"
    # never showed up. Add the possessive spellings of every one-word name.
    names += [f"{n}{a}s" for n in list(names) if '"' not in n for a in ("’", "'")]
    return f"({' OR '.join(names)}) lang:en filter:native_video -filter:retweets -filter:replies"


def _to_candidate(kw):
    if kw.get("type") != "video":
        return None
    date = kw.get("date")
    if isinstance(date, str):
        try:
            date = datetime.fromisoformat(date.replace(" ", "T")).replace(tzinfo=timezone.utc)
        except ValueError:
            date = None
    author = kw.get("author") or kw.get("user") or {}
    return {
        "tweet_id": str(kw.get("tweet_id")),
        "media_key": f"{kw.get('tweet_id')}_{kw.get('num', 1)}",
        "text": kw.get("content", ""),
        "author": author.get("name", ""),
        "likes": int(kw.get("favorite_count") or 0),
        "date": date.isoformat() if date else None,
        "width": kw.get("width"),
        "height": kw.get("height"),
        "duration": kw.get("duration"),
        "url": f"https://x.com/{author.get('name', 'i')}/status/{kw.get('tweet_id')}",
    }


def _parse_json(stdout):
    try:
        data = json.loads(stdout)
    except json.JSONDecodeError:
        return []
    out = []
    for msg in data:
        # gallery-dl -j messages: [3, url, kwdict] for every file
        if isinstance(msg, list) and len(msg) >= 3 and msg[0] == 3 and isinstance(msg[2], dict):
            cand = _to_candidate(msg[2])
            if cand:
                out.append(cand)
    return out


_last_top = 0.0


def discover(cfg, log):
    """Return video tweets from the search plus every watched account."""
    auth = _auth_args(cfg)
    rng = f"1-{cfg['search_results_per_poll']}"
    q = urllib.parse.quote(search_query(cfg))
    urls = [f"https://x.com/search?f=live&q={q}"]                                 # newest, every pass
    # "Top" only every 10 min: searching both every 2 min hit X's rate limit (gallery-dl then waits = hangs).
    global _last_top
    if time.time() - _last_top >= 600:
        urls.append(f"https://x.com/search?q={q}")
        _last_top = time.time()
    urls += [f"https://x.com/{h.lstrip('@')}/media" for h in cfg.get("watch_accounts", [])]
    found = {}
    for url in urls:
        try:
            # One slow search (X's "Top" tab sometimes hangs) must not wipe out the others' results.
            out, err = _gallery_dl([*auth, "-j", "--range", rng, url], timeout=90)
        except Exception as e:
            log(f"search skipped this pass ({'top' if 'f=live' not in url else 'latest'}): {str(e)[-60:]}")
            continue
        items = _parse_json(out)
        if not items and err.strip():
            log(f"gallery-dl: {err.strip().splitlines()[-1][:200]}")
        watched = "/media" in url
        for c in items:
            c["watched"] = watched
            found.setdefault(c["media_key"], c)
    return list(found.values())


def download(cfg, cand, dest: Path):
    dest.mkdir(parents=True, exist_ok=True)
    _gallery_dl([*_auth_args(cfg), "-D", str(dest), "-f", "{tweet_id}_{num}.{extension}",
                 "--filter", "type == 'video'", f"https://x.com/i/status/{cand['tweet_id']}"], timeout=600)
    num = cand["media_key"].split("_")[-1]
    hits = sorted(dest.glob(f"{cand['tweet_id']}_{num}.*"))
    return hits[0] if hits else None


def current_likes(cfg, tweet_ids):
    """Like counts right now for these tweets, in one gallery-dl call. {tweet_id: likes}"""
    if not tweet_ids:
        return {}
    try:
        out, _ = _gallery_dl([*_auth_args(cfg), "-j", *[f"https://x.com/i/status/{t}" for t in tweet_ids]],
                             timeout=120)
    except Exception:
        return {}
    return {c["tweet_id"]: c["likes"] for c in _parse_json_all(out)}


def _parse_json_all(stdout):
    """Like _parse_json, but copes with one JSON document per URL."""
    out, dec, i = [], json.JSONDecoder(), 0
    while i < len(stdout):
        while i < len(stdout) and stdout[i].isspace():
            i += 1
        if i >= len(stdout):
            break
        try:
            data, i = dec.raw_decode(stdout, i)
        except json.JSONDecodeError:
            break
        out += _parse_json(json.dumps(data))
    return out
