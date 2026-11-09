"""Clean copy of a watermarked clip, straight from the streamer's own Twitch recording (owner 2026-11-08).

A clip page's tweet carries their watermark; the same moment in Max's / Jason's own stream recording carries
only the streamer's overlay, which the campaigns want. Find it by sound: download the stream's audio for
the minutes before the tweet, line the clip's audio up against it, then download just that stretch of video.
"""
import nowin  # noqa: F401  (no console windows for child processes)
import json
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np

import fingerprint

LOOKBACK = (16, 50)       # minutes before the tweet to search: the recent stretch first, then further back
PIECE = 8.0               # seconds of the clip matched at its start and at its end
MAX_BER = 0.30


def _ytdlp(args, timeout=300):
    res = subprocess.run([sys.executable, "-m", "yt_dlp", "--no-warnings", *args], capture_output=True,
                         text=True, encoding="utf-8", errors="replace", timeout=timeout)
    return res.stdout, res.stderr


def find_vod(channel, when):
    """(vod_url, start_timestamp) of the recording that was running at `when` (unix seconds)."""
    out, _ = _ytdlp(["--flat-playlist", "--playlist-end", "3", "--print", "%(id)s",
                     f"https://www.twitch.tv/{channel}/videos?filter=archives&sort=time"], timeout=90)
    for vid in out.split():
        url = f"https://www.twitch.tv/videos/{vid.lstrip('v')}"
        meta, _ = _ytdlp(["-J", url], timeout=90)
        try:
            d = json.loads(meta)
        except json.JSONDecodeError:
            continue
        start, dur = d.get("timestamp") or 0, d.get("duration") or 0
        # A recording still going (live) keeps growing; its duration lags a few minutes behind.
        if start <= when <= start + dur + (900 if d.get("is_live") else 120):
            return url, start
    return None


def _section(url, fmt, a, b, dest):
    """Seconds a..b of the recording, fetched as the stream's own ~10 s segment files, 8 at a time.
    (yt-dlp's --download-sections hands a still-live recording to ffmpeg, which reads it at playback
    speed: 18 MB in 10 minutes on Jason's live stream.) Returns (path, where that file starts) or None."""
    out, _ = _ytdlp(["-g", "-f", fmt, url], timeout=90)
    m3u8 = out.strip().splitlines()[-1] if out.strip() else ""
    if not m3u8.startswith("http"):
        return None
    with urllib.request.urlopen(m3u8, timeout=30) as r:
        lines = r.read().decode("utf-8", "replace").splitlines()
    segs, t, length, init = [], 0.0, None, None
    for line in lines:
        if line.startswith("#EXT-X-MAP:") and 'URI="' in line:          # fMP4 recordings: header file first
            init = urllib.parse.urljoin(m3u8, line.split('URI="')[1].split('"')[0])
        elif line.startswith("#EXTINF:"):
            length = float(line[8:].split(",")[0])
        elif line and not line.startswith("#") and length is not None:
            if t + length > a and t < b:
                segs.append((t, urllib.parse.urljoin(m3u8, line)))
            t, length = t + length, None
    if not segs:
        return None

    def get(u):
        for _ in range(3):
            try:
                with urllib.request.urlopen(u, timeout=30) as r:
                    return r.read()
            except Exception:
                time.sleep(1)
        return b""
    with ThreadPoolExecutor(8) as ex:
        parts = list(ex.map(get, ([init] if init else []) + [u for _, u in segs]))
    if not all(parts):
        return None
    dest.write_bytes(b"".join(parts))                    # MPEG-TS segments join by simple appending
    return dest, segs[0][0]


def _locate(piece, long):
    """Offset in seconds where `piece` sits inside `long` (both fingerprints), or None."""
    a, b = np.frombuffer(long, np.uint32), np.frombuffer(piece, np.uint32)
    if len(b) < 40 or len(a) < len(b):
        return None
    # Bit errors at every offset, vectorised over offsets in chunks.
    win = np.lib.stride_tricks.sliding_window_view(a, len(b))
    best, best_at = 1.0, None
    for i in range(0, len(win), 4000):
        x = win[i:i + 4000] ^ b
        ber = np.unpackbits(x.view(np.uint8), axis=1).sum(1) / (32 * len(b))
        j = int(ber.argmin())
        if ber[j] < best:
            best, best_at = float(ber[j]), i + j
    return best_at * fingerprint.STEP if best < MAX_BER else None


def _clip_fp(src, start, length, work):
    part = work / f"piece_{start:.0f}.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{start:.2f}", "-t", f"{length:.2f}", "-i", str(src),
                    "-vn", "-ac", "1", str(part)], capture_output=True)
    return fingerprint.compute(part) if part.exists() else None


def clean_copy(channel, src, tweet_time, dur, work: Path, log):
    """Download the same moment from `channel`'s Twitch recording. Returns the new video path or None."""
    work.mkdir(parents=True, exist_ok=True)
    when = datetime.fromisoformat(tweet_time).timestamp()
    vod = find_vod(channel, when)
    if not vod:
        log(f"  twitch: no {channel} recording covering that time")
        return None
    url, vstart = vod
    head = _clip_fp(src, 0.5, PIECE, work)
    tail = _clip_fp(src, max(0.5, dur - PIECE - 0.5), PIECE, work)
    if not head:
        return None
    now_at = when - vstart                               # tweet time, as a position in the recording
    hi = now_at + 60
    for mins in LOOKBACK:
        lo = max(0.0, now_at - mins * 60)
        got = _section(url, "Audio_Only/bestaudio", lo, hi, work / "vod_audio.ts")
        if not got:
            log("  twitch: couldn't download the recording's audio")
            return None
        audio, start = got
        long = fingerprint.compute(audio)
        at = _locate(head, long) if long else None
        if at is not None:
            a = start + at - 0.5                         # where the clip starts in the recording
            b = a + dur
            end_at = _locate(tail, long) if tail else None
            if end_at is not None and 0 < start + end_at + PIECE + 0.5 - a <= 1.5 * dur:
                b = start + end_at + PIECE + 0.5         # clip pages trim the middle sometimes: take it all
            log(f"  twitch: found it at {int(a // 3600)}h{int(a % 3600 // 60):02d}m in {url}")
            got = _section(url, "1080p60/1080p/720p60/best[height<=1080]", a, b, work / "twitch_raw.ts")
            if not got:
                return None
            raw, vs = got
            out = work / "twitch.mp4"
            import layout                                # GTX 1660 NVENC, CPU fallback
            layout.encode(lambda v: ["ffmpeg", "-v", "error", "-y", "-ss", f"{a - vs:.2f}", "-i", str(raw),
                                     "-t", f"{b - a:.2f}", *v, "-c:a", "aac", "-b:a", "192k", str(out)], 18)
            return out if out.exists() and out.stat().st_size > 1000 else None
        hi = lo + 30                                     # next round: further back, a little overlap
    log("  twitch: moment not found in the recording")
    return None
