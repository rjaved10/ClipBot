"""Reject clips that carry someone else's watermark or an already-edited headline.

Reads text from frames across the clip with OCR. Text that stays on screen in the
same spot is part of the video file (a watermark or burned-in caption), unlike chat
or game text that keeps changing.
"""
import nowin  # noqa: F401  (no console windows for child processes)
import os
import re
import sys
from pathlib import Path
from difflib import SequenceMatcher

import cv2

_ocr = None
_last = {}  # path -> OCR text per frame, reused by shows_name()

STRONG = re.compile(r"tiktok|capcut")
# Spinning/animated logo stamps (the YTClips circle) only read as garbled bits: OYTCLIPY, @VTGLIPL, OVTCUIPI.
# everythingDDG's blue "e" + DDG stamp reads as eDDG (other frames: DOG, em).
LOGO = re.compile(r"[yv]t[cg][lu1i]|ytclip|^\W*eddg\W*$")   # not bare "clipz": chat usernames like scrapclipz
BAD_WORDS = re.compile(r"tiktok|clips|clipz|subscribe|follow|instagram|youtube\.com|\big:|snapchat|capcut|reels|#fyp|shorts")


_on_gpu = False


def _engine():
    """OCR engine on the GTX 1660 (owner 2026-11-08: keep work off the CPU), CPU if that can't load.
    The GPU onnxruntime lives in clipbot/gpu_libs, separate from the system one, so nothing else changes."""
    global _ocr, _on_gpu
    if _ocr is None:
        import layout
        gpu_dir = Path(__file__).parent / "gpu_libs"
        if gpu_dir.exists() and not layout.blender_running() and str(gpu_dir) not in sys.path:
            sys.path.insert(0, str(gpu_dir))
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"      # device 0 = the 1660, as in nvidia-smi
        from rapidocr_onnxruntime import RapidOCR
        try:
            import onnxruntime as ort
            # Hide "No registered plugin EP device found for 'CUDAExecutionProvider'": harmless, the session
            # still runs on CUDA through the built-in provider (checked 2026-11-08).
            ort.set_default_logger_severity(3)
            if hasattr(ort, "preload_dlls"):
                ort.preload_dlls()                               # CUDA/cuDNN DLLs from the nvidia-* packages
            if "CUDAExecutionProvider" in ort.get_available_providers():
                _ocr = RapidOCR(det_use_cuda=True, cls_use_cuda=True, rec_use_cuda=True)
                _on_gpu = True
        except Exception:
            _ocr = None
        if _ocr is None:
            _ocr = RapidOCR()
    return _ocr


def _read(frame):
    """OCR one frame. If the 1660 errors (out of memory next to Ollama), carry on with the CPU engine."""
    global _ocr, _on_gpu
    try:
        return _engine()(frame)
    except Exception:
        if not _on_gpu:
            raise
        from rapidocr_onnxruntime import RapidOCR
        _ocr, _on_gpu = RapidOCR(), False
        return _ocr(frame)


def _frames(path, n=10):
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    frames = []
    for i in range(n):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * (i + 0.5) / n))
        ok, f = cap.read()
        if ok:
            s = 800 / max(f.shape[:2])
            frames.append(cv2.resize(f, None, fx=s, fy=s) if s < 1 else f)
    cap.release()
    return frames


# Event/sponsor overlays on our creators' own streams (owner 2026-11-06: Red Bull on PlaqueBoyMax's
# Rising Star stream is fine). Matched with spaces removed, so "Red Bull" and "RedBull" both count.
# OCR often cuts or misreads them ("RedBu", "edBull"), so any 5-letter piece of the name counts.
# Rising Star Challenge overlay/set logo reads as RISI, RISIN, RISIAV, STKR (2026-11-06: 5 clips wrongly skipped).
OK_OVERLAYS = ("redbu", "edbul", "dbull", "risi", "ngstar", "stkr", "stak", "challen", "hallenge")


OK_WORDS = ("risingstarchallenge", "redbull")


def _ok_overlay(text):
    flat = re.sub(r"[^a-z]", "", text.lower())
    if any(o in flat for o in OK_OVERLAYS):
        return True
    # Any 3+ letter piece of those names, allowing one misread letter (RIS, RISIAV, STKR).
    if 3 <= len(flat) <= 22:
        for word in OK_WORDS:
            for i in (range(len(word) - len(flat) + 1) if len(flat) >= 4 else [0]):   # 3 letters: only RIS/RED
                if sum(a != b for a, b in zip(flat, word[i:i + len(flat)])) <= (1 if len(flat) >= 4 else 0):
                    return True
    return False


def _is_allowed(word, allowed):
    """Allowed creator name, tolerating OCR misreads like plagueboymax for plaqueboymax."""
    word = word.strip(".,:!")
    name = word.rsplit("/", 1)[-1]          # twitch.tv/lacy, kick.com/n3on
    return any(name == a or (len(name) >= 5 and SequenceMatcher(None, name, a).ratio() >= 0.85) for a in allowed)


def check(path, allowed_handles):
    """Return None if clean, otherwise the reason it was rejected."""
    allowed = {h.lower().lstrip("@") for h in allowed_handles}
    frames = _frames(path)
    if not frames:
        return "could not read video"
    h, w = frames[0].shape[:2]
    seen = []  # (text, cx, cy, box_w) per frame
    logo = 0   # frames with a garbled logo-stamp read (kept even at low OCR confidence)
    for f in frames:
        res, _ = _read(f)
        items = []
        logo += any(score >= 0.45 and LOGO.search(text.lower()) for _, text, score in res or [])
        for box, text, score in res or []:
            if score < 0.6:
                continue
            xs = [p[0] for p in box]
            ys = [p[1] for p in box]
            items.append((text.strip(), sum(xs) / 4 / w, sum(ys) / 4 / h, (max(xs) - min(xs)) / w))
        seen.append(items)
    _last[str(path)] = seen
    if logo >= 1:                 # never seen on a clean clip; the spinning stamp is often only read once
        return "clip-page logo stamp (ytclips / everythingDDG)"

    # Where the Rising Star / Red Bull logo was read correctly at least once: other reads in that same spot are
    # the same logo misread (its neon font reads as RIENR, RENER, RIEINER; owner 2026-11-07).
    spots = [(x, y) for items in seen for t, x, y, _ in items if _ok_overlay(t)]

    def ok_spot(cx, cy):
        return any(abs(cx - x) < 0.08 and abs(cy - y) < 0.08 for x, y in spots)

    def steady(text, cx, cy):
        """Same text in the same spot for at least half the clip (chat and game text change)."""
        hits = sum(any(abs(cx - x) < 0.06 and abs(cy - y) < 0.06 and SequenceMatcher(None, text, t).ratio() > 0.7
                       for t, x, y, _ in items) for items in seen)
        return hits >= max(2, len(seen) // 2)

    for items in seen:
        for text, cx, cy, _ in items:
            low = text.lower()
            if any(_is_allowed(word.lstrip("@"), allowed) for word in low.split()) or _ok_overlay(low) or ok_spot(cx, cy):
                continue
            if STRONG.search(low):                       # TikTok's own watermark bounces around
                return f"watermark text '{text}'"
            if not steady(text, cx, cy):
                continue
            for handle in re.findall(r"(?:^|\s)@\s?([a-z0-9_.]{3,})", low):
                return f"watermark handle @{handle}"
            if BAD_WORDS.search(low):
                return f"watermark text '{text}'"

    # Text that sits still in most frames, is wide and wordy = someone's burned-in headline.
    still = 0
    # OCR often glues caption words together ("N3onIMMEDIATELYEndsStream"), so long text counts as wordy too.
    for text, cx, cy, bw in seen[len(seen) // 2]:
        wordy = len(text.split()) >= 3 or len(re.sub(r"[^A-Za-z]", "", text)) >= 18
        if not wordy or bw < 0.35:
            continue
        hits = sum(any(abs(cx - x) < 0.05 and abs(cy - y) < 0.05 for _, x, y, _ in items) for items in seen)
        if hits >= 0.75 * len(seen):
            still += 1
    if still:
        return "already has a burned-in headline/caption"

    # Owner's rule: ANY text fixed on screen is a watermark (e.g. "VIEWSCEO", no @ needed), except our
    # creator's own Twitch badge/name. Text in the room (a logo on the wall) moves or vanishes when the
    # stream's camera cuts; an overlay stays put across cuts. Single-camera clips: fixed = overlay.
    import layout
    cuts = layout.scene_cuts(path)
    cap = cv2.VideoCapture(str(path))
    dur = (cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1) / (cap.get(cv2.CAP_PROP_FPS) or 30)
    cap.release()
    shot_of = [sum(c <= dur * (i + 0.5) / len(seen) for c in cuts) for i in range(len(seen))]
    checked = set()
    for items in seen:
        for text, cx, cy, _ in items:
            t = text.strip()
            key = (t.lower(), round(cx, 1), round(cy, 1))
            if key in checked or len(re.sub(r"[^A-Za-z]", "", t)) < 3:
                continue
            checked.add(key)
            if any(_is_allowed(word.lstrip("@"), allowed) for word in t.lower().split()) or _ok_overlay(t) or ok_spot(cx, cy):
                continue
            frames = [i for i, its in enumerate(seen)
                      if any(abs(cx - x) < 0.05 and abs(cy - y) < 0.05 and SequenceMatcher(None, t.lower(), u.lower()).ratio() > 0.7
                             for u, x, y, _ in its)]
            shots = {shot_of[i] for i in frames}
            if len(frames) >= max(3, 0.6 * len(seen)) and (not cuts or len(shots) >= 2):
                return f"watermark/overlay text '{t}'"
    return None


def rising_star(path):
    """True if the OCR from check() saw the Rising Star overlay/set logo in 2+ frames."""
    seen = _last.get(str(path)) or []
    keys = ("risi", "ngstar", "stkr", "stak", "challen", "hallenge")
    return sum(any(k in re.sub(r"[^a-z]", "", t.lower()) for t, *_ in items for k in keys) for items in seen) >= 2


def shows_name(path, names):
    """True if the clip already has one of these names on screen (e.g. the creator's own
    Twitch badge from their clipping overlay), so we don't add a second one."""
    seen = _last.get(str(path)) or []
    want = [n.lower().replace(" ", "") for n in names]
    frames = sum(any(_is_allowed(t.lower().replace(" ", "").lstrip("@"), want) for t, *_ in items)
                 for items in seen)
    return frames >= 2


def has_platform_bar(path):
    """True if the clip already shows a Kick/Twitch channel bar (kick.com/name, twitch.tv/name)."""
    seen = _last.get(str(path)) or []
    return sum(any(re.search(r"kick\.com|twitch\.tv", t.lower()) for t, *_ in items) for items in seen) >= 2


def foreign_bar(path, allowed_handles):
    """Someone else's Kick/Twitch channel bar (e.g. kick.com/cuffem).
    Returns ("bottom", y) or ("top", y) as a fraction of the height to cut at, "elsewhere" if it
    can't be cropped off, or None if there's no foreign bar."""
    allowed = {h.lower().lstrip("@") for h in allowed_handles}
    seen = _last.get(str(path)) or []
    ys = []
    for items in seen:
        for t, cx, cy, _ in items:
            m = re.search(r"(?:kick\.com|twitch\.tv)/([a-z0-9_]+)", t.lower().replace(" ", ""))
            if m and not _is_allowed(m.group(1), allowed):
                ys.append(cy)
    if len(ys) < 2:
        return None
    y = float(sorted(ys)[len(ys) // 2])
    if y > 0.8:
        return ("bottom", y - 0.06)       # cut a little above the text to take the logo strip too
    if y < 0.2:
        return ("top", y + 0.06)
    return "elsewhere"


def name_box(path, names):
    """Where the creator's own name badge sits in the source (pixels): (name, x, y, w, h), padded to
    cover the Twitch/Kick icon left of the text. None if no steady badge is found."""
    want = [n.lower().replace(" ", "") for n in names]
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    boxes = []
    for i in range(5):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * (i + 0.5) / 5))
        ok, f = cap.read()
        if not ok:
            continue
        for box, text, score in _read(f)[0] or []:
            flat = text.lower().replace(" ", "").lstrip("@")
            hit = next((n for n in want if _is_allowed(flat, [n])), None)
            if hit and score >= 0.6:
                xs, ys = [p[0] for p in box], [p[1] for p in box]
                boxes.append((hit, min(xs), min(ys), max(xs), max(ys)))
    fh, fw = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap.release()
    if len(boxes) < 3:
        return None
    name = max(set(b[0] for b in boxes), key=[b[0] for b in boxes].count)
    boxes = [b for b in boxes if b[0] == name]
    x0, y0, x1, y1 = (sorted(b[k] for b in boxes)[len(boxes) // 2] for k in (1, 2, 3, 4))
    # A badge sits still; the name in chat or a headline moves around. Need 3+ reads in the same spot.
    steady = [b for b in boxes if abs((b[1] + b[3]) / 2 - (x0 + x1) / 2) < 0.04 * fw
              and abs((b[2] + b[4]) / 2 - (y0 + y1) / 2) < 0.04 * fh]
    if len(steady) < 3:
        return None
    th = y1 - y0
    x0, x1 = max(1, x0 - 1.8 * th), min(fw - 2, x1 + 0.5 * th)      # the icon sits left of the name
    y0, y1 = max(1, y0 - 0.45 * th), min(fh - 2, y1 + 0.45 * th)
    return name, int(x0), int(y0), int(x1 - x0), int(y1 - y0)
