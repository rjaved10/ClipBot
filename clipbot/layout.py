"""Turn a clip into a 1080x1920 TikTok video in the clip-page style.

- Reaction (wide video with a small facecam): facecam on the top half, what they're
  watching on the bottom half, headline on the seam.
- Single (anything else): the clip in the middle over a blurred copy of itself,
  headline over its top edge. Already-vertical clips fill the screen.
"""
import nowin  # noqa: F401  (no console windows for child processes)
import os
import re
import subprocess
import urllib.request
from pathlib import Path

import cv2

cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)   # hide the harmless dnn WARN line
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).parent
ASSETS = ROOT / "assets"
W, H = 1080, 1920
HALF = H // 2

_det = None


def _detector(w, h):
    global _det
    if _det is None:
        _det = cv2.FaceDetectorYN.create(str(ASSETS / "face_yunet.onnx"), "", (w, h), 0.7)
    _det.setInputSize((w, h))
    return _det


def _sample(path, n=12):
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    frames = []
    for i in range(n):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * (i + 0.5) / n))
        ok, f = cap.read()
        if ok:
            frames.append(f)
    cap.release()
    return frames


def _faces(frame):
    h, w = frame.shape[:2]
    s = 640 / max(w, h)
    small = cv2.resize(frame, (int(w * s), int(h * s)))
    _, res = _detector(small.shape[1], small.shape[0]).detect(small)
    if res is None:
        return []
    return [(x / s, y / s, fw / s, fh / s) for x, y, fw, fh, *_ in res]


def _cam_border(gray, fx, fy, fw, fh):
    """Find the facecam's rectangle edges around a face from strong straight lines."""
    h, w = gray.shape
    gx = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
    gy = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
    r0, r1 = int(max(0, fy - fh)), int(min(h, fy + 2 * fh))
    c0, c1 = int(max(0, fx - fw)), int(min(w, fx + 2 * fw))

    def pick(profile, lo, hi, default):
        lo, hi = int(max(0, lo)), int(min(len(profile), hi))
        if hi - lo < 3:
            return default
        seg = profile[lo:hi]
        i = int(np.argmax(seg))
        return lo + i if seg[i] > 3 * (np.median(profile) + 1e-3) else default

    col = gx[r0:r1].mean(axis=0)
    row = gy[:, c0:c1].mean(axis=1)
    left = pick(col, fx - 4 * fw, fx - 0.4 * fw, 0)
    right = pick(col, fx + 1.4 * fw, fx + 5 * fw, w)
    top = pick(row, fy - 3 * fh, fy - 0.3 * fh, 0)
    bottom = pick(row, fy + 1.4 * fh, fy + 4 * fh, h)
    return left, top, right, bottom


def _fit(box, aspect, frame_w, frame_h, anchor=None):
    """Largest crop of `aspect` (w/h) inside box, centred on anchor if given."""
    l, t, r, b = box
    bw, bh = r - l, b - t
    if bw / bh > aspect:
        cw, ch = bh * aspect, bh
    else:
        cw, ch = bw, bw / aspect
    ax, ay = anchor if anchor else ((l + r) / 2, (t + b) / 2)
    x = min(max(ax - cw / 2, l), r - cw)
    y = min(max(ay - ch / 2, t), b - ch)
    cw, ch = int(cw) // 2 * 2, int(ch) // 2 * 2
    x = int(max(0, min(x, frame_w - cw)))
    y = int(max(0, min(y, frame_h - ch)))
    return x, y, cw, ch


def _analyse(path):
    frames = _sample(path)
    if not frames:
        raise RuntimeError("unreadable video")
    fh_, fw_ = frames[0].shape[:2]
    if fh_ > fw_ * 1.2:
        return {"mode": "vertical", "w": fw_, "h": fh_,
                "faces": [f for fr in frames for f in _faces(fr)]}

    per_frame = [_faces(f) for f in frames]
    all_faces = [f for fs in per_frame for f in fs]

    # Facecam = a small face that stays in about the same place for most of the clip. There can be several
    # steady small faces (DDG's cam + someone sitting still in the video he reacts to): try each spot.
    small = [f for f in all_faces if f[2] < 0.13 * fw_]
    spots = []
    for f in sorted(small, key=lambda f: f[2]):
        x, y = f[0] + f[2] / 2, f[1] + f[3] / 2
        if not any(abs(x - sx) < 0.05 * fw_ and abs(y - sy) < 0.05 * fh_ for sx, sy in spots):
            spots.append((x, y))
    median = None
    for sx, sy in spots:
        steady = [f for f in small if abs(f[0] + f[2] / 2 - sx) < 0.05 * fw_ and abs(f[1] + f[3] / 2 - sy) < 0.05 * fh_]
        mx = float(np.median([f[0] + f[2] / 2 for f in steady]))
        my = float(np.median([f[1] + f[3] / 2 for f in steady]))
        frames_with = sum(any(abs(f[0] + f[2] / 2 - mx) < 0.05 * fw_ and abs(f[1] + f[3] / 2 - my) < 0.05 * fh_
                              for f in fs) for fs in per_frame)
        # MORE than half the clip: lower let a face in a screenshot of an X post pass as a webcam (FaZe Banks
        # clip); exactly half was people standing in a wide IRL shot (NitroCamden on Max's stream, 2026-11-06).
        if frames_with <= 0.5 * len(frames):
            continue
        # A facecam sits off to a side or corner; a face in the middle is just the main shot.
        # Middle third across counts as the main shot whatever the height (2026-11-06: Max standing / Cuffem
        # sitting mid-room on the Rising Star set got the split). Real cams so far all sit in the outer quarter.
        # Tightened to the outer quarter: Coi Leray sitting at 72% across got the split (2026-11-06).
        # Real cams: DDG 10%, Lacy 13%, Jason 78-82%.
        if 0.25 * fw_ < mx < 0.75 * fw_:
            continue
        fx, fy, fw, fh = np.median(np.array(steady), axis=0)
        # People sitting side by side in one room (Max, Lil Yachty, Cuffem on the Rising Star set, 2026-11-06):
        # the still one in the middle isn't a webcam if, in most frames, someone else at least as big sits
        # level with him (same height on screen) on BOTH sides. A facecam sits off to one side of the video it
        # reacts to, so a face level with it on one side only is normal (Jason's cam next to Jshock's mukbang).
        def level(fs):
            near = [o[0] + o[2] / 2 - mx for o in fs if o[2] >= 0.6 * fw and abs(o[1] + o[3] / 2 - my) < 1.5 * fh]
            return any(d < -1.5 * fw for d in near) and any(d > 1.5 * fw for d in near)
        if sum(level(fs) for fs in per_frame) > 0.5 * len(frames):
            continue
        if median is None:
            median = np.median(np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]), axis=0).astype(np.uint8)
        l, t, r, b = _cam_border(median, fx, fy, fw, fh)
        # A real facecam is a box: at least 2 of its edges sit inside the frame, and it's small.
        inside = sum([l > 0.02 * fw_, t > 0.02 * fh_, r < 0.98 * fw_, b < 0.98 * fh_])
        if inside < 2 or (r - l) * (b - t) > 0.3 * fw_ * fh_:
            continue
        # Never let the cam crop grow past a sensible size around the face.
        l, r = max(l, fx - 4 * fw), min(r, fx + 5 * fw)
        t, b = max(t, fy - 3 * fh), min(b, fy + 4 * fh)
        cam = _fit((l, t, r, b), W / HALF, fw_, fh_, (fx + fw / 2, fy + fh / 2))
        # Lines INSIDE the cam (LED strips, a door frame) can fool the border finder into a tiny box and a
        # 10x zoom: never crop tighter than ~5.5 face widths (about the whole cam on a normal stream layout).
        min_w = 5.5 * fw
        if cam[2] < min_w:
            cw_ = min(min_w, fw_)
            ch_ = min(cw_ * HALF / W, fh_)
            cam = _fit((max(0, fx + fw / 2 - cw_ / 2), max(0, fy + fh / 2 - ch_ / 2),
                        min(fw_, fx + fw / 2 + cw_ / 2), min(fh_, fy + fh / 2 + ch_ / 2)),
                       W / HALF, fw_, fh_, (fx + fw / 2, fy + fh / 2))
            if cam[2] < 0.8 * min_w:     # cam squashed in a corner: shift the box inward instead
                cw_, ch_ = int(min_w) // 2 * 2, int(min_w * HALF / W) // 2 * 2
                cam = (int(max(0, min(fx + fw / 2 - cw_ / 2, fw_ - cw_))),
                       int(max(0, min(fy + fh / 2 - ch_ / 2, fh_ - ch_))), cw_, ch_)
        # Content: the rest of the frame, pushed away from the side the cam is on.
        ccw = int(min(fw_, fh_ * W / HALF)) // 2 * 2
        cam_left = mx < fw_ / 2
        cx0 = fw_ - ccw if cam_left else 0
        content = (cx0, 0, ccw, int(ccw * HALF / W) // 2 * 2)
        content = (content[0], max(0, (fh_ - content[3]) // 2), content[2], min(content[3], fh_ // 2 * 2))
        # (No overlap check any more: the bottom shows the whole frame, cam included, so Lacy's wide cam is fine.)
        return {"mode": "reaction", "w": fw_, "h": fh_, "cam": cam, "content": content, "faces": all_faces}

    return _single(fw_, fh_, all_faces)


def _main_faces(faces):
    """Drop small background faces (people at other tables, crowds): keep ones near the biggest size."""
    if not faces:
        return []
    big = float(np.percentile([f[2] for f in faces], 90))
    return [f for f in faces if f[2] >= 0.5 * big]


def _group_crop(faces, fw_, fh_):
    """Tightest crop that still shows every main person: 3:4 when they fit, wider only as much as needed."""
    faces = _main_faces(faces)
    if not faces:
        return _fit((0, 0, fw_, fh_), 3 / 4, fw_, fh_, (fw_ / 2, fh_ / 2))
    lefts = np.percentile([f[0] for f in faces], 5)
    rights = np.percentile([f[0] + f[2] for f in faces], 95)
    pad = 0.6 * float(np.median([f[2] for f in faces]))     # a little room for shoulders
    left, right = max(0, lefts - pad), min(fw_, rights + pad)
    aspect = max(3 / 4, min(fw_, right - left) / fh_)
    return _fit((0, 0, fw_, fh_), aspect, fw_, fh_, ((left + right) / 2, fh_ / 2))


def _single(fw_, fh_, all_faces):
    """Crop to 3:4 around where the people are (wider only if they don't all fit)."""
    crop = _group_crop(all_faces, fw_, fh_)
    return {"mode": "single", "w": fw_, "h": fh_, "crop": crop, "faces": all_faces}


def cam_segments(path, info, step=0.5, min_run=2.0):
    """The streamer can switch scenes mid-clip (reaction view -> full-screen cam). Returns
    [(start, end, top_crop)]: the facecam box while the cam face is in it, otherwise a crop around the
    big face on screen. None when the layout never changes."""
    fw_, fh_ = info["w"], info["h"]
    x, y, w, h = info["cam"]
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    dur = total / fps
    labels, big = [], []
    t = 0.0
    while t < dur:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ok, f = cap.read()
        if not ok:
            break
        fs = _faces(f)
        in_cam = any(x <= fx + fw / 2 <= x + w and y <= fy + fh / 2 <= y + h and fw < 0.13 * fw_
                     for fx, fy, fw, fh in fs)
        large = [ff for ff in fs if ff[2] >= 0.08 * fw_]
        labels.append(in_cam or not large)          # no big face either: keep the cam view
        big.append(max(large, key=lambda ff: ff[2]) if large else None)
        t += step
    cap.release()
    # Smooth: a short blip (face missed, hand over face) doesn't count as a scene switch.
    n, k = len(labels), max(1, int(min_run / step))
    runs, i = [], 0
    while i < n:
        j = i
        while j < n and labels[j] == labels[i]:
            j += 1
        runs.append([labels[i], i, j])
        i = j
    merged = []
    for r in runs:
        if merged and (r[2] - r[1] < k or merged[-1][0] == r[0]):
            merged[-1][2] = r[2]
        else:
            merged.append(r)
    if len(merged) < 2:
        return None
    cuts = scene_cuts(path)
    segs = []
    for cam_view, a, b in merged:
        start, end = a * step, min(dur, b * step)
        if segs:   # snap the switch to the real scene cut nearby
            near = [c for c in cuts if abs(c - start) < 1.0]
            if near:
                start = min(near, key=lambda c: abs(c - start))
                segs[-1][1] = start
        if cam_view:
            crop = info["cam"]
        else:
            fs = [ff for ff in big[a:b] if ff]
            fx, fy, fw, fh = np.median(np.array(fs), axis=0)
            cw_ = min(fw_, 4.5 * fw, fh_ * W / HALF)
            ch_ = cw_ * HALF / W
            crop = _fit((max(0, fx + fw / 2 - cw_ / 2), max(0, fy + fh / 2 - ch_ * 0.45),
                         min(fw_, fx + fw / 2 + cw_ / 2), min(fh_, fy + fh / 2 + ch_ * 0.55)),
                        W / HALF, fw_, fh_, (fx + fw / 2, fy + fh / 2 + 0.05 * ch_))
        segs.append([start, end, crop])
    segs[-1][1] = dur
    return [tuple(s) for s in segs]


def face_start(path, step=0.5, big=0.08):
    """Seconds until a big face is first on screen (2 looks in a row), e.g. after a screen-share of an X
    post. 0 if a face is there from the start, None if no big face ever shows (cam-only reactions)."""
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    t, prev, first = 0.0, None, None
    while t < total / fps:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ok, f = cap.read()
        if not ok:
            break
        has = any(ff[2] >= big * f.shape[1] for ff in _faces(f))
        if has and prev is not None:
            first = prev
            break
        prev = t if has else None
        t += step
    cap.release()
    if first is None or first <= 0:
        return None if first is None else 0.0
    cuts = [c for c in scene_cuts(path) if first - 1.0 <= c <= first + step]
    return max(cuts) if cuts else first


def trim_start(src, start, dest):
    """Copy of the clip from `start` seconds (re-encoded so the cut is frame-exact)."""
    encode(lambda v: ["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{start:.3f}", "-i", str(src), *v,
                      "-c:a", "aac", "-b:a", "192k", str(dest)], 16)
    return dest


def analyse(path):
    info = _analyse(path)
    if info["mode"] == "reaction":
        try:
            info["segments"] = cam_segments(path, info)
        except Exception:
            info["segments"] = None
    # Owner 2026-11-06: ONLY two layouts. The normal one is the 'Max scolds NitroCamden' look: one steady
    # crop on everyone, centred over blur, headline on top, no zooming in on whoever talks. So no
    # speaker_shots any more. The other is the reaction split (PC reactions only).
    return info


# ---------- shot framing: wide for groups, tight 3:4 on the stream's own close-ups ----------

STEP = 1 / 6          # seconds between looks at the video
WINDOW = 0.5          # decide the shot every half second
MIN_SHOT = 2.0        # never cut faster than this


def _people(faces, frame_w):
    """Group face sightings into people by where they sit left-to-right. Returns centre x per person."""
    xs = sorted(f[0] + f[2] / 2 for f in faces)
    if not xs:
        return []
    gap = float(np.median([f[2] for f in faces])) * 1.2
    groups = [[xs[0]]]
    for x in xs[1:]:
        if x - groups[-1][-1] > gap:
            groups.append([x])
        else:
            groups[-1].append(x)
    # ignore one-off sightings (passers-by, false detections)
    return [float(np.mean(g)) for g in groups if len(g) >= max(2, 0.15 * len(xs) / len(groups))]


def _audio_rms(path, step):
    pcm = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", "16000", "-f", "s16le", "-"],
                         capture_output=True).stdout
    a = np.frombuffer(pcm, np.int16).astype(np.float32) / 32768
    hop = int(16000 * step)
    return np.array([np.sqrt(np.mean(a[i:i + hop] ** 2)) for i in range(0, max(1, len(a) - hop), hop)])


def speaker_shots(path, info):
    """[(start, end, crop)]: wide (everyone) while several people are on screen, a tight 3:4 while the
    stream itself is on a close-up of one person. None if there's only one person or nothing to cut."""
    fw_, fh_ = info["w"], info["h"]
    wide = info["crop"]
    if wide[2] <= fh_ * 3 / 4 + 4:                      # already tight: nothing to cut between
        return None
    people = _people(info.get("faces", []), fw_)
    if len(people) < 2:
        return None

    cuts = scene_cuts(path)
    if not cuts:
        return None                                      # one continuous camera: keep the wide shot

    # Each camera shot of the stream gets one framing, switching on the exact frame the stream cuts:
    # one person in shot = tight 3:4 on them, several = wide so everyone's in frame.
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    bounds = [0.0, *cuts, total / fps]
    labels, groups = [], []
    for t0, t1 in zip(bounds, bounds[1:]):
        seen, shot_faces = set(), []
        for k in range(3):                               # look at a few frames inside this shot
            cap.set(cv2.CAP_PROP_POS_FRAMES, int((t0 + (t1 - t0) * (k + 1) / 4) * fps))
            ok, frame = cap.read()
            if not ok:
                continue
            for fx, fy, fw, fh in _faces(frame):
                shot_faces.append((fx, fy, fw, fh))
                cx = fx + fw / 2
                pid = min(range(len(people)), key=lambda i: abs(people[i] - cx))
                if abs(people[pid] - cx) < max(fw, 0.08 * fw_):
                    seen.add(pid)
        labels.append(next(iter(seen)) if len(seen) == 1 else (None if seen else (labels[-1] if labels else None)))
        # Group shots are framed on just the people in THIS camera shot, so they zoom in when they can.
        groups.append(_group_crop(shot_faces, fw_, fh_) if len(seen) > 1 else None)
    cap.release()

    crops = {p: _fit((0, 0, fw_, fh_), 3 / 4, fw_, fh_, (people[p], fh_ / 2)) for p in range(len(people))}
    shots = []
    for (t0, t1), lab, grp in zip(zip(bounds, bounds[1:]), labels, groups):
        crop = (grp or wide) if lab is None else crops[lab]
        if shots and shots[-1][2] == crop:
            shots[-1] = (shots[-1][0], t1, crop)         # same framing as the shot before: no cut
        else:
            shots.append((t0, t1, crop))
    if len(shots) == 1 and shots[0][2] == wide:
        return None
    shots[-1] = (shots[-1][0], None, shots[-1][2])
    return shots


def _thumbs_gpu(path):
    """Every frame as a 64x36 BGR thumbnail, decoded and shrunk on the GTX 1660 (owner 2026-11-08: off the
    CPU; decoding full 1080p60 on the CPU took ~16 s a clip). None if the GPU can't do it."""
    gpu = None if blender_running() else gpu_index()
    if gpu is None:
        return None
    try:
        res = subprocess.run(["ffmpeg", "-v", "error", "-hwaccel", "cuda", "-hwaccel_output_format", "cuda",
                              "-hwaccel_device", str(gpu), "-i", str(path), "-an",
                              "-vf", "scale_cuda=64:36,hwdownload,format=nv12,format=bgr24", "-fps_mode", "passthrough",
                              "-f", "rawvideo", "-"], capture_output=True, timeout=300,
                             env={**os.environ, "CUDA_DEVICE_ORDER": "PCI_BUS_ID"})
    except Exception:
        return None
    n = len(res.stdout) // (64 * 36 * 3)
    if res.returncode != 0 or n < 2:
        return None
    return np.frombuffer(res.stdout[:n * 64 * 36 * 3], np.uint8).reshape(n, 36, 64, 3)


def scene_cuts(path):
    """Times (seconds) where the stream's own camera cuts, found from big jumps between frames."""
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    thumbs = _thumbs_gpu(path)
    if thumbs is not None:
        cap.release()
    frames = iter(thumbs) if thumbs is not None else None
    prev, diffs = None, []
    while True:
        if frames is not None:
            small = next(frames, None)
            if small is None:
                break
        else:
            ok, frame = cap.read()
            if not ok:
                break
            small = cv2.resize(frame, (64, 36))
        small = cv2.cvtColor(np.ascontiguousarray(small), cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([small], [0, 1], None, [16, 8], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        gray = small[:, :, 2].astype(np.float32)
        if prev is not None:
            hd = cv2.compareHist(prev[0], hist, cv2.HISTCMP_BHATTACHARYYA)
            pd = float(np.mean(np.abs(gray - prev[1]))) / 255
            diffs.append(hd + pd)
        else:
            diffs.append(0.0)
        prev = (hist, gray)
    cap.release()
    d = np.array(diffs)
    if len(d) < 10:
        return []
    thresh = max(0.35, float(np.median(d)) * 6)
    cuts, last = [], -1e9
    for i, v in enumerate(d):
        t = i / fps
        if v > thresh and t - last > 0.4 and 0.3 < t < len(d) / fps - 0.3:
            cuts.append(t)
            last = t
    return cuts


# ---------- headline (white boxes, black TikTok Sans, emoji at the end) ----------

def _font(size):
    f = ImageFont.truetype(str(ASSETS / "TikTokSans.ttf"), size)
    try:
        f.set_variation_by_axes([36, 104, 760, 0])  # opsz, wdth, wght, slnt
    except Exception:
        try:
            f.set_variation_by_name("ExtraBold")
        except Exception:
            pass
    return f


def _emoji_img(ch, size):
    code = "-".join(f"{ord(c):x}" for c in ch if ord(c) != 0xFE0F)
    path = ASSETS / "emoji" / f"{code}.png"
    if not path.exists():
        try:
            urllib.request.urlretrieve(f"https://cdn.jsdelivr.net/gh/jdecked/twemoji@15.1.0/assets/72x72/{code}.png", path)
        except Exception:
            return None
    return Image.open(path).convert("RGBA").resize((size, size), Image.LANCZOS)


def _wrap(text, font, max_w):
    lines, cur = [], ""
    for word in text.split():
        trial = f"{cur} {word}".strip()
        if font.getlength(trial) <= max_w or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


def headline_png(text, emoji, out_path, size=52, max_w=860):
    font = _font(size)
    # The font has no emoji: one the AI slips into the text itself draws as an empty box.
    text = re.sub(r"\s+", " ", re.sub(r"[\U0001F000-\U0001FFFF☀-➿⬀-⯿️‍]", "", text)).strip()
    lines = _wrap(text.upper(), font, max_w - (size + 10 if emoji else 0))
    pad_x, pad_y, line_h, radius = 16, 6, int(size * 1.08), 14
    img = Image.new("RGBA", (W, line_h * len(lines) + pad_y * 2 + 8), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    for i, line in enumerate(lines):
        last = i == len(lines) - 1
        tw = font.getlength(line) + (size * 0.25 + size if last and emoji else 0)
        x0 = (W - tw) / 2
        y0 = pad_y + i * line_h
        d.rounded_rectangle([x0 - pad_x, y0 - pad_y, x0 + tw + pad_x, y0 + line_h + pad_y], radius, fill="white")
    for i, line in enumerate(lines):
        last = i == len(lines) - 1
        tw = font.getlength(line) + (size * 0.25 + size if last and emoji else 0)
        x0 = (W - tw) / 2
        y0 = pad_y + i * line_h
        d.text((x0, y0 + line_h / 2), line, font=font, fill="black", anchor="lm")
        if last and emoji:
            em = _emoji_img(emoji, size)
            if em:
                img.alpha_composite(em, (int(x0 + font.getlength(line) + size * 0.25), int(y0 + (line_h - size) / 2)))
    img.save(out_path)
    return img.size


# ---------- render ----------

BADGE_W = 540
HL_MIN_Y = 230      # below TikTok's LIVE / Following / For You bar
HL_TOP = 160        # highest the headline goes (just under TikTok's tabs). Owner wants this look (2026-11-06): moving it to 250 pushed tall clips off-centre.
HL_OVERLAP = 12     # the headline's bottom edge just touches the clip's top edge


def _faces_out(info, crop, dest, hair=0.35):
    """Faces (top, bottom) in output pixels for a source crop drawn at dest = (x, y, w, h).
    The box is stretched up by `hair` (share of the face height) to cover hair and hats."""
    x, y, w, h = crop
    dx, dy, dw, dh = dest
    out = []
    for fx, fy, fw, fh in info.get("faces", []):
        if x <= fx + fw / 2 <= x + w and y <= fy + fh / 2 <= y + h:
            t = dy + (fy - hair * fh - y) * dh / h
            b = dy + (fy + fh - y) * dh / h
            out.append((max(dy, t), min(dy + dh, b)))
    return out


def _place(want, hl_h, faces, lo, hi):
    """Headline y nearest `want` (between lo and hi) that covers no face; None if there's no such spot."""
    def free(yy):
        return all(yy + hl_h <= ft or yy >= fb for ft, fb in faces)
    for d in range(0, 2000, 10):
        for yy in (want - d, want + d):
            if lo <= yy <= hi and free(yy):
                return yy
    return None


def badge_for(creators):
    """The creator's Twitch name badge, if we have one (only some creators do)."""
    for k in creators:
        path = ASSETS / "badges" / f"{k}.png"
        if path.exists():
            return path
    return None


def _render_mixed(src, info, headline, emoji, out_path, work_dir, badge):
    """Owner 2026-11-06: the split only while they react to something on the PC. When the stream switches
    to the full camera, that stretch goes back to the normal layout (clip centred over blur), then back
    to the split when the PC view returns. Each stretch is rendered on its own and joined."""
    work = Path(work_dir) / "mixed"
    work.mkdir(parents=True, exist_ok=True)
    parts = []
    for i, (a, b, c) in enumerate(info["segments"]):
        piece = work / f"src{i}.mp4"
        encode(lambda v: ["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{a:.3f}", "-to", f"{b:.3f}", "-i", str(src),
                          *v, "-c:a", "aac", "-b:a", "192k", str(piece)], 16)
        out = work / f"out{i}.mp4"
        if tuple(c) == tuple(info["cam"]):
            render(piece, dict(info, segments=None), headline, emoji, out, work, badge)
        else:
            faces = [f for fr in _sample(piece) for f in _faces(fr)]
            render(piece, _single(info["w"], info["h"], faces), headline, emoji, out, work, badge)
        parts.append(out)
    lst = work / "list.txt"
    lst.write_text("".join(f"file '{p.as_posix()}'\n" for p in parts), encoding="utf-8")
    encode(lambda v: ["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst),
                      *v, "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart",
                      str(out_path)], 18)
    return out_path


def render(src, info, headline, emoji, out_path, work_dir, badge=None):
    m = info["mode"]
    segs = info.get("segments")
    if m == "reaction" and segs and any(tuple(c) != tuple(info["cam"]) for _, _, c in segs):
        return _render_mixed(src, info, headline, emoji, out_path, work_dir, badge)
    hl = Path(work_dir) / "headline.png"
    _, hl_h = headline_png(headline, emoji, hl)
    if m == "reaction":
        x, y, w, h = info["cam"]
        # The thing they're reacting to: the WHOLE frame, fitted to the width (no side cut off), over a blur.
        fw_, fh_ = info["w"], info["h"]
        vh = min(HALF, int(W * fh_ / fw_) // 2 * 2)
        vy = HALF + (HALF - vh) // 2
        faces = _faces_out(info, (x, y, w, h), (0, 0, W, HALF)) + _faces_out(info, (0, 0, fw_, fh_), (0, vy, W, vh))
        hl_y = _place(HALF - hl_h // 2, hl_h, faces, HL_MIN_Y, 1500) or HALF - hl_h // 2
        # Facecams are tiny (DDG's: 428x240) and get zoomed ~4x: denoise, lanczos, then sharpen so it isn't mush.
        def top_chain(c):
            cx_, cy_, cw_, ch_ = (int(v) // 2 * 2 for v in c)
            sharp = ",unsharp=5:5:0.8:5:5:0" if W / cw_ > 2 else ""
            return f"crop={cw_}:{ch_}:{cx_}:{cy_},hqdn3d=2:2:3:3,scale={W}:{HALF}:flags=lanczos{sharp},setsar=1"
        segs = info.get("segments")
        if segs:
            # Scene switches: each stretch gets its own top crop (cam box, or the full-screen face).
            n = len(segs)
            parts = "".join(f"[a{i}]trim=start={a:.3f}:end={b:.3f},setpts=PTS-STARTPTS,{top_chain(c)}[t{i}];"
                            for i, (a, b, c) in enumerate(segs))
            top = (f"[0:v]split=2[a][b];[a]split={n}" + "".join(f"[a{i}]" for i in range(n)) + ";" + parts
                   + "".join(f"[t{i}]" for i in range(n)) + f"concat=n={n}:v=1:a=0[top];")
            faces = sum((_faces_out(info, c, (0, 0, W, HALF)) for _, _, c in segs), []) + faces[len(_faces_out(info, (x, y, w, h), (0, 0, W, HALF))):]
            hl_y = _place(HALF - hl_h // 2, hl_h, faces, HL_MIN_Y, 1500) or HALF - hl_h // 2
        else:
            top = f"[0:v]split[a][b];[a]{top_chain((x, y, w, h))}[top];"
        graph = (top +
                 f"[b]split[b1][b2];[b1]scale={W}:{HALF}:force_original_aspect_ratio=increase,crop={W}:{HALF},"
                 f"boxblur=20:2,eq=brightness=-0.08,setsar=1[bg];[b2]scale={W}:{vh}:flags=lanczos,setsar=1[fg];"
                 f"[bg][fg]overlay=0:{vy - HALF}[bot];"
                 f"[top][bot]vstack[v];[v][1:v]overlay=0:{hl_y}[hv]")
        clip_bottom = 1760          # keep clear of TikTok's caption area
    elif m == "single":
        return _render_single(src, info, hl, hl_h, out_path, badge)
    else:
        sc = max(W / info["w"], H / info["h"])
        vis = ((info["w"] * sc - W) / 2 / sc, (info["h"] * sc - H) / 2 / sc, W / sc, H / sc)
        hl_y = _place(330, hl_h, _faces_out(info, vis, (0, 0, W, H)), HL_MIN_Y, 900) or 330
        graph = (f"[0:v]scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},setsar=1[v];"
                 f"[v][1:v]overlay=0:{hl_y}[hv]")
        clip_bottom = 1760
    inputs = ["-i", str(src), "-i", str(hl)]
    if badge:
        from PIL import Image
        bw, bh = Image.open(badge).size
        b_h = int(BADGE_W * bh / bw)
        inputs += ["-i", str(badge)]
        graph += (f";[2:v]scale={BADGE_W}:{b_h}[bd];"
                  f"[hv][bd]overlay=(W-w)/2:{clip_bottom - b_h - 28}[out]")
    else:
        graph += ";[hv]null[out]"
    return _encode(inputs, graph, out_path)


def _render_single(src, info, hl, hl_h, out_path, badge):
    """Clip in the middle over a blurred copy of itself. With "shots", it cuts between a wide shot
    (everyone) and a tight 3:4 on the middle person while they talk; the headline never moves."""
    shots = info.get("shots") or [(0.0, None, info["crop"])]
    # Like yakclipz: the headline sits right ON TOP of the clip (bottom edge touching the clip's top
    # edge), so it never covers anyone. The clip is as close to the middle as that allows.
    sizes = [[W, min(int(W * h / w) // 2 * 2, 1900)] for _, _, (x, y, w, h) in shots]
    tall = max(fh for _, fh in sizes)
    top = max((H - tall) // 2, HL_TOP + hl_h - HL_OVERLAP) // 2 * 2
    hl_y = top - hl_h + HL_OVERLAP
    centre = top + tall // 2
    # Shorter (wide) shots sit on the same middle line as the tall one.
    tops = [max(top, centre - fh // 2) // 2 * 2 for _, fh in sizes]

    n = len(shots)
    inputs = ["-i", str(src), "-i", str(hl)]
    parts = [f"[0:v]fps=30,split={n}" + "".join(f"[s{i}]" for i in range(n))]
    if badge:
        from PIL import Image
        bw, bh = Image.open(badge).size
        b_h = int(BADGE_W * bh / bw)
        inputs += ["-i", str(badge)]
        parts.append(f"[2:v]scale={BADGE_W}:{b_h},split={n}" + "".join(f"[bd{i}]" for i in range(n)))
    for i, (t0, t1, (x, y, w, h)) in enumerate(shots):
        fw, fh = sizes[i]
        sy = tops[i]
        end = f":end={t1:.3f}" if t1 is not None else ""
        parts.append(f"[s{i}]trim=start={t0:.3f}{end},setpts=PTS-STARTPTS,split[a{i}][b{i}]")
        parts.append(f"[a{i}]scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},"
                     f"boxblur=24:2,eq=brightness=-0.12[bg{i}]")
        parts.append(f"[b{i}]crop={w}:{h}:{x}:{y},scale={fw}:{fh},setsar=1[fg{i}]")
        parts.append(f"[bg{i}][fg{i}]overlay={(W - fw) // 2}:{sy}" + ("[v%d]" % i if badge else "[c%d]" % i))
        if badge:
            parts.append(f"[v{i}][bd{i}]overlay=(W-w)/2:{min(sy + fh, 1760) - b_h - 28}[c{i}]")
    parts.append("".join(f"[c{i}]" for i in range(n)) + f"concat=n={n}:v=1:a=0[cv]")
    parts.append(f"[cv][1:v]overlay=0:{hl_y}[out]")
    return _encode(inputs, ";".join(parts), out_path)


def encode(build, crf):
    """Run ffmpeg command build(video_codec_args) on the GTX 1660's NVENC chip, or on the CPU (x264) when
    Blender is open or NVENC fails. Owner 2026-11-08: keep the encoding off the CPU."""
    gpu = None if blender_running() else gpu_index()
    if gpu is not None:
        nvenc = ["-c:v", "h264_nvenc", "-gpu", str(gpu), "-preset", "p5", "-rc", "vbr", "-cq", str(crf + 1), "-b:v", "0"]
        env = {**os.environ, "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}     # numbered like nvidia-smi
        if subprocess.run(build(nvenc), env=env, capture_output=True).returncode == 0:
            return
    subprocess.run(build(["-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf)]), check=True)


def _encode(inputs, graph, out_path):
    def cmd(video_codec):
        return ["ffmpeg", "-y", "-loglevel", "error", *inputs,
                "-filter_complex", graph, "-map", "[out]", "-map", "0:a?", "-r", "30", *video_codec,
                "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", str(out_path)]

    encode(cmd, 20)                     # GTX 1660 NVENC (CPU only while Blender is open)
    return out_path


def blender_running():
    """True while Blender is open (the owner renders on the 1660), so the bot uses the CPU instead."""
    try:
        out = subprocess.run(["tasklist", "/FI", "IMAGENAME eq blender.exe", "/NH"],
                             capture_output=True, text=True, creationflags=subprocess.CREATE_NO_WINDOW).stdout
        return "blender.exe" in out.lower()
    except Exception:
        return False


_gpu = "unset"


def gpu_index(name="1660"):
    """nvidia-smi index of the card to encode on (None if not found)."""
    global _gpu
    if _gpu == "unset":
        _gpu = None
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=index,name", "--format=csv,noheader"],
                                 capture_output=True, text=True).stdout
            for line in out.splitlines():
                idx, gname = line.split(",", 1)
                if name in gname:
                    _gpu = int(idx)
        except Exception:
            pass
    return _gpu


def erase_box(src, box, dest):
    """Paint out a rectangle of the source (the creator's own off-centre name badge) with ffmpeg's
    delogo, so our centred badge can replace it."""
    x, y, w, h = box
    encode(lambda v: ["ffmpeg", "-v", "error", "-y", "-i", str(src), "-vf", f"delogo=x={x}:y={y}:w={w}:h={h}",
                      *v, "-c:a", "copy", str(dest)], 16)
    return dest


def center_on(info, cx):
    """Slide the single-mode crop sideways so x=cx (the stream's own name badge) ends up in the middle,
    instead of the badge sitting off to one side (owner 2026-11-06). Only when every main person
    still fits inside the moved crop; otherwise the crop stays as it was. No blur, no erasing."""
    if info.get("mode") != "single":
        return info
    faces = _main_faces(info.get("faces", []))
    fw_ = info["w"]

    def moved(c):
        x, y, w, h = c
        nx = int(min(max(cx - w / 2, 0), fw_ - w)) // 2 * 2
        if abs(nx + w / 2 - cx) > 0.03 * w:          # can't centre it (badge too near the frame edge)
            return None
        if any(f[0] < nx or f[0] + f[2] > nx + w for f in faces):
            return None
        return (nx, y, w, h)

    new = dict(info)
    if info.get("shots"):
        shots = [(a, b, moved(c)) for a, b, c in info["shots"]]
        if all(s[2] for s in shots):
            new["shots"] = shots
            return new
        return info
    c = moved(info["crop"])
    if c:
        new["crop"] = c
    return new
