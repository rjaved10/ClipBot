"""Same footage check: two X accounts post the same moment with different captions.

The text check can't always tell ("lumberjack outfit" vs "mogging him on desktop" was one moment), but the
sound is identical however each account crops or captions the clip. Each clip gets an audio fingerprint
(Haitsma-Kalker style: 32 bits per 50 ms from band-energy changes); two clips that line up for a few
seconds with few differing bits are the same footage.
"""
import nowin  # noqa: F401  (no console windows for child processes)
import subprocess

import numpy as np

RATE = 5512
STEP = 0.05                 # seconds per fingerprint frame
WIN = 2048                  # ~0.37 s analysis window
BANDS = np.geomspace(300, 2000, 34)
MIN_OVERLAP = 6.0           # seconds that have to line up
MAX_BER = 0.25              # fraction of differing bits (unrelated audio sits around 0.5)


def compute(path):
    """Fingerprint of a video's audio as bytes (None if it has no usable sound)."""
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", str(RATE),
                          "-f", "s16le", "-"], capture_output=True).stdout
    pcm = np.frombuffer(raw, np.int16).astype(np.float32)
    hop = int(RATE * STEP)
    if len(pcm) < WIN + hop * 10 or np.abs(pcm).mean() < 50:
        return None
    frames = np.lib.stride_tricks.sliding_window_view(pcm, WIN)[::hop] * np.hanning(WIN)
    spec = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    freqs = np.fft.rfftfreq(WIN, 1 / RATE)
    idx = np.searchsorted(freqs, BANDS)
    energy = np.stack([spec[:, a:b].sum(1) for a, b in zip(idx[:-1], idx[1:])], 1)
    d = energy[:, :-1] - energy[:, 1:]
    bits = (d[1:] - d[:-1]) > 0
    return np.packbits(bits, axis=1, bitorder="little").view(np.uint32).ravel().tobytes()


def _popcount(x):
    return np.unpackbits(x.view(np.uint8)).reshape(len(x), -1).sum(1)


def same(a, b):
    """Do two fingerprints share a stretch of the same audio?"""
    if not a or not b:
        return False
    a, b = np.frombuffer(a, np.uint32), np.frombuffer(b, np.uint32)
    need = int(MIN_OVERLAP / STEP)
    for off in range(-(len(b) - need), len(a) - need + 1, 2):
        x, y = (a[off:], b) if off >= 0 else (a, b[-off:])
        n = min(len(x), len(y))
        if _popcount(x[:n] ^ y[:n]).sum() / (32 * n) < MAX_BER:
            return True
    return False
