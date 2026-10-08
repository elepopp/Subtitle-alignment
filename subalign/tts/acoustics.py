"""Matching the dub to the sound of the original recording (声学环境匹配).

A synthetic voice is dry and has its own tonal balance; laid over the original
background it sounds like a second recording.  From the speaker's own lines:

* **tonal balance** - long-term average spectrum in 1/3-octave bands (speech frames
  only); the dub gets the difference as a linear-phase EQ, smoothed and capped, and
  never above ``max_hz`` (a 22 kHz model has nothing there to lift)
* **room** - how fast sound dies away after a phrase ends (dB/s, from the energy
  decay after phrase offsets); a dry dub decays almost at once, so an exponential
  noise reverb is added at the wet level whose decay comes closest to the original's
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

log = logging.getLogger("subalign")
SR = 48000

# 1/3-octave band centres 80 Hz .. 12.5 kHz
CENTRES = 1000.0 * 2.0 ** (np.arange(-11, 12) / 3.0)


def _frames_spec(y: np.ndarray, sr: int, n_fft: int = 2048, hop: int = 512) -> Tuple[np.ndarray, np.ndarray]:
    if len(y) < n_fft:
        y = np.pad(y, (0, n_fft - len(y)))
    n = 1 + (len(y) - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n)[:, None]
    win = np.hanning(n_fft)
    spec = np.abs(np.fft.rfft(y[idx] * win, axis=1)) ** 2
    return spec, np.fft.rfftfreq(n_fft, 1 / sr)


def band_levels(y: np.ndarray, sr: int, depth: float = 30.0) -> np.ndarray:
    """Long-term average spectrum (dB) per 1/3-octave band over the speech frames
    (within ``depth`` dB of the loud end), normalised to 0 dB at 1 kHz."""
    spec, f = _frames_spec(y, sr)
    e = 10 * np.log10(spec.sum(axis=1) + 1e-12)
    keep = e > np.percentile(e, 95) - depth
    p = spec[keep].mean(axis=0) if keep.any() else spec.mean(axis=0)
    out = []
    for c in CENTRES:
        lo, hi = c / 2 ** (1 / 6), c * 2 ** (1 / 6)
        m = (f >= lo) & (f < hi)
        out.append(10 * np.log10(p[m].sum() + 1e-12) if m.any() else -120.0)       # band power
    out = np.array(out)
    return out - out[np.argmin(np.abs(CENTRES - 1000))]


def eq_gains(target: np.ndarray, dub: np.ndarray, max_db: float = 9.0, max_hz: float = 10000.0,
             smooth: int = 3) -> np.ndarray:
    """Per-band gains (dB) that bring ``dub``'s balance to ``target``'s: smoothed over
    ``smooth`` bands, capped, zero above ``max_hz`` and where either side is silent."""
    g = np.where((target > -90) & (dub > -90), target - dub, 0.0)
    if smooth > 1:
        k = np.ones(smooth) / smooth
        g = np.convolve(np.pad(g, smooth // 2, mode="edge"), k, mode="valid")
    g = g - np.mean(g[(CENTRES >= 300) & (CENTRES <= 3000)])         # keep the level, change the balance
    g = np.clip(g, -max_db, max_db)
    g[CENTRES > max_hz] = 0.0
    return g


def apply_eq(y: np.ndarray, sr: int, gains: np.ndarray, taps: int = 2049) -> np.ndarray:
    """Linear-phase FIR with the band gains (interpolated on a log-frequency axis)."""
    from scipy.signal import fftconvolve, firwin2

    if not np.any(np.abs(gains) > 0.05):
        return y
    f = np.concatenate([[0], CENTRES[CENTRES < sr / 2], [sr / 2]])
    g = np.concatenate([[gains[0]], gains[CENTRES < sr / 2], [0.0]])
    h = firwin2(taps, f / (sr / 2), 10 ** (g / 20))
    out = fftconvolve(y, h)[taps // 2:taps // 2 + len(y)]
    return out.astype(np.float32)


# ------------------------------------------------------------------ room
def decay_rate(y: np.ndarray, sr: int, hop_s: float = 0.01, min_pause: float = 0.25) -> Optional[float]:
    """Median energy decay (dB/s) over the first 20 dB after phrase offsets: a dry
    voice falls by hundreds of dB/s, a reverberant room by tens."""
    hop = int(hop_s * sr)
    n = len(y) // hop
    if n < 50:
        return None
    db = 10 * np.log10(np.mean(y[:n * hop].reshape(n, hop) ** 2, axis=1) + 1e-12)
    floor = np.percentile(db, 10)
    top = np.percentile(db, 95)
    loud = db > top - 25
    rates = []
    i = 1
    while i < n:
        if loud[i - 1] and not loud[i]:                  # an offset
            j = i
            while j < n and not loud[j]:
                j += 1
            if (j - i) * hop_s >= min_pause:
                seg = db[i - 1:j]
                start = seg[0]
                stop = max(start - 20, floor + 3)
                k = int(np.argmax(seg <= stop)) if np.any(seg <= stop) else 0
                if k >= 2:
                    slope = np.polyfit(np.arange(k + 1) * hop_s, seg[:k + 1], 1)[0]
                elif k == 1:                             # gone within a frame: a dry sound
                    slope = (seg[1] - seg[0]) / hop_s
                else:
                    slope = 0.0
                if slope < 0:
                    rates.append(-slope)
            i = j
        i += 1
    return float(np.median(rates)) if len(rates) >= 3 else None


def room_ir(sr: int, decay_db_s: float, length: Optional[float] = None, seed: int = 5) -> np.ndarray:
    """Exponentially decaying noise (the late field of a room), darker as it decays."""
    from scipy.signal import butter, sosfilt

    rt = 60.0 / max(decay_db_s, 1.0)
    length = length or min(2.0, rt)
    n = int(length * sr)
    t = np.arange(n) / sr
    rng = np.random.default_rng(seed)
    ir = rng.standard_normal(n) * 10 ** (-decay_db_s * t / 20)
    ir = sosfilt(butter(2, 6000 / (sr / 2), "low", output="sos"), ir)
    ir[: int(0.005 * sr)] = 0                          # a few ms after the direct sound
    return (ir / (np.sqrt(np.sum(ir ** 2)) + 1e-12)).astype(np.float32)


def add_room(y: np.ndarray, sr: int, decay_db_s: float, wet: float) -> np.ndarray:
    from scipy.signal import fftconvolve

    if wet <= 0:
        return y
    rev = fftconvolve(y, room_ir(sr, decay_db_s))[:len(y)]
    return (y + wet * rev).astype(np.float32)


def fit_room(dub: np.ndarray, sr: int, target_decay: float,
             wets: Sequence[float] = (0.0, 0.03, 0.06, 0.1, 0.15, 0.22, 0.3, 0.4)) -> Tuple[float, Optional[float]]:
    """The wet level whose processed ``dub`` decays closest to ``target_decay`` (on a log
    scale); returns (wet, resulting decay)."""
    best = (0.0, decay_rate(dub, sr))
    best_err = abs(np.log(best[1] / target_decay)) if best[1] else np.inf
    for w in wets[1:]:
        d = decay_rate(add_room(dub, sr, target_decay, w), sr)
        if d is None:
            continue
        err = abs(np.log(d / target_decay))
        if err < best_err:
            best, best_err = (w, d), err
    return best


def profile(clips: List[np.ndarray], sr: int) -> Dict:
    """Tonal balance and decay of a speaker's original lines (joined with short gaps)."""
    gap = np.zeros(int(0.4 * sr), np.float32)
    y = np.concatenate([c for x in clips for c in (x, gap)]) if clips else np.zeros(sr, np.float32)
    return {"bands": [round(float(v), 2) for v in band_levels(y, sr)], "decay": decay_rate(y, sr)}
