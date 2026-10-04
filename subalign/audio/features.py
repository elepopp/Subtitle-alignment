"""Acoustic analysis used by the aligners (pure numpy / scipy).

Everything runs on 16 kHz mono with a 10 ms hop:

* ``onset``      SuperFlux-style spectral flux on a log filterbank (robust to
                 vibrato thanks to the frequency max-filter)
* ``f0``         YIN pitch + periodicity
* ``pitch_change`` note-change strength derived from the pitch track; this is
                 what catches legato syllable changes in singing, where there
                 is no energy onset at all
* ``active``     vocal-activity probability from adaptive energy thresholds
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.ndimage import maximum_filter1d, median_filter, uniform_filter1d

SR = 16000
HOP = 160            # 10 ms
WIN = 1024           # 64 ms
ONSET_LATENCY = 0.015


@dataclass
class Features:
    hop_s: float
    onset: np.ndarray            # (T,) normalised onset envelope (~0..1)
    pitch_change: np.ndarray     # (T,) 0..1
    f0: np.ndarray               # (T,) Hz, 0 where unvoiced
    periodicity: np.ndarray      # (T,) 0..1
    db: np.ndarray               # (T,) frame energy in dB
    active: np.ndarray           # (T,) vocal-activity probability 0..1
    boundary: np.ndarray = field(default=None)  # combined boundary strength

    @property
    def n(self) -> int:
        return len(self.onset)

    def t2i(self, t: float) -> int:
        return int(np.clip(round(t / self.hop_s), 0, self.n - 1))

    @property
    def times(self) -> np.ndarray:
        return np.arange(self.n) * self.hop_s


def _frames(y: np.ndarray, win: int, hop: int) -> np.ndarray:
    pad = win // 2
    yp = np.pad(y, (pad, pad + hop), mode="constant")
    n = 1 + (len(y)) // hop
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    return yp[idx]


def _log_filterbank(n_fft: int, sr: int, n_bands: int = 72, fmin: float = 60.0, fmax: float = 6000.0) -> np.ndarray:
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    edges = np.geomspace(fmin, fmax, n_bands + 2)
    fb = np.zeros((len(freqs), n_bands), dtype=np.float32)
    for b in range(n_bands):
        lo, c, hi = edges[b], edges[b + 1], edges[b + 2]
        up = (freqs - lo) / max(c - lo, 1e-6)
        down = (hi - freqs) / max(hi - c, 1e-6)
        fb[:, b] = np.clip(np.minimum(up, down), 0, None)
    fb /= np.maximum(fb.sum(axis=0, keepdims=True), 1e-9)
    return fb


def onset_envelope(y: np.ndarray, sr: int = SR, hop: int = HOP, win: int = WIN, lag: int = 2) -> tuple:
    fr = _frames(y, win, hop) * np.hanning(win)[None, :].astype(np.float32)
    mag = np.abs(np.fft.rfft(fr, axis=1)).astype(np.float32)
    spec = np.log1p(100.0 * (mag @ _log_filterbank(win, sr)))
    ref = maximum_filter1d(spec, size=3, axis=1)
    flux = np.zeros(len(spec), dtype=np.float32)
    flux[lag:] = np.clip(spec[lag:] - ref[:-lag], 0, None).sum(axis=1)
    rms = np.sqrt(np.mean(fr ** 2, axis=1) + 1e-12)
    db = 20 * np.log10(rms + 1e-9)
    return flux, db


def yin(y: np.ndarray, sr: int = SR, hop: int = HOP, fmin: float = 65.0, fmax: float = 1050.0,
        win: int = 512, threshold: float = 0.15, block: int = 2048) -> tuple:
    tmax = int(sr / fmin) + 1
    tmin = max(2, int(sr / fmax))
    L = win + tmax
    n = 1 + len(y) // hop
    yp = np.pad(y, (win // 2, L), mode="constant")
    nfft = 1 << int(np.ceil(np.log2(L + win)))
    f0 = np.zeros(n, dtype=np.float32)
    per = np.zeros(n, dtype=np.float32)
    taus = np.arange(tmax)
    for s in range(0, n, block):
        e = min(n, s + block)
        idx = np.arange(L)[None, :] + hop * np.arange(s, e)[:, None]
        x = yp[idx].astype(np.float64)
        a = x[:, :win]
        C = np.fft.irfft(np.conj(np.fft.rfft(a, nfft, axis=1)) * np.fft.rfft(x, nfft, axis=1), nfft, axis=1)[:, :tmax]
        cs = np.concatenate([np.zeros((len(x), 1)), np.cumsum(x ** 2, axis=1)], axis=1)
        e0 = cs[:, win][:, None]
        et = cs[:, taus + win] - cs[:, taus]
        d = np.maximum(e0 + et - 2 * C, 0.0)
        d[:, 0] = 0
        cum = np.cumsum(d[:, 1:], axis=1)
        cmnd = np.ones_like(d)
        cmnd[:, 1:] = d[:, 1:] * taus[1:][None, :] / np.maximum(cum, 1e-12)
        seg = cmnd[:, tmin:tmax]
        below = seg < threshold
        first = np.where(below.any(axis=1), below.argmax(axis=1), seg.argmin(axis=1))
        # descend to the local minimum after the first threshold crossing
        rows = np.arange(len(seg))
        tau = first.copy()
        for _ in range(40):
            nxt = np.minimum(tau + 1, seg.shape[1] - 1)
            better = seg[rows, nxt] < seg[rows, tau]
            if not better.any():
                break
            tau = np.where(better, nxt, tau)
        val = seg[rows, tau]
        # parabolic interpolation
        l = seg[rows, np.maximum(tau - 1, 0)]
        r = seg[rows, np.minimum(tau + 1, seg.shape[1] - 1)]
        den = l - 2 * val + r
        with np.errstate(divide="ignore", invalid="ignore"):
            off = np.where(np.abs(den) > 1e-9, 0.5 * (l - r) / den, 0.0)
        tt = tau + tmin + np.clip(off, -1, 1)
        energy = e0[:, 0] / win
        voiced = (val < 0.35) & (energy > 1e-7)
        f0[s:e] = np.where(voiced, sr / tt, 0.0)
        per[s:e] = np.clip(1.0 - val, 0, 1) * (energy > 1e-7)
    return f0, per


def _pitch_change(f0: np.ndarray, lag: int = 4) -> np.ndarray:
    voiced = f0 > 0
    semis = np.where(voiced, 12 * np.log2(np.maximum(f0, 1.0) / 440.0), np.nan)
    # interpolate over short gaps so the median filter is meaningful
    idx = np.arange(len(f0))
    if voiced.sum() < 2:
        return np.zeros_like(f0)
    filled = np.interp(idx, idx[voiced], semis[voiced])
    sm = median_filter(filled, size=7, mode="nearest")
    delta = np.zeros_like(sm)
    delta[lag:-lag] = np.abs(sm[2 * lag:] - sm[:-2 * lag])
    both = np.zeros_like(voiced)
    both[lag:-lag] = voiced[:-2 * lag] & voiced[2 * lag:]
    # octave errors of the tracker are not note changes
    delta = np.where(np.abs(delta - 12) < 0.6, 0, delta)
    score = np.clip((delta - 0.6) / 1.4, 0, 1) * both
    return score.astype(np.float32)


def _activity(db: np.ndarray, per: np.ndarray, hop_s: float) -> np.ndarray:
    floor = np.percentile(db, 5)
    peak = np.percentile(db, 98)
    thr = max(floor + 0.3 * (peak - floor), peak - 38.0)
    if peak - floor < 6:  # basically flat (silence or constant noise)
        thr = peak - 3
    prob = 1.0 / (1.0 + np.exp(-(db - thr) / 2.5))
    # periodic frames slightly more likely to be voice
    prob = np.clip(prob * (0.85 + 0.3 * per), 0, 1)
    k = max(1, int(round(0.03 / hop_s)))
    return uniform_filter1d(prob, size=k, mode="nearest").astype(np.float32)


def _normalize(x: np.ndarray, pct: float = 97.0) -> np.ndarray:
    ref = np.percentile(x[x > 0], pct) if np.any(x > 0) else 1.0
    return np.clip(x / max(ref, 1e-9), 0, 1.5).astype(np.float32) / 1.5


def analyze(y: np.ndarray, sr: int = SR, pitch_weight: float = 0.8) -> Features:
    """Compute alignment features from 16 kHz mono audio.

    ``pitch_weight``: contribution of note changes to the boundary curve (high for
    singing, low for speech where intonation glides continuously).
    """
    if sr != SR:
        from .io import to_mono_16k

        y, sr = to_mono_16k(y, sr)
    y = np.asarray(y, dtype=np.float32)
    flux, db = onset_envelope(y, sr)
    f0, per = yin(y, sr)
    n = min(len(flux), len(f0))
    flux, db, f0, per = flux[:n], db[:n], f0[:n], per[:n]
    hop_s = HOP / sr
    # adaptive whitening of the onset envelope (removes slow loudness changes)
    local = uniform_filter1d(flux, size=int(0.4 / hop_s), mode="nearest")
    onset = _normalize(np.clip(flux - 0.6 * local, 0, None))
    pc = _pitch_change(f0)
    act = _activity(db, per, hop_s)
    boundary = np.maximum(onset, pitch_weight * pc) * np.clip(act + 0.2, 0, 1)
    # latency compensation: with centred 64 ms frames the boundary evidence
    # appears ~20 ms before the true syllable onset (measured on synthetic
    # material), so the curves are delayed accordingly.
    lag = int(round(ONSET_LATENCY / hop_s))
    if lag > 0:
        onset = np.concatenate([np.zeros(lag, np.float32), onset[:-lag]])
        boundary = np.concatenate([np.zeros(lag, np.float32), boundary[:-lag]]).astype(np.float32)
    feats = Features(hop_s=hop_s, onset=onset, pitch_change=pc, f0=f0, periodicity=per, db=db, active=act)
    feats.boundary = boundary
    return feats


def pick_peaks(x: np.ndarray, hop_s: float, min_dist: float = 0.05, delta: float = 0.06,
               avg_win: float = 0.2) -> np.ndarray:
    """Indices of local maxima above a moving average + delta."""
    k = max(1, int(round(min_dist / hop_s)))
    mx = maximum_filter1d(x, size=2 * k + 1, mode="nearest")
    avg = uniform_filter1d(x, size=max(1, int(avg_win / hop_s)), mode="nearest")
    cand = np.flatnonzero((x == mx) & (x >= avg + delta) & (x > 0))
    # de-duplicate plateaus
    if len(cand) > 1:
        keep = np.concatenate([[True], np.diff(cand) > k])
        cand = cand[keep]
    return cand


def active_segments(feats: Features, thr: float = 0.5, min_gap: float = 0.25,
                    min_len: float = 0.12) -> list:
    """Vocal activity regions [(start_s, end_s), ...] after gap closing."""
    a = feats.active > thr
    segs = []
    i, n = 0, len(a)
    while i < n:
        if not a[i]:
            i += 1
            continue
        j = i
        while j < n and a[j]:
            j += 1
        segs.append([i * feats.hop_s, j * feats.hop_s])
        i = j
    merged = []
    for s in segs:
        if merged and s[0] - merged[-1][1] < min_gap:
            merged[-1][1] = s[1]
        else:
            merged.append(s)
    return [(s, e) for s, e in merged if e - s >= min_len]


def detect_content_type(feats: Features) -> str:
    """Heuristic speech / song classifier based on pitch stability.

    Sung notes hold a stable pitch for long runs; speech intonation glides
    continuously.  Accompaniment (also stable-pitched) pushes towards "song".
    """
    f0 = feats.f0
    voiced = f0 > 0
    if voiced.sum() < 50:
        return "speech"
    semis = 12 * np.log2(np.maximum(f0, 1.0) / 440.0)
    stable = np.zeros_like(voiced)
    stable[1:] = voiced[1:] & voiced[:-1] & (np.abs(np.diff(semis)) < 0.25)
    run_frames = int(0.2 / feats.hop_s)
    total_stable = 0
    i, n = 0, len(stable)
    while i < n:
        if stable[i]:
            j = i
            while j < n and stable[j]:
                j += 1
            if j - i >= run_frames:
                total_stable += j - i
            i = j
        else:
            i += 1
    ratio = total_stable / max(1, voiced.sum())
    activity = float(np.mean(feats.active > 0.5))
    score = ratio + 0.25 * max(0.0, activity - 0.7)
    return "song" if score > 0.3 else "speech"


def slice_features(feats: Features, t0: float, t1: Optional[float] = None) -> Features:
    i0 = feats.t2i(t0)
    i1 = feats.n if t1 is None else feats.t2i(t1) + 1
    sl = slice(i0, i1)
    return Features(hop_s=feats.hop_s, onset=feats.onset[sl], pitch_change=feats.pitch_change[sl],
                    f0=feats.f0[sl], periodicity=feats.periodicity[sl], db=feats.db[sl],
                    active=feats.active[sl], boundary=feats.boundary[sl])
