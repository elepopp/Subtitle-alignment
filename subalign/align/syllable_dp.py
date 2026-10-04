"""Syllable-to-boundary dynamic programming (the core character-level aligner).

Problem
-------
Given N timing units (characters / words) grouped into lines, find their start
times t_1 < t_2 < ... < t_N in the audio.  Acoustic evidence is noisy, ASR/CTC
anchors (when present) are approximate (+-50..300 ms), and singing breaks most
speech assumptions: syllables can be sustained for seconds, legato note
changes have no energy onset, and lines are separated by instrumental gaps.

Model
-----
We build a sparse, over-complete set of *boundary candidates*:

* peaks of the combined boundary curve  max(spectral-flux onset, pitch-change)
* rising edges of vocal activity
* a dense grid inside voiced regions (so no syllable is ever impossible) and a
  sparse grid in silence (keeps the problem feasible).

Then a first-order (semi-Markov) DP assigns every unit to one candidate:

    cost = sum_i unary(i, c_i) + sum_i trans(i, c_{i-1}, c_i)

unary  : - onset strength at the candidate
         + penalty for starting inside silence
         - bonus for line-initial units that follow a pause
         + robust (Huber) distance to the unit's anchor time, weighted by the
           anchor confidence
trans  : log-normal duration likelihood of the *previous* unit computed on
         its **voiced** duration (so instrumental gaps between lines are free),
         with separate statistics for line-final sustained syllables
         + cost for strong onsets that were skipped inside the unit (an unused
           onset is evidence of a missed boundary; melisma keeps this moderate)
         + cost for silence inside a non-final unit (mid-line breaths are allowed
           but must pay)

The DP is banded around anchors (or around a voiced-time proportional prior
when no anchors exist), so it runs in O(N * W * B) vector operations with W
the number of candidates within the maximum unit duration and B the band size.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import List, Optional, Sequence

import numpy as np

from ..audio.features import Features, pick_peaks

INF = 1e18


@dataclass
class DPParams:
    base_dur: float = 0.32         # expected voiced seconds per unit weight
    dur_sigma: float = 0.75        # log-normal sigma
    final_dur: float = 0.9         # expected duration of line-final unit (sustain)
    final_sigma: float = 1.0
    max_unit_voiced: float = 4.0   # hard cap on a non-final unit's voiced duration
    min_unit: float = 0.05
    w_onset: float = 2.0
    w_skip: float = 0.7
    w_dur: float = 1.0
    w_anchor: float = 1.2
    w_line_gap: float = 2.0
    w_unvoiced_start: float = 2.5
    w_inner_gap: float = 3.0       # per second of silence inside a non-final unit
    free_inner_gap: float = 0.12
    grid: float = 0.03
    silence_grid: float = 0.1
    band: float = 2.5              # band half-width around confident anchors (s)
    prior_band: float = 12.0       # band half-width around proportional prior (s)


SONG_PARAMS = DPParams()
SPEECH_PARAMS = DPParams(base_dur=0.17, dur_sigma=0.6, final_dur=0.3, final_sigma=0.8,
                         max_unit_voiced=1.6, w_onset=1.5, w_skip=0.4, w_anchor=1.0,
                         w_line_gap=1.0, w_inner_gap=2.0, free_inner_gap=0.15, band=1.5,
                         prior_band=8.0, grid=0.02)


@dataclass
class Unit:
    weight: float = 1.0
    line_start: bool = False
    line_end: bool = False
    anchor: Optional[float] = None       # approximate start time
    anchor_conf: float = 0.0             # 0..1
    anchor_sigma: float = 0.1            # expected anchor error (s): CTC ~0.05, Whisper ~0.2
    lo: Optional[float] = None           # hard window for the start time
    hi: Optional[float] = None


@dataclass
class Candidates:
    frame: np.ndarray
    time: np.ndarray
    onset: np.ndarray      # boundary evidence at the candidate (0..1)
    peak: np.ndarray       # >0 only for detected peaks / activity edges
    act: np.ndarray        # vocal activity at the candidate
    gap: np.ndarray        # 0..1 amount of silence just before the candidate
    voiced_cum: np.ndarray  # cumulative voiced seconds at the candidate


def build_candidates(feats: Features, p: DPParams, t0: float = 0.0, t1: Optional[float] = None) -> Candidates:
    hop = feats.hop_s
    n = feats.n
    f0 = max(0, int(t0 / hop))
    f1 = n if t1 is None else min(n, int(np.ceil(t1 / hop)) + 1)
    bnd = feats.boundary
    act = feats.active
    peaks = pick_peaks(bnd, hop, min_dist=0.04, delta=0.04)
    peaks = peaks[(peaks >= f0) & (peaks < f1) & (act[peaks] > 0.15)]
    on = act > 0.5
    edges = np.flatnonzero(on[1:] & ~on[:-1]) + 1
    edges = edges[(edges >= f0) & (edges < f1)]
    g = max(1, int(round(p.grid / hop)))
    gs = max(1, int(round(p.silence_grid / hop)))
    grid = np.arange(f0, f1, g)
    grid = grid[act[grid] > 0.3]
    sgrid = np.arange(f0, f1, gs)
    frames = np.unique(np.concatenate([peaks, edges, grid, sgrid, [f0]]).astype(np.int64))
    peak_val = np.zeros(n, dtype=np.float64)
    peak_val[peaks] = bnd[peaks]
    peak_val[edges] = np.maximum(peak_val[edges], 0.8)
    onset = np.maximum(bnd[frames], peak_val[frames])
    vcum = np.concatenate([[0.0], np.cumsum(act > 0.5) * hop])
    ucum = np.concatenate([[0.0], np.cumsum(act <= 0.5) * hop])
    look = int(0.5 / hop)
    prev = np.maximum(frames - look, 0)
    gap = np.clip((ucum[frames] - ucum[prev]) / 0.3, 0, 1)
    return Candidates(frame=frames, time=frames * hop, onset=onset.astype(np.float64),
                      peak=peak_val[frames], act=act[frames].astype(np.float64), gap=gap,
                      voiced_cum=vcum[frames])


def _huber(x: np.ndarray) -> np.ndarray:
    ax = np.abs(x)
    return np.where(ax < 1.0, 0.5 * x * x, ax - 0.5)


def _offset_after(feats: Features, t: float, min_silence: float = 0.12, limit: Optional[float] = None) -> float:
    """First time >= t where vocal activity stays low for ``min_silence``."""
    hop = feats.hop_s
    i = feats.t2i(t)
    j_lim = feats.n if limit is None else min(feats.n, int(np.ceil(limit / hop)) + 1)
    on = feats.active[i:j_lim] > 0.5
    k = max(1, int(min_silence / hop))
    off = ~on
    if len(off) < k:
        return (i + len(on)) * hop
    run = np.convolve(off.astype(np.int32), np.ones(k, dtype=np.int32), mode="valid")
    hits = np.flatnonzero(run >= k)
    if len(hits) == 0:
        return (i + len(on)) * hop
    return (i + hits[0]) * hop


def _prior_anchors(units: Sequence[Unit], feats: Features, t0: float, t1: float,
                   base_dur: float = 0.3) -> np.ndarray:
    """Fill missing anchors: interpolate between known anchors in *voiced time*;
    with no anchors at all, map cumulative unit weight to cumulative voiced time."""
    n = len(units)
    hop = feats.hop_s
    i0, i1 = feats.t2i(t0), max(feats.t2i(t1), feats.t2i(t0) + 1)
    vc = np.concatenate([[0.0], np.cumsum(feats.active[i0:i1] > 0.5)]).astype(np.float64)
    times = t0 + np.arange(len(vc)) * hop
    w = np.array([max(u.weight, 0.2) for u in units])
    cw = np.concatenate([[0.0], np.cumsum(w)])[:-1]
    known = [i for i, u in enumerate(units) if u.anchor is not None]
    out = np.zeros(n)
    if not known:
        if vc[-1] <= 0:
            return t0 + (t1 - t0) * cw / max(cw[-1] + w[-1], 1e-9)
        target = vc[-1] * cw / (cw[-1] + w[-1])
        return np.interp(target, vc, times)
    kv = np.array([np.interp(units[i].anchor, times, vc) for i in known])
    kc = cw[known]
    # extrapolate beyond the first/last anchors with the mean voiced rate
    rate = (kv[-1] - kv[0]) / max(kc[-1] - kc[0], 1e-9) if len(known) > 1 else 0.0
    if rate <= 0:
        rate = base_dur / hop
    v_est = np.interp(cw, kc, kv, left=np.nan, right=np.nan)
    left = np.isnan(v_est) & (cw < kc[0])
    right = np.isnan(v_est) & (cw > kc[-1])
    v_est[left] = kv[0] - (kc[0] - cw[left]) * rate
    v_est[right] = kv[-1] + (cw[right] - kc[-1]) * rate
    out = np.interp(v_est, vc, times)
    for i in known:
        out[i] = units[i].anchor
    return out


def align_units(units: Sequence[Unit], feats: Features, params: DPParams = SONG_PARAMS,
                t0: float = 0.0, t1: Optional[float] = None) -> np.ndarray:
    """Return an (N, 2) array of [start, end] times for the units."""
    N = len(units)
    if N == 0:
        return np.zeros((0, 2))
    p = params
    total = feats.n * feats.hop_s
    t1 = total if t1 is None else min(t1, total)
    cand = build_candidates(feats, p, t0, t1)
    M = len(cand.time)
    T, V = cand.time, cand.voiced_cum
    prior = _prior_anchors(units, feats, t0, t1, p.base_dur)
    has_anchor = np.array([u.anchor is not None and u.anchor_conf > 0 for u in units])
    any_anchor = bool(has_anchor.any())

    # per-unit band [lo, hi] in candidate indices
    bands = []
    for i, u in enumerate(units):
        half = max(p.band, 6 * u.anchor_sigma) if (u.anchor is not None and u.anchor_conf >= 0.5) else (
            p.band * 2.5 if any_anchor else p.prior_band)
        lo, hi = prior[i] - half, prior[i] + half
        if u.lo is not None:
            lo = max(lo, u.lo) if u.lo <= hi else u.lo
        if u.hi is not None:
            hi = min(hi, u.hi) if u.hi >= lo else u.hi
        if hi < lo:
            lo, hi = min(lo, hi), max(lo, hi)
        a, b = np.searchsorted(T, lo, "left"), np.searchsorted(T, hi, "right")
        a, b = int(min(a, M - 1)), int(max(b, a + 1))
        bands.append((a, min(b, M)))
    # make bands monotone so that a path always exists
    fixed = []
    prev_a = 0
    for a, b in bands:
        a = max(a, prev_a + 1) if fixed else a
        a = min(a, M - 1)
        b = max(b, a + 1)
        fixed.append((a, min(b, M)))
        prev_a = a
    bands = fixed

    pk_cum = np.concatenate([[0.0], np.cumsum(cand.peak)])  # pk_cum[j] = sum peaks < j

    def unary(i: int, sl: slice) -> np.ndarray:
        u = units[i]
        c = -p.w_onset * cand.onset[sl] + p.w_unvoiced_start * (1.0 - cand.act[sl])
        if u.line_start:
            c = c - p.w_line_gap * cand.gap[sl]
        if u.anchor is not None and u.anchor_conf > 0:
            sg = max(u.anchor_sigma, 0.02)
            c = c + p.w_anchor * u.anchor_conf * _huber((T[sl] - u.anchor) / sg)
        if u.lo is not None:
            c = np.where(T[sl] < u.lo - 1e-6, INF, c)
        if u.hi is not None:
            c = np.where(T[sl] > u.hi + 1e-6, INF, c)
        return c

    def dur_cost(prev: Unit, vd: np.ndarray) -> np.ndarray:
        if prev.line_end:
            mu, sg = max(p.final_dur, p.base_dur * prev.weight), p.final_sigma
        else:
            mu, sg = p.base_dur * max(prev.weight, 0.3), p.dur_sigma
        z = (np.log(vd + 0.03) - np.log(mu)) / sg
        return p.w_dur * 0.5 * z * z

    a0, b0 = bands[0]
    cost = np.full(M, INF)
    cost[a0:b0] = unary(0, slice(a0, b0))
    backs: List[np.ndarray] = [np.zeros(0, dtype=np.int32)]
    for i in range(1, N):
        a, b = bands[i]
        pa, pb = bands[i - 1]
        prev_u = units[i - 1]
        js = np.arange(a, b)
        best = np.full(len(js), INF)
        arg = np.full(len(js), -1, dtype=np.int32)
        max_vd = p.max_unit_voiced * (2.5 if prev_u.line_end else 1.0)
        # loop over predecessor index jp = j - k, vectorised over j
        jp_lo = np.maximum(np.searchsorted(V, V[js] - max_vd, "left"), pa)
        kmax = int(np.max(js - jp_lo)) if len(js) else 0
        kmax = max(1, min(kmax, 600))
        for k in range(1, kmax + 1):
            jp = js - k
            ok = (jp >= jp_lo) & (jp >= pa) & (jp < pb)
            if not ok.any():
                if (jp < pa).all():
                    break
                continue
            jpc = np.clip(jp, 0, M - 1)
            prevc = cost[jpc]
            ok &= prevc < INF
            if not ok.any():
                continue
            td = T[js] - T[jpc]
            vd = V[js] - V[jpc]
            ok &= td >= p.min_unit
            tc = prevc + dur_cost(prev_u, vd)
            tc += p.w_skip * (pk_cum[js] - pk_cum[jpc + 1])
            if not prev_u.line_end:
                tc += p.w_inner_gap * np.clip(td - vd - p.free_inner_gap, 0, None)
            tc = np.where(ok, tc, INF)
            better = tc < best
            best = np.where(better, tc, best)
            arg = np.where(better, jpc, arg)
        new = np.full(M, INF)
        u = unary(i, slice(a, b))
        if np.all(best >= INF):
            # infeasible (should be rare): restart the chain at this unit
            best = np.full(len(js), float(np.min(cost)))
            arg = np.full(len(js), int(np.argmin(cost)), dtype=np.int32)
        new[a:b] = best + u
        cost = new
        full_arg = np.full(M, -1, dtype=np.int32)
        full_arg[a:b] = arg
        backs.append(full_arg)

    # final unit: add its own duration cost up to the next silence
    a, b = bands[-1]
    last = units[-1]
    final = cost.copy()
    for j in range(a, b):
        if final[j] >= INF:
            continue
        end = _offset_after(feats, T[j] + p.min_unit, limit=t1)
        vd = np.array([max(end - T[j], p.min_unit)])
        final[j] += dur_cost(replace(last, line_end=True), vd)[0]
    j = int(np.argmin(final))
    starts = np.zeros(N)
    for i in range(N - 1, -1, -1):
        starts[i] = T[j]
        if i == 0:
            break
        jp = int(backs[i][j])
        if jp < 0:
            jp = max(0, j - 1)
        j = jp
    starts = np.maximum.accumulate(starts)
    return _assign_ends(starts, units, feats, p, t1)


def _assign_ends(starts: np.ndarray, units: Sequence[Unit], feats: Features, p: DPParams, t1: float) -> np.ndarray:
    N = len(starts)
    out = np.zeros((N, 2))
    for i in range(N):
        s = starts[i]
        nxt = starts[i + 1] if i + 1 < N else t1
        if nxt <= s:
            nxt = s + p.min_unit
        off = _offset_after(feats, s + p.min_unit, min_silence=0.12 if units[i].line_end else 0.15, limit=nxt)
        e = min(nxt, off)
        if not units[i].line_end and nxt - e < 0.08:
            e = nxt   # tiny gap inside a line: close it
        out[i] = (s, max(e, s + p.min_unit))
    # never overlap the next unit
    for i in range(N - 1):
        out[i, 1] = min(out[i, 1], max(out[i + 1, 0], out[i, 0] + 1e-3))
    return out
