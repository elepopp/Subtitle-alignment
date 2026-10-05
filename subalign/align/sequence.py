"""Text-to-text alignment (script <-> ASR hypothesis) with phonetic costs.

Strategy:
  * up to ``FULL_DP_CELLS`` (a few thousand units each side) one global
    weighted Levenshtein DP whose substitution cost is ``1 - phonetic_similarity``
    (vectorised per row) - robust to repeated choruses / refrains;
  * longer texts: exact-match blocks via ``difflib.SequenceMatcher`` (``autojunk``
    off so that frequent CJK characters such as 的/了 are kept), then the same DP
    inside every gap between blocks (banded when the gap is large).
"""
from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import List, Optional, Sequence

import numpy as np

from ..models import Token
from ..text.phonetic import unit_similarity
from ..text.tokenize import normalize_key
from .timing import fill_missing_times


@dataclass
class AlignOp:
    op: str                 # "match" | "sub" | "del" (ref only) | "ins" (hyp only)
    ref: Optional[int]
    hyp: Optional[int]
    sim: float = 0.0


INDEL = 0.8  # insertion / deletion cost (substitution of dissimilar units costs 1.0)


def _sim_matrix(ref: Sequence[str], hyp: Sequence[str]) -> np.ndarray:
    """(n, m) phonetic similarity, computed once per distinct key pair."""
    ur, ri = np.unique(np.array(ref, dtype=object).astype(str), return_inverse=True)
    uh, hi = np.unique(np.array(hyp, dtype=object).astype(str), return_inverse=True)
    small = np.array([[1.0 if a == b else unit_similarity(a, b) for b in uh] for a in ur], dtype=np.float32)
    return small[np.ix_(ri.ravel(), hi.ravel())]


def _dp_gap(ref: Sequence[str], hyp: Sequence[str], r0: int, h0: int, band: int = 400) -> List[AlignOp]:
    """Weighted Levenshtein (sub = 1 - similarity, indel = INDEL), vectorised per row.

    Within a row the insertion chain ``c[j] = min(t[j], c[j-1] + INDEL)`` is solved
    with a cumulative minimum: ``c[j] = min_k (t[k] - INDEL*k) + INDEL*j``.
    """
    n, m = len(ref), len(hyp)
    if n == 0:
        return [AlignOp("ins", None, h0 + j) for j in range(m)]
    if m == 0:
        return [AlignOp("del", r0 + i, None) for i in range(n)]
    inf = 1e18
    sim = _sim_matrix(ref, hyp)
    use_band = n * m > 4_000_000
    back = np.zeros((n + 1, m + 1), dtype=np.int8)  # 1 diag, 2 up(del), 3 left(ins)
    jj = np.arange(m + 1, dtype=np.float64)
    prev = jj * INDEL
    back[0, 1:] = 3
    for i in range(1, n + 1):
        diag = np.full(m + 1, inf)
        diag[1:] = prev[:-1] + (1.0 - sim[i - 1])
        up = prev + INDEL
        t = np.minimum(diag, up)
        arg = np.where(diag <= up, 1, 2).astype(np.int8)
        if use_band:
            c = i * m / n
            out = (jj < c - band) | (jj > c + band)
            t[out] = inf
        left = np.minimum.accumulate(t - INDEL * jj) + INDEL * jj
        ins = left < t - 1e-12
        cur = np.where(ins, left, t)
        arg[ins] = 3
        back[i] = arg
        prev = cur
    ops: List[AlignOp] = []
    i, j = n, m
    while i > 0 or j > 0:
        a = back[i, j] if i > 0 else 3
        if j == 0:
            a = 2
        if a == 1:
            sv = float(sim[i - 1, j - 1])
            ops.append(AlignOp("match" if sv >= 0.999 else "sub", r0 + i - 1, h0 + j - 1, sv))
            i, j = i - 1, j - 1
        elif a == 2:
            ops.append(AlignOp("del", r0 + i - 1, None))
            i -= 1
        else:
            ops.append(AlignOp("ins", None, h0 + j - 1))
            j -= 1
    ops.reverse()
    return ops


# global DP up to this many cells (memory: one int8 per cell); longer texts are split at
# exact-match blocks first
FULL_DP_CELLS = 25_000_000


def align_keys(ref: Sequence[str], hyp: Sequence[str]) -> List[AlignOp]:
    if len(ref) * len(hyp) <= FULL_DP_CELLS:
        # one global alignment: block matching can pair a repeated chorus with the wrong repeat
        return _dp_gap(list(ref), list(hyp), 0, 0)
    sm = SequenceMatcher(None, list(ref), list(hyp), autojunk=False)
    ops: List[AlignOp] = []
    ri = hi = 0
    for blk in sm.get_matching_blocks():
        ops.extend(_dp_gap(ref[ri:blk.a], hyp[hi:blk.b], ri, hi))
        for k in range(blk.size):
            ops.append(AlignOp("match", blk.a + k, blk.b + k, 1.0))
        ri, hi = blk.a + blk.size, blk.b + blk.size
    return ops


def align_tokens(ref: Sequence[Token], hyp: Sequence[Token]) -> List[AlignOp]:
    return align_keys([normalize_key(t.text) for t in ref], [normalize_key(t.text) for t in hyp])


def transfer_times(ref: Sequence[Token], hyp: Sequence[Token], min_sim: float = 0.5,
                   ops: Optional[List[AlignOp]] = None, interpolate: bool = True) -> List[AlignOp]:
    """Copy timestamps from aligned hypothesis tokens onto reference tokens.

    Unaligned reference tokens are interpolated between their neighbours.
    Returns the alignment ops (useful for proofreading reports).
    """
    if ops is None:
        ops = align_tokens(ref, hyp)
    for t in ref:
        t.start = t.end = None
    for op in ops:
        if op.ref is None or op.hyp is None:
            continue
        if op.op == "match" or op.sim >= min_sim:
            h = hyp[op.hyp]
            r = ref[op.ref]
            r.start, r.end = h.start, h.end
            r.confidence = h.confidence * (1.0 if op.op == "match" else 0.7)
    if interpolate:
        fill_missing_times(list(ref))
    return ops


def error_rate(ops: Sequence[AlignOp]) -> float:
    """Character / word error rate of hyp w.r.t. ref."""
    n_ref = sum(1 for o in ops if o.ref is not None)
    errs = sum(1 for o in ops if o.op != "match")
    return errs / max(1, n_ref)
