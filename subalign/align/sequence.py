"""Text-to-text alignment (script <-> ASR hypothesis) with phonetic costs.

Strategy (fast and robust for long texts):
  1. exact-match blocks via ``difflib.SequenceMatcher`` (``autojunk`` off so
     that frequent CJK characters such as 的/了 are not discarded);
  2. every unmatched gap between two blocks is aligned with a weighted
     Levenshtein DP whose substitution cost is ``1 - phonetic_similarity``
     (banded when the gap is large).
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


def _dp_gap(ref: Sequence[str], hyp: Sequence[str], r0: int, h0: int, band: int = 400) -> List[AlignOp]:
    n, m = len(ref), len(hyp)
    if n == 0:
        return [AlignOp("ins", None, h0 + j) for j in range(m)]
    if m == 0:
        return [AlignOp("del", r0 + i, None) for i in range(n)]
    inf = 1e18
    cost = np.full((n + 1, m + 1), inf)
    back = np.zeros((n + 1, m + 1), dtype=np.int8)  # 1 diag, 2 up(del), 3 left(ins)
    cost[0, 0] = 0.0
    use_band = n * m > 250_000
    for i in range(0, n + 1):
        if use_band:
            c = i * m / n
            jlo, jhi = max(0, int(c - band)), min(m, int(c + band))
        else:
            jlo, jhi = 0, m
        for j in range(jlo, jhi + 1):
            if i == 0 and j == 0:
                continue
            best, arg = inf, 0
            if i > 0 and j > 0 and cost[i - 1, j - 1] < inf:
                s = unit_similarity(ref[i - 1], hyp[j - 1])
                v = cost[i - 1, j - 1] + (1.0 - s)
                if v < best:
                    best, arg = v, 1
            if i > 0 and cost[i - 1, j] < inf:
                v = cost[i - 1, j] + INDEL
                if v < best:
                    best, arg = v, 2
            if j > 0 and cost[i, j - 1] < inf:
                v = cost[i, j - 1] + INDEL
                if v < best:
                    best, arg = v, 3
            cost[i, j], back[i, j] = best, arg
    ops: List[AlignOp] = []
    i, j = n, m
    while i > 0 or j > 0:
        a = back[i, j]
        if a == 1:
            s = unit_similarity(ref[i - 1], hyp[j - 1])
            ops.append(AlignOp("match" if s >= 0.999 else "sub", r0 + i - 1, h0 + j - 1, s))
            i, j = i - 1, j - 1
        elif a == 2:
            ops.append(AlignOp("del", r0 + i - 1, None))
            i -= 1
        else:
            ops.append(AlignOp("ins", None, h0 + j - 1))
            j -= 1
    ops.reverse()
    return ops


def align_keys(ref: Sequence[str], hyp: Sequence[str]) -> List[AlignOp]:
    sm = SequenceMatcher(None, list(ref), list(hyp), autojunk=False)
    ops: List[AlignOp] = []
    ri = hi = 0
    for blk in sm.get_matching_blocks():
        # short isolated matches inside large gaps are unreliable anchors: let
        # the DP decide them together with the gap.
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
