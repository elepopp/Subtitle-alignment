"""Timing utilities: interpolation of missing times and sanity post-processing."""
from __future__ import annotations

from typing import List, Optional, Sequence

from ..models import Document, Line, Token
from ..text.tokenize import syllable_weight


def _interp_run(tokens: Sequence[Token], lo: float, hi: float) -> None:
    """Distribute [lo, hi] over tokens proportionally to syllable weight."""
    if not tokens:
        return
    hi = max(hi, lo)
    ws = [syllable_weight(t) for t in tokens]
    total = sum(ws) or 1.0
    t = lo
    for tok, w in zip(tokens, ws):
        d = (hi - lo) * w / total
        tok.start, tok.end = t, t + d
        tok.confidence = min(tok.confidence, 0.3)
        t += d


def fill_missing_times(doc_or_tokens, default_unit: float = 0.25) -> None:
    """Interpolate tokens without times between their timed neighbours."""
    if isinstance(doc_or_tokens, Document):
        toks: List[Token] = list(doc_or_tokens.tokens())
    else:
        toks = list(doc_or_tokens)
    n = len(toks)
    i = 0
    while i < n:
        if toks[i].timed:
            i += 1
            continue
        j = i
        while j < n and not toks[j].timed:
            j += 1
        run = toks[i:j]
        prev_end: Optional[float] = toks[i - 1].end if i > 0 else None
        next_start: Optional[float] = toks[j].start if j < n else None
        est = sum(syllable_weight(t) for t in run) * default_unit
        if prev_end is None and next_start is None:
            lo, hi = 0.0, est
        elif prev_end is None:
            lo, hi = max(0.0, next_start - est), next_start
        elif next_start is None:
            lo, hi = prev_end, prev_end + est
        else:
            lo, hi = prev_end, max(prev_end, next_start)
            if hi - lo < 1e-3:   # squeezed: steal a little time from neighbours
                lo = max(toks[i - 1].start or lo, lo - 0.04 * len(run))
                toks[i - 1].end = lo
        _interp_run(run, lo, hi)
        i = j


def enforce_monotonic(tokens: Sequence[Token], min_dur: float = 0.02) -> None:
    """Make token times non-overlapping and non-decreasing."""
    last = 0.0
    for t in tokens:
        if not t.timed:
            continue
        if t.start < last:
            t.start = last
        if t.end < t.start + min_dur:
            t.end = t.start + min_dur
        last = t.start + min_dur if t.end is None else t.end
    # fix overlaps created by min_dur pushing
    toks = [t for t in tokens if t.timed]
    for a, b in zip(toks, toks[1:]):
        if a.end > b.start:
            a.end = max(a.start + 1e-3, b.start)


def shift(doc: Document, offset: float) -> Document:
    for ln in doc.lines:
        for t in ln.tokens:
            if t.start is not None:
                t.start = max(0.0, t.start + offset)
            if t.end is not None:
                t.end = max(0.0, t.end + offset)
        if ln.start is not None:
            ln.start = max(0.0, ln.start + offset)
        if ln.end is not None:
            ln.end = max(0.0, ln.end + offset)
    return doc


def line_tokens_flat(lines: Sequence[Line]) -> List[Token]:
    return [t for ln in lines for t in ln.tokens]
