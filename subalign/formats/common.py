from __future__ import annotations

import re
from typing import List, Optional

from ..models import Line, Token
from ..text.tokenize import strip_punct, tokenize


def _split(t: float):
    ms_total = int(round(max(0.0, t) * 1000))
    h, rem = divmod(ms_total, 3600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return h, m, s, ms


def ts_srt(t: float) -> str:
    h, m, s, ms = _split(t)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def ts_vtt(t: float) -> str:
    h, m, s, ms = _split(t)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def ts_sbv(t: float) -> str:
    h, m, s, ms = _split(t)
    return f"{h}:{m:02d}:{s:02d}.{ms:03d}"


def ts_ass(t: float) -> str:
    cs_total = int(round(max(0.0, t) * 100))
    h, rem = divmod(cs_total, 360_000)
    m, rem = divmod(rem, 6000)
    s, cs = divmod(rem, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def ts_lrc(t: float, ms: bool = False) -> str:
    if ms:
        total = int(round(max(0.0, t) * 1000))
        m, rem = divmod(total, 60_000)
        s, x = divmod(rem, 1000)
        return f"{m:02d}:{s:02d}.{x:03d}"
    total = int(round(max(0.0, t) * 100))
    m, rem = divmod(total, 6000)
    s, cs = divmod(rem, 100)
    return f"{m:02d}:{s:02d}.{cs:02d}"


def ms(t: Optional[float]) -> int:
    return int(round(max(0.0, t or 0.0) * 1000))


_TS_RE = re.compile(r"(?:(\d+):)?(\d+):(\d+)(?:[.,:](\d+))?")


def parse_ts(s: str) -> float:
    """Parse HH:MM:SS,mmm | MM:SS.xx | H:MM:SS.cc etc."""
    s = s.strip()
    m = _TS_RE.fullmatch(s)
    if not m:
        raise ValueError(f"bad timestamp {s!r}")
    h = int(m.group(1) or 0)
    mi = int(m.group(2))
    se = int(m.group(3))
    frac = m.group(4) or "0"
    return h * 3600 + mi * 60 + se + int(frac) / (10 ** len(frac))


def line_text(ln: Line, punct: str = "keep", sep: Optional[str] = None) -> str:
    """Line text after the punctuation policy; with ``sep`` multi-row cues are
    joined with it (e.g. newline), otherwise rows are joined with a space."""
    if ln.breaks and sep is not None:
        return sep.join(strip_punct(r, punct) for r in ln.rows())
    return strip_punct(ln.text, punct)


def token_display(tokens: List[Token], i: int) -> str:
    t = tokens[i]
    return t.text + (" " if t.space_after and i < len(tokens) - 1 else "")


def tokens_with_times(text_parts, starts, ends) -> List[Token]:
    """Build timed tokens from (text, start, end) karaoke chunks; chunks that
    contain several units (e.g. '你好') are split evenly."""
    from ..align.timing import _interp_run

    out: List[Token] = []
    for txt, a, b in zip(text_parts, starts, ends):
        trailing_space = txt.endswith(" ")
        leading_space = txt.startswith(" ")
        if leading_space and out:
            out[-1].space_after = True
        toks = tokenize(txt)
        if not toks:
            if txt.strip() == "" and out:
                out[-1].space_after = True
            continue
        if len(toks) == 1:
            toks[0].start, toks[0].end = a, b
        else:
            _interp_run(toks, a, b)
            for t in toks:
                t.confidence = 1.0
        if trailing_space:
            toks[-1].space_after = True
        out.extend(toks)
    if out:
        out[-1].space_after = False
    return out


def strip_tags(s: str) -> str:
    s = re.sub(r"\{[^}]*\}", "", s)
    s = re.sub(r"<[^>]+>", "", s)
    return s
