"""Line-oriented subtitle formats: SRT, WebVTT, SBV, plain text
(+ karaoke variants of SRT and WebVTT)."""
from __future__ import annotations

import re
from typing import List

from ..models import Document, Line
from ..text.tokenize import make_line
from .common import (line_text, parse_ts, strip_tags, token_display, tokens_with_times, ts_sbv, ts_srt,
                     ts_vtt)


def _lines(doc: Document):
    return [ln for ln in doc.lines if ln.tokens and ln.start is not None and ln.end is not None]


def _text(ln: Line, bilingual: bool, punct: str) -> str:
    t = line_text(ln, punct, sep="\n")
    if bilingual and ln.translation:
        t += "\n" + ln.translation
    return t


# ---------------------------------------------------------------- SRT
def write_srt(doc: Document, bilingual: bool = True, punct: str = "keep", **_) -> str:
    out = []
    for i, ln in enumerate(_lines(doc), 1):
        out.append(f"{i}\n{ts_srt(ln.start)} --> {ts_srt(ln.end)}\n{_text(ln, bilingual, punct)}\n")
    return "\n".join(out)


def write_srt_karaoke(doc: Document, highlight: str = "#FFD700", bilingual: bool = True, **_) -> str:
    """Progressive per-character highlight using one cue per unit (works in
    any player that renders <font color>)."""
    out = []
    k = 1
    for ln in _lines(doc):
        toks = ln.tokens
        for i, t in enumerate(toks):
            a = t.start
            b = toks[i + 1].start if i + 1 < len(toks) else ln.end
            if b is None or a is None or b <= a:
                continue
            sung = "".join(token_display(toks, j) for j in range(i + 1))
            rest = "".join(token_display(toks, j) for j in range(i + 1, len(toks)))
            txt = f'<font color="{highlight}">{sung}</font>{rest}'
            if bilingual and ln.translation:
                txt += "\n" + ln.translation
            out.append(f"{k}\n{ts_srt(a)} --> {ts_srt(b)}\n{txt}\n")
            k += 1
    return "\n".join(out)


_CUE_RE = re.compile(r"((?:\d+:)?\d+:\d+[.,]\d+)\s*-->\s*((?:\d+:)?\d+:\d+[.,]\d+)")
_INLINE_TS = re.compile(r"<((?:\d+:)?\d+:\d+\.\d+)>")


def read_srt(text: str) -> Document:
    """SRT and WebVTT reader (WebVTT inline <timestamps> become token timings)."""
    text = text.replace("\r\n", "\n").lstrip("﻿")
    blocks = re.split(r"\n\s*\n", text)
    lines: List[Line] = []
    for blk in blocks:
        rows = [r for r in blk.split("\n") if r.strip()]
        for idx, r in enumerate(rows):
            m = _CUE_RE.search(r)
            if not m:
                continue
            a, b = parse_ts(m.group(1)), parse_ts(m.group(2))
            body = rows[idx + 1:]
            if not body:
                break
            first = body[0]
            if _INLINE_TS.search(first):
                parts = _INLINE_TS.split(first)
                texts, starts = [strip_tags(parts[0])], [a]
                for j in range(1, len(parts), 2):
                    starts.append(parse_ts(parts[j]))
                    texts.append(strip_tags(parts[j + 1]))
                ends = starts[1:] + [b]
                toks = tokens_with_times(texts, starts, ends)
                ln = Line(tokens=toks, start=a, end=b)
            else:
                ln = make_line(strip_tags(first), start=a, end=b)
            if len(body) > 1:
                ln.translation = strip_tags(" ".join(body[1:]))
            if ln.tokens:
                lines.append(ln)
            break
    return Document(lines=lines)


# ---------------------------------------------------------------- WebVTT
def write_vtt(doc: Document, bilingual: bool = True, punct: str = "keep", **_) -> str:
    out = ["WEBVTT", ""]
    for ln in _lines(doc):
        out.append(f"{ts_vtt(ln.start)} --> {ts_vtt(ln.end)}\n{_text(ln, bilingual, punct)}\n")
    return "\n".join(out)


def write_vtt_karaoke(doc: Document, bilingual: bool = True, past: str = "#FFD700", future: str = "#FFFFFF",
                      **_) -> str:
    out = ["WEBVTT", "", "STYLE",
           f"::cue(:past) {{ color: {past}; }}\n::cue(:future) {{ color: {future}; }}", ""]
    for ln in _lines(doc):
        parts = []
        for i, t in enumerate(ln.tokens):
            if i > 0 and t.start is not None:
                parts.append(f"<{ts_vtt(t.start)}>")
            parts.append(f"<c>{token_display(ln.tokens, i)}</c>")
        txt = "".join(parts)
        if bilingual and ln.translation:
            txt += "\n" + ln.translation
        out.append(f"{ts_vtt(ln.start)} --> {ts_vtt(ln.end)}\n{txt}\n")
    return "\n".join(out)


# ---------------------------------------------------------------- SBV (YouTube)
def write_sbv(doc: Document, bilingual: bool = True, punct: str = "keep", **_) -> str:
    return "\n".join(f"{ts_sbv(ln.start)},{ts_sbv(ln.end)}\n{_text(ln, bilingual, punct)}\n" for ln in _lines(doc))


def read_sbv(text: str) -> Document:
    lines = []
    for blk in re.split(r"\n\s*\n", text.replace("\r\n", "\n").strip()):
        rows = blk.split("\n")
        if len(rows) < 2 or "," not in rows[0]:
            continue
        a, b = rows[0].split(",", 1)
        ln = make_line(rows[1], start=parse_ts(a), end=parse_ts(b))
        if len(rows) > 2:
            ln.translation = " ".join(rows[2:])
        lines.append(ln)
    return Document(lines=lines)


# ---------------------------------------------------------------- plain text
def write_txt(doc: Document, bilingual: bool = True, punct: str = "keep", **_) -> str:
    return "\n".join(_text(ln, bilingual, punct) for ln in doc.lines) + "\n"


def read_txt(text: str) -> Document:
    from ..text.tokenize import lines_from_text

    return Document(lines=lines_from_text(text))
