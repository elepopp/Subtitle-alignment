"""Lyric formats.

=============  =========  =================================================
format         level      example
=============  =========  =================================================
lrc            line       ``[00:12.34]歌词``
lrc-enhanced   word       ``[00:12.34]<00:12.34>歌 <00:12.80>词<00:13.20>`` (A2 / ESLyric)
lrc-word       word       ``[00:12.34]歌[00:12.80]词[00:13.20]`` (逐字 LRC)
qrc            word       ``[12340,860]歌(12340,460)词(12800,400)`` (QQ 音乐)
krc            word       ``[12340,860]<0,460,0>歌<460,400,0>词`` (酷狗, plain text)
krc-encrypted  word       binary ``.krc`` (krc1 + zlib + xor)
yrc            word       ``[12340,860](12340,460,0)歌(12800,400,0)词`` (网易云)
=============  =========  =================================================
"""
from __future__ import annotations

import base64
import json
import re
import zlib
from typing import List, Optional

from ..models import Document, Line
from ..text.tokenize import make_line
from .common import line_text, ms, parse_ts, token_display, tokens_with_times, ts_lrc

_META_KEYS = {"title": "ti", "artist": "ar", "album": "al", "author": "au", "by": "by", "lyricist": "lr"}


def _header(doc: Document, tool_tag: bool = True) -> List[str]:
    out = []
    for k, tag in _META_KEYS.items():
        if doc.metadata.get(k):
            out.append(f"[{tag}:{doc.metadata[k]}]")
    if tool_tag and not doc.metadata.get("by"):
        out.append("[by:subalign]")
    return out


def _timed(doc: Document):
    return [ln for ln in doc.lines if ln.tokens and ln.start is not None]


# ---------------------------------------------------------------- LRC
def write_lrc(doc: Document, bilingual: bool = True, punct: str = "keep", clear_gaps: float = 2.0,
              ms_precision: bool = False, **_) -> str:
    out = _header(doc)
    lines = _timed(doc)
    for i, ln in enumerate(lines):
        stamp = f"[{ts_lrc(ln.start, ms_precision)}]"
        out.append(stamp + line_text(ln, punct))
        if bilingual and ln.translation:
            out.append(stamp + ln.translation)
        nxt = lines[i + 1].start if i + 1 < len(lines) else None
        if clear_gaps and ln.end is not None and (nxt is None or nxt - ln.end >= clear_gaps):
            out.append(f"[{ts_lrc(ln.end, ms_precision)}]")
    return "\n".join(out) + "\n"


def write_lrc_enhanced(doc: Document, bilingual: bool = True, ms_precision: bool = False, **_) -> str:
    out = _header(doc)
    for ln in _timed(doc):
        parts = [f"[{ts_lrc(ln.start, ms_precision)}]"]
        for i, t in enumerate(ln.tokens):
            parts.append(f"<{ts_lrc(t.start, ms_precision)}>{token_display(ln.tokens, i)}")
        parts.append(f"<{ts_lrc(ln.end, ms_precision)}>")
        out.append("".join(parts))
        if bilingual and ln.translation:
            out.append(f"[{ts_lrc(ln.start, ms_precision)}]{ln.translation}")
    return "\n".join(out) + "\n"


def write_lrc_word(doc: Document, bilingual: bool = True, ms_precision: bool = False, **_) -> str:
    out = _header(doc)
    for ln in _timed(doc):
        parts = []
        for i, t in enumerate(ln.tokens):
            parts.append(f"[{ts_lrc(t.start, ms_precision)}]{token_display(ln.tokens, i)}")
        parts.append(f"[{ts_lrc(ln.end, ms_precision)}]")
        out.append("".join(parts))
        if bilingual and ln.translation:
            out.append(f"[{ts_lrc(ln.start, ms_precision)}]{ln.translation}")
    return "\n".join(out) + "\n"


_LRC_TAG = re.compile(r"\[(\d+:\d+(?:[.:]\d+)?)\]")
_LRC_WORD = re.compile(r"<(\d+:\d+(?:[.:]\d+)?)>")
_META_RE = re.compile(r"^\[([a-zA-Z#]+):(.*)\]\s*$")


def read_lrc(text: str) -> Document:
    """Reads plain, enhanced (<mm:ss.xx>) and bracket-word LRC.  Repeated
    timestamps (translation lines) become ``Line.translation``."""
    doc = Document(kind="song")
    inv_meta = {v: k for k, v in _META_KEYS.items()}
    entries = []  # (time, raw_text_after_first_stamp)
    for raw in text.replace("\r\n", "\n").lstrip("﻿").split("\n"):
        raw = raw.strip()
        if not raw:
            continue
        mm = _META_RE.match(raw)
        if mm and not _LRC_TAG.match(raw):
            key = mm.group(1).lower()
            if key in inv_meta:
                doc.metadata[inv_meta[key]] = mm.group(2).strip()
            elif key == "offset":
                doc.metadata["offset_ms"] = int(mm.group(2) or 0)
            continue
        stamps = []
        rest = raw
        while True:
            m = _LRC_TAG.match(rest)
            if not m:
                break
            stamps.append(parse_ts(m.group(1)))
            rest = rest[m.end():]
        if not stamps:
            continue
        if len(stamps) == 1 and _LRC_TAG.search(rest):  # bracket word-level LRC
            entries.append((stamps[0], rest, "word"))
        else:
            for s in stamps:   # compressed LRC: [t1][t2]text
                entries.append((s, rest, "line"))
    entries.sort(key=lambda e: e[0])
    lines: List[Line] = []
    for t, body, kind in entries:
        if kind == "word":
            ln = _parse_bracket_words(t, body)
        elif _LRC_WORD.search(body):
            ln = _parse_enhanced(t, body)
        else:
            if not body.strip():
                if lines and lines[-1].end is None:
                    lines[-1].end = t
                continue
            ln = make_line(body.strip(), start=t)
        if lines and abs(lines[-1].start - t) < 1e-3 and ln.tokens:
            # same timestamp twice -> translation
            if lines[-1].translation is None:
                lines[-1].translation = ln.text
            continue
        if ln.tokens:
            lines.append(ln)
    for i, ln in enumerate(lines):
        if ln.end is None:
            last_tok_end = ln.tokens[-1].end if ln.tokens[-1].end is not None else None
            nxt = lines[i + 1].start if i + 1 < len(lines) else None
            ln.end = last_tok_end or (min(nxt, ln.start + 8.0) if nxt else ln.start + 5.0)
    off = doc.metadata.pop("offset_ms", 0)
    doc.lines = lines
    if off:
        from ..align.timing import shift

        shift(doc, -off / 1000.0)  # LRC: positive offset => lyrics earlier
    return doc


def _parse_enhanced(t: float, body: str) -> Line:
    parts = _LRC_WORD.split(body)
    texts, starts = [], []
    lead = parts[0]
    for j in range(1, len(parts), 2):
        starts.append(parse_ts(parts[j]))
        texts.append(parts[j + 1])
    if lead.strip():
        texts.insert(0, lead)
        starts.insert(0, t)
    # trailing stamp with empty text = end of last word
    end = None
    if texts and not texts[-1].strip():
        end = starts[-1]
        texts, starts = texts[:-1], starts[:-1]
    ends = starts[1:] + [end if end is not None else (starts[-1] + 0.5 if starts else t)]
    toks = tokens_with_times(texts, starts, ends)
    return Line(tokens=toks, start=t, end=end)


def _parse_bracket_words(t: float, body: str) -> Line:
    return _parse_enhanced(t, _LRC_TAG.sub(lambda m: f"<{m.group(1)}>", f"<{_fmt(t)}>" + body))


def _fmt(t: float) -> str:
    return ts_lrc(t, True)


# ---------------------------------------------------------------- QRC
def write_qrc(doc: Document, xml: bool = False, **_) -> str:
    out = _header(doc, tool_tag=False)
    for ln in _timed(doc):
        s, e = ms(ln.start), ms(ln.end)
        parts = [f"[{s},{e - s}]"]
        for i, t in enumerate(ln.tokens):
            a, b = ms(t.start), ms(t.end)
            parts.append(f"{token_display(ln.tokens, i)}({a},{max(0, b - a)})")
        out.append("".join(parts))
    body = "\n".join(out) + "\n"
    if xml:
        esc = body.replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;").replace(">", "&gt;")
        return ('<?xml version="1.0" encoding="utf-8"?>\n<QrcInfos>\n<QrcHeadInfo SaveTime="0" Version="100"/>\n'
                f'<LyricInfo LyricCount="1">\n<Lyric_1 LyricType="1" LyricContent="{esc}"/>\n</LyricInfo>\n</QrcInfos>\n')
    return body


_QRC_LINE = re.compile(r"^\[(\d+),(\d+)\](.*)$")
_QRC_WORD = re.compile(r"(.*?)\((\d+),(\d+)\)")


def read_qrc(text: str) -> Document:
    m = re.search(r'LyricContent="(.*?)"\s*/>', text, re.S)
    if m:
        import html

        text = html.unescape(m.group(1))
    lines = []
    for raw in text.replace("\r\n", "\n").split("\n"):
        mm = _QRC_LINE.match(raw.strip())
        if not mm:
            continue
        s, d = int(mm.group(1)) / 1000, int(mm.group(2)) / 1000
        texts, starts, ends = [], [], []
        for w in _QRC_WORD.finditer(mm.group(3)):
            texts.append(w.group(1))
            a = int(w.group(2)) / 1000
            starts.append(a)
            ends.append(a + int(w.group(3)) / 1000)
        toks = tokens_with_times(texts, starts, ends)
        if toks:
            lines.append(Line(tokens=toks, start=s, end=s + d))
    return Document(lines=lines, kind="song")


# ---------------------------------------------------------------- KRC
KRC_KEY = bytes([64, 71, 97, 119, 94, 50, 116, 71, 81, 54, 49, 45, 206, 210, 110, 105])


def write_krc(doc: Document, bilingual: bool = True, **_) -> str:
    out = ["[id:$00000000]"] + _header(doc, tool_tag=False) + ["[offset:0]"]
    lines = _timed(doc)
    if bilingual and any(ln.translation for ln in lines):
        payload = {"content": [{"language": 0, "type": 1,
                                "lyricContent": [[ln.translation or ""] for ln in lines]}], "version": 1}
        b64 = base64.b64encode(json.dumps(payload, ensure_ascii=False).encode("utf-8")).decode()
        out.append(f"[language:{b64}]")
    for ln in lines:
        s, e = ms(ln.start), ms(ln.end)
        parts = [f"[{s},{e - s}]"]
        for i, t in enumerate(ln.tokens):
            a, b = ms(t.start), ms(t.end)
            parts.append(f"<{a - s},{max(0, b - a)},0>{token_display(ln.tokens, i)}")
        out.append("".join(parts))
    return "\n".join(out) + "\n"


def write_krc_encrypted(doc: Document, **kw) -> bytes:
    return krc_encrypt(write_krc(doc, **kw))


def krc_encrypt(text: str) -> bytes:
    z = zlib.compress(("﻿" + text).encode("utf-8"))
    return b"krc1" + bytes(b ^ KRC_KEY[i % 16] for i, b in enumerate(z))


def krc_decrypt(data: bytes) -> str:
    if data[:4] != b"krc1":
        raise ValueError("not an encrypted KRC file")
    z = bytes(b ^ KRC_KEY[i % 16] for i, b in enumerate(data[4:]))
    return zlib.decompress(z).decode("utf-8").lstrip("﻿")


_KRC_WORD = re.compile(r"<(-?\d+),(\d+),\d+>([^<]*)")


def read_krc(text) -> Document:
    if isinstance(text, (bytes, bytearray)):
        text = krc_decrypt(bytes(text))
    doc = Document(kind="song")
    translations: Optional[list] = None
    lines = []
    for raw in text.replace("\r\n", "\n").split("\n"):
        raw = raw.strip()
        if raw.startswith("[language:"):
            try:
                payload = json.loads(base64.b64decode(raw[len("[language:"):-1]).decode("utf-8"))
                for c in payload.get("content", []):
                    if c.get("type") == 1:
                        translations = [x[0] if x else "" for x in c.get("lyricContent", [])]
            except Exception:
                pass
            continue
        mm = _QRC_LINE.match(raw)
        if not mm:
            mt = _META_RE.match(raw)
            if mt and mt.group(1) in ("ti", "ar", "al"):
                doc.metadata[{"ti": "title", "ar": "artist", "al": "album"}[mt.group(1)]] = mt.group(2)
            continue
        s, d = int(mm.group(1)) / 1000, int(mm.group(2)) / 1000
        texts, starts, ends = [], [], []
        for w in _KRC_WORD.finditer(mm.group(3)):
            a = s + int(w.group(1)) / 1000
            texts.append(w.group(3))
            starts.append(a)
            ends.append(a + int(w.group(2)) / 1000)
        toks = tokens_with_times(texts, starts, ends)
        if toks:
            lines.append(Line(tokens=toks, start=s, end=s + d))
    if translations:
        for ln, tr in zip(lines, translations):
            ln.translation = tr or None
    doc.lines = lines
    return doc


# ---------------------------------------------------------------- YRC
def write_yrc(doc: Document, **_) -> str:
    out = []
    for ln in _timed(doc):
        s, e = ms(ln.start), ms(ln.end)
        parts = [f"[{s},{e - s}]"]
        for i, t in enumerate(ln.tokens):
            a, b = ms(t.start), ms(t.end)
            parts.append(f"({a},{max(0, b - a)},0){token_display(ln.tokens, i)}")
        out.append("".join(parts))
    return "\n".join(out) + "\n"


_YRC_WORD = re.compile(r"\((\d+),(\d+),-?\d+\)([^(]*)")


def read_yrc(text: str) -> Document:
    lines = []
    for raw in text.replace("\r\n", "\n").split("\n"):
        mm = _QRC_LINE.match(raw.strip())
        if not mm:
            continue
        s, d = int(mm.group(1)) / 1000, int(mm.group(2)) / 1000
        texts, starts, ends = [], [], []
        for w in _YRC_WORD.finditer(mm.group(3)):
            a = int(w.group(1)) / 1000
            texts.append(w.group(3))
            starts.append(a)
            ends.append(a + int(w.group(2)) / 1000)
        toks = tokens_with_times(texts, starts, ends)
        if toks:
            lines.append(Line(tokens=toks, start=s, end=s + d))
    return Document(lines=lines, kind="song")
