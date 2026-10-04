"""TTML (W3C Timed Text).  Word-level output follows the Apple Music lyric
convention (``itunes:timing="Word"`` with one <span> per unit and translations
as ``ttm:role="x-translation"`` spans)."""
from __future__ import annotations

import xml.etree.ElementTree as ET
from xml.sax.saxutils import escape

from ..models import Document, Line
from .common import parse_ts, tokens_with_times, ts_vtt
from ..text.tokenize import make_line

NS = {"tt": "http://www.w3.org/ns/ttml", "ttm": "http://www.w3.org/ns/ttml#metadata",
      "itunes": "http://music.apple.com/lyric-ttml-internal"}


def write_ttml(doc: Document, word_level: bool = True, bilingual: bool = True, target_lang: str = "", **_) -> str:
    lang = doc.language or "und"
    lines = [ln for ln in doc.lines if ln.tokens and ln.start is not None]
    timing = "Word" if word_level else "Line"
    out = ['<?xml version="1.0" encoding="UTF-8"?>',
           f'<tt xmlns="{NS["tt"]}" xmlns:ttm="{NS["ttm"]}" xmlns:itunes="{NS["itunes"]}" '
           f'itunes:timing="{timing}" xml:lang="{escape(lang)}">',
           "  <head>", "    <metadata>"]
    if doc.metadata.get("title"):
        out.append(f"      <ttm:title>{escape(str(doc.metadata['title']))}</ttm:title>")
    out += ["    </metadata>", "  </head>"]
    end = max((ln.end or 0) for ln in lines) if lines else 0
    out.append(f'  <body dur="{ts_vtt(end)}">')
    if lines:
        out.append(f'    <div begin="{ts_vtt(lines[0].start)}" end="{ts_vtt(end)}">')
    for i, ln in enumerate(lines, 1):
        attrs = f'begin="{ts_vtt(ln.start)}" end="{ts_vtt(ln.end)}" itunes:key="L{i}"'
        if word_level:
            spans = []
            for j, t in enumerate(ln.tokens):
                sp = f'<span begin="{ts_vtt(t.start)}" end="{ts_vtt(t.end)}">{escape(t.text)}</span>'
                if t.space_after and j < len(ln.tokens) - 1:
                    sp += " "
                spans.append(sp)
            body = "".join(spans)
        else:
            body = escape(ln.text)
        if bilingual and ln.translation:
            tl = f' xml:lang="{escape(target_lang)}"' if target_lang else ""
            body += f'<span ttm:role="x-translation"{tl}>{escape(ln.translation)}</span>'
        out.append(f"      <p {attrs}>{body}</p>")
    if lines:
        out.append("    </div>")
    out += ["  </body>", "</tt>"]
    return "\n".join(out) + "\n"


def _t(el, name):
    v = el.get(name)
    return parse_ts(v.rstrip("s")) if v and ":" in v else (float(v.rstrip("s")) if v else None)


def read_ttml(text: str) -> Document:
    root = ET.fromstring(text.encode("utf-8") if isinstance(text, str) else text)
    lang = root.get("{http://www.w3.org/XML/1998/namespace}lang")
    lines = []
    for p in root.iter(f"{{{NS['tt']}}}p"):
        a, b = _t(p, "begin"), _t(p, "end")
        spans = [s for s in p if s.tag == f"{{{NS['tt']}}}span"]
        translation = None
        texts, starts, ends = [], [], []
        for s in spans:
            role = s.get(f"{{{NS['ttm']}}}role")
            if role in ("x-translation", "x-roman"):
                if role == "x-translation":
                    translation = "".join(s.itertext())
                continue
            if s.get("begin") is None:
                continue
            texts.append("".join(s.itertext()) + (" " if (s.tail or "").startswith(" ") else ""))
            starts.append(_t(s, "begin"))
            ends.append(_t(s, "end"))
        if texts:
            ln = Line(tokens=tokens_with_times(texts, starts, ends), start=a, end=b)
        else:
            plain = [p.text or ""]
            for ch in p:
                if ch.get(f"{{{NS['ttm']}}}role") not in ("x-translation", "x-roman"):
                    plain.append("".join(ch.itertext()))
                plain.append(ch.tail or "")
            ln = make_line("".join(plain).strip(), start=a, end=b)
        ln.translation = translation
        if ln.tokens:
            lines.append(ln)
    return Document(lines=lines, language=lang)
