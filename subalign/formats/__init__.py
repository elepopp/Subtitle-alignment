"""Format registry: ``write(doc, fmt)`` / ``read(path)`` / ``save(doc, path)``."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Union

from ..models import Document
from . import ass, lyrics, subtitle, ttml


def write_json(doc: Document, **_) -> str:
    d = doc.to_dict()
    d["format"] = "subalign"
    d["version"] = 1
    return json.dumps(d, ensure_ascii=False, indent=2)


def read_json(text: str) -> Document:
    return Document.from_dict(json.loads(text))


@dataclass
class Format:
    name: str
    ext: str
    level: str                 # "line" | "word"
    writer: Callable
    reader: Optional[Callable] = None
    binary: bool = False
    description: str = ""


FORMATS: Dict[str, Format] = {f.name: f for f in [
    Format("srt", ".srt", "line", subtitle.write_srt, subtitle.read_srt, description="SubRip"),
    Format("srt-karaoke", ".srt", "word", subtitle.write_srt_karaoke, None,
           description="SubRip, one cue per unit with <font> highlight"),
    Format("vtt", ".vtt", "line", subtitle.write_vtt, subtitle.read_srt, description="WebVTT"),
    Format("vtt-karaoke", ".vtt", "word", subtitle.write_vtt_karaoke, subtitle.read_srt,
           description="WebVTT with inline <timestamp> karaoke"),
    Format("ass", ".ass", "word", ass.write_ass, ass.read_ass, description="ASS with styles / karaoke effects"),
    Format("lrc", ".lrc", "line", lyrics.write_lrc, lyrics.read_lrc, description="LRC"),
    Format("lrc-enhanced", ".lrc", "word", lyrics.write_lrc_enhanced, lyrics.read_lrc,
           description="Enhanced LRC (A2 <mm:ss.xx>)"),
    Format("lrc-word", ".lrc", "word", lyrics.write_lrc_word, lyrics.read_lrc,
           description="逐字 LRC ([mm:ss.xx] per unit)"),
    Format("qrc", ".qrc", "word", lyrics.write_qrc, lyrics.read_qrc, description="QQ Music QRC (plain)"),
    Format("krc", ".krc.txt", "word", lyrics.write_krc, lyrics.read_krc, description="Kugou KRC (plain text)"),
    Format("krc-encrypted", ".krc", "word", lyrics.write_krc_encrypted, lyrics.read_krc, binary=True,
           description="Kugou KRC (encrypted binary)"),
    Format("yrc", ".yrc", "word", lyrics.write_yrc, lyrics.read_yrc, description="NetEase YRC"),
    Format("ttml", ".ttml", "word", ttml.write_ttml, ttml.read_ttml,
           description="TTML (Apple Music word timing)"),
    Format("ttml-line", ".ttml", "line", lambda d, **k: ttml.write_ttml(d, word_level=False, **k), ttml.read_ttml,
           description="TTML line timing"),
    Format("sbv", ".sbv", "line", subtitle.write_sbv, subtitle.read_sbv, description="YouTube SBV"),
    Format("txt", ".txt", "line", subtitle.write_txt, subtitle.read_txt, description="plain text"),
    Format("json", ".json", "word", write_json, read_json, description="full subalign document"),
]}

_EXT_READERS = {".srt": "srt", ".vtt": "vtt", ".ass": "ass", ".ssa": "ass", ".lrc": "lrc", ".qrc": "qrc",
                ".krc": "krc", ".yrc": "yrc", ".ttml": "ttml", ".xml": "ttml", ".sbv": "sbv", ".txt": "txt",
                ".json": "json"}


def write(doc: Document, fmt: str, **opts) -> Union[str, bytes]:
    if fmt not in FORMATS:
        raise ValueError(f"unknown format {fmt!r}; available: {', '.join(FORMATS)}")
    return FORMATS[fmt].writer(doc, **opts)


def save(doc: Document, path: Union[str, Path], fmt: Optional[str] = None, **opts) -> Path:
    path = Path(path)
    fmt = fmt or guess_format(path)
    data = write(doc, fmt, **opts)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, bytes):
        path.write_bytes(data)
    else:
        path.write_text(data, encoding="utf-8")
    return path


def guess_format(path: Union[str, Path]) -> str:
    ext = Path(path).suffix.lower()
    if ext not in _EXT_READERS:
        raise ValueError(f"cannot infer format from extension {ext!r}")
    return _EXT_READERS[ext]


def detect_text_format(text: str) -> str:
    """Sniff the format of subtitle / lyric text content."""
    import re

    head = text.lstrip("﻿")[:4000]
    if head.startswith("WEBVTT"):
        return "vtt"
    if "[Script Info]" in head or "\nDialogue:" in head:
        return "ass"
    if "<tt" in head and "ttml" in head:
        return "ttml"
    if re.search(r"^\[\d+,\d+\]\(\d+,\d+,-?\d+\)", head, re.M):
        return "yrc"
    if "QrcInfos" in head or re.search(r"^\[\d+,\d+\][^\n]*\(\d+,\d+\)", head, re.M):
        return "qrc"
    if re.search(r"^\[\d+,\d+\]<\d+,\d+,\d+>", head, re.M):
        return "krc"
    if re.search(r"^\d+\s*\n\s*\d+:\d+:\d+[,.]\d+\s*-->", head, re.M):
        return "srt"
    if re.search(r"^\[\d+:\d+(?:[.:]\d+)?\]", head, re.M):
        return "lrc"
    if head.lstrip().startswith("{") and '"lines"' in head:
        return "json"
    if re.search(r"^\d+:\d+:\d+\.\d+,\d+:\d+:\d+\.\d+$", head, re.M):
        return "sbv"
    return "txt"


def read(path: Union[str, Path], fmt: Optional[str] = None) -> Document:
    path = Path(path)
    raw = path.read_bytes()
    if raw[:4] == b"krc1":
        return lyrics.read_krc(raw)
    text = raw.decode("utf-8-sig", errors="replace")
    fmt = fmt or detect_text_format(text)
    reader = FORMATS[fmt].reader
    if reader is None:
        raise ValueError(f"format {fmt!r} is write-only")
    return reader(text)


def read_text(text: str, fmt: Optional[str] = None) -> Document:
    fmt = fmt or detect_text_format(text)
    return FORMATS[fmt].reader(text)
