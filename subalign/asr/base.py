"""ASR backend interface and conversion of transcripts to timed tokens."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from ..models import Document, Line, Token
from ..text.tokenize import is_cjk, syllable_weight, tokenize


@dataclass
class Word:
    text: str
    start: float
    end: float
    prob: float = 1.0


@dataclass
class Segment:
    start: float
    end: float
    text: str
    words: List[Word] = field(default_factory=list)


@dataclass
class Transcript:
    segments: List[Segment] = field(default_factory=list)
    language: Optional[str] = None

    @property
    def text(self) -> str:
        return "\n".join(s.text.strip() for s in self.segments)


class ASRBackend:
    name = "base"

    def transcribe(self, audio_path: str, language: Optional[str] = None,
                   prompt: Optional[str] = None, **kw) -> Transcript:  # pragma: no cover
        raise NotImplementedError


def word_to_tokens(w: Word) -> List[Token]:
    """Split one ASR word into timing units, distributing its time span.

    Whisper often emits multi-character 'words' for CJK; we split them by
    syllable weight (later refined acoustically)."""
    toks = tokenize(w.text)
    if not toks:
        return []
    if len(toks) == 1:
        toks[0].start, toks[0].end, toks[0].confidence = w.start, w.end, w.prob
        return toks
    ws = [syllable_weight(t) for t in toks]
    total = sum(ws) or 1.0
    t = w.start
    for tok, wt in zip(toks, ws):
        d = (w.end - w.start) * wt / total
        tok.start, tok.end = t, t + d
        tok.confidence = w.prob * 0.8
        t += d
    return toks


def segment_tokens(seg: Segment) -> List[Token]:
    if not seg.words:
        toks = tokenize(seg.text)
        from ..align.timing import _interp_run

        _interp_run(toks, seg.start, seg.end)
        return toks
    out: List[Token] = []
    for w in seg.words:
        wt = word_to_tokens(w)
        if not wt:
            continue
        # whisper words carry their leading space: " hello"
        if out and w.text[:1].isspace():
            out[-1].space_after = True
        out.extend(wt)
    return out


def transcript_tokens(tr: Transcript) -> List[Token]:
    out: List[Token] = []
    for seg in tr.segments:
        toks = segment_tokens(seg)
        if out and toks and not (is_cjk(toks[0].text[:1] or "a") and is_cjk(out[-1].text[-1:] or "a")):
            out[-1].space_after = True
        out.extend(toks)
    return out


def transcript_to_document(tr: Transcript, kind: str = "speech") -> Document:
    lines = []
    for seg in tr.segments:
        toks = segment_tokens(seg)
        if not toks:
            continue
        toks[-1].space_after = False
        ln = Line(tokens=toks)
        ln.update_bounds()
        lines.append(ln)
    return Document(lines=lines, language=tr.language, kind=kind)
