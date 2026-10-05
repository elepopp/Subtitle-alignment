"""Core data model shared by every stage of the pipeline.

All times are in seconds (float).  A :class:`Document` is a list of
:class:`Line` objects, each line being a list of timed :class:`Token` units.
A token is the smallest unit that gets its own timestamp: one character for
CJK / kana / hangul, one word for alphabetic scripts.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterator, List, Optional


@dataclass
class Token:
    text: str                      # display text (may carry attached punctuation)
    start: Optional[float] = None
    end: Optional[float] = None
    confidence: float = 1.0
    space_after: bool = False      # render a space after this token (alphabetic scripts)

    @property
    def duration(self) -> float:
        if self.start is None or self.end is None:
            return 0.0
        return max(0.0, self.end - self.start)

    @property
    def timed(self) -> bool:
        return self.start is not None and self.end is not None


@dataclass
class Line:
    tokens: List[Token] = field(default_factory=list)
    start: Optional[float] = None
    end: Optional[float] = None
    translation: Optional[str] = None
    style: Optional[str] = None
    speaker: Optional[str] = None
    breaks: List[int] = field(default_factory=list)  # token indices after which a new display row starts

    @property
    def text(self) -> str:
        parts = []
        for i, t in enumerate(self.tokens):
            parts.append(t.text)
            if t.space_after and i < len(self.tokens) - 1:
                parts.append(" ")
        return "".join(parts)

    def rows(self) -> List[str]:
        """Display rows (multi-row cues produced by the line breaker)."""
        out, cur = [], []
        stops = set(self.breaks)
        for i, t in enumerate(self.tokens):
            cur.append(t.text)
            if i in stops and i < len(self.tokens) - 1:
                out.append("".join(cur).strip())
                cur = []
            elif t.space_after and i < len(self.tokens) - 1:
                cur.append(" ")
        out.append("".join(cur).strip())
        return out

    def update_bounds(self) -> None:
        """Recompute line start/end from token times (if tokens are timed)."""
        starts = [t.start for t in self.tokens if t.start is not None]
        ends = [t.end for t in self.tokens if t.end is not None]
        if starts:
            self.start = min(starts)
        if ends:
            self.end = max(ends)

    @property
    def duration(self) -> float:
        if self.start is None or self.end is None:
            return 0.0
        return max(0.0, self.end - self.start)


@dataclass
class Document:
    lines: List[Line] = field(default_factory=list)
    language: Optional[str] = None
    kind: str = "speech"           # "speech" | "song"
    metadata: Dict[str, Any] = field(default_factory=dict)  # title, artist, album, ...

    def __iter__(self) -> Iterator[Line]:
        return iter(self.lines)

    def __len__(self) -> int:
        return len(self.lines)

    def tokens(self) -> Iterator[Token]:
        for ln in self.lines:
            yield from ln.tokens

    def copy(self) -> "Document":
        return copy.deepcopy(self)

    def finalize(self) -> "Document":
        """Fill missing token times by interpolation and refresh line bounds."""
        from .align.timing import fill_missing_times

        fill_missing_times(self)
        for ln in self.lines:
            ln.update_bounds()
        return self

    # --- (de)serialisation -------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Document":
        lines = []
        for ld in d.get("lines", []):
            toks = [Token(**td) for td in ld.get("tokens", [])]
            lines.append(Line(tokens=toks, start=ld.get("start"), end=ld.get("end"),
                              translation=ld.get("translation"), style=ld.get("style"),
                              speaker=ld.get("speaker"), breaks=list(ld.get("breaks", []))))
        return cls(lines=lines, language=d.get("language"), kind=d.get("kind", "speech"),
                   metadata=dict(d.get("metadata", {})))
