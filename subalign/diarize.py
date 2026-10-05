"""Speaker diarisation (说话人区分) for transcripts.

Speaker embeddings come from CAM++ (3D-Speaker, through FunASR / ModelScope -
not gated, downloaded into the model cache on first use).  Each line is split
into *chunks* at its internal pauses so a line that contains a speaker change
can be split; chunks long enough for a reliable embedding are clustered
(average-linkage agglomerative clustering on cosine distance, by threshold or
to a given number of speakers) and short chunks inherit the speaker of their
nearest neighbour in time.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np

from .models import Document, Line

log = logging.getLogger("subalign")

_MODEL = {}


def _embedder(device: Optional[str] = None):
    key = device or "auto"
    if key not in _MODEL:
        try:
            from funasr import AutoModel  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise ImportError("speaker diarisation needs `pip install funasr modelscope`") from e
        kw = {"model": "cam++", "disable_update": True}
        if device:
            kw["device"] = device
        _MODEL[key] = AutoModel(**kw)
    return _MODEL[key]


def embed(segments: List[np.ndarray], device: Optional[str] = None) -> np.ndarray:
    """(n, d) L2-normalised speaker embeddings for 16 kHz mono clips."""
    m = _embedder(device)
    out = []
    for y in segments:
        r = m.generate(input=np.ascontiguousarray(y, dtype=np.float32), disable_pbar=True)
        e = r[0]["spk_embedding"]
        e = e.detach().cpu().numpy() if hasattr(e, "detach") else np.asarray(e)
        out.append(e.reshape(-1))
    E = np.vstack(out).astype(np.float64)
    return E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)


def cluster(E: np.ndarray, n_speakers: Optional[int] = None, threshold: float = 0.45) -> np.ndarray:
    """Labels 0..k-1.  ``threshold`` is the cosine *distance* below which clusters merge."""
    if len(E) == 1:
        return np.zeros(1, dtype=int)
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import pdist

    Z = linkage(pdist(E, metric="cosine"), method="average")
    if n_speakers:
        lab = fcluster(Z, t=max(1, int(n_speakers)), criterion="maxclust")
    else:
        lab = fcluster(Z, t=threshold, criterion="distance")
    return lab - 1


@dataclass
class _Chunk:
    line: int
    a: int          # first token index in the line
    b: int          # last token index (inclusive)
    start: float
    end: float
    label: int = -1


def _chunks(doc: Document, pause: float) -> List[_Chunk]:
    out = []
    for li, ln in enumerate(doc.lines):
        toks = ln.tokens
        timed = [i for i, t in enumerate(toks) if t.start is not None and t.end is not None]
        if not timed:
            continue
        a = timed[0]
        for p, q in zip(timed, timed[1:]):
            if toks[q].start - toks[p].end >= pause:
                out.append(_Chunk(li, a, p, toks[a].start, toks[p].end))
                a = q
        out.append(_Chunk(li, a, timed[-1], toks[a].start, toks[timed[-1]].end))
    return out


def diarize(doc: Document, audio: Union[str, Path, np.ndarray], n_speakers: Optional[int] = None,
            threshold: float = 0.45, min_chunk: float = 0.8, pause: float = 0.35,
            device: Optional[str] = None) -> int:
    """Set ``Line.speaker`` ("S1", "S2", ...) on ``doc``, splitting lines at speaker
    changes.  Returns the number of speakers found."""
    from .audio.io import load_audio

    y = audio if isinstance(audio, np.ndarray) else load_audio(audio, 16000)
    sr = 16000
    chunks = _chunks(doc, pause)
    if not chunks:
        return 0
    long_ = [c for c in chunks if c.end - c.start >= min_chunk]
    if not long_:                       # very short utterances only: embed what there is
        long_ = sorted(chunks, key=lambda c: c.start - c.end)[:max(1, len(chunks) // 2)]
    clips = []
    for c in long_:
        a, b = int(max(0.0, c.start - 0.05) * sr), int((c.end + 0.05) * sr)
        if b - a > 12 * sr:            # long monologue: a centred 12 s is plenty
            mid = (a + b) // 2
            a, b = mid - 6 * sr, mid + 6 * sr
        clips.append(y[a:b])
    if n_speakers == 1:
        labels = np.zeros(len(long_), dtype=int)
    else:
        labels = cluster(embed(clips, device), n_speakers, threshold)
    for c, lab in zip(long_, labels):
        c.label = int(lab)
    anchored = [c for c in chunks if c.label >= 0]
    for c in chunks:                    # short chunks follow the nearest labelled chunk
        if c.label < 0:
            mid = (c.start + c.end) / 2
            c.label = min(anchored, key=lambda d: abs((d.start + d.end) / 2 - mid)).label
    # stable names by first appearance
    names = {}
    for c in sorted(chunks, key=lambda c: c.start):
        names.setdefault(c.label, f"S{len(names) + 1}")
    # rebuild lines: split where consecutive chunks of a line change speaker
    by_line = {}
    for c in chunks:
        by_line.setdefault(c.line, []).append(c)
    new_lines: List[Line] = []
    for li, ln in enumerate(doc.lines):
        cs = sorted(by_line.get(li, []), key=lambda c: c.a)
        if not cs:
            new_lines.append(ln)
            continue
        groups: List[Tuple[str, int, int]] = []
        for c in cs:
            spk = names[c.label]
            if groups and groups[-1][0] == spk:
                groups[-1] = (spk, groups[-1][1], c.b)
            else:
                groups.append((spk, c.a, c.b))
        if len(groups) == 1:
            ln.speaker = groups[0][0]
            new_lines.append(ln)
            continue
        bounds = [g[1] for g in groups[1:]]
        cuts = [0] + bounds + [len(ln.tokens)]
        for (spk, _, _), lo, hi in zip(groups, cuts, cuts[1:]):
            toks = ln.tokens[lo:hi]
            if not toks:
                continue
            toks[-1].space_after = False
            nl = Line(tokens=toks, style=ln.style, speaker=spk)
            nl.update_bounds()
            new_lines.append(nl)
        if ln.translation and new_lines:
            new_lines[-len(groups)].translation = ln.translation
    doc.lines = new_lines
    doc.metadata["speakers"] = len(names)
    return len(names)
