"""High-level aligner: audio (+ optional script / lyrics) -> timed Document.

Pipeline
--------
1. decide speech vs song (auto-detected unless forced)
2. songs: separate vocals, all analysis runs on the vocal stem
3. acoustic features (onsets, pitch changes, vocal activity)
4. anchors - best available source:
     CTC forced alignment (torch + transformers)  ~ +-40 ms, sigma 0.05
     ASR word timestamps aligned to the script     ~ +-150 ms, sigma 0.15-0.25
     none (pure acoustic, songs only)
5. syllable DP refinement (``syllable_dp``) snaps every unit to the onsets
   / note changes consistent with the anchors and duration priors
6. keyless tokens (pure punctuation) and line bounds are filled in
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from ..audio.features import Features, analyze, detect_content_type
from ..audio.io import load_audio, save_audio
from ..models import Document, Line, Token
from ..text.tokenize import is_cjk, lines_from_text, normalize_key, syllable_weight
from .sequence import align_tokens, transfer_times
from .syllable_dp import SONG_PARAMS, SPEECH_PARAMS, DPParams, Unit, align_units
from .timing import enforce_monotonic, fill_missing_times

log = logging.getLogger("subalign")


@dataclass
class AlignConfig:
    mode: str = "auto"                  # auto | speech | song
    language: Optional[str] = None
    asr_backend: str = "auto"           # auto | faster-whisper | whisper | openai | funasr | none
    asr_model: Optional[str] = None
    asr_options: Dict[str, Any] = field(default_factory=dict)
    ctc: str = "auto"                   # auto | on | off
    ctc_model: Optional[str] = None
    separation: str = "auto"            # auto | uvr | demucs | dsp | none   (songs only)
    separation_model: Optional[str] = None
    refine: bool = True                 # acoustic DP refinement
    proofread: bool = True              # with a script, also run ASR to report differences
    device: Optional[str] = None
    chunk_units: int = 400


@dataclass
class AlignResult:
    document: Document
    mode: str
    anchor_source: str
    report: Optional[Dict] = None
    stems: Dict[str, Path] = field(default_factory=dict)
    transcript: Optional[Any] = None


# ------------------------------------------------------------------ helpers
def _as_document(script: Union[None, str, Document]) -> Optional[Document]:
    if script is None:
        return None
    if isinstance(script, Document):
        return script.copy()
    from ..formats import detect_text_format, read_text

    fmt = detect_text_format(script)
    if fmt == "txt":
        return Document(lines=lines_from_text(script))
    return read_text(script, fmt)


def _keyed(tokens: Sequence[Token]) -> List[int]:
    return [i for i, t in enumerate(tokens) if normalize_key(t.text)]


def _split_by_pauses(doc: Document, gap: float, max_units: int = 40) -> Document:
    """Song transcripts: one line per sung phrase."""
    out: List[Line] = []
    for ln in doc.lines:
        cur: List[Token] = []
        for t in ln.tokens:
            if cur and t.start is not None and cur[-1].end is not None and (
                    t.start - cur[-1].end >= gap or len(cur) >= max_units):
                cur[-1].space_after = False
                out.append(Line(tokens=cur))
                cur = []
            cur.append(t)
        if cur:
            cur[-1].space_after = False
            out.append(Line(tokens=cur))
    for ln in out:
        ln.update_bounds()
    doc.lines = out
    return doc


def _get_asr(cfg: AlignConfig):
    from ..asr import get_backend

    kw: Dict[str, Any] = {}
    if cfg.asr_model:
        kw["model"] = cfg.asr_model
    if cfg.device and cfg.asr_backend in ("faster-whisper", "whisper", "funasr"):
        kw["device"] = cfg.device
    return get_backend(cfg.asr_backend, **kw)


def _try_ctc(cfg: AlignConfig):
    if cfg.ctc == "off":
        return None
    try:
        from .ctc import HFCTCEmitter

        return HFCTCEmitter(cfg.ctc_model, cfg.language, cfg.device)
    except Exception as e:  # missing torch/transformers or model download failure
        if cfg.ctc == "on":
            raise
        log.info("CTC aligner unavailable (%s); falling back", e)
        return None


# ------------------------------------------------------------------ main entry
def align_audio(audio_path: Union[str, Path], script: Union[None, str, Document] = None,
                cfg: Optional[AlignConfig] = None, workdir: Union[str, Path, None] = None) -> AlignResult:
    cfg = cfg or AlignConfig()
    audio_path = Path(audio_path)
    workdir = Path(workdir) if workdir else audio_path.parent / f"{audio_path.stem}.subalign"
    workdir.mkdir(parents=True, exist_ok=True)

    y = load_audio(audio_path, 16000)
    mode = cfg.mode
    if mode == "auto":
        mode = detect_content_type(analyze(y))
        log.info("content type detected: %s", mode)

    stems: Dict[str, Path] = {}
    analysis_path = audio_path
    if mode == "song" and cfg.separation != "none":
        from ..separate import separate

        try:
            stems = separate(audio_path, workdir, backend=cfg.separation, model=cfg.separation_model,
                             device=cfg.device)
            analysis_path = stems["vocals"]
            y = load_audio(analysis_path, 16000)
        except Exception as e:
            log.warning("vocal separation failed (%s); analysing the mix", e)
    feats = analyze(y)
    params = SONG_PARAMS if mode == "song" else SPEECH_PARAMS

    doc = _as_document(script)
    transcript = None
    report = None
    anchor_source = "none"

    if doc is None:
        # ---------------------------------------------------- recognition mode
        if cfg.asr_backend == "none":
            raise ValueError("no script given and ASR disabled - nothing to align")
        from ..asr import transcript_to_document

        asr = _get_asr(cfg)
        transcript = asr.transcribe(str(analysis_path), language=cfg.language, **cfg.asr_options)
        doc = transcript_to_document(transcript, kind=mode)
        doc.language = cfg.language or transcript.language
        if mode == "song":
            _split_by_pauses(doc, gap=0.6)
        for t in doc.tokens():
            t.confidence = min(t.confidence, 0.9)
        anchor_source = "asr"
        _set_anchors_from_times(doc, sigma=0.12 if mode == "speech" else 0.2)
    else:
        doc.kind = mode
        doc.language = doc.language or cfg.language
        prior_times = _existing_line_times(doc)
        for t in doc.tokens():
            t.start = t.end = None
        emitter = _try_ctc(cfg)
        if emitter is not None:
            _anchors_from_ctc(doc, emitter, y)
            anchor_source = "ctc"
        need_asr = (anchor_source == "none" or cfg.proofread) and cfg.asr_backend != "none"
        if need_asr:
            try:
                asr = _get_asr(cfg)
                prompt = " ".join(ln.text for ln in doc.lines)[:600]
                transcript = asr.transcribe(str(analysis_path), language=cfg.language, prompt=prompt,
                                            **cfg.asr_options)
                report = _anchors_from_asr(doc, transcript, use_times=anchor_source == "none",
                                           sigma=0.15 if mode == "speech" else 0.25)
                if anchor_source == "none":
                    anchor_source = "asr"
            except ImportError as e:
                log.warning("ASR unavailable (%s)", e)
        if anchor_source == "none" and prior_times:
            _anchors_from_line_times(doc, prior_times)
            anchor_source = "script-timestamps"
        if anchor_source == "none" and mode == "speech":
            log.warning("no anchors available for speech - result relies on acoustics only")

    if cfg.refine:
        _refine(doc, feats, params, cfg.chunk_units)
    else:
        _apply_anchor_times(doc)
    _finish(doc)
    return AlignResult(document=doc, mode=mode, anchor_source=anchor_source, report=report, stems=stems,
                       transcript=transcript)


# ------------------------------------------------------------------ anchors
_ANCHOR = "_anchor"


def _set_anchor(tok: Token, t: Optional[float], conf: float, sigma: float) -> None:
    setattr(tok, _ANCHOR, (t, conf, sigma) if t is not None else None)


def _get_anchor(tok: Token) -> Optional[Tuple[float, float, float]]:
    return getattr(tok, _ANCHOR, None)


def _set_anchors_from_times(doc: Document, sigma: float) -> None:
    for t in doc.tokens():
        _set_anchor(t, t.start, max(0.3, min(1.0, t.confidence)), sigma)


def _existing_line_times(doc: Document) -> List[Optional[Tuple[float, float]]]:
    out = [(ln.start, ln.end) if ln.start is not None else None for ln in doc.lines]
    return out if any(out) else []


def _anchors_from_line_times(doc: Document, times) -> None:
    for ln, tt in zip(doc.lines, times):
        if tt and ln.tokens:
            _set_anchor(ln.tokens[0], tt[0], 0.7, 0.3)


def _anchors_from_ctc(doc: Document, emitter, y: np.ndarray) -> None:
    from .ctc import ctc_align_tokens

    spans = ctc_align_tokens(emitter, y, [[t.text for t in ln.tokens] for ln in doc.lines])
    for ln, sp in zip(doc.lines, spans):
        for tok, s in zip(ln.tokens, sp):
            if s.start is None:
                _set_anchor(tok, None, 0, 0)
                continue
            conf = float(np.clip(math.exp(max(s.score, -20.0)) * 1.2, 0.2, 1.0))
            _set_anchor(tok, s.start, conf, 0.05)
            tok.start, tok.end, tok.confidence = s.start, s.end, conf


def _anchors_from_asr(doc: Document, transcript, use_times: bool, sigma: float) -> Dict:
    from ..asr import transcript_tokens
    from ..proofread import diff_report

    hyp = transcript_tokens(transcript)
    ref = [t for ln in doc.lines for t in ln.tokens]
    ops = align_tokens(ref, hyp)
    report = diff_report(doc.lines, hyp, ops)
    if use_times:
        saved = [(t.start, t.end) for t in ref]
        transfer_times(ref, hyp, ops=ops, interpolate=False)
        for t in ref:
            if t.start is not None:
                _set_anchor(t, t.start, 0.8 * max(0.3, t.confidence), sigma)
            else:
                _set_anchor(t, None, 0, 0)
        for t, (a, b) in zip(ref, saved):
            if a is not None and t.start is None:
                t.start, t.end = a, b
    return report


def _apply_anchor_times(doc: Document) -> None:
    for t in doc.tokens():
        a = _get_anchor(t)
        if a and t.start is None:
            t.start = a[0]


# ------------------------------------------------------------------ refinement
def _refine(doc: Document, feats: Features, params: DPParams, chunk_units: int) -> None:
    entries: List[Tuple[Token, Unit]] = []
    for ln in doc.lines:
        keyed = _keyed(ln.tokens)
        for k, idx in enumerate(keyed):
            tok = ln.tokens[idx]
            a = _get_anchor(tok)
            u = Unit(weight=syllable_weight(tok), line_start=k == 0, line_end=k == len(keyed) - 1)
            if a:
                u.anchor, u.anchor_conf, u.anchor_sigma = a
            entries.append((tok, u))
    if not entries:
        return
    total = feats.n * feats.hop_s
    # chunk long inputs at line boundaries between confident anchors
    chunks: List[Tuple[int, int]] = []
    s = 0
    while s < len(entries):
        e = min(len(entries), s + chunk_units)
        if e < len(entries):
            # move e back to a line start that has an anchor
            k = e
            while k > s + chunk_units // 2 and not (entries[k][1].line_start and entries[k][1].anchor is not None):
                k -= 1
            if k > s + chunk_units // 2:
                e = k
        chunks.append((s, e))
        s = e
    has_anchor_global = any(u.anchor is not None for _, u in entries)
    prev_end = 0.0
    for ci, (s, e) in enumerate(chunks):
        units = [u for _, u in entries[s:e]]
        if len(chunks) > 1 and has_anchor_global:
            anchors = [u.anchor for u in units if u.anchor is not None]
            nxt = next((u.anchor for _, u in entries[e:] if u.anchor is not None), None) if e < len(entries) else None
            t0 = max(prev_end, (min(anchors) - 4.0) if anchors else prev_end)
            t1 = (nxt + 0.5) if nxt is not None else total
            t1 = max(t1, t0 + 1.0)
        else:
            t0, t1 = prev_end, total
        times = align_units(units, feats, params, t0=t0, t1=t1)
        for (tok, _), (a, b) in zip(entries[s:e], times):
            tok.start, tok.end = float(a), float(b)
        prev_end = float(times[-1, 1]) if len(times) else prev_end


def _finish(doc: Document) -> None:
    for ln in doc.lines:
        # keyless tokens (lone symbols) borrow time from neighbours
        for i, t in enumerate(ln.tokens):
            if not normalize_key(t.text):
                t.start = t.end = None
    fill_missing_times(doc)
    enforce_monotonic(list(doc.tokens()))
    for t in doc.tokens():
        if hasattr(t, _ANCHOR):
            delattr(t, _ANCHOR)
    for ln in doc.lines:
        ln.update_bounds()


# ------------------------------------------------------------------ convenience
def align_with_features(doc: Document, feats: Features, mode: str = "song",
                        anchors: Optional[Sequence[Optional[float]]] = None, sigma: float = 0.1) -> Document:
    """Align a document directly against precomputed features (testing / custom
    pipelines).  ``anchors``: optional start-time guess per keyed token."""
    toks = [t for ln in doc.lines for t in ln.tokens if normalize_key(t.text)]
    if anchors is not None:
        for t, a in zip(toks, anchors):
            _set_anchor(t, a, 0.9 if a is not None else 0.0, sigma)
    _refine(doc, feats, SONG_PARAMS if mode == "song" else SPEECH_PARAMS, 400)
    _finish(doc)
    return doc
