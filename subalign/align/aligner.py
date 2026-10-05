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
from ..text.tokenize import (guess_language, is_cjk, is_credit_line, lines_from_text, normalize_key,
                              syllable_weight)
from .sequence import align_tokens, transfer_times
from .syllable_dp import SONG_PARAMS, SPEECH_PARAMS, DPParams, Unit, align_units
from .timing import _interp_run, enforce_monotonic, fill_missing_times

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
    # ---- recognition (no script) quality
    verbatim: bool = False              # keep 嗯/呃/repeats: Whisper is prompted not to tidy them
    asr_prompt: str = ""                # names / terms / topic given to the recogniser as a hint
    screen_hallucinations: bool = True  # drop text written over music / silence (asr.hallucination)
    cross_check: Optional[str] = None   # second ASR backend; disagreements are marked for review


@dataclass
class AlignResult:
    document: Document
    mode: str
    anchor_source: str
    report: Optional[Dict] = None
    stems: Dict[str, Path] = field(default_factory=dict)
    transcript: Optional[Any] = None
    # recognition quality notes: {"hallucinations": [...], "disagreements": [...], "cross_check": name}
    asr_notes: Dict[str, Any] = field(default_factory=dict)


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


def _try_ctc(cfg: AlignConfig, language: Optional[str] = None):
    if cfg.ctc == "off":
        return None
    try:
        from .ctc import HFCTCEmitter

        return HFCTCEmitter(cfg.ctc_model, language or cfg.language, cfg.device)
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
    asr_notes: Dict[str, Any] = {}
    credits: List[Tuple[int, Line]] = []
    transcript = None
    report = None
    anchor_source = "none"

    if doc is None:
        # ---------------------------------------------------- recognition mode
        if cfg.asr_backend == "none":
            raise ValueError("no script given and ASR disabled - nothing to align")
        from ..asr import transcript_to_document

        asr = _get_asr(cfg)
        opts = dict(cfg.asr_options)
        prompt = _recognition_prompt(cfg)
        if prompt and "prompt" not in opts:
            opts["prompt"] = prompt
        transcript = asr.transcribe(str(analysis_path), language=cfg.language, **opts)
        if cfg.screen_hallucinations:
            from ..asr.hallucination import screen

            transcript, notes = screen(transcript, feats, song=mode == "song")
            if notes:
                asr_notes["hallucinations"] = notes
                log.warning("ASR: %d likely hallucinated segment(s) removed, %d flagged",
                            sum(n["action"] == "removed" for n in notes), sum(n["action"] == "flagged" for n in notes))
        doc = transcript_to_document(transcript, kind=mode)
        doc.language = cfg.language or transcript.language
        if mode == "song":
            _split_by_pauses(doc, gap=0.6)
        for t in doc.tokens():
            t.confidence = min(t.confidence, 0.9)
        anchor_source = "asr"
        _set_anchors_from_times(doc, sigma=0.12 if mode == "speech" else 0.2)
        # second pass: CTC forced alignment of the recognised text.  ASR word times are coarse and
        # put pauses in the wrong places; CTC onsets are ~±40 ms.  Checked against the ASR times.
        emitter = _try_ctc(cfg, doc.language) if any(True for _ in doc.tokens()) else None
        if emitter is not None:
            toks = list(doc.tokens())
            saved = [(t.start, t.end, t.confidence, _get_anchor(t)) for t in toks]
            quality = _anchors_from_ctc(doc, emitter, y)
            pairs = [(t, s[0]) for t, s in zip(toks, saved) if _get_anchor(t) is not None]
            problem = "no aligned units" if not np.isfinite(quality) else _disagreement(doc, pairs, "ASR timestamps")
            if problem:
                log.warning("CTC alignment of the transcript rejected (%s); keeping ASR timings", problem)
                for t, (a, b, c, anc) in zip(toks, saved):
                    t.start, t.end, t.confidence = a, b, c
                    setattr(t, _ANCHOR, anc)
            else:
                # units CTC could not place keep their ASR anchor
                for t, (a, b, c, anc) in zip(toks, saved):
                    if _get_anchor(t) is None and anc is not None:
                        setattr(t, _ANCHOR, anc)
                        t.start, t.end = a, b
                    if c < 0.5:      # the recogniser itself was unsure of this word: keep that visible
                        t.confidence = min(t.confidence, c)
                anchor_source = "asr+ctc"
    else:
        doc.kind = mode
        doc.language = doc.language or cfg.language or guess_language("\n".join(ln.text for ln in doc.lines))
        # credit lines (作词：xx, "Artist - Title") are not sung: keep them out of the acoustic alignment
        credits = _pop_credit_lines(doc)
        prior_times = _existing_line_times(doc)
        for t in doc.tokens():
            t.start = t.end = None
        emitter = _try_ctc(cfg, doc.language)
        if emitter is not None:
            quality = _anchors_from_ctc(doc, emitter, y)
            problem = "no aligned units" if not np.isfinite(quality) else _disagreement(
                doc, [(ln.tokens[0], tt[0]) for ln, tt in zip(doc.lines, prior_times) if tt and ln.tokens],
                "script line timestamps")
            if problem:
                log.warning("CTC alignment rejected (%s)", problem)
                _clear_anchors(doc)
            else:
                anchor_source = "ctc"
        need_asr = (anchor_source == "none" or cfg.proofread) and cfg.asr_backend != "none"
        if need_asr:
            try:
                asr = _get_asr(cfg)
                prompt = " ".join(ln.text for ln in doc.lines)[:600]
                transcript = asr.transcribe(str(analysis_path), language=cfg.language or doc.language,
                                            prompt=prompt, **cfg.asr_options)
                if anchor_source == "ctc":
                    # cross-check CTC against the recogniser's own timing on exactly matching units
                    problem = _disagreement(doc, _asr_matches(doc, transcript), "ASR timestamps")
                    if problem:
                        log.warning("CTC alignment rejected (%s); using ASR anchors", problem)
                        _clear_anchors(doc)
                        anchor_source = "none"
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
    if credits:
        n_sung = len(doc.lines)
        _restore_credit_lines(doc, credits)
        if report is not None:   # report line numbers refer to sung lines: map back
            drop = {i for i, _ in credits}
            orig = [i for i in range(n_sung + len(credits)) if i not in drop]
            for it in report.get("issues", []):
                it["line"] = orig[min(it["line"], len(orig) - 1)]
    if cfg.cross_check and script is None and doc.lines:
        try:
            asr_notes.update(_cross_check(doc, cfg, str(analysis_path)))
        except Exception as e:  # the second opinion is optional
            log.warning("cross-check with %s failed: %s", cfg.cross_check, e)
    if asr_notes.get("disagreements"):
        doc.metadata["review"] = [{"start": d["start"], "end": d["end"], "text": d["text"], "other": d["other"]}
                                  for d in asr_notes["disagreements"]]
    return AlignResult(document=doc, mode=mode, anchor_source=anchor_source, report=report, stems=stems,
                       transcript=transcript, asr_notes=asr_notes)


def _recognition_prompt(cfg: AlignConfig) -> str:
    """Initial prompt for the recogniser: disfluency style (verbatim) + the user's names / terms."""
    parts = []
    if cfg.verbatim:
        from ..asr.base import DISFLUENCY_PROMPT

        parts.append(DISFLUENCY_PROMPT)
    if cfg.asr_prompt.strip():
        parts.append(cfg.asr_prompt.strip())
    return " ".join(parts)[:400]


def _cross_check(doc: Document, cfg: AlignConfig, audio: str) -> Dict[str, Any]:
    """Transcribe again with a second backend; every place the two disagree is marked
    (confidence lowered, listed) so a human checks exactly those spots."""
    from ..asr import get_backend, transcript_tokens

    kw: Dict[str, Any] = {}
    if cfg.device and cfg.cross_check in ("faster-whisper", "whisper", "funasr"):
        kw["device"] = cfg.device
    other = get_backend(cfg.cross_check, **kw).transcribe(audio, language=doc.language)
    hyp = transcript_tokens(other)
    ref = [t for ln in doc.lines for t in ln.tokens]
    ops = align_tokens(ref, hyp)
    out: List[Dict[str, Any]] = []
    run: List = []

    def flush():
        if not run:
            return
        r_idx = [o.ref for o in run if o.ref is not None]
        h_txt = "".join(hyp[o.hyp].text for o in run if o.hyp is not None)
        homophone = all(o.op == "sub" and o.sim >= 0.5 for o in run)
        if r_idx:
            for i in r_idx:
                ref[i].confidence = min(ref[i].confidence, 0.45 if homophone else 0.3)
            out.append({"start": ref[r_idx[0]].start, "end": ref[r_idx[-1]].end,
                        "text": "".join(ref[i].text for i in r_idx), "other": h_txt,
                        "kind": "homophone" if homophone else ("missing" if not h_txt else "different")})
        elif h_txt.strip():
            near = next((ref[o.ref] for o in ops[ops.index(run[0]):] if o.ref is not None), ref[-1])
            out.append({"start": near.start, "end": near.start, "text": "", "other": h_txt, "kind": "extra"})
        run.clear()

    for o in ops:
        if o.op == "match":
            flush()
        else:
            run.append(o)
    flush()
    return {"cross_check": cfg.cross_check, "disagreements": out}


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


# CTC scores do not separate good from broken alignments (a wrong blank id scored -5 vs -5 for
# a correct model), so CTC is validated against independent timing evidence instead
MAX_ANCHOR_SPREAD = 2.0   # s, median deviation from the reference after removing a global offset


def _disagreement(doc: Document, pairs: Sequence[Tuple[Token, float]], what: str) -> Optional[str]:
    """``pairs``: (token with CTC time, independent reference time).  Returns a reason
    string when they disagree beyond a constant offset (the LRC may be shifted)."""
    d = np.array([tok.start - ref for tok, ref in pairs if tok.start is not None and ref is not None])
    if len(d) < 3:
        return None
    spread = float(np.median(np.abs(d - np.median(d))))
    if spread > MAX_ANCHOR_SPREAD:
        return f"differs from {what} by {spread:.1f}s (median, {len(d)} points)"
    return None


def _asr_matches(doc: Document, transcript) -> List[Tuple[Token, float]]:
    from ..asr import transcript_tokens

    hyp = transcript_tokens(transcript)
    ref = [t for ln in doc.lines for t in ln.tokens]
    return [(ref[o.ref], hyp[o.hyp].start) for o in align_tokens(ref, hyp)
            if o.op == "match" and hyp[o.hyp].start is not None]


def _clear_anchors(doc: Document) -> None:
    for t in doc.tokens():
        _set_anchor(t, None, 0, 0)
        t.start = t.end = None


def _pop_credit_lines(doc: Document) -> List[Tuple[int, Line]]:
    """Remove credit lines from ``doc``; returns (original index, line) pairs."""
    if len(doc.lines) < 3:
        return []
    credits = [(i, ln) for i, ln in enumerate(doc.lines) if is_credit_line(ln.text, doc.metadata, i)]
    if len(credits) >= len(doc.lines) - 1:
        return []
    drop = {i for i, _ in credits}
    doc.lines = [ln for i, ln in enumerate(doc.lines) if i not in drop]
    return credits


def _restore_credit_lines(doc: Document, credits: List[Tuple[int, Line]]) -> None:
    """Put credit lines back: keep their own timestamps when they had some, otherwise
    share the time before the first sung line (or after the previous line)."""
    lines = list(doc.lines)
    for idx, ln in credits:
        lines.insert(min(idx, len(lines)), ln)
    first_sung = next((l.start for l in doc.lines if l.start is not None), 0.0)
    for k, ln in enumerate(lines):
        if not any(ln is c for _, c in credits):
            continue
        nxt = next((l.start for l in lines[k + 1:] if l.start is not None and not any(l is c for _, c in credits)),
                   None)
        start = ln.start
        if start is None:
            prev = next((l.end for l in reversed(lines[:k]) if l.end is not None), None)
            start = prev if prev is not None else 0.0
        end = start + 0.4 * max(1, len(ln.tokens))
        limit = nxt if nxt is not None else (first_sung if k == 0 else None)
        if limit is not None and limit > start:
            end = min(end, limit)
        for t in ln.tokens:
            t.start = t.end = None
        _interp_run(ln.tokens, start, max(end, start + 0.01 * len(ln.tokens)))
        ln.update_bounds()
    doc.lines = lines


def _anchors_from_ctc(doc: Document, emitter, y: np.ndarray) -> float:
    """Set CTC anchors; returns the median per-unit score (alignment quality)."""
    from .ctc import ctc_align_tokens

    spans = ctc_align_tokens(emitter, y, [[t.text for t in ln.tokens] for ln in doc.lines])
    scores = [s.score for sp in spans for s in sp if s.start is not None and np.isfinite(s.score)]
    for ln, sp in zip(doc.lines, spans):
        for tok, s in zip(ln.tokens, sp):
            if s.start is None:
                _set_anchor(tok, None, 0, 0)
                continue
            conf = float(np.clip(math.exp(max(s.score, -20.0)) * 1.2, 0.2, 1.0))
            _set_anchor(tok, s.start, conf, 0.05)
            tok.start, tok.end, tok.confidence = s.start, s.end, conf
    return float(np.median(scores)) if scores else -np.inf


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
