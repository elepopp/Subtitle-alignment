"""Choosing the best take (多候选择优) of a dubbed sentence.

The engine is stochastic: with the same references one seed reads a line flat,
another lively, a third drifts away from the voice.  Rejecting a take only when
words are misread keeps the first readable one however it sounds, so here every
take is measured and the candidates are ranked by:

* **misreading** - the character / word error rate of :func:`.dubbing.check_take`
  (above ``max_error`` a take only wins when nothing else passes)
* **voice** - CAM++ cosine similarity between the take and its speaker's voice
  reference: a take that drifts to another timbre loses
* **performance** - on a translated dub, how close the take's pitch range and
  pitch height (relative to the voice reference) are to the original line's: a
  flat reading of a shouted line loses to a lively one, and the other way round
* **fit** - on a timeline, how much faster than ``max_stretch`` allows the take
  would have to be played to fit before the next sentence

Each term is ~0..1 for an ordinary difference; missing measurements are left out.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np

log = logging.getLogger("subalign")

AN_SR = 16000
W_ERR, W_VOICE, W_PERF, W_FIT = 4.0, 2.0, 1.0, 5.0
SEMITONES = 6.0          # a pitch difference that costs 1
# CAM++ is unreliable on short clips: takes of the same voice under ~1.3 s scored down
# to 0.42 (vs 0.6-0.83 for longer ones; another person's voice: 0.0-0.2)
MIN_VOICE_S = 1.5
_EMB: Dict[str, np.ndarray] = {}
_PITCH: Dict[str, tuple] = {}


def _load16(path) -> np.ndarray:
    from ..audio.io import load_audio

    return load_audio(path, AN_SR).astype(np.float32)


def _trimmed(path) -> np.ndarray:
    from .dubbing import trim_take

    return trim_take(_load16(path), AN_SR)


def embedding(path) -> Optional[np.ndarray]:
    """CAM++ speaker embedding (L2-normalised) of a file; references are cached."""
    key = str(path)
    if key in _EMB:
        return _EMB[key]
    try:
        from ..diarize import embed

        y = _trimmed(path)
        if len(y) < AN_SR * 0.5:
            return None
        e = embed([y[:AN_SR * 15]])[0]
    except Exception as ex:             # optional model / dependency
        log.warning("speaker similarity unavailable: %s", ex)
        return None
    _EMB[key] = e
    return e


def voice_similarity(take, ref) -> Optional[float]:
    a, b = embedding(take), embedding(ref)
    _EMB.pop(str(take), None)           # only references are worth keeping
    if a is None or b is None:
        return None
    return round(float(a @ b), 3)


def ref_pitch(ref) -> Optional[float]:
    """Median pitch (semitones re 55 Hz) of a voice reference, cached."""
    from .expressive import pitch_stats

    key = str(ref)
    if key not in _PITCH:
        try:
            _PITCH[key] = pitch_stats(_trimmed(ref))
        except Exception as ex:
            log.warning("reference pitch unavailable: %s", ex)
            _PITCH[key] = (None, None)
    return _PITCH[key][0]


def slot_for(seg: Dict, nxt: Optional[Dict]) -> Optional[float]:
    """Seconds the sentence may take on the original timeline (until the next one)."""
    if seg.get("src_start") is None:
        return None
    if nxt is not None and nxt.get("src_start") is not None:
        return max(0.2, nxt["src_start"] - seg["src_start"] - 0.08)
    if seg.get("src_end") is not None:
        return seg["src_end"] - seg["src_start"] + 1.0
    return None


def measure(take_path: Path, seg: Dict, spk_ref: Optional[str], *, voice: bool = True,
            performance: bool = True, slot: Optional[float] = None) -> Dict:
    """What the ranking needs of one take (``None`` where not measured)."""
    from .expressive import pitch_stats

    out: Dict = {"voice": None, "f0": None, "spread": None, "length": None}
    y = _trimmed(take_path)
    out["length"] = round(len(y) / AN_SR, 3)
    if voice and spk_ref and out["length"] >= MIN_VOICE_S:
        out["voice"] = voice_similarity(take_path, spk_ref)
    if performance and (seg.get("src_spread") is not None or seg.get("src_f0") is not None):
        f0, spread = pitch_stats(y)
        base = ref_pitch(spk_ref) if spk_ref else None
        out["spread"] = None if spread is None else round(spread, 2)
        out["f0"] = None if f0 is None or base is None else round(f0 - base, 2)
    if slot:
        out["slot"] = round(slot, 3)
    return out


def score(m: Dict, seg: Dict, err: Optional[float], max_error: float, max_stretch: float) -> Dict:
    """Cost terms and their weighted ``total`` (lower is better)."""
    terms: Dict[str, float] = {}
    if err is not None:
        terms["err"] = W_ERR * err + (10.0 if err > max_error else 0.0)
    if m.get("voice") is not None:
        terms["voice"] = W_VOICE * (1.0 - m["voice"])
    perf = []
    if m.get("spread") is not None and seg.get("src_spread") is not None:
        perf.append(abs(m["spread"] - seg["src_spread"]) / SEMITONES)
    if m.get("f0") is not None and seg.get("src_f0") is not None:
        perf.append(abs(m["f0"] - seg["src_f0"]) / SEMITONES)
    if perf:
        terms["perf"] = W_PERF * float(np.mean(perf))
    if m.get("slot") and m.get("length"):
        terms["fit"] = W_FIT * max(0.0, m["length"] / m["slot"] - max_stretch)
    return {**{k: round(v, 3) for k, v in terms.items()}, "total": round(sum(terms.values()), 3)}
