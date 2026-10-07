"""Per-line voice references (逐句参考): every dubbed sentence is generated with its
own original line as the reference, so voice and delivery come from that line.

Subtitle timing - auto-generated captions above all - is often a few hundred ms
off, and a clip cut at the subtitle times carries the end of the line before or
the start of the next one into the reference (and so into the dub).  So:

1. **forced alignment** - each run of sentences is aligned against the vocals with
   CTC, with the sentences around it as context and a garbage state between lines
   that absorbs speech the subtitles do not have; a line's span is its first to
   last aligned word
2. **cut in the gap** - a clip ends half-way between this line's last word and the
   next line's first word (never inside a word), with at most ``pad`` of air
3. **listen to it** - Whisper transcribes every clip; words heard before the line's
   first word or after its last one are another line's and are trimmed off (at the
   gap between the words), and a clip that misses much of its text is flagged and
   not used
4. **long enough** - IndexTTS clones poorly from a second of audio: a line shorter
   than ``min_ref_s`` is joined with the speaker's own neighbouring lines (each cut
   and checked the same way)
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

log = logging.getLogger("subalign")

AN_SR = 16000


@dataclass
class LineRefConfig:
    pad: float = 0.12               # air kept before / after the line
    min_ref_s: float = 3.0          # shorter lines are joined with the speaker's neighbouring lines
    max_join: int = 3
    group: int = 6                  # sentences per forced-alignment window (+1 of context each side)
    margin: float = 1.5             # the window extends this far past the subtitle times
    max_shift: float = 3.0          # an aligned span further than this from the subtitle is not trusted
    verify: bool = True             # Whisper check / trim of every clip
    max_missing: float = 0.34       # a clip missing more of its words than this is not used


_TAGS = re.compile(r"[\[(（【][^\])）】]*[\])）】]|♪+|>>")


def spoken(text: str) -> str:
    """The words actually said: sound-event tags ([music], (applause), ♪, >>) removed."""
    return re.sub(r"\s+", " ", _TAGS.sub(" ", text or "")).strip()


def keys(text: str) -> List[str]:
    from ..text.tokenize import normalize_key, tokenize

    return [k for k in (normalize_key(t.text) for t in tokenize(spoken(text))) if k]


# ------------------------------------------------------------------ 1. forced alignment
def _device_for(min_free_gb: float = 2.5) -> str:
    """GPU when it has room (an IndexTTS worker may hold most of an 11 GB card)."""
    try:
        import torch

        if torch.cuda.is_available() and torch.cuda.mem_get_info()[0] / 2 ** 30 >= min_free_gb:
            return "cuda"
    except Exception:
        pass
    return "cpu"


def refine_spans(spans: Sequence[Tuple[float, float]], texts: Sequence[str], y16: np.ndarray, lang: str,
                 cfg: LineRefConfig, emitter=None) -> List[Optional[Tuple[float, float]]]:
    """Aligned (start, end) of every line's words, None where alignment was not
    possible or not plausible."""
    from ..align.ctc import HFCTCEmitter, ctc_align_tokens

    n = len(spans)
    out: List[Optional[Tuple[float, float]]] = [None] * n
    ks = [keys(t) for t in texts]
    if not any(ks):
        return out
    own = emitter is None
    if own:
        emitter = HFCTCEmitter(None, lang, _device_for())
    total = len(y16) / AN_SR
    try:
        for i in range(0, n, cfg.group):
            idx = list(range(max(0, i - 1), min(n, i + cfg.group + 1)))
            w0 = max(0.0, min(spans[k][0] for k in idx) - cfg.margin)
            w1 = min(total, max(spans[k][1] for k in idx) + cfg.margin)
            if w1 - w0 < 0.2:
                continue
            res = ctc_align_tokens(emitter, y16[int(w0 * AN_SR):int(w1 * AN_SR)], [ks[k] for k in idx])
            for k, sp in zip(idx, res):
                if not (i <= k < i + cfg.group):
                    continue
                timed = [s for s in sp if s.start is not None and s.end is not None]
                if not timed:
                    continue
                a, b = w0 + timed[0].start, w0 + timed[-1].end
                if b > a and abs(a - spans[k][0]) <= cfg.max_shift and abs(b - spans[k][1]) <= cfg.max_shift:
                    out[k] = (round(a, 3), round(b, 3))
    finally:
        if own:
            _release(emitter)
    return out


def _release(emitter) -> None:
    try:
        import torch

        del emitter.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


# ------------------------------------------------------------------ 2. cut in the gap
def cut_bounds(spans: Sequence[Tuple[float, float]], i: int, pad: float, total: float) -> Tuple[float, float]:
    """Clip of line ``i``: its words plus at most ``pad``, never past the middle of the
    gap to the neighbouring lines (or into them when they overlap)."""
    a, b = spans[i]
    lo, hi = max(0.0, a - pad), min(total, b + pad)
    if i > 0:
        pe = spans[i - 1][1]
        lo = max(lo, (pe + a) / 2 if pe < a else a)
    if i + 1 < len(spans):
        ns = spans[i + 1][0]
        hi = min(hi, (b + ns) / 2 if ns > b else b)
    return lo, max(lo, hi)


# ------------------------------------------------------------------ 3. listen to it
def check_clip(heard_words: Sequence[Tuple[str, float, float]], text: str) -> Dict:
    """Compare what Whisper heard in a clip (word, start, end) with the line's text.

    Compared character by character (``follow-up`` heard as ``follow`` + ``-up``, or a
    misspelt name, is still the line's): a heard word belongs to the line when most of
    its characters align with the text.  Returns ``{"heard", "extra_head",
    "extra_tail", "missing", "trim": [a, b]}``: heard words before the first / after the
    last word of the line are another line's, ``trim`` is where to cut them off (None =
    nothing to cut), ``missing`` the share of the line's characters not heard."""
    from ..align.sequence import align_keys
    from ..text.tokenize import normalize_key

    ref = list("".join(keys(text)))
    words = [(normalize_key(w), s, e) for w, s, e in heard_words]
    words = [(k, s, e) for k, s, e in words if k]
    hyp, owner = [], []
    for wi, (k, _, _) in enumerate(words):
        hyp += list(k)
        owner += [wi] * len(k)
    heard = "".join(w for w, _, _ in heard_words).strip()
    out = {"heard": heard, "extra_head": 0, "extra_tail": 0, "missing": 1.0 if ref else 0.0, "trim": [None, None]}
    if not ref or not hyp:
        return out
    ops = align_keys(ref, hyp)
    hit = [o for o in ops if o.op == "match"]
    if not hit:
        return out
    per_word = np.zeros(len(words))
    for o in hit:
        per_word[owner[o.hyp]] += 1
    mine = [wi for wi, (k, _, _) in enumerate(words) if per_word[wi] >= 0.5 * len(k)]
    if not mine:
        return out
    first, last = mine[0], mine[-1]
    out["missing"] = round(1 - len({o.ref for o in hit}) / len(ref), 3)
    out["extra_head"], out["extra_tail"] = first, len(words) - 1 - last
    if first > 0:
        out["trim"][0] = round((words[first - 1][2] + words[first][1]) / 2, 3)
    if last < len(words) - 1:
        out["trim"][1] = round((words[last][2] + words[last + 1][1]) / 2, 3)
    return out


def _listen(asr, clip16: np.ndarray, lang: str, tmp: Path) -> List[Tuple[str, float, float]]:
    from ..audio.io import save_audio

    save_audio(tmp, clip16, AN_SR)
    try:
        tr = asr.transcribe(str(tmp), language=lang, vad=False)
    finally:
        tmp.unlink(missing_ok=True)
    return [(w.text, w.start, w.end) for s in tr.segments for w in s.words]


# ------------------------------------------------------------------ all together
def _fade(y: np.ndarray, sr: int, ms: float = 10.0) -> np.ndarray:
    y = y.astype(np.float32).copy()
    f = min(int(ms / 1000 * sr), len(y) // 4)
    if f > 1:
        y[:f] *= np.linspace(0, 1, f)
        y[-f:] *= np.linspace(1, 0, f)
    return y


def build(spans: List[Tuple[float, float]], aligned: List[bool], texts: List[str], speakers: List[str],
          usable: List[bool], y: np.ndarray, sr: int, y16: np.ndarray, lang: Optional[str], out_dir: Path,
          names: List[str], cfg: Optional[LineRefConfig] = None, asr=None) -> List[Dict]:
    """One entry per line: ``{"line": path | None, "line_check": {"aligned", "heard",
    "trimmed", "missing", "joined", "ok"}}``.  ``spans``: the lines' words (aligned by
    :func:`refine_spans` where ``aligned``); ``usable``: False for lines that get no
    reference (sound events, silence); ``names``: file stems."""
    cfg = cfg or LineRefConfig()
    n = len(spans)
    total = len(y) / sr
    best = list(spans)
    # 2. cut, 3. listen and trim
    if cfg.verify and lang and asr is None:
        try:
            from .dubbing import asr_backend

            asr = asr_backend()
        except Exception as e:
            log.warning("reference check skipped: %s", e)
    res: List[Dict] = []
    clips: List[Optional[np.ndarray]] = []
    for i in range(n):
        if not usable[i]:
            res.append({"line": None, "line_check": None})
            clips.append(None)
            continue
        lo, hi = cut_bounds(best, i, cfg.pad, total)
        chk = {"aligned": bool(aligned[i]), "heard": None, "trimmed": [0, 0], "missing": None,
               "joined": [], "ok": True}
        if asr is not None and hi - lo >= 0.3:
            try:
                c16 = y16[int(lo * AN_SR):int(hi * AN_SR)]
                r = check_clip(_listen(asr, c16, lang, out_dir / f".check_{i}.wav"), texts[i])
                chk.update(heard=r["heard"], missing=r["missing"], trimmed=[r["extra_head"], r["extra_tail"]])
                a2, b2 = r["trim"]             # seconds into the clip
                base = lo
                if a2 is not None:
                    lo = base + a2
                if b2 is not None:
                    hi = base + b2
                chk["ok"] = r["missing"] <= cfg.max_missing
            except Exception as e:
                log.warning("reference check of line %d failed: %s", i + 1, e)
        clip = y[int(lo * sr):int(hi * sr)]
        clips.append(clip if chk["ok"] and len(clip) > 0.3 * sr else None)
        res.append({"line": None, "line_check": chk})
    # 4. long enough: join the speaker's own neighbouring lines
    gap = np.zeros(int(0.15 * sr), np.float32)
    for i in range(n):
        if clips[i] is None:
            continue
        parts = {i: clips[i]}
        length = len(clips[i]) / sr
        order = sorted((k for k in range(n) if k != i and speakers[k] == speakers[i] and clips[k] is not None),
                       key=lambda k: abs(k - i))
        for k in order:
            if length >= cfg.min_ref_s or len(parts) > cfg.max_join:
                break
            parts[k] = clips[k]
            length += len(clips[k]) / sr + 0.15
        seq = []
        for k in sorted(parts):
            seq += [parts[k], gap]
        ref = np.concatenate(seq[:-1])[:int(15 * sr)]
        path = out_dir / f"{names[i]}.wav"
        _save(path, _fade(ref, sr), sr)
        res[i]["line"] = str(path)
        res[i]["line_check"]["joined"] = [names[k] for k in sorted(parts) if k != i]
    return res


def _save(path: Path, y: np.ndarray, sr: int) -> None:
    from ..audio.io import save_audio

    path.parent.mkdir(parents=True, exist_ok=True)
    save_audio(path, y, sr)
