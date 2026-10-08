"""AI voice-over projects (AI 配音): sentence-level synthesis with a cloned voice
(reference A) and an optional emotion reference (B), automatic QA, re-generation
of single sentences, assembly with natural pacing, and post-processing.

Engine: IndexTTS-2.5 in its own environment (``webui/tts_worker.py``), driven
through :class:`TTSWorker`.

What makes AI speech sound less synthetic (``naturalize``):
* every take is trimmed (models often add a breath of noise or a click at the edges)
* sentence loudness is matched (takes come out at different levels)
* speaking rate is evened out: sentences more than ``rate_tolerance`` off the
  median are time-stretched (pitch-preserving) toward it
* pauses follow punctuation and vary a little (perfectly regular pauses are robotic)
* real breaths harvested from the voice reference are placed before sentences
  that follow a pause
* a faint room tone runs under everything: digital silence between sentences is
  one of the clearest tells
* the vocoder's metallic top end is softened, and a gentle exciter adds back the
  "air" a 22 kHz model cannot produce (nothing above 11 kHz otherwise)
* a little harmonic warmth (soft saturation)
* emphasised words (<重|...>) are located with CTC and lifted a few dB
For a translated dub with the original recording (:mod:`.expressive`, :mod:`.linerefs`)
each sentence is generated from its own original line, and loudness and in-sentence
pauses follow the original instead of being evened out.
Then the regular voice-over chain (:mod:`subalign.studio`) runs: EQ, de-ess,
compression, optional reverb / BGM, loudness, export.
"""
from __future__ import annotations

import json
import logging
import math
import os
import random
import re
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from . import textprep

log = logging.getLogger("subalign")
ROOT = Path(__file__).resolve().parents[2]
SR = 48000


@dataclass
class DubConfig:
    speed: float = 1.0                # 0.95 - 1.05 (engine duration_factor = 1 / speed)
    emo_alpha: float = 0.8            # how strongly the emotion reference is applied
    lang: str = "ZH"
    qa: bool = True                   # transcribe every take and compare with the text
    max_tries: int = 3
    max_error: float = 0.12           # character error rate (homophones count as correct)
    rate_tolerance: float = 0.06      # even out sentences faster / slower than this vs the median
    pause_jitter: float = 0.10
    breaths: bool = True
    room_tone_db: float = -66.0       # 0 = off
    deharsh_db: float = 2.0           # high-shelf cut on the vocoder's metallic band
    air: float = 0.4                  # exciter amount (0 = off)
    warmth: float = 0.25              # soft saturation (0 = off)
    emphasis_db: float = 3.0
    # dubbing over a video (from a translation with the original timing): every sentence
    # starts where the original sentence started; takes that overrun the slot are sped up
    # (pitch-preserving) by at most ``max_stretch``, short ones slowed by at most ``min_stretch``
    timeline: bool = False
    max_stretch: float = 1.15
    min_stretch: float = 0.93
    # translated dub with the original recording (:mod:`.linerefs`): every sentence is
    # generated from its own original line (aligned, cut in the gaps, checked by ear) as
    # the only reference - voice and delivery both come from it, no separate emotion
    # reference (IndexTTS-2.5 adds the emotion vector to the speaker embedding, so a
    # second reference moved the timbre and made the Chinese dub ~3 semitones higher)
    line_ref: bool = False
    # ... while the timbre comes from the speaker's stable voice reference: IndexTTS's
    # acoustic renderer (which largely decides the timbre) is conditioned on it, the
    # language model (rhythm, tone) on the line (``timbre`` in ``webui/tts_worker.py``) -
    # a short line alone as the reference made the voice drift (similarity 0.65 vs 0.75)
    stable_timbre: bool = True
    # on a timeline every take is generated at the pace that fits it before the next
    # sentence: the renderer speaks faster itself (``target`` in ``webui/tts_worker.py``,
    # down to ``pace_min`` of its natural length) instead of the take being sped up
    # afterwards or pushed late.  A take shorter than the original line is only slowed a
    # little (``pace_max``) and not slowed again on assembly: Chinese is shorter than the
    # English it replaces (here 0.81 of the original's length at a normal 0.97x Chinese
    # rate), and filling the original's length made it ~20% too slow
    pace: bool = True
    pace_min: float = 0.82
    pace_max: float = 1.05
    # IndexTTS's acoustic renderer: guidance strength (how closely it follows the voice
    # reference) and diffusion steps.  1.0 / 50 (library: 0.7 / 25) was closer to the
    # speaker on all of 8 test sentences (similarity +0.026), ~1.5x the time
    render_cfg: float = 1.0
    render_steps: int = 50
    # every speaker's sentences get an EQ that brings the finished dub's tonal balance
    # (after the de-mechanising chain) to the original speaker's (:mod:`.acoustics`),
    # 100 Hz - 10 kHz, at most ``tone_max_db``.  No room reverb: the originals measured
    # fairly dry, and matching their decay with reverb lowered the voice similarity
    match_tone: bool = False
    tone_max_db: float = 6.0
    # sentence loudness follows the original line (offset from the median, capped)
    # instead of being evened out
    follow_dynamics: bool = False
    dynamics_range_db: float = 8.0
    # pauses inside an original line are re-placed at the matching clause boundary
    source_pauses: bool = False
    # several takes per sentence, ranked by misreading, voice similarity, closeness to
    # the original line's performance and fit on the timeline (:mod:`.takeqa`)
    pick_best: bool = False
    candidates: int = 2


# ------------------------------------------------------------------ engine worker
class TTSWorker:
    """IndexTTS-2.5 in tools/index-tts/.venv, kept alive between requests."""

    def __init__(self):
        self.proc: Optional[subprocess.Popen] = None
        self.lock = threading.Lock()
        self._id = 0

    @staticmethod
    def python() -> Path:
        exe = ROOT / "tools" / "index-tts" / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        return exe

    def available(self) -> bool:
        return self.python().exists() and (ROOT / "models" / "indextts-2.5" / "config.yaml").exists()

    def start(self) -> None:
        if self.proc and self.proc.poll() is None:
            return
        if not self.available():
            raise RuntimeError("IndexTTS-2.5 is not installed (tools/index-tts/.venv + models/indextts-2.5)")
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
        log_f = open(ROOT / "webui_data" / "tts_worker.log", "ab")
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        self.proc = subprocess.Popen([str(self.python()), str(ROOT / "webui" / "tts_worker.py")], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=log_f, env=env, creationflags=flags,
                                     text=True, encoding="utf-8", bufsize=1)
        self._read()                                         # the "ready" line

    def _read(self) -> Dict:
        assert self.proc and self.proc.stdout
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("TTS worker exited (see webui_data/tts_worker.log)")
            if line.startswith("@@RESULT "):
                return json.loads(line[len("@@RESULT "):])

    # after one of these the CUDA context of the worker is broken: every later request
    # fails the same way until the process is restarted
    FATAL = ("CUDA error", "AcceleratorError", "CUBLAS_STATUS", "cuDNN error", "TTS worker exited")

    def request(self, **req) -> Dict:
        try:
            return self._request(dict(req))
        except (RuntimeError, OSError, ValueError) as e:
            if not any(k in str(e) for k in self.FATAL) or req.get("cmd") == "quit":
                raise
            log.warning("TTS worker failed (%s): restarting it and retrying once", str(e)[:200])
            self._kill()
            return self._request(dict(req))

    def _request(self, req: Dict) -> Dict:
        with self.lock:
            self.start()
            self._id += 1
            req["id"] = self._id
            assert self.proc and self.proc.stdin
            self.proc.stdin.write(json.dumps(req, ensure_ascii=False) + "\n")
            self.proc.stdin.flush()
            while True:
                r = self._read()
                if r.get("id") == self._id:
                    if not r.get("ok"):
                        raise RuntimeError(r.get("error", "TTS failed"))
                    return r

    def _kill(self) -> None:
        with self.lock:
            if self.proc and self.proc.poll() is None:
                self.proc.kill()
                try:
                    self.proc.wait(timeout=10)
                except Exception:
                    pass
            self.proc = None

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            try:
                self.request(cmd="quit")
            except Exception:
                pass
            self.proc.kill()
        self.proc = None


WORKER = TTSWorker()


# ------------------------------------------------------------------ project
def _now() -> float:
    return round(time.time(), 3)


def create_project(pdir: Path, script: str, spk: Path, emo: Optional[Path], cfg: DubConfig, title: str = "",
                   sentences: Optional[List[Dict]] = None, source: Optional[Dict] = None,
                   speakers: Optional[Dict] = None, acoustics: Optional[Dict] = None) -> Dict:
    """``script``: the text, raw or already reviewed (markup allowed).  It is always
    normalised (idempotent), so notes / emoji / digits never reach the engine.

    ``sentences`` (from a dubbing translation): ``[{"text", "start", "end"}]`` - one
    segment each, never re-split, with the original timing kept for timeline assembly;
    performance data from :func:`.expressive.analyze_source` rides along (``EXPR_KEYS``).
    ``speakers``: the per-speaker voice references of that analysis."""
    pdir.mkdir(parents=True, exist_ok=True)
    lang = cfg.lang.lower()
    if sentences is None:
        script = textprep.normalize(script, lang=lang)
        segs = [dict(asdict(s)) for s in textprep.split_segments(script, lang=lang)]
    else:
        segs = _timed_segments(sentences, lang)
        script = "\n".join(s["text"] for s in segs)
    proj = {"version": 1, "title": title or "AI 配音", "created": _now(), "spk": str(spk), "emo": str(emo) if emo else None,
            "script": script, "config": asdict(cfg), "mix": None, "final": None, "source": source,
            "speakers": speakers or {}, "acoustics": acoustics or {},
            "segments": [dict(s, id=i + 1, status="pending", audio=None, takes=[], qa=None) for i, s in enumerate(segs)]}
    save(pdir, proj)
    return proj


EXPR_KEYS = ("speaker", "spk", "line", "line_check", "expr", "src_level", "src_pauses", "src_spread", "src_f0")


def _timed_segments(sentences: List[Dict], lang: str) -> List[Dict]:
    cjk = textprep.is_cjk_lang(lang)
    ends = textprep.SENT_END + textprep.CLAUSE if cjk else textprep.LATIN_END + textprep.LATIN_CLAUSE + "…"
    out = []
    for i, s in enumerate(sentences):
        t = textprep.normalize(s.get("text") or "", lang=lang)
        if not t:
            continue
        if t[-1] not in ends:
            t += "。" if cjk else "."
        tts, emph = textprep.to_engine(t)
        nxt = sentences[i + 1].get("start") if i + 1 < len(sentences) else None
        pause = 0.45
        if s.get("end") is not None and nxt is not None:
            pause = round(max(0.1, min(3.0, nxt - s["end"])), 2)
        out.append({"text": t, "tts_text": tts, "pause_after": pause, "emphasis": emph, "paragraph_end": False,
                    "src_start": s.get("start"), "src_end": s.get("end"), "source_text": s.get("source_text"),
                    "dt_id": s.get("dt_id"), **{k: s[k] for k in EXPR_KEYS if s.get(k) is not None}})
    return out


def load(pdir: Path) -> Dict:
    return json.loads((pdir / "project.json").read_text(encoding="utf-8"))


_save_lock = threading.Lock()


def save(pdir: Path, proj: Dict) -> None:
    with _save_lock:
        _write(pdir, proj)


def _write(pdir: Path, proj: Dict) -> None:
    tmp = pdir / "project.json.tmp"
    tmp.write_text(json.dumps(proj, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(pdir / "project.json")


def update(pdir: Path, fn) -> Dict:
    """Read-modify-write under the lock (sentences are edited while others generate)."""
    with _save_lock:
        proj = load(pdir)
        fn(proj)
        _write(pdir, proj)
        return proj


def find(proj: Dict, sid: int) -> Dict:
    for s in proj["segments"]:
        if s["id"] == sid:
            return s
    raise KeyError(f"no sentence {sid}")


def edit_segment(pdir: Path, sid: int, text: Optional[str] = None, pause_after: Optional[float] = None) -> Dict:
    """Change a sentence's text (normalised again, markup kept) / the pause after it."""
    def fn(proj):
        s = find(proj, sid)
        if text is not None and text.strip():
            t = textprep.normalize(text.strip(), lang=proj["config"].get("lang", "ZH").lower())
            if t != s["text"]:
                s["text"] = t
                s["tts_text"], s["emphasis"] = textprep.to_engine(t)
                s["status"] = "edited" if s.get("audio") else "pending"
        if pause_after is not None:
            s["pause_after"] = max(0.0, min(5.0, float(pause_after)))
    return find(update(pdir, fn), sid)


def set_config(pdir: Path, **kw) -> Dict:
    def fn(proj):
        known = DubConfig.__dataclass_fields__
        proj["config"].update({k: v for k, v in kw.items() if k in known and v is not None})
    return update(pdir, fn)


# ------------------------------------------------------------------ QA
def _plain(text: str) -> str:
    from ..text.tokenize import normalize_key

    t, _ = textprep.to_engine(text)
    t = re.sub(r"<([^<>|]+)\|[^<>]+>", r"\1", t)
    return "".join(normalize_key(c) for c in t)


def _units(text: str, lang: str) -> List[str]:
    """What is compared in QA: characters for CJK, words (numbers read out) otherwise."""
    if textprep.is_cjk_lang(lang):
        return list(_plain(text))
    from ..text.tokenize import normalize_key
    from ..translate.isochrony import number_to_en

    t, _ = textprep.to_engine(text)
    t = re.sub(r"<([^<>|]+)\|[^<>]+>", r"\1", t)
    if lang.startswith("en"):
        t = re.sub(r"[$¥€£]?\d+(?:[.,:]\d+)*%?", lambda m: number_to_en(m.group(0)), t)
    return [k for k in (normalize_key(w) for w in re.split(r"[\s\-]+", t)) if k]


_ASR = {}


def asr_backend():
    """The recogniser used to check takes and reference clips (loaded once), 8-bit.  In
    float16 next to the IndexTTS worker (~8.7 GB) it filled an 11 GB card and generation
    slowed from ~6 s to over 100 s a sentence as Windows paged GPU memory; 8-bit takes
    ~1.1 GB (0.3 s a check, generation unaffected).  On the CPU (7 s a check, whatever
    the length: Whisper always encodes 30 s) only when the GPU has no room for it."""
    from ..asr import get_backend
    from .linerefs import _device_for

    if "asr" not in _ASR:
        dev = _device_for(min_free_gb=1.5)
        _ASR["asr"] = get_backend("faster-whisper", model="large-v3-turbo", device=dev,
                                  compute_type="int8_float16" if dev == "cuda" else "int8")
        log.info("take check (Whisper) on %s", dev)
    return _ASR["asr"]


def check_take(path: Path, text: str, lang: str = "zh") -> Dict:
    """Transcribe a take and compare with the intended text: character (CJK) / word
    error rate where a homophone (same reading, other character) counts as correct."""
    from ..align.sequence import align_keys

    tr = asr_backend().transcribe(str(path), language=lang, vad=False)
    joiner = "" if textprep.is_cjk_lang(lang) else " "
    heard = joiner.join(s.text.strip() for s in tr.segments)
    # whisper writes numbers as digits: read them out the same way as the script
    ref, hyp = _units(text, lang), _units(textprep.normalize(heard, lang=lang), lang)
    if not ref:
        return {"error": 0.0, "heard": heard}
    ops = align_keys(ref, hyp)
    bad = sum(1 for o in ops if o.op in ("del", "ins") or (o.op == "sub" and o.sim < 0.6))
    return {"error": round(bad / len(ref), 3), "heard": heard.strip()}


# ------------------------------------------------------------------ audio helpers
def _load(path, sr=SR) -> np.ndarray:
    from ..audio.io import load_audio

    return load_audio(path, sr).astype(np.float32)


def trim_take(y: np.ndarray, sr: int = SR, pad: float = 0.03) -> np.ndarray:
    """Cut leading / trailing silence and junk: keep from the first to the last
    50 ms window that is within 35 dB of the take's loudest window."""
    win = int(0.05 * sr)
    n = len(y) // win
    if n < 3:
        return y
    r = 20 * np.log10(np.sqrt(np.mean(y[:n * win].reshape(n, win) ** 2, axis=1)) + 1e-9)
    on = np.flatnonzero(r > r.max() - 35)
    a = max(0, on[0] * win - int(pad * sr))
    b = min(len(y), (on[-1] + 1) * win + int(pad * sr))
    out = y[a:b].copy()
    f = min(int(0.008 * sr), len(out) // 4)
    if f > 1:
        out[:f] *= np.linspace(0, 1, f)
        out[-f:] *= np.linspace(1, 0, f)
    return out


def speaking_rate(y: np.ndarray, text: str, sr: int = SR) -> float:
    """Characters per second of voiced time (pauses inside the take excluded)."""
    win = int(0.02 * sr)
    n = len(y) // win
    if n == 0:
        return 0.0
    r = 20 * np.log10(np.sqrt(np.mean(y[:n * win].reshape(n, win) ** 2, axis=1)) + 1e-9)
    voiced = np.sum(r > r.max() - 30) * win / sr
    return len(_plain(text)) / max(voiced, 0.2)


def stretch(y: np.ndarray, factor: float, sr: int = SR) -> np.ndarray:
    """Pitch-preserving tempo change (factor > 1 = faster) via ffmpeg (rubberband if
    available, else atempo)."""
    if abs(factor - 1) < 0.005:
        return y
    from ..studio import _ffmpeg, ff_filter

    out = subprocess.run([_ffmpeg(), "-hide_banner", "-filters"], capture_output=True, text=True).stdout
    af = f"rubberband=tempo={factor:.4f}:formant=preserved" if " rubberband " in out else f"atempo={factor:.4f}"
    cmd_len = int(len(y) / factor)
    z = ff_filter(np.pad(y, (0, int(0.2 * sr))), af, sr)
    return z[:cmd_len]


def _loudness(y: np.ndarray, sr: int = SR) -> float:
    from ..studio import measure_loudness

    return measure_loudness(y, sr)["lufs"]


def harvest_breaths(ref: Path, max_n: int = 6) -> List[np.ndarray]:
    """Real breaths from the voice reference (the speaker's own), for insertion."""
    try:
        from ..audio.features import analyze
        from ..roughcut import RoughCutConfig, breaths_and_coughs, voiced_units

        y16 = _load(ref, 16000)
        feats = analyze(y16)
        found = breaths_and_coughs(voiced_units(feats), feats, RoughCutConfig(breaths="reduce", coughs=False))
        y = _load(ref)
        out = []
        for it in found[:max_n]:
            a, b = int(it.start * SR), int(it.end * SR)
            clip = y[a:b].copy()
            f = min(int(0.03 * SR), len(clip) // 3)
            if f > 1:
                clip[:f] *= np.linspace(0, 1, f)
                clip[-f:] *= np.linspace(1, 0, f)
            out.append(clip)
        return out
    except Exception as e:  # breaths are optional
        log.info("no breaths harvested: %s", e)
        return []


def naturalize(y: np.ndarray, cfg: DubConfig, sr: int = SR) -> np.ndarray:
    """Spectral finishing for synthetic speech: soften the metallic top, add air
    and a little warmth."""
    from scipy.signal import butter, sosfilt

    from ..studio import ff_filter

    if cfg.deharsh_db > 0:
        y = ff_filter(y, f"highshelf=f=7500:g=-{cfg.deharsh_db}:t=q:w=0.7", sr)
    if cfg.air > 0:
        # exciter: saturate the 3-7 kHz band, keep only the new harmonics above 8 kHz
        band = sosfilt(butter(2, [3000 / (sr / 2), 7000 / (sr / 2)], "band", output="sos"), y)
        gen = np.tanh(band * 6.0)
        top = sosfilt(butter(4, 8000 / (sr / 2), "high", output="sos"), gen)
        rms_y, rms_t = np.sqrt(np.mean(y ** 2)) + 1e-9, np.sqrt(np.mean(top ** 2)) + 1e-9
        y = y + top * (rms_y / rms_t) * 0.06 * cfg.air
    if cfg.warmth > 0:
        d = 1 + 2 * cfg.warmth
        peak = np.max(np.abs(y)) + 1e-9
        sat = np.tanh(y / peak * d) / np.tanh(d) * peak
        y = (1 - 0.5 * cfg.warmth) * y + 0.5 * cfg.warmth * sat
    return y.astype(np.float32)


def emphasize(y: np.ndarray, text: str, words: List[str], db: float, lang: str = "zh") -> np.ndarray:
    """Lift emphasised words: locate them with CTC forced alignment, +``db`` with ramps."""
    if not words or db <= 0:
        return y
    try:
        from ..align.ctc import HFCTCEmitter, ctc_align_tokens
        from ..text.tokenize import tokenize

        plain, _ = textprep.to_engine(text)
        plain = re.sub(r"<([^<>|]+)\|[^<>]+>", r"\1", plain)
        toks = tokenize(plain)
        if "ctc" not in _ASR:
            _ASR["ctc"] = HFCTCEmitter(None, lang, None)
        y16 = _resample(y, SR, 16000)
        spans = ctc_align_tokens(_ASR["ctc"], y16, [[t.text for t in toks]])[0]
        keys = "".join(t.text for t in toks)
        g = np.ones(len(y), dtype=np.float32)
        up = 10 ** (db / 20)
        r = int(0.03 * SR)
        for w in words:
            pos = keys.find(w)
            if pos < 0:
                continue
            # character offset -> token index
            acc, i0, i1 = 0, None, None
            for k, t in enumerate(toks):
                if i0 is None and acc + len(t.text) > pos:
                    i0 = k
                if acc < pos + len(w):
                    i1 = k
                acc += len(t.text)
            sts = [s for s in spans[i0:i1 + 1] if s.start is not None]
            if not sts:
                continue
            a, b = int(sts[0].start * SR), int((sts[-1].end + 0.05) * SR)
            g[a:b] = np.maximum(g[a:b], up)
            g[max(0, a - r):a] = np.maximum(g[max(0, a - r):a], np.linspace(1, up, min(r, a)))
            g[b:b + r] = np.maximum(g[b:b + r], np.linspace(up, 1, len(g[b:b + r])))
        return (y * g).astype(np.float32)
    except Exception as e:
        log.warning("emphasis skipped: %s", e)
        return y


def _resample(y: np.ndarray, a: int, b: int) -> np.ndarray:
    from math import gcd

    from scipy.signal import resample_poly

    g = gcd(a, b)
    return resample_poly(y, b // g, a // g).astype(np.float32)


# ------------------------------------------------------------------ synthesis
def config(proj: Dict) -> DubConfig:
    """The project's settings (keys of older versions ignored)."""
    return DubConfig(**{k: v for k, v in proj["config"].items() if k in DubConfig.__dataclass_fields__})


def speaker_ref(proj: Dict, seg: Dict) -> str:
    """The sentence's speaker's voice reference (the project's voice without one)."""
    return seg["spk"] if seg.get("spk") and Path(seg["spk"]).exists() else proj["spk"]


def references(proj: Dict, seg: Dict, cfg: DubConfig) -> tuple:
    """(voice reference, emotion reference, emotion weight) for one sentence: its own
    original line when the project uses line references (IndexTTS then takes the
    delivery from it too), else its speaker's voice + the project's emotion reference."""
    if cfg.line_ref and seg.get("line") and Path(seg["line"]).exists():
        return seg["line"], None, cfg.emo_alpha
    return speaker_ref(proj, seg), proj.get("emo"), cfg.emo_alpha


def _err(take: Dict) -> Optional[float]:
    return (take.get("qa") or {}).get("error")


def _best_take(takes: List[Dict], ranked: bool) -> Dict:
    """Lowest ``score.total`` when ranked, else the fewest misread characters (a take
    that could not be checked only wins when no take was checked)."""
    if ranked and all(t.get("score") for t in takes):
        return min(takes, key=lambda t: t["score"]["total"])
    checked = [t for t in takes if _err(t) is not None]
    return min(checked, key=_err) if checked else takes[0]


DEFAULT_EDGE = 0.25


def take_edges(proj: Dict, n: int = 30) -> float:
    """Seconds of silence / noise a raw take has around its speech (removed by
    :func:`trim_take`), from the project's recent takes."""
    d = [t["duration"] - t["speech"] for s in proj["segments"] for t in s.get("takes", [])
         if t.get("duration") and t.get("speech")]
    return float(np.median(d[-n:])) if d else DEFAULT_EDGE


def pace_target(proj: Dict, seg: Dict, cfg: DubConfig) -> Optional[float]:
    """Raw length (seconds) a take of ``seg`` should have: the original line's length,
    never more than the time until the next sentence, at the project's speed, plus the
    usual silence around a take's speech.  None when the sentence is not on a timeline."""
    from .takeqa import slot_for

    if not (cfg.timeline and cfg.pace and seg.get("src_start") is not None and seg.get("src_end") is not None):
        return None
    i = proj["segments"].index(seg)
    slot = slot_for(seg, proj["segments"][i + 1] if i + 1 < len(proj["segments"]) else None)
    want = seg["src_end"] - seg["src_start"]
    if slot:
        want = min(want, slot)
    if want <= 0.2:
        return None
    return round(want / max(0.5, min(2.0, cfg.speed)) + take_edges(proj), 3)


def synth_segment(pdir: Path, sid: int, worker: Optional[TTSWorker] = None) -> Dict:
    """(Re)generate one sentence.  With QA it tries up to ``max_tries`` seeds until a
    take reads correctly; with ``pick_best`` at least ``candidates`` takes are made and
    the best by :mod:`.takeqa` is kept (still up to ``max_tries`` while none reads
    correctly)."""
    from . import takeqa

    worker = worker or WORKER
    proj = update(pdir, lambda p: find(p, sid).update(status="running"))
    cfg = config(proj)
    seg = find(proj, sid)
    text, tts_text = seg["text"], seg["tts_text"]
    spk, emo, alpha = references(proj, seg, cfg)
    voice = speaker_ref(proj, seg)          # what a take is compared with for voice drift
    timbre = voice if cfg.line_ref and cfg.stable_timbre and spk != voice and Path(voice).exists() else None
    ranked = cfg.pick_best
    at_least = max(1, int(cfg.candidates)) if ranked else 1
    tries = max(at_least, cfg.max_tries if cfg.qa else 1)
    target = pace_target(proj, seg, cfg)
    slot = None
    if ranked and cfg.timeline:
        i = proj["segments"].index(seg)
        slot = takeqa.slot_for(seg, proj["segments"][i + 1] if i + 1 < len(proj["segments"]) else None)
    (pdir / "seg").mkdir(exist_ok=True)
    takes: List[Dict] = []
    try:
        for k in range(tries):
            seed = random.randint(1, 2 ** 31 - 1)
            out = pdir / "seg" / f"{sid:04d}_{int(time.time() * 1000) % 10 ** 9:09d}.wav"
            r = worker.request(cmd="synth", text=tts_text, spk=spk, emo=emo,
                               emo_alpha=alpha, duration_factor=round(1 / max(0.5, min(2.0, cfg.speed)), 4),
                               seed=seed, out=str(out), lang=cfg.lang, timbre=timbre, target=target,
                               pace_range=[cfg.pace_min, cfg.pace_max], cfg_rate=cfg.render_cfg,
                               steps=cfg.render_steps)
            take = {"file": f"seg/{out.name}", "seed": seed, "duration": r.get("duration"), "seconds": r.get("seconds"),
                    "text": text, "emo_alpha": alpha if emo else None, "timbre": bool(timbre),
                    "target": target, "natural": r.get("natural"), "pace": r.get("pace")}
            try:
                take["speech"] = round(len(trim_take(_load(out))) / SR, 3)
            except Exception as e:
                log.warning("could not measure take %s: %s", out.name, e)
            if cfg.qa:
                try:
                    take["qa"] = check_take(out, text, cfg.lang.lower())
                except Exception as e:
                    log.warning("QA failed: %s", e)
                    take["qa"] = {"error": None, "heard": f"(校验失败: {e})"}
            if ranked:
                try:
                    m = takeqa.measure(out, seg, voice, performance=cfg.line_ref, slot=slot)
                    take["measure"] = m
                    take["score"] = takeqa.score(m, seg, _err(take), cfg.max_error, cfg.max_stretch)
                except Exception as e:
                    log.warning("take scoring failed: %s", e)
            takes.append(take)
            if k + 1 >= at_least and any(_err(t) is None or _err(t) <= cfg.max_error for t in takes):
                break
        best = _best_take(takes, ranked)
    except Exception as e:
        def failed(p):
            s = find(p, sid)
            s["takes"] = s.get("takes", []) + takes
            s.update(status="error", error=str(e)[:500])
        update(pdir, failed)
        raise

    def done(p):
        s = find(p, sid)
        s["takes"] = s.get("takes", []) + takes
        s["audio"], s["qa"], s["error"] = best["file"], best.get("qa"), None
        s["measure"], s["score"], s["paced"] = best.get("measure"), best.get("score"), best.get("pace") is not None
        bad = cfg.qa and (s["qa"] or {}).get("error") is not None and s["qa"]["error"] > cfg.max_error
        s["status"] = "edited" if s["text"] != text else ("check" if bad else "done")
        s["updated"] = _now()
    return find(update(pdir, done), sid)


def use_take(pdir: Path, sid: int, file: str) -> Dict:
    """Pick an earlier take of a sentence."""
    def fn(p):
        s = find(p, sid)
        t = next(t for t in s.get("takes", []) if t["file"] == file)
        s["audio"], s["qa"], s["status"] = t["file"], t.get("qa"), "done"
        s["measure"], s["score"], s["paced"] = t.get("measure"), t.get("score"), t.get("pace") is not None
    return find(update(pdir, fn), sid)


def assemble(pdir: Path, proj: Dict, out: Optional[Path] = None) -> Dict:
    """Takes -> one natural-sounding voice track (48 kHz) + per-sentence timing.  With
    ``timeline`` (a translated dub) sentences sit at the original times instead."""
    cfg = config(proj)
    rng = np.random.default_rng(int(proj.get("created", 7)) % 2 ** 32)
    segs = [s for s in proj["segments"] if s.get("audio")]
    if not segs:
        raise RuntimeError("no generated sentences yet")
    from . import expressive

    clips, placed_pauses = [], []
    for s in segs:
        y = trim_take(_load(pdir / s["audio"]))
        if s.get("emphasis"):
            y = emphasize(y, s["text"], s["emphasis"], cfg.emphasis_db, cfg.lang.lower())
        n_p = 0
        if cfg.source_pauses and s.get("src_pauses"):
            y, n_p = expressive.place_pauses(y, s["text"], s["src_pauses"], cfg.lang.lower(), SR)
        clips.append(y)
        placed_pauses.append(n_p)
    timeline = cfg.timeline and all(s.get("src_start") is not None for s in segs)
    # loudness: evened out (capped), or on a translated dub following the original lines
    louds = [_loudness(c) for c in clips]
    if timeline and cfg.follow_dynamics and any(s.get("src_level") is not None for s in segs):
        gains = expressive.dynamics_gains(louds, [s.get("src_level") for s in segs], cfg.dynamics_range_db)
    else:
        target = float(np.median([l for l in louds if l > -60] or [-23]))
        gains = [max(-6, min(6, target - l)) if l > -60 else 0.0 for l in louds]
    clips = [c * 10 ** (g / 20) for c, g in zip(clips, gains)]
    eq_db = match_tone(clips, segs, proj, cfg) if cfg.match_tone else {}
    peak = max(float(np.max(np.abs(c))) if len(c) else 0.0 for c in clips)
    if peak > 0.95:                     # loud lines lifted: scale everything, keep the contrast
        clips = [c * (0.95 / peak) for c in clips]
    # speaking-rate evening (pitch-preserving); on a timeline every take is fitted to its slot instead
    if cfg.rate_tolerance > 0 and len(clips) > 2 and not timeline:
        rates = [speaking_rate(c, s["text"]) for c, s in zip(clips, segs)]
        med = float(np.median(rates))
        for i, (c, rt) in enumerate(zip(clips, rates)):
            dev = rt / med - 1
            if abs(dev) > cfg.rate_tolerance:
                factor = 1 - dev + math.copysign(cfg.rate_tolerance / 2, dev)   # bring within half the tolerance
                factor = max(0.95, min(1.05, factor))
                clips[i] = stretch(c, factor)
    # breaths of each sentence's own speaker (their voice reference)
    breaths: List[List[np.ndarray]] = [[] for _ in segs]
    if cfg.breaths:
        refs = [speaker_ref(proj, s) for s in segs]
        found = {r: harvest_breaths(Path(r)) for r in set(refs)}
        if any(found.values()):
            # a breath sits ~24 dB under the voice (the sentences were loudness-matched, the
            # reference was not)
            def _rms(x):
                return float(np.sqrt(np.mean(x ** 2)) + 1e-9)
            voice = float(np.median([_rms(c[np.abs(c) > 0.02 * np.abs(c).max()]) for c in clips]))
            found = {r: [b * (voice * 10 ** (-24 / 20) / _rms(b)) for b in bs] for r, bs in found.items()}
            breaths = [found[r] for r in refs]
    tone_lvl = 10 ** (cfg.room_tone_db / 20) if cfg.room_tone_db < 0 else 0.0
    if timeline:
        y, timing = _place_on_timeline(clips, segs, cfg, breaths)
        for t, g, n_p in zip(timing, gains, placed_pauses):
            t.update(gain_db=round(g, 1), pauses=n_p)
        return _finish(pdir, proj, y, timing, cfg, tone_lvl, out, tone_eq=eq_db)
    parts: List[np.ndarray] = [np.zeros(int(0.25 * SR), np.float32)]
    timing = []
    t = len(parts[0])
    for i, (c, s) in enumerate(zip(clips, segs)):
        if i > 0:
            pause = segs[i - 1]["pause_after"] * (1 + rng.uniform(-cfg.pause_jitter, cfg.pause_jitter))
            gap = np.zeros(int(pause * SR), np.float32)
            if breaths[i] and pause >= 0.4:
                b = breaths[i][i % len(breaths[i])]
                if len(b) < len(gap) - int(0.1 * SR):
                    end = len(gap) - int(0.06 * SR)
                    gap[end - len(b):end] += b
            parts.append(gap)
            t += len(gap)
        timing.append({"id": s["id"], "start": round(t / SR, 3), "end": round((t + len(c)) / SR, 3)})
        parts.append(c)
        t += len(c)
    parts.append(np.zeros(int(0.4 * SR), np.float32))
    return _finish(pdir, proj, np.concatenate(parts), timing, cfg, tone_lvl, out, tone_eq=eq_db)


def match_tone(clips: List[np.ndarray], segs: List[Dict], proj: Dict, cfg: DubConfig) -> Dict[str, List[float]]:
    """EQ every speaker's clips (in place) toward the original speaker's tonal balance,
    measured on the clips as the de-mechanising chain will leave them.  Returns the
    band gains used per speaker."""
    from . import acoustics

    targets = proj.get("acoustics") or {}
    used: Dict[str, List[float]] = {}
    for spk in sorted({s.get("speaker") or "S1" for s in segs}):
        prof = targets.get(spk) or (next(iter(targets.values())) if len(targets) == 1 else None)
        idx = [i for i, s in enumerate(segs) if (s.get("speaker") or "S1") == spk]
        if not prof or not idx:
            continue
        gap = np.zeros(int(0.4 * SR), np.float32)
        joined = np.concatenate([c for i in idx for c in (clips[i], gap)])
        now = acoustics.band_levels(naturalize(joined, cfg), SR)
        g = acoustics.eq_gains(np.array(prof["bands"]), now, max_db=cfg.tone_max_db)
        for i in idx:
            clips[i] = acoustics.apply_eq(clips[i], SR, g)
        used[spk] = [round(float(x), 1) for x in g]
    return used


def fit_factor(length: float, slot: float, natural: float, cfg: DubConfig, paced: bool = False) -> float:
    """Tempo factor (> 1 = faster) for a take of ``length`` s in a slot of ``slot`` s
    (until the next sentence) where the original took ``natural`` s.  A ``paced`` take
    (generated at its pace already) is only sped up when it still does not fit."""
    if length > slot > 0:
        return min(cfg.max_stretch, length / slot)
    if not paced and natural > 0 and length < natural * 0.85:
        return max(cfg.min_stretch, length / natural)
    return 1.0


def _place_on_timeline(clips: List[np.ndarray], segs: List[Dict], cfg: DubConfig,
                       breaths: List[List[np.ndarray]]) -> tuple:
    """Every sentence starts at its original start time; a take longer than its slot is
    sped up (at most ``max_stretch``) and, if still too long, pushes the next one later.
    ``breaths``: the breaths to use before each sentence ([] = none at all)."""
    breaths = breaths or [[] for _ in clips]
    gap_min = int(0.08 * SR)
    placed, timing = [], []
    cursor = 0
    for i, (c, s) in enumerate(zip(clips, segs)):
        start = int(s["src_start"] * SR)
        nxt = segs[i + 1]["src_start"] if i + 1 < len(segs) else None
        natural = (s.get("src_end") or s["src_start"]) - s["src_start"]
        slot = (nxt - s["src_start"] - 0.08) if nxt is not None else natural + 1.0
        f = fit_factor(len(c) / SR, slot, natural, cfg, paced=bool(s.get("paced")))
        if abs(f - 1) >= 0.01:
            c = stretch(c, f)
        at = max(start, cursor + (gap_min if placed else 0))
        if breaths[i] and placed and at - cursor >= int(0.5 * SR):
            b = breaths[i][i % len(breaths[i])]
            if len(b) < at - cursor - int(0.1 * SR):
                placed.append((at - int(0.06 * SR) - len(b), b))
        placed.append((at, c))
        timing.append({"id": s["id"], "start": round(at / SR, 3), "end": round((at + len(c)) / SR, 3),
                       "src_start": s["src_start"], "late": round(max(0, at - start) / SR, 3), "tempo": round(f, 3)})
        cursor = at + len(c)
    total = max(cursor, int(((segs[-1].get("src_end") or 0) + 0.4) * SR)) + int(0.4 * SR)
    y = np.zeros(total, np.float32)
    for at, c in placed:
        y[at:at + len(c)] += c[:max(0, total - at)]
    return y, timing


def _finish(pdir: Path, proj: Dict, y: np.ndarray, timing: List[Dict], cfg: DubConfig, tone_lvl: float,
            out: Optional[Path], tone_eq: Optional[Dict] = None) -> Dict:
    from ..audio.io import save_audio

    if tone_lvl:
        from scipy.signal import lfilter

        w = np.random.default_rng(3).standard_normal(len(y))
        pink = lfilter([0.049922035, -0.095993537, 0.050612699, -0.004408786], [1, -2.494956002, 2.017265875, -0.522189400], w)
        y = y + (pink / (np.sqrt(np.mean(pink ** 2)) + 1e-12) * tone_lvl).astype(np.float32)
    y = naturalize(y, cfg)
    out = out or pdir / "mix.wav"
    save_audio(out, np.clip(y, -1, 1), SR)
    mix = {"file": out.name, "built": _now(), "duration": round(len(y) / SR, 2), "timing": timing,
           "missing": [s["id"] for s in proj["segments"] if not s.get("audio")], "tone_eq": tone_eq or {}}
    update(pdir, lambda p: p.update(mix=mix))
    return mix


def export_mix(pdir: Path, fmt: str = "mp3", bitrate: str = "320k", bit_depth: int = 24, out_sr: int = 48000,
               channels: int = 1) -> Path:
    """The assembled (unprocessed) voice track in another format."""
    from ..studio import EXT, StudioConfig, export

    y = _load(pdir / "mix.wav")
    cfg = StudioConfig(format=fmt, bitrate=bitrate, bit_depth=bit_depth, out_sr=out_sr, channels=channels)
    return export(y, pdir / f"mix{EXT.get(fmt, '.wav')}", cfg)


# AI voice has no room noise, breaths are re-added on purpose and the vocoder's top
# end was already treated: the voice-over chain runs without denoise / breath cleanup
# and with gentler de-essing / presence.
STUDIO_DEFAULTS = {"denoise": "off", "cleanup": False, "declick": False, "plosives": True, "deess": 4.0,
                   "presence_gain": 1.5, "mud_gain": -2.0}
