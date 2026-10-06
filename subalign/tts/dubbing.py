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

    def request(self, **req) -> Dict:
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


def create_project(pdir: Path, script: str, spk: Path, emo: Optional[Path], cfg: DubConfig, title: str = "") -> Dict:
    """``script``: the text, raw or already reviewed (markup allowed).  It is always
    normalised (idempotent), so notes / emoji / digits never reach the engine."""
    pdir.mkdir(parents=True, exist_ok=True)
    script = textprep.normalize(script)
    segs = textprep.split_segments(script)
    proj = {"version": 1, "title": title or "AI 配音", "created": _now(), "spk": str(spk), "emo": str(emo) if emo else None,
            "script": script, "config": asdict(cfg), "mix": None, "final": None,
            "segments": [dict(asdict(s), id=i + 1, status="pending", audio=None, takes=[], qa=None)
                         for i, s in enumerate(segs)]}
    save(pdir, proj)
    return proj


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
            t = textprep.normalize(text.strip())
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


_ASR = {}


def check_take(path: Path, text: str, lang: str = "zh") -> Dict:
    """Transcribe a take and compare with the intended text: character error rate
    where a homophone (same reading, other character) counts as correct."""
    from ..align.sequence import align_keys
    from ..asr import get_backend

    if "asr" not in _ASR:
        _ASR["asr"] = get_backend("faster-whisper", model="large-v3-turbo")
    tr = _ASR["asr"].transcribe(str(path), language=lang, vad=False)
    heard = "".join(s.text for s in tr.segments)
    # whisper writes numbers as digits: read them out the same way as the script
    ref, hyp = list(_plain(text)), list(_plain(textprep.normalize(heard)))
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
def synth_segment(pdir: Path, sid: int, worker: Optional[TTSWorker] = None) -> Dict:
    """(Re)generate one sentence; with QA it tries up to ``max_tries`` seeds and keeps
    the take with the fewest misread characters."""
    worker = worker or WORKER
    proj = update(pdir, lambda p: find(p, sid).update(status="running"))
    cfg = DubConfig(**proj["config"])
    seg = find(proj, sid)
    text, tts_text = seg["text"], seg["tts_text"]
    (pdir / "seg").mkdir(exist_ok=True)
    takes: List[Dict] = []
    best = None
    try:
        for _ in range(max(1, cfg.max_tries if cfg.qa else 1)):
            seed = random.randint(1, 2 ** 31 - 1)
            out = pdir / "seg" / f"{sid:04d}_{int(time.time() * 1000) % 10 ** 9:09d}.wav"
            r = worker.request(cmd="synth", text=tts_text, spk=proj["spk"], emo=proj.get("emo"),
                               emo_alpha=cfg.emo_alpha, duration_factor=round(1 / max(0.5, min(2.0, cfg.speed)), 4),
                               seed=seed, out=str(out), lang=cfg.lang)
            take = {"file": f"seg/{out.name}", "seed": seed, "duration": r.get("duration"), "seconds": r.get("seconds"),
                    "text": text}
            if cfg.qa:
                try:
                    take["qa"] = check_take(out, text, cfg.lang.lower())
                except Exception as e:
                    log.warning("QA failed: %s", e)
                    take["qa"] = {"error": None, "heard": f"(校验失败: {e})"}
            takes.append(take)
            err = (take.get("qa") or {}).get("error")
            best_err = (best or {}).get("qa", {}) or {}
            if best is None or (err is not None and (best_err.get("error") is None or err < best_err["error"])):
                best = take
            if err is None or err <= cfg.max_error:
                break
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
    return find(update(pdir, fn), sid)


def assemble(pdir: Path, proj: Dict, out: Optional[Path] = None) -> Dict:
    """Takes -> one natural-sounding voice track (48 kHz) + per-sentence timing."""
    from ..audio.io import save_audio

    cfg = DubConfig(**proj["config"])
    rng = np.random.default_rng(int(proj.get("created", 7)) % 2 ** 32)
    segs = [s for s in proj["segments"] if s.get("audio")]
    if not segs:
        raise RuntimeError("no generated sentences yet")
    clips = []
    for s in segs:
        y = trim_take(_load(pdir / s["audio"]))
        if s.get("emphasis"):
            y = emphasize(y, s["text"], s["emphasis"], cfg.emphasis_db, cfg.lang.lower())
        clips.append(y)
    # loudness matching between sentences (capped)
    louds = [_loudness(c) for c in clips]
    target = float(np.median([l for l in louds if l > -60] or [-23]))
    clips = [c * 10 ** (max(-6, min(6, target - l)) / 20) if l > -60 else c for c, l in zip(clips, louds)]
    # speaking-rate evening (pitch-preserving)
    if cfg.rate_tolerance > 0 and len(clips) > 2:
        rates = [speaking_rate(c, s["text"]) for c, s in zip(clips, segs)]
        med = float(np.median(rates))
        for i, (c, rt) in enumerate(zip(clips, rates)):
            dev = rt / med - 1
            if abs(dev) > cfg.rate_tolerance:
                factor = 1 - dev + math.copysign(cfg.rate_tolerance / 2, dev)   # bring within half the tolerance
                factor = max(0.95, min(1.05, factor))
                clips[i] = stretch(c, factor)
    breaths = harvest_breaths(Path(proj["spk"])) if cfg.breaths else []
    if breaths:
        # a breath sits ~24 dB under the voice (the sentences were loudness-matched, the
        # reference was not)
        def _rms(x):
            return float(np.sqrt(np.mean(x ** 2)) + 1e-9)
        voice = float(np.median([_rms(c[np.abs(c) > 0.02 * np.abs(c).max()]) for c in clips]))
        breaths = [b * (voice * 10 ** (-24 / 20) / _rms(b)) for b in breaths]
    tone_lvl = 10 ** (cfg.room_tone_db / 20) if cfg.room_tone_db < 0 else 0.0
    parts: List[np.ndarray] = [np.zeros(int(0.25 * SR), np.float32)]
    timing = []
    t = len(parts[0])
    for i, (c, s) in enumerate(zip(clips, segs)):
        if i > 0:
            pause = segs[i - 1]["pause_after"] * (1 + rng.uniform(-cfg.pause_jitter, cfg.pause_jitter))
            gap = np.zeros(int(pause * SR), np.float32)
            if breaths and pause >= 0.4:
                b = breaths[i % len(breaths)]
                if len(b) < len(gap) - int(0.1 * SR):
                    end = len(gap) - int(0.06 * SR)
                    gap[end - len(b):end] += b
            parts.append(gap)
            t += len(gap)
        timing.append({"id": s["id"], "start": round(t / SR, 3), "end": round((t + len(c)) / SR, 3)})
        parts.append(c)
        t += len(c)
    parts.append(np.zeros(int(0.4 * SR), np.float32))
    y = np.concatenate(parts)
    if tone_lvl:
        from scipy.signal import lfilter

        w = np.random.default_rng(3).standard_normal(len(y))
        pink = lfilter([0.049922035, -0.095993537, 0.050612699, -0.004408786], [1, -2.494956002, 2.017265875, -0.522189400], w)
        y = y + (pink / (np.sqrt(np.mean(pink ** 2)) + 1e-12) * tone_lvl).astype(np.float32)
    y = naturalize(y, cfg)
    out = out or pdir / "mix.wav"
    save_audio(out, np.clip(y, -1, 1), SR)
    mix = {"file": out.name, "built": _now(), "duration": round(len(y) / SR, 2), "timing": timing,
           "missing": [s["id"] for s in proj["segments"] if not s.get("audio")]}
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
