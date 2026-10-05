"""Voice-over / talking-head audio processing (口播音频处理).

The chain follows the usual post-production order for spoken voice::

    cleanup     (optional) cut fillers / breaths / coughs / long pauses   -> roughcut
    low cut     80 Hz high-pass, 24 dB/oct                                -> rumble, handling noise
    denoise     RNNoise (speech model) / FFT denoiser                     -> background noise
    declick     impulsive mouth clicks                                    -> "咔哒"
    plosives    dynamic low-band reduction on pops                        -> 喷麦 / 爆破音
    de-ess      dynamic 4.5-10 kHz reduction on sibilants                 -> 齿音
    EQ          -mud at ~250 Hz, +presence at ~3 kHz, optional air        -> 去闷 / 清晰度
    compressor  3:1 above -20 dBFS                                        -> even level
    reverb      (optional) light synthetic room                           -> space
    BGM         (optional) music bed at 15-25 % of the voice, ducked under speech
    loudness    EBU R128 integrated loudness + 4x oversampled true-peak limiter
    export      wav / mp3 / aac (m4a) / flac / opus

Every stage reports what it did, so the result can be checked objectively.
Processing runs at 48 kHz; the voice is mono until the BGM / export stage.
"""
from __future__ import annotations

import json
import logging
import math
import os
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.signal import fftconvolve, istft, lfilter, stft

log = logging.getLogger("subalign")

SR = 48000
ROOT = Path(__file__).resolve().parent.parent
RNNOISE_DIR = ROOT / "models" / "rnnoise"
# a release build is preferred: nightly builds have crashed in arnndn (and returned NaN)
BUNDLED_FFMPEG = ROOT / "tools" / "ffmpeg" / "bin" / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")


@dataclass
class StudioConfig:
    # cleanup (needs recognition for fillers; breaths / coughs are acoustic)
    cleanup: bool = False
    fillers: bool = True
    breaths: str = "reduce"          # off | reduce | remove
    breath_reduce_db: float = 12.0
    coughs: bool = True
    pauses: bool = False             # also shorten long pauses
    extra_fillers: Tuple[str, ...] = ()
    language: Optional[str] = None
    # repair
    highpass: float = 80.0           # Hz, 0 = off
    denoise: str = "medium"          # off | light | medium | strong
    declick: bool = True
    plosives: bool = True
    deess: float = 6.0               # max reduction (dB), 0 = off
    # tone
    eq: bool = True
    mud_freq: float = 250.0
    mud_gain: float = -3.0
    presence_freq: float = 3000.0
    presence_gain: float = 2.5
    air_gain: float = 0.0            # high shelf at 10 kHz
    # dynamics
    compress: bool = True
    comp_threshold: float = -20.0    # dBFS
    comp_ratio: float = 3.0
    comp_attack: float = 10.0        # ms
    comp_release: float = 120.0      # ms
    # space
    reverb: float = 0.0              # wet amount 0..1 (0 = off; 0.1-0.2 = light)
    reverb_time: float = 0.45        # RT60 seconds (small room)
    # background music
    bgm_ratio: float = 0.2           # BGM loudness relative to the voice (0.15-0.25 typical)
    bgm_duck: bool = True            # extra dip while the voice speaks
    bgm_duck_db: float = 4.0
    bgm_fade_in: float = 1.5
    bgm_fade_out: float = 3.0
    bgm_loop: bool = True
    # loudness
    loudness: Optional[float] = -16.0   # LUFS integrated; None = off
    true_peak: float = -1.5              # dBTP
    # export
    format: str = "wav"              # wav | mp3 | aac | flac | opus
    bitrate: str = "320k"            # lossy formats
    bit_depth: int = 24              # wav / flac
    out_sr: int = 48000
    channels: int = 2


# ------------------------------------------------------------------ helpers
def _ffmpeg() -> str:
    for cand in (os.environ.get("SUBALIGN_FFMPEG"), str(BUNDLED_FFMPEG) if BUNDLED_FFMPEG.exists() else None,
                 shutil.which("ffmpeg")):
        if cand:
            return cand
    raise RuntimeError("audio processing needs ffmpeg")


def ff_filter(y: np.ndarray, af: str, sr: int = SR, cwd: Optional[Path] = None, retries: int = 1) -> np.ndarray:
    """Run a mono or (ch, n) float buffer through an ffmpeg audio filter graph.

    ``retries``: some nightly ffmpeg builds crash intermittently in arnndn (access
    violation in roughly 1 run out of 10, even single-threaded); the filters are
    deterministic, so running again gives the identical result."""
    y2 = y if y.ndim == 2 else y[None]
    ch, n = y2.shape
    cmd = [_ffmpeg(), "-nostdin", "-v", "error", "-filter_threads", "1", "-f", "f32le", "-ar", str(sr), "-ac", str(ch),
           "-i", "pipe:0", "-af", af, "-f", "f32le", "-ar", str(sr), "-ac", str(ch), "pipe:1"]
    data = np.ascontiguousarray(y2.T, dtype=np.float32).tobytes()
    for attempt in range(max(1, retries)):
        try:
            out = subprocess.run(cmd, input=data, capture_output=True, check=True, cwd=str(cwd) if cwd else None).stdout
            # the same arnndn bug sometimes returns garbage instead of crashing
            if not np.isfinite(np.frombuffer(out, dtype=np.float32)).all():
                raise subprocess.CalledProcessError(1, cmd, b"", b"non-finite output")
            break
        except subprocess.CalledProcessError:
            if attempt + 1 >= max(1, retries):
                raise
            log.info("ffmpeg filter crashed (%s); retrying", af.split("=")[0])
    z = np.frombuffer(out, dtype=np.float32).reshape(-1, ch).T.copy()
    if z.shape[1] < n:
        z = np.pad(z, ((0, 0), (0, n - z.shape[1])))
    z = z[:, :n]
    return z if y.ndim == 2 else z[0]


def measure_loudness(y: np.ndarray, sr: int = SR) -> Dict[str, float]:
    """EBU R128 integrated loudness (LUFS), true peak (dBTP) and loudness range (LU)."""
    y2 = y if y.ndim == 2 else y[None]
    cmd = [_ffmpeg(), "-nostdin", "-hide_banner", "-nostats", "-f", "f32le", "-ar", str(sr), "-ac", str(y2.shape[0]),
           "-i", "pipe:0", "-af", "loudnorm=print_format=json", "-f", "null", "-"]
    err = subprocess.run(cmd, input=np.ascontiguousarray(y2.T, dtype=np.float32).tobytes(), capture_output=True,
                         check=True).stderr.decode("utf-8", "replace")
    m = re.findall(r"\{[^{}]*\}", err)
    d = json.loads(m[-1]) if m else {}

    def f(k, default=-70.0):
        try:
            v = float(d.get(k, default))
            return v if math.isfinite(v) else default
        except (TypeError, ValueError):
            return default
    return {"lufs": f("input_i"), "true_peak": f("input_tp"), "lra": f("input_lra", 0.0)}


def noise_floor_db(y: np.ndarray, sr: int = SR) -> float:
    """10th percentile of 20 ms frame RMS (dBFS): the level between words."""
    hop = int(0.02 * sr)
    n = len(y) // hop
    if n < 5:
        return -120.0
    r = np.sqrt(np.mean(y[:n * hop].reshape(n, hop) ** 2, axis=1) + 1e-12)
    return float(20 * np.log10(np.percentile(r, 10) + 1e-12))


def _db(x):
    return 20 * np.log10(np.maximum(x, 1e-12))


def _smooth_gain(g: np.ndarray, attack: int, release: int) -> np.ndarray:
    """Per-frame gain smoothing: fast going down (attack), slow coming back (release)."""
    out = np.empty_like(g)
    cur = 1.0
    a, r = 1.0 / max(1, attack), 1.0 / max(1, release)
    for i, v in enumerate(g):
        cur += (v - cur) * (a if v < cur else r)
        out[i] = cur
    return out


# ------------------------------------------------------------------ repair stages
def _local_speech_median(x: np.ndarray, speech: np.ndarray, hop_s: float, span: float = 3.0) -> np.ndarray:
    """Median of ``x`` over speech frames within +-span/2 (computed on 0.25 s blocks)."""
    blk = max(1, int(0.25 / hop_s))
    nb = int(math.ceil(len(x) / blk))
    meds = np.full(nb, np.nan)
    for k in range(nb):
        sl = slice(k * blk, (k + 1) * blk)
        if speech[sl].any():
            meds[k] = np.median(x[sl][speech[sl]])
    half = max(1, int(span / 2 / 0.25))
    out = np.empty(nb)
    glob = np.nanmedian(meds) if np.isfinite(meds).any() else 0.0
    for k in range(nb):
        win = meds[max(0, k - half):k + half + 1]
        win = win[np.isfinite(win)]
        out[k] = np.median(win) if len(win) else glob
    return np.repeat(out, blk)[:len(x)]


def reduce_plosives(y: np.ndarray, sr: int = SR, lo: float = 40.0, hi: float = 160.0, threshold_db: float = 12.0,
                    ratio: float = 4.0) -> Tuple[np.ndarray, int]:
    """Pops / mic blasts (喷麦) with a dynamic low band, the way a mixing engineer does it:
    the 40-160 Hz band is compressed (fast attack, ``ratio``:1) only where it rises more
    than ``threshold_db`` above *the speaker's own usual* low level (median over the
    surrounding 3 s of speech) *and* is louder than the mids (a pop is a low-frequency
    blast; a vowel carries its energy in the mids).  Normal speech rarely meets both
    and then only its excess is touched, so a deep voice keeps its body.  The rest of
    the spectrum is unchanged.  Returns the audio and the number of pops (>4 dB)."""
    nfft, hop = 1024, 128                     # ~21 ms frames: a pop is short
    f, _, Z = stft(y, sr, nperseg=nfft, noverlap=nfft - hop)
    P = np.abs(Z) ** 2
    band = (f >= lo) & (f < hi)
    e_lo = 10 * np.log10(np.mean(P[band], axis=0) + 1e-18)
    e_mid = 10 * np.log10(np.mean(P[(f >= 300) & (f < 3000)], axis=0) + 1e-18)
    speech = e_mid > np.percentile(e_mid, 50)
    over = e_lo - _local_speech_median(e_lo, speech, hop / sr) - threshold_db
    loud = e_lo > (np.median(e_mid[speech]) if speech.any() else np.median(e_mid)) - 25     # pops are loud
    red = np.clip(over, 0, None) * (1 - 1 / ratio) * (e_lo > e_mid) * loud
    edge = nfft // hop                       # zero-padded analysis at the file edges is not a pop
    red[:edge] = 0
    red[-edge:] = 0
    g = _smooth_gain(10 ** (-red / 20), attack=1, release=20)          # ~3 ms attack, ~50 ms release
    Z[band] *= g[None, :]
    _, out = istft(Z, sr, nperseg=nfft, noverlap=nfft - hop)
    out = out[:len(y)]
    if len(out) < len(y):
        out = np.pad(out, (0, len(y) - len(out)))
    strong = red > 4
    count = int(np.sum(np.diff(np.concatenate([[0], strong.astype(np.int8)])) == 1))
    return out.astype(np.float32), count


def declick(y: np.ndarray, sr: int = SR, ratio: float = 3.0) -> Tuple[np.ndarray, int]:
    """Mouth clicks (咔哒): isolated impulses of at most ~3 ms in the >2 kHz band that
    stand ``ratio`` x above the *loudest* other moment of the surrounding +-10 ms.
    Glottal pulses of a pressed voice also make sharp high-band peaks, but every
    pitch period has one, so they are never isolated; consonant bursts (t, k, p) are
    followed by aspiration.  A click is replaced by a median-filtered version of
    itself (the impulse goes, the waveform underneath stays) with short crossfades."""
    from scipy.ndimage import maximum_filter1d
    from scipy.signal import butter as _butter, medfilt, sosfilt as _sosfilt

    blk = int(0.001 * sr)                                   # 1 ms blocks
    hpf = _sosfilt(_butter(4, 2000 / (sr / 2), btype="high", output="sos"), y)
    nb = len(y) // blk
    if nb < 50:
        return y, 0
    bmax = np.abs(hpf[:nb * blk]).reshape(nb, blk).max(axis=1)
    floor = np.percentile(bmax, 20)
    # loudest block in +-10 ms, excluding the candidate and its immediate neighbours
    left = maximum_filter1d(np.concatenate([np.zeros(2), bmax[:-2]]), size=9, origin=4)
    right = maximum_filter1d(np.concatenate([bmax[3:], np.zeros(3)]), size=9, origin=-4)
    around = np.maximum(left, right) + 1e-6
    hit = (bmax > ratio * around) & (bmax > 4 * floor)
    out = y.copy()
    count = 0
    i = 0
    while i < nb:
        if not hit[i]:
            i += 1
            continue
        j = i
        while j < nb and hit[j]:
            j += 1
        if j - i <= 3:
            a, b = max(0, (i - 1) * blk), min(len(y), (j + 1) * blk)
            lo = max(0, a - blk)
            seg = y[lo:min(len(y), b + blk)]
            fixed = medfilt(seg, 2 * blk + 1)[a - lo:a - lo + (b - a)]
            x = np.ones(b - a)
            r = min(blk // 2, (b - a) // 4)
            if r > 0:
                x[:r] = np.linspace(0, 1, r)
                x[-r:] = np.linspace(1, 0, r)
            out[a:b] = y[a:b] * (1 - x) + fixed * x
            count += 1
        i = j
    return out.astype(np.float32), count


def deess(y: np.ndarray, sr: int = SR, max_db: float = 6.0, lo: float = 4500.0, hi: float = 10000.0
          ) -> Tuple[np.ndarray, Dict[str, float]]:
    """Dynamic de-esser: where the sibilant band dominates the frame it is turned down
    (by up to ``max_db``), proportionally to how much it sticks out; elsewhere untouched."""
    nfft, hop = 1024, 256                     # ~5 ms hops: fast enough for an "s"
    f, _, Z = stft(y, sr, nperseg=nfft, noverlap=nfft - hop)
    band = (f >= lo) & (f <= hi)
    rest = (f >= 200) & (f < lo)
    be = np.sqrt(np.sum(np.abs(Z[band]) ** 2, axis=0) + 1e-18)
    re_ = np.sqrt(np.sum(np.abs(Z[rest]) ** 2, axis=0) + 1e-18)
    loud = _db(be) > np.percentile(_db(be), 50)
    ratio_db = _db(be) - _db(re_)
    thr = 0.0                                  # sibilant band louder than the voice body
    excess = np.clip(ratio_db - thr, 0, None) * loud
    red = np.minimum(max_db, excess * 0.8)
    g = _smooth_gain(10 ** (-red / 20), attack=1, release=4)
    Z[band] *= g[None, :]
    _, out = istft(Z, sr, nperseg=nfft, noverlap=nfft - hop)
    out = out[:len(y)]
    if len(out) < len(y):
        out = np.pad(out, (0, len(y) - len(out)))
    frames = int(np.sum(red > 1.0))
    return out.astype(np.float32), {"sibilant_ms": round(frames * hop / sr * 1000), "max_reduction_db":
                                    round(float(red.max()) if len(red) else 0.0, 1)}


def room_ir(sr: int = SR, rt60: float = 0.45, seed: int = 7) -> np.ndarray:
    """Synthetic small-room impulse response: a few early reflections + a damped
    exponential tail (high frequencies decay faster, like a real room)."""
    rng = np.random.default_rng(seed)
    n = int(sr * min(2.0, rt60 * 1.3))
    t = np.arange(n) / sr
    tail = rng.standard_normal(n) * np.exp(-6.9 * t / rt60)
    # progressive damping: one-pole low-pass whose cutoff falls over time
    out = np.empty(n)
    state = 0.0
    for i in range(n):
        a = 0.15 + 0.8 * (1 - math.exp(-t[i] / (rt60 / 3)))
        state = a * state + (1 - a) * tail[i]
        out[i] = state
    pre = int(0.012 * sr)
    ir = np.zeros(n + pre)
    ir[pre:] = out
    for d, g in ((0.007, 0.5), (0.011, 0.35), (0.017, 0.3), (0.023, 0.22)):
        ir[int(d * sr)] += g * (1 if rng.random() > 0.5 else -1)
    return (ir / np.sqrt(np.sum(ir ** 2))).astype(np.float32)


def add_reverb(y: np.ndarray, amount: float, rt60: float = 0.45, sr: int = SR) -> np.ndarray:
    """Dry + wet mix; ``amount`` 0.15 puts the room about 16 dB under the voice."""
    if amount <= 0:
        return y
    wet = fftconvolve(y, room_ir(sr, rt60))[:len(y)]
    wet *= np.sqrt(np.sum(y ** 2) / (np.sum(wet ** 2) + 1e-12))
    return (y + amount * wet).astype(np.float32)


def _speech_envelope(y: np.ndarray, sr: int = SR) -> np.ndarray:
    """0..1 'voice is speaking' envelope (50 ms attack, 600 ms release) for ducking."""
    hop = int(0.01 * sr)
    n = len(y) // hop + 1
    yy = np.pad(y, (0, n * hop - len(y)))
    r = _db(np.sqrt(np.mean(yy.reshape(n, hop) ** 2, axis=1)))
    on = (r > max(np.percentile(r, 40), r.max() - 40)).astype(float)
    env = _smooth_gain(1 - on, attack=5, release=60)          # 0 while speaking
    env = 1 - env
    return np.interp(np.arange(len(y)) / hop, np.arange(n), env)


def mix_bgm(voice: np.ndarray, bgm: np.ndarray, cfg: StudioConfig, sr: int = SR) -> Tuple[np.ndarray, Dict]:
    """Voice (n,) + music (2, m) -> (2, n).  The music is looped / trimmed to the voice,
    faded, and set so its loudness is ``bgm_ratio`` of the voice's (0.2 -> -14 dB)."""
    n = len(voice)
    if bgm.ndim == 1:
        bgm = np.vstack([bgm, bgm])
    if bgm.shape[1] < n and cfg.bgm_loop and bgm.shape[1] > sr:
        reps = int(math.ceil(n / bgm.shape[1]))
        xf = int(0.5 * sr)
        out = bgm.copy()
        for _ in range(reps - 1):
            t = np.linspace(0, 1, xf, dtype=np.float32)
            out[:, -xf:] = out[:, -xf:] * np.cos(t * np.pi / 2) + bgm[:, :xf] * np.sin(t * np.pi / 2)
            out = np.concatenate([out, bgm[:, xf:]], axis=1)
        bgm = out
    bgm = bgm[:, :n] if bgm.shape[1] >= n else np.pad(bgm, ((0, 0), (0, n - bgm.shape[1])))
    fi, fo = int(cfg.bgm_fade_in * sr), int(cfg.bgm_fade_out * sr)
    if fi:
        bgm[:, :fi] *= np.linspace(0, 1, fi)[None]
    if fo:
        bgm[:, -fo:] *= np.linspace(1, 0, fo)[None]
    lv, lb = measure_loudness(voice, sr)["lufs"], measure_loudness(bgm, sr)["lufs"]
    target = lv + 20 * math.log10(max(cfg.bgm_ratio, 1e-3))
    gain_db = target - lb
    bgm = bgm * (10 ** (gain_db / 20))
    if cfg.bgm_duck and cfg.bgm_duck_db > 0:
        env = _speech_envelope(voice, sr)
        bgm = bgm * (10 ** (-cfg.bgm_duck_db * env / 20))[None]
    mix = np.vstack([voice, voice]) + bgm
    return mix.astype(np.float32), {"voice_lufs": round(lv, 1), "bgm_lufs_before": round(lb, 1),
                                    "bgm_gain_db": round(gain_db, 1), "bgm_relative_db": round(target - lv, 1)}


def normalize_loudness(y: np.ndarray, target: float, true_peak: float, sr: int = SR) -> Tuple[np.ndarray, Dict]:
    """Gain to the target integrated loudness, then a 4x-oversampled lookahead limiter.
    The measured true peak and loudness are checked afterwards; the limiter ceiling is
    tightened / the gain corrected until both are within tolerance."""
    before = measure_loudness(y, sr)
    gain = target - before["lufs"]
    ceiling = true_peak - 0.3
    out, after = y, before
    for _ in range(4):
        lim = 10 ** (ceiling / 20)
        chain = (f"aresample={sr * 4},alimiter=limit={lim:.6f}:attack=1:release=50:level=0:latency=1,"
                 f"aresample={sr}")
        out = ff_filter(y * (10 ** (gain / 20)), chain, sr)
        after = measure_loudness(out, sr)
        tp_over = after["true_peak"] - true_peak
        miss = target - after["lufs"]
        if tp_over <= 0.05 and abs(miss) < 0.3:
            break
        if tp_over > 0.05:
            ceiling -= tp_over + 0.1
        if abs(miss) >= 0.3:
            gain += miss
    info = {"lufs_before": round(before["lufs"], 1), "lufs": round(after["lufs"], 1),
            "true_peak": round(after["true_peak"], 1), "lra": round(after["lra"], 1), "gain_db": round(gain, 1)}
    if abs(target - after["lufs"]) > 1.0:
        # very peaky material: reaching the target would need heavy limiting
        info["warning"] = (f"峰值相对平均音量太高，在不超过 {true_peak} dBTP 且不过度压限的前提下只能做到 "
                           f"{after['lufs']:.1f} LUFS；可以加大压缩后再试")
    return out, info


# ------------------------------------------------------------------ export
def export(y: np.ndarray, path: Path, cfg: StudioConfig, sr: int = SR) -> Path:
    path = Path(path)
    y2 = y if y.ndim == 2 else y[None]
    if cfg.channels == 2 and y2.shape[0] == 1:
        y2 = np.vstack([y2, y2])
    elif cfg.channels == 1 and y2.shape[0] == 2:
        y2 = y2.mean(axis=0, keepdims=True)
    fmt = cfg.format.lower()
    codec: list
    if fmt == "wav":
        codec = ["-c:a", {16: "pcm_s16le", 24: "pcm_s24le", 32: "pcm_f32le"}.get(cfg.bit_depth, "pcm_s24le")]
    elif fmt == "flac":
        codec = ["-c:a", "flac", "-sample_fmt", "s16" if cfg.bit_depth == 16 else "s32"]
    elif fmt == "mp3":
        codec = ["-c:a", "libmp3lame", "-b:a", cfg.bitrate]
    elif fmt in ("aac", "m4a"):
        enc = "libfdk_aac" if _has_encoder("libfdk_aac") else "aac"
        codec = ["-c:a", enc, "-b:a", cfg.bitrate]
    elif fmt == "opus":
        codec = ["-c:a", "libopus", "-b:a", cfg.bitrate]
    else:
        raise ValueError(f"unknown export format {cfg.format!r}")
    out_sr = 48000 if fmt == "opus" else cfg.out_sr
    cmd = [_ffmpeg(), "-nostdin", "-v", "error", "-y", "-f", "f32le", "-ar", str(sr), "-ac", str(y2.shape[0]),
           "-i", "pipe:0"]
    if fmt in ("wav", "flac") and cfg.bit_depth < 32:
        cmd += ["-af", "aresample=dither_method=triangular_hp"]       # dither when reducing to 16/24 bit
    cmd += ["-ar", str(out_sr)] + codec + [str(path)]
    subprocess.run(cmd, input=np.ascontiguousarray(np.clip(y2, -1, 1).T, dtype=np.float32).tobytes(), check=True)
    return path


_ENC: Dict[str, bool] = {}


def _has_encoder(name: str) -> bool:
    if name not in _ENC:
        out = subprocess.run([_ffmpeg(), "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
        _ENC[name] = re.search(rf"\s{re.escape(name)}\s", out) is not None
    return _ENC[name]


EXT = {"wav": ".wav", "flac": ".flac", "mp3": ".mp3", "aac": ".m4a", "m4a": ".m4a", "opus": ".opus"}


# ------------------------------------------------------------------ chain
def process(voice_path: Path, out_dir: Path, cfg: StudioConfig, bgm_path: Optional[Path] = None,
            basename: Optional[str] = None, asr_backend: str = "auto", asr_model: Optional[str] = None,
            device: Optional[str] = None) -> Dict[str, Any]:
    from .audio.io import load_audio

    voice_path, out_dir = Path(voice_path), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = basename or voice_path.stem
    report: Dict[str, Any] = {"steps": []}

    def step(name: str, **info):
        report["steps"].append({"step": name, **info})
        log.info("studio: %s %s", name, info)

    y = load_audio(voice_path, SR, mono=True).astype(np.float32)
    report["input"] = {"duration": round(len(y) / SR, 2), "noise_floor_db": round(noise_floor_db(y), 1),
                       **{k: round(v, 1) for k, v in measure_loudness(y).items()}}

    if cfg.highpass > 0:
        y = ff_filter(y, f"highpass=f={cfg.highpass}:p=2,highpass=f={cfg.highpass}:p=2")
        step("lowcut", freq=cfg.highpass)
    if cfg.denoise != "off":
        nf0 = noise_floor_db(y)
        chain = {
            "light": "afftdn=nr=10:nf=-50:tn=1",
            "medium": "arnndn=m=sh.rnnn:mix=0.85",
            "strong": "arnndn=m=sh.rnnn:mix=1,afftdn=nr=8:nf=-55:tn=1",
        }[cfg.denoise]
        if "arnndn" in chain and not (RNNOISE_DIR / "sh.rnnn").exists():
            log.warning("RNNoise model missing (%s); using the FFT denoiser", RNNOISE_DIR)
            chain = "afftdn=nr=14:nf=-50:tn=1"
        used = chain
        try:
            y = ff_filter(y, chain, cwd=RNNOISE_DIR if RNNOISE_DIR.exists() else None, retries=6)
        except subprocess.CalledProcessError:
            used = "afftdn=nr=14:nf=-50:tn=1"
            log.warning("RNNoise kept crashing in this ffmpeg build; used the FFT denoiser instead")
            y = ff_filter(y, used)
        step("denoise", mode=cfg.denoise, engine="RNNoise" if "arnndn" in used else "FFT",
             noise_floor_before=round(nf0, 1), noise_floor_after=round(noise_floor_db(y), 1))
    if cfg.cleanup:
        # breaths / coughs / fillers are found far more reliably once the noise is gone
        from .audio.io import save_audio
        from .roughcut import RoughCutConfig, rough_cut

        work = out_dir / "work"
        tmp = save_audio(work / f"{base}.denoised.wav", y, SR)
        rc = RoughCutConfig(fillers=cfg.fillers, repeats=False, retakes=False, unrecognized=cfg.fillers,
                            pauses=cfg.pauses, breaths=cfg.breaths, breath_reduce_db=cfg.breath_reduce_db,
                            coughs=cfg.coughs, extra_fillers=cfg.extra_fillers, acoustic_only=not cfg.fillers)
        res = rough_cut(tmp, work, rc, language=cfg.language, asr_backend=asr_backend, asr_model=asr_model,
                        device=device, audio_format="wav", video=False)
        y = load_audio(res["audio"], SR, mono=True).astype(np.float32)
        report["cleanup_plan"] = next((str(f) for f in res["files"] if str(f).endswith(".roughcut.json")), None)
        step("cleanup", **{k: v for k, v in res["stats"].items() if k in ("duration", "output_duration", "counts")})
    if cfg.declick:
        y, n = declick(y)
        step("declick", count=n)
    if cfg.plosives:
        y, n = reduce_plosives(y)
        step("plosives", count=n)
    if cfg.deess > 0:
        y, info = deess(y, max_db=cfg.deess)
        step("deess", **info)
    if cfg.eq:
        eq = [f"equalizer=f={cfg.mud_freq}:t=o:w=1.0:g={cfg.mud_gain}",
              f"equalizer=f={cfg.presence_freq}:t=o:w=1.2:g={cfg.presence_gain}"]
        if cfg.air_gain:
            eq.append(f"highshelf=f=10000:g={cfg.air_gain}")
        y = ff_filter(y, ",".join(eq))
        step("eq", mud=f"{cfg.mud_gain:+g} dB @ {cfg.mud_freq:g} Hz", presence=f"{cfg.presence_gain:+g} dB @ {cfg.presence_freq:g} Hz")
    if cfg.compress:
        thr = 10 ** (cfg.comp_threshold / 20)
        y = ff_filter(y, f"acompressor=threshold={thr:.6f}:ratio={cfg.comp_ratio}:attack={cfg.comp_attack}:"
                         f"release={cfg.comp_release}:knee=2.5:makeup=1")
        step("compressor", threshold_db=cfg.comp_threshold, ratio=cfg.comp_ratio)
    if cfg.reverb > 0:
        y = add_reverb(y, cfg.reverb, cfg.reverb_time)
        step("reverb", amount=cfg.reverb, rt60=cfg.reverb_time)

    out: np.ndarray = y
    if bgm_path:
        bgm = load_audio(bgm_path, SR, mono=False)
        out, info = mix_bgm(y, bgm, cfg)
        step("bgm", **info)
    if cfg.loudness is not None:
        out, info = normalize_loudness(out, cfg.loudness, cfg.true_peak)
        step("loudness", target=cfg.loudness, **info)
    path = export(out, out_dir / f"{base}.processed{EXT.get(cfg.format.lower(), '.wav')}", cfg)
    mono = out if out.ndim == 1 else out.mean(axis=0)
    report["output"] = {"file": path.name, "duration": round(len(mono) / SR, 2),
                        "noise_floor_db": round(noise_floor_db(mono), 1),
                        **{k: round(v, 1) for k, v in measure_loudness(out).items()}}
    report["config"] = asdict(cfg)
    rp = out_dir / f"{base}.studio.json"
    rp.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    return {"audio": path, "report": report, "files": [path, rp]}
