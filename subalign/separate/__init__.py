"""Vocal / accompaniment separation (人声分离 / 伴奏提取).

Backends (``backend="auto"`` tries them in this order):

* ``uvr``    - ``audio-separator`` with a BS-RoFormer / MDX model (best quality)
* ``demucs`` - Meta's Hybrid Transformer Demucs (``htdemucs`` / ``htdemucs_ft``)
* ``dsp``    - dependency-free fallback: stereo centre-channel extraction with
  harmonic masking, or REPET-SIM for mono.  Lower quality, but good enough
  as an analysis signal for lyric alignment.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import istft, stft

from ..audio.io import load_audio, save_audio

STEMS = ("vocals", "instrumental")


def _have(mod: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(mod) is not None


def separate(input_path, out_dir, backend: str = "auto", stems: Iterable[str] = STEMS, fmt: str = "wav",
             model: Optional[str] = None, device: Optional[str] = None, sr: int = 44100) -> Dict[str, Path]:
    """Separate ``input_path``; returns {"vocals": path, "instrumental": path} (requested stems only)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stems = tuple(stems)
    for s in stems:
        if s not in STEMS:
            raise ValueError(f"unknown stem {s!r}; choose from {STEMS}")
    order = {"auto": ["uvr", "demucs", "dsp"], "uvr": ["uvr"], "demucs": ["demucs"], "dsp": ["dsp"]}[backend]
    last: Optional[Exception] = None
    for b in order:
        try:
            if b == "uvr" and _have("audio_separator"):
                raw = _sep_uvr(input_path, out_dir, model)
            elif b == "demucs" and _have("demucs"):
                raw = _sep_demucs(input_path, out_dir, model, device)
            elif b == "dsp":
                raw = _sep_dsp(input_path, sr)
            else:
                continue
            break
        except Exception as e:  # try the next backend
            last = e
            if backend != "auto":
                raise
    else:
        raise RuntimeError(f"no separation backend succeeded: {last}")
    stem_name = Path(str(input_path)).stem
    result: Dict[str, Path] = {}
    for s in stems:
        dst = out_dir / f"{stem_name}.{s}.{fmt}"
        src = raw[s]
        if isinstance(src, tuple):            # (audio array, sr)
            save_audio(dst, src[0], src[1])
        else:
            if src.suffix.lower() == f".{fmt}":
                shutil.move(str(src), dst)
            else:
                y = load_audio(src, sr=sr, mono=False)
                save_audio(dst, y, sr)
        result[s] = dst
    # drop the backend's temp folder (demucs_* / uvr_* created inside out_dir)
    for src in raw.values():
        if isinstance(src, Path):
            for parent in src.parents:
                if parent.parent == out_dir and parent.name.startswith(("demucs_", "uvr_")):
                    shutil.rmtree(parent, ignore_errors=True)
                    break
    return result


# ------------------------------------------------------------------ backends
def _sep_demucs(input_path, out_dir: Path, model: Optional[str], device: Optional[str]) -> Dict[str, Path]:
    name = model or "htdemucs"
    tmp = Path(tempfile.mkdtemp(prefix="demucs_", dir=out_dir))
    cmd = [sys.executable, "-m", "demucs", "--two-stems", "vocals", "-n", name, "-o", str(tmp),
           "--filename", "{stem}.{ext}"]
    if device:
        cmd += ["-d", device]
    cmd.append(str(input_path))
    subprocess.run(cmd, check=True)
    base = tmp / name
    return {"vocals": base / "vocals.wav", "instrumental": base / "no_vocals.wav"}


def _sep_uvr(input_path, out_dir: Path, model: Optional[str]) -> Dict[str, Path]:
    from audio_separator.separator import Separator  # type: ignore

    tmp = Path(tempfile.mkdtemp(prefix="uvr_", dir=out_dir))
    kw = {"model_file_dir": os.environ["SUBALIGN_UVR_MODEL_DIR"]} if os.environ.get("SUBALIGN_UVR_MODEL_DIR") else {}
    sep = Separator(output_dir=str(tmp), output_format="WAV", **kw)
    sep.load_model(model_filename=model or "model_bs_roformer_ep_317_sdr_12.9755.ckpt")
    files = [Path(f) if Path(f).is_absolute() else tmp / f for f in sep.separate(str(input_path))]
    found: Dict[str, Path] = {}
    for f in files:
        low = f.name.lower()
        if "(vocals)" in low:
            found["vocals"] = f
        elif "(instrumental)" in low or "(no_vocals)" in low or "(other)" in low:
            found["instrumental"] = f
    if "vocals" in found and "instrumental" not in found:  # derive the accompaniment
        mix = load_audio(input_path, 44100, mono=False)
        voc = load_audio(found["vocals"], 44100, mono=False)
        n = min(mix.shape[1], voc.shape[1])
        p = tmp / "instrumental.wav"
        save_audio(p, mix[:, :n] - voc[:, :n], 44100)
        found["instrumental"] = p
    if set(found) != set(STEMS):
        raise RuntimeError(f"unexpected audio-separator outputs: {files}")
    return found


def _sep_dsp(input_path, sr: int) -> Dict[str, Tuple[np.ndarray, int]]:
    y = load_audio(input_path, sr=sr, mono=False)
    voc, inst = dsp_separate(y, sr)
    return {"vocals": (voc, sr), "instrumental": (inst, sr)}


def dsp_separate(y: np.ndarray, sr: int, n_fft: int = 4096) -> Tuple[np.ndarray, np.ndarray]:
    """y: (2, n) stereo or (n,) mono -> (vocals, instrumental) with y's shape."""
    hop = n_fft // 4
    stereo = y.ndim == 2 and y.shape[0] == 2 and not np.allclose(y[0], y[1], atol=1e-4)
    if y.ndim == 2 and not stereo:
        mono_in = y.mean(axis=0)
    else:
        mono_in = y if y.ndim == 1 else None
    if stereo:
        _, _, L = stft(y[0], sr, nperseg=n_fft, noverlap=n_fft - hop)
        _, _, R = stft(y[1], sr, nperseg=n_fft, noverlap=n_fft - hop)
        aL, aR = np.abs(L), np.abs(R)
        sim = 2 * np.abs(L * np.conj(R)) / (aL ** 2 + aR ** 2 + 1e-12)        # 1 == centre panned
        level = 1 - np.abs(aL - aR) / (aL + aR + 1e-12)
        center = np.clip(sim * level, 0, 1) ** 6
        M = 0.5 * (L + R)
        mask = center * (0.5 + 0.5 * _harmonic_mask(np.abs(M))) * _vocal_band(sr, n_fft)[:, None]
        mask = median_filter(mask, size=(3, 3))
        V = mask * M
        _, v = istft(V, sr, nperseg=n_fft, noverlap=n_fft - hop)
        n = y.shape[1]
        v = _fit(v, n)
        voc = np.vstack([v, v]).astype(np.float32)
        return voc, (y - voc).astype(np.float32)
    x = mono_in
    _, _, X = stft(x, sr, nperseg=n_fft, noverlap=n_fft - hop)
    mag = np.abs(X)
    bg = _repet_sim(mag, sr, hop)
    fg = np.maximum(mag - bg, 0)
    mask = fg ** 2 / (fg ** 2 + bg ** 2 + 1e-12) * _vocal_band(sr, n_fft)[:, None]
    _, v = istft(mask * X, sr, nperseg=n_fft, noverlap=n_fft - hop)
    v = _fit(v, len(x)).astype(np.float32)
    inst = (x - v).astype(np.float32)
    if y.ndim == 2:
        return np.vstack([v, v]), np.vstack([inst, inst])
    return v, inst


def _fit(v: np.ndarray, n: int) -> np.ndarray:
    return v[:n] if len(v) >= n else np.pad(v, (0, n - len(v)))


def _vocal_band(sr: int, n_fft: int, lo: float = 110.0, hi: float = 9000.0) -> np.ndarray:
    f = np.fft.rfftfreq(n_fft, 1 / sr)
    up = 1 / (1 + np.exp(-(f - lo) / 15))
    down = 1 / (1 + np.exp((f - hi) / 600))
    return up * down


def _harmonic_mask(mag: np.ndarray) -> np.ndarray:
    H = median_filter(mag, size=(1, 17))
    P = median_filter(mag, size=(17, 1))
    return H ** 2 / (H ** 2 + P ** 2 + 1e-12)


def _repet_sim(mag: np.ndarray, sr: int, hop: int, k: int = 12, min_sep_s: float = 1.0,
               block: int = 512) -> np.ndarray:
    """REPET-SIM: the repeating background of each frame is the median of its
    k most similar (non-adjacent) frames."""
    F, T = mag.shape
    feat = np.log1p(mag[: F // 2])
    feat = feat / (np.linalg.norm(feat, axis=0, keepdims=True) + 1e-9)
    min_sep = int(min_sep_s * sr / hop)
    bg = np.empty_like(mag)
    k = min(k, max(1, T - 2 * min_sep - 1))
    for s in range(0, T, block):
        e = min(T, s + block)
        sim = feat[:, s:e].T @ feat                                   # (b, T)
        idx = np.arange(s, e)[:, None]
        sim[np.abs(np.arange(T)[None, :] - idx) < min_sep] = -np.inf
        top = np.argpartition(-sim, kth=min(k, T - 1) - 1, axis=1)[:, :k]
        for r, frame in enumerate(range(s, e)):
            bg[:, frame] = np.median(mag[:, top[r]], axis=1)
    return np.minimum(bg, mag)
