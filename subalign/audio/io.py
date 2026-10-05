"""Audio loading / saving.  Decoding goes through ffmpeg when available so any
container (mp3, m4a, flac, mp4, mkv ...) works; soundfile is the fallback."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Tuple, Union

import numpy as np

PathLike = Union[str, Path]


def has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def load_audio(path: PathLike, sr: int = 16000, mono: bool = True) -> np.ndarray:
    """Decode ``path`` to float32 at ``sr``.  Returns (n,) if mono else (channels, n)."""
    path = str(path)
    if has_ffmpeg():
        ch = 1 if mono else 2
        cmd = ["ffmpeg", "-nostdin", "-v", "error", "-i", path, "-f", "f32le",
               "-acodec", "pcm_f32le", "-ac", str(ch), "-ar", str(sr), "-"]
        out = subprocess.run(cmd, capture_output=True, check=True).stdout
        y = np.frombuffer(out, dtype=np.float32).copy()
        if mono:
            return y
        return y.reshape(-1, 2).T.copy()
    import soundfile as sf

    y, file_sr = sf.read(path, dtype="float32", always_2d=True)
    y = y.T  # (channels, n)
    if file_sr != sr:
        from scipy.signal import resample_poly
        from math import gcd

        g = gcd(file_sr, sr)
        y = resample_poly(y, sr // g, file_sr // g, axis=1).astype(np.float32)
    if mono:
        return y.mean(axis=0)
    if y.shape[0] == 1:
        y = np.vstack([y, y])
    return y[:2]


def audio_duration(path: PathLike) -> float:
    try:
        import soundfile as sf

        return float(sf.info(str(path)).duration)
    except Exception:
        return len(load_audio(path, 16000)) / 16000.0


def save_audio(path: PathLike, y: np.ndarray, sr: int) -> Path:
    """Save (n,) or (channels, n) float audio.  Format follows the extension;
    wav/flac/ogg are written natively, anything else (mp3, m4a) via ffmpeg."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = y.T if y.ndim == 2 else y
    data = np.clip(data, -1.0, 1.0)
    ext = path.suffix.lower()
    import soundfile as sf

    if ext in (".wav", ".flac", ".ogg"):
        sf.write(str(path), data, sr)
        return path
    if not has_ffmpeg():
        raise RuntimeError(f"Writing {ext} requires ffmpeg")
    tmp = path.with_suffix(".tmp.wav")
    sf.write(str(tmp), data, sr)
    try:
        args = ["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(tmp)]
        if ext == ".mp3":
            args += ["-codec:a", "libmp3lame", "-b:a", "320k"]
        args.append(str(path))
        subprocess.run(args, check=True)
    finally:
        tmp.unlink(missing_ok=True)
    return path


def to_mono_16k(y: np.ndarray, sr: int) -> Tuple[np.ndarray, int]:
    if y.ndim == 2:
        y = y.mean(axis=0)
    if sr != 16000:
        from math import gcd

        from scipy.signal import resample_poly

        g = gcd(sr, 16000)
        y = resample_poly(y, 16000 // g, sr // g).astype(np.float32)
    return y.astype(np.float32), 16000
