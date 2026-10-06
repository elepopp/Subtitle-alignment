"""Translated video (视频翻译): the AI voice-over of a translation put back into the
original video.

1. **background**: the original soundtrack is separated (:mod:`subalign.separate`);
   the voice is dropped, music / ambience are kept.
2. **mix**: the dubbed voice (assembled on the original timeline) goes over the
   background at the original balance - the background sits as far below the
   new voice as it sat below the original one (adjustable, optional ducking) -
   and the result is brought to the loudness of the original soundtrack.
3. **subtitles**: the translation (optionally with the original line) timed to
   where the dubbed sentences are actually spoken, laid out for the video's
   own resolution; burned in, added as a switchable track, or both.
4. **video**: ``keep`` copies the video stream untouched when nothing is burned
   in (no quality loss, same container) and otherwise re-encodes with the same
   codec family, size, frame rate and about the original bit rate; ``custom``
   picks resolution / frame rate / codec / quality / container.
"""
from __future__ import annotations

import json
import os
import logging
import math
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

log = logging.getLogger("subalign")
SR = 48000
QUALITY_CRF = {"high": 18, "standard": 23, "small": 28}


@dataclass
class VideoSpec:
    mode: str = "keep"              # keep | custom
    height: Optional[int] = None    # custom: short side in px (2160 / 1440 / 1080 / 720 / 480); None = source
    fps: Optional[float] = None     # custom: None = source
    codec: str = "h264"             # custom: h264 | h265
    quality: str = "high"           # custom: high | standard | small
    container: str = "mp4"          # custom: mp4 | mkv | mov


@dataclass
class ComposeConfig:
    separation: str = "auto"        # auto | uvr | demucs | dsp | none (no background)
    bg_gain_db: float = 0.0         # background relative to the original balance
    duck: bool = False              # lower the background a little under the dubbed voice
    duck_db: float = 4.0
    extend: bool = True             # dub longer than the video: hold the last frame (else the end is cut)
    subtitles: str = "burn"         # none | burn | soft | both
    bilingual: bool = False         # second row: the original line
    audio_bitrate: str = "256k"
    spec: VideoSpec = field(default_factory=VideoSpec)


# ------------------------------------------------------------------ media
def _ffmpeg() -> str:
    from .studio import _ffmpeg as f

    return f()


def _ffprobe() -> str:
    ff = Path(_ffmpeg())
    cand = ff.with_name(ff.name.replace("ffmpeg", "ffprobe"))
    return str(cand) if cand.exists() else (shutil.which("ffprobe") or "ffprobe")


def probe(path: Path) -> Dict[str, Any]:
    """Video stream facts needed to keep the format: codec, size (rotation applied),
    frame rate, pixel format, bit rate, container; and whether there is a video at all."""
    from .videocut import video_info

    d = json.loads(subprocess.run([_ffprobe(), "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
                                  capture_output=True, check=True, text=True, encoding="utf-8").stdout)
    vs = next((s for s in d.get("streams", []) if s.get("codec_type") == "video"
               and not s.get("disposition", {}).get("attached_pic")), None)
    if vs is None:
        raise ValueError("源文件没有视频画面，不能生成翻译视频")
    info = video_info(path)
    br = vs.get("bit_rate") or (vs.get("tags") or {}).get("BPS")
    if not br:                                       # mkv / webm: container rate minus audio
        tot = d.get("format", {}).get("bit_rate")
        aud = sum(int(s.get("bit_rate") or 0) for s in d.get("streams", []) if s.get("codec_type") == "audio")
        br = int(tot) - aud if tot else None
    info.update(codec=vs.get("codec_name", ""), pix_fmt=vs.get("pix_fmt", ""), bit_rate=int(br) if br else None,
                ext=path.suffix.lower().lstrip("."), format=d.get("format", {}).get("format_name", ""))
    return info


# ------------------------------------------------------------------ audio
def _loud(y: np.ndarray) -> float:
    from .studio import measure_loudness

    try:
        return measure_loudness(y, SR)["lufs"]
    except Exception:
        return -70.0


def _fit(y: np.ndarray, n: int) -> np.ndarray:
    return y[:, :n] if y.shape[1] >= n else np.pad(y, ((0, 0), (0, n - y.shape[1])))


def _stereo(y: np.ndarray) -> np.ndarray:
    if y.ndim == 1:
        return np.vstack([y, y])
    return y if y.shape[0] == 2 else np.vstack([y[0], y[0]])


def separate_background(video: Path, stems_dir: Path, backend: str = "auto", isolate: bool = True) -> Dict[str, Path]:
    """vocals / instrumental of the original soundtrack (cached in ``stems_dir``)."""
    from .separate import separate

    stems_dir.mkdir(parents=True, exist_ok=True)
    cached = {k: stems_dir / f"{k}.wav" for k in ("vocals", "instrumental")}
    meta = stems_dir / "stems.json"
    if all(p.exists() for p in cached.values()) and meta.exists() and \
            json.loads(meta.read_text(encoding="utf-8")).get("backend") == backend:
        return cached
    if isolate:
        # in its own process: the separator's GPU runtime next to already loaded recognition
        # models can crash the host process (seen as a segfault in the web server)
        import sys

        out = subprocess.run([sys.executable, "-m", "subalign", "separate", str(video), "-o", str(stems_dir / "tmp"),
                              "--backend", backend, "--stems", "vocals,instrumental", "--format", "wav"],
                             capture_output=True, text=True, encoding="utf-8", errors="replace",
                             env=dict(os.environ, PYTHONIOENCODING="utf-8"))
        if out.returncode:
            raise RuntimeError(f"人声分离失败（{out.returncode}）：{(out.stderr or out.stdout)[-800:]}")
        k = out.stdout.rfind("{\n")                    # the result is the last (indented) JSON block
        res = {k2: Path(v) for k2, v in json.loads(out.stdout[k if k >= 0 else out.stdout.index("{"):]).items()}
    else:
        res = separate(video, stems_dir / "tmp", backend=backend, stems=("vocals", "instrumental"), fmt="wav")
    for k, p in res.items():
        shutil.move(str(p), cached[k])
    shutil.rmtree(stems_dir / "tmp", ignore_errors=True)
    meta.write_text(json.dumps({"backend": backend}), encoding="utf-8")
    return cached


def mix_tracks(voice: np.ndarray, background: Optional[np.ndarray], n: int, original_balance_db: Optional[float],
               target_lufs: Optional[float], cfg: ComposeConfig) -> tuple:
    """Dubbed voice + background -> stereo mix of ``n`` samples at the original
    balance, at the loudness of the original soundtrack."""
    from .studio import _speech_envelope, normalize_loudness

    v = _fit(_stereo(voice.astype(np.float32)), n)
    lv = _loud(v)
    info: Dict[str, Any] = {"voice_lufs": round(lv, 1)}
    mix = v
    if background is not None:
        b = _fit(_stereo(background.astype(np.float32)), n)
        lb = _loud(b)
        rel = (original_balance_db if original_balance_db is not None else -12.0) + cfg.bg_gain_db
        gain = (lv + rel) - lb if lb > -69 else 0.0
        b = b * 10 ** (gain / 20)
        if cfg.duck and cfg.duck_db > 0:
            env = _speech_envelope(v[0], SR)
            b = b * (10 ** (-cfg.duck_db * env / 20))[None]
        mix = v + b
        info.update(background_relative_db=round(rel, 1), background_gain_db=round(gain, 1),
                    original_balance_db=None if original_balance_db is None else round(original_balance_db, 1))
    if target_lufs is not None and target_lufs > -60:
        mix, ln = normalize_loudness(mix, target_lufs, -1.0, SR)
        info["loudness"] = ln
    tail = int(0.03 * SR)
    mix[:, -tail:] *= np.linspace(1, 0, tail)[None]
    return mix.astype(np.float32), info


# ------------------------------------------------------------------ subtitles
def subtitle_document(sentences: Sequence[Dict[str, Any]], bilingual: bool = False, language: str = ""):
    """``sentences``: [{"text": translation, "original": source text, "start", "end"}] in
    output time -> a Document (token times interpolated inside each line)."""
    from .models import Document, Line
    from .text.tokenize import syllable_weight, tokenize

    lines = []
    for s in sentences:
        if not (s.get("text") or "").strip() or s.get("start") is None:
            continue
        a, b = float(s["start"]), float(max(s["end"], s["start"] + 0.3))
        toks = tokenize(s["text"].strip())
        # the line's time spread over its words by syllables (line breaking needs word times)
        w = [syllable_weight(t) for t in toks]
        tot, acc = sum(w) or 1.0, 0.0
        for t, x in zip(toks, w):
            t.start = round(a + (b - a) * acc / tot, 3)
            acc += x
            t.end = round(a + (b - a) * acc / tot, 3)
        lines.append(Line(tokens=toks, start=a, end=b, translation=(s.get("original") or None) if bilingual else None))
    return Document(lines=lines, language=language or None, kind="speech")


# on-screen subtitles: no full stops, commas become two half-width spaces (numbers untouched)
_COMMA = re.compile(r"[ \t]*(?:(?<!\d)[,，]|[,，](?!\d))[ \t]*")
_STOP = re.compile(r"(?<!\d)[.。]+|[.。]+(?!\d)")


def clean_punct(text: str) -> str:
    t = _STOP.sub("", _COMMA.sub("  ", text))
    return re.sub(r" {3,}", "  ", t)


def clean_ass(ass_text: str) -> str:
    """:func:`clean_punct` on the text of every Dialogue line (override tags and \\N kept)."""
    out = []
    for line in ass_text.splitlines():
        if line.startswith("Dialogue:"):
            head = line.split(",", 9)
            if len(head) == 10:
                parts = re.split(r"(\{[^}]*\}|\\N|\\n)", head[9])
                body = "".join(p if p.startswith("{") or p in ("\\N", "\\n") else clean_punct(p) for p in parts)
                body = re.sub(r" *(\\N) *", r"\1", body).strip()
                line = ",".join(head[:9] + [body])
        out.append(line)
    return "\n".join(out) + ("\n" if ass_text.endswith("\n") else "")


def clean_srt(srt_text: str) -> str:
    out = []
    for line in srt_text.splitlines():
        if line.strip() and not line.strip().isdigit() and "-->" not in line:
            line = clean_punct(line).strip()
        out.append(line)
    return "\n".join(out) + ("\n" if srt_text.endswith("\n") else "")


def subtitle_files(doc, out_dir: Path, base: str, width: int, height: int, style=None, bilingual: bool = False,
                   font_size: Optional[int] = None, punct: bool = True) -> Dict[str, Path]:
    from .formats import write
    from .formats.ass import write_ass
    from .segment.layout import get_layout
    from .segment.linebreak import segment_document

    layout = get_layout(f"{width}x{height}", font_size)
    doc = segment_document(doc, layout)
    ass = out_dir / f"{base}.ass"
    a_txt = write_ass(doc, style=style, layout=layout, bilingual=bilingual)
    ass.write_text(clean_ass(a_txt) if punct else a_txt, encoding="utf-8")
    srt = out_dir / f"{base}.srt"
    s_txt = write(doc, "srt", bilingual=bilingual)
    srt.write_text(clean_srt(s_txt) if punct else s_txt, encoding="utf-8")
    js = out_dir / f"{base}.json"
    js.write_text(json.dumps(doc.to_dict(), ensure_ascii=False), encoding="utf-8")
    return {"ass": ass, "srt": srt, "json": js}


def prepare_fonts(ass: Path, fonts_dir: Path, library=None, lang_main: str = "", lang_second: str = "") -> List[Dict[str, str]]:
    """Make the subtitles use only fonts of the free-for-commercial-use library
    (:mod:`subalign.fontlib`) that cover their text, and copy those fonts into
    ``fonts_dir`` - libass reads them from there.  Returns the substitutions made."""
    from .fontlib import FontLibrary

    lib = library or FontLibrary.load()
    if not lib.entries:
        raise RuntimeError("可商用字体库是空的：运行 python webui/install_fonts.py（fonts/ 下放字体压缩包）")
    text, subs, used = lib.enforce(ass.read_text(encoding="utf-8"), lang_main, lang_second)
    ass.write_text(text, encoding="utf-8")
    fonts_dir.mkdir(parents=True, exist_ok=True)
    for e in used:
        src = lib.path(e)
        dst = fonts_dir / src.name
        if not dst.exists():
            shutil.copy(src, dst)
    return subs


# ------------------------------------------------------------------ video
def _scale_size(w: int, h: int, short: Optional[int]) -> tuple:
    if not short or short >= min(w, h):
        return w, h
    k = short / min(w, h)
    return int(round(w * k / 2)) * 2, int(round(h * k / 2)) * 2


def encode_args(info: Dict[str, Any], spec: VideoSpec, burn: bool) -> Dict[str, Any]:
    """Container, video codec arguments and filters for the output."""
    copy = spec.mode == "keep" and not burn
    if spec.mode == "keep":
        container = info["ext"] if info["ext"] in ("mp4", "mov", "mkv", "webm", "m4v") else "mp4"
        if container == "m4v":
            container = "mp4"
    else:
        container = spec.container if spec.container in ("mp4", "mkv", "mov") else "mp4"
    vf: List[str] = []
    if copy:
        vargs = ["-c:v", "copy"]
    elif spec.mode == "keep":
        codec = {"h264": "libx264", "hevc": "libx265", "vp9": "libvpx-vp9", "av1": "libx264"}.get(info["codec"], "libx264")
        if container == "webm" and codec != "libvpx-vp9":
            codec = "libvpx-vp9"
        ten_bit = "10" in (info.get("pix_fmt") or "") and codec in ("libx265", "libvpx-vp9")
        vargs = ["-c:v", codec, "-pix_fmt", "yuv420p10le" if ten_bit else "yuv420p"]
        br = info.get("bit_rate")
        if br and br > 200_000:                       # about the original bit rate (+5 % for the subtitles)
            k = int(br * 1.05 / 1000)
            vargs += ["-b:v", f"{k}k", "-maxrate", f"{int(k * 1.5)}k", "-bufsize", f"{k * 2}k"]
        else:
            vargs += ["-crf", "18"] if codec != "libvpx-vp9" else ["-crf", "24", "-b:v", "0"]
        if codec in ("libx264", "libx265"):
            vargs += ["-preset", "medium"]
        if codec == "libx265":
            vargs += ["-tag:v", "hvc1"]
    else:
        codec = "libx265" if spec.codec == "h265" else "libx264"
        vargs = ["-c:v", codec, "-crf", str(QUALITY_CRF.get(spec.quality, 18) + (2 if codec == "libx265" else 0)),
                 "-preset", "medium", "-pix_fmt", "yuv420p"]
        if codec == "libx265":
            vargs += ["-tag:v", "hvc1"]
        w, h = _scale_size(info["width"], info["height"], spec.height)
        if (w, h) != (info["width"], info["height"]):
            vf.append(f"scale={w}:{h}:flags=lanczos")
        if spec.fps and abs(spec.fps - float(info["fps"])) > 0.01:
            vf.append(f"fps={spec.fps}")
    acodec = ["-c:a", "libopus", "-b:a", "192k"] if container == "webm" else ["-c:a", "aac"]
    scodec = {"mp4": "mov_text", "mov": "mov_text", "mkv": "ass", "webm": "webvtt"}[container]
    return {"container": container, "vargs": vargs, "vf": vf, "copy": copy, "acodec": acodec, "scodec": scodec}


def _run_ffmpeg(cmd: List[str], duration: float, cwd: Path, progress: Optional[Callable[[float], None]]) -> None:
    p = subprocess.Popen(cmd + ["-progress", "pipe:1", "-nostats"], cwd=str(cwd), stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace")
    assert p.stdout is not None
    for line in p.stdout:
        if line.startswith("out_time_us=") and progress and duration > 0:
            try:
                progress(min(1.0, int(line.split("=")[1]) / 1e6 / duration))
            except ValueError:
                pass
    err = p.stderr.read() if p.stderr else ""
    if p.wait():
        raise RuntimeError(f"ffmpeg failed: {err[-1500:]}")


# ------------------------------------------------------------------ entry point
def compose(video: Path, voice_wav: Path, sentences: Sequence[Dict[str, Any]], out_dir: Path, cfg: ComposeConfig,
            stems_dir: Optional[Path] = None, style=None, font_size: Optional[int] = None,
            font_library=None, source_language: str = "", base: str = "translated",
            language: str = "", progress: Optional[Callable[[str, float], None]] = None) -> Dict[str, Any]:
    """Background + dubbed voice + subtitles -> a translated video in ``out_dir``."""
    from .audio.io import load_audio, save_audio

    note = progress or (lambda *_: None)
    out_dir.mkdir(parents=True, exist_ok=True)
    info = probe(video)
    dur = float(info["duration"]) or 0.0
    original = _stereo(load_audio(video, SR, mono=False))
    n = original.shape[1] if original.size else int(dur * SR)
    report: Dict[str, Any] = {"source": {k: (str(v) if k == "fps" else v) for k, v in info.items()}, "config": asdict(cfg)}

    # 1. background
    background, balance = None, None
    if cfg.separation != "none":
        note("separate", 0.0)
        stems = separate_background(video, stems_dir or out_dir / "stems", cfg.separation)
        background = load_audio(stems["instrumental"], SR, mono=False)
        vocals = load_audio(stems["vocals"], SR, mono=False)
        lvoc, lbg = _loud(_stereo(vocals)), _loud(_stereo(background))
        if lvoc > -60 and lbg > -69:
            balance = lbg - lvoc
        report["original"] = {"vocals_lufs": round(lvoc, 1), "background_lufs": round(lbg, 1)}

    # 2. mix
    note("mix", 0.0)
    voice = load_audio(voice_wav, SR, mono=True)
    # where the dubbed speech really ends (the voice track has room tone / a tail after it)
    hop = int(0.02 * SR)
    env = np.sqrt(np.mean(voice[:len(voice) // hop * hop].reshape(-1, hop) ** 2, axis=1)) if len(voice) >= hop else np.zeros(1)
    loud = np.flatnonzero(env > max(env.max() * 10 ** (-40 / 20), 1e-4)) if env.size else np.array([], int)
    speech_end = (loud[-1] + 1) * hop / SR if len(loud) else 0.0
    over = max(0.0, speech_end + 0.3 - n / SR)
    extend = over if (over > 0.05 and cfg.extend) else 0.0
    total = n + int(extend * SR)
    target = _loud(original) if original.size else None
    mix, minfo = mix_tracks(voice, background, total, balance, target, cfg)
    if extend:
        minfo["extended_s"] = round(extend, 2)       # last frame held until the dub has finished
    elif over > 0.05:
        minfo["truncated_s"] = round(over, 2)
    report["mix"] = minfo
    audio = save_audio(out_dir / f"{base}.mix.wav", mix, SR)

    # 3. subtitles
    files: Dict[str, Path] = {"audio": audio}
    want_subs = cfg.subtitles in ("burn", "soft", "both")
    if want_subs:
        doc = subtitle_document(sentences, cfg.bilingual, language)
        files.update(subtitle_files(doc, out_dir, base, info["width"], info["height"], style, cfg.bilingual, font_size))
    burn = cfg.subtitles in ("burn", "both")

    # 4. video
    note("encode", 0.0)
    ea = encode_args(info, cfg.spec, burn or extend > 0)       # holding a frame needs re-encoding
    dst = out_dir / f"{base}.{ea['container']}"
    vf = list(ea["vf"])
    if extend:
        vf.append(f"tpad=stop_mode=clone:stop_duration={extend + 0.1:.3f}")
    if burn:
        subs = prepare_fonts(files["ass"], out_dir / "fonts", font_library, language, source_language)
        if subs:
            report["font_substituted"] = subs
        vf.append(f"subtitles={files['ass'].name}:fontsdir=fonts")
    cmd = [_ffmpeg(), "-nostdin", "-v", "error", "-y", "-i", str(video.resolve()), "-i", str(audio.resolve())]
    soft = cfg.subtitles in ("soft", "both")
    if soft:
        cmd += ["-i", str((files["ass"] if ea["scodec"] == "ass" else files["srt"]).resolve())]
    cmd += ["-map", "0:v:0", "-map", "1:a:0"] + (["-map", "2:s:0"] if soft else [])
    if vf:
        cmd += ["-vf", ",".join(vf)]
    cmd += ea["vargs"] + ea["acodec"] + (["-b:a", cfg.audio_bitrate] if ea["container"] != "webm" else [])
    if soft:
        cmd += ["-c:s", ea["scodec"]]
        lang3 = {"en": "eng", "zh": "chi", "ja": "jpn", "ko": "kor", "es": "spa", "fr": "fre", "de": "ger", "ru": "rus"}.get(
            (language or "").split("-")[0])
        if lang3:
            cmd += ["-metadata:s:s:0", f"language={lang3}"]
    if ea["container"] in ("mp4", "mov"):
        cmd += ["-movflags", "+faststart"]
    # the length of the original video (-shortest would end at the last subtitle)
    cmd += ["-t", f"{(total / SR if original.size else dur):.3f}", str(dst.resolve())]
    _run_ffmpeg(cmd, dur, out_dir, lambda f: note("encode", f))
    files["video"] = dst
    out = probe(dst)
    report["output"] = {"container": ea["container"], "video_copied": ea["copy"], "burned": burn, "soft": soft,
                        "width": out["width"], "height": out["height"], "fps": str(out["fps"]), "codec": out["codec"],
                        "bit_rate": out["bit_rate"], "duration": out["duration"], "size": dst.stat().st_size}
    rp = out_dir / f"{base}.report.json"
    rp.write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    files["report"] = rp
    note("done", 1.0)
    return {"files": files, "report": report}
