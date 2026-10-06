import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from subalign import dubvideo as dv
from subalign.dubvideo import ComposeConfig, VideoSpec

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not available")
INFO = {"width": 1920, "height": 1080, "fps": 30, "codec": "h264", "pix_fmt": "yuv420p", "bit_rate": 8_000_000, "ext": "mp4"}


def test_keep_copies_the_video_unless_something_is_burned_in():
    a = dv.encode_args(INFO, VideoSpec(), burn=False)
    assert a["copy"] and a["vargs"] == ["-c:v", "copy"] and a["container"] == "mp4" and a["scodec"] == "mov_text"
    b = dv.encode_args(INFO, VideoSpec(), burn=True)
    assert not b["copy"] and b["vargs"][:2] == ["-c:v", "libx264"] and "8400k" in b["vargs"]   # ~ original bit rate
    hevc = dv.encode_args(dict(INFO, codec="hevc", pix_fmt="yuv420p10le", ext="mkv"), VideoSpec(), burn=True)
    assert hevc["vargs"][:4] == ["-c:v", "libx265", "-pix_fmt", "yuv420p10le"] and hevc["container"] == "mkv"
    assert hevc["scodec"] == "ass"


def test_custom_spec():
    a = dv.encode_args(INFO, VideoSpec(mode="custom", height=720, fps=25, codec="h265", quality="small", container="mov"),
                       burn=False)
    assert a["container"] == "mov" and not a["copy"]
    assert "libx265" in a["vargs"] and a["vargs"][a["vargs"].index("-crf") + 1] == "30"
    assert a["vf"] == ["scale=1280:720:flags=lanczos", "fps=25"]
    # never upscaled; portrait keeps its short side
    assert dv._scale_size(640, 360, 720) == (640, 360)
    assert dv._scale_size(1080, 1920, 720) == (720, 1280)


def test_subtitle_document_and_fonts(tmp_path):
    doc = dv.subtitle_document([{"text": "Hello there.", "original": "你好。", "start": 1.0, "end": 2.0},
                                {"text": "", "original": "x", "start": 3.0, "end": 4.0}], bilingual=True)
    assert len(doc.lines) == 1 and doc.lines[0].translation == "你好。" and doc.lines[0].tokens[0].start == 1.0
    files = dv.subtitle_files(doc, tmp_path, "t", 1280, 720, bilingual=True)
    ass = files["ass"].read_text(encoding="utf-8")
    assert "PlayResX: 1280" in ass and ",Hello there\n" in ass and "你好" in ass   # full stops removed
    pytest.importorskip("fontTools")
    from test_fontlib import LATIN, make_font

    from subalign.fontlib import FontEntry, FontLibrary

    (tmp_path / "lib" / "H").mkdir(parents=True)
    make_font(tmp_path / "lib" / "H" / "h.ttf", "Free Hans", "你好。" + LATIN)
    lib = FontLibrary([FontEntry(id="H", file="H/h.ttf", family="Free Hans", display="H")], tmp_path / "lib")
    subs = dv.prepare_fonts(files["ass"], tmp_path / "fonts", lib, "en", "zh")
    ass2 = files["ass"].read_text(encoding="utf-8")
    assert subs and "Noto Sans CJK SC" not in ass2 and "Free Hans" in ass2
    assert (tmp_path / "fonts" / "h.ttf").exists()


@needs_ffmpeg
def test_mix_keeps_the_original_balance():
    sr = dv.SR
    t = np.arange(3 * sr) / sr
    voice = (0.3 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
    bg = (0.05 * np.sin(2 * np.pi * 90 * t)).astype(np.float32)
    mix, info = dv.mix_tracks(voice, bg, len(t), -10.0, None, ComposeConfig())
    assert mix.shape == (2, len(t))
    assert info["background_relative_db"] == -10.0
    measured = dv._loud(np.vstack([bg, bg]) * 10 ** (info["background_gain_db"] / 20)) - dv._loud(np.vstack([voice, voice]))
    assert abs(measured + 10.0) < 0.6


def _video(path, seconds=4.0):
    """A small test video whose sound is a tone (the 'original voice') over noise."""
    sr = 48000
    t = np.arange(int(seconds * sr)) / sr
    rng = np.random.default_rng(1)
    a = 0.3 * np.sin(2 * np.pi * 220 * t) * (t < 2) + 0.02 * rng.standard_normal(len(t))
    wav = path.with_suffix(".wav")
    sf.write(wav, a.astype(np.float32), sr)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc=size=320x180:rate=25:duration={seconds}",
                    "-i", str(wav), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(path)], check=True)
    return path


def _streams(p):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(p)],
                         capture_output=True, text=True, check=True).stdout
    return json.loads(out)


@needs_ffmpeg
@pytest.mark.parametrize("subs,spec,copied", [("soft", VideoSpec(), True), ("burn", VideoSpec(), False),
                                              ("both", VideoSpec(mode="custom", height=120, container="mkv"), False)])
def test_compose(tmp_path, subs, spec, copied):
    src = _video(tmp_path / "src.mp4")
    voice = tmp_path / "voice.wav"
    sr = 48000
    t = np.arange(int(3.0 * sr)) / sr
    sf.write(voice, (0.3 * np.sin(2 * np.pi * 330 * t) * ((t > 0.5) & (t < 2.5))).astype(np.float32), sr)
    sents = [{"text": "Hello", "original": "你好", "start": 0.5, "end": 2.5}]
    lib = None
    if subs in ("burn", "both"):
        pytest.importorskip("fontTools")
        from test_fontlib import LATIN, make_font

        from subalign.fontlib import FontEntry, FontLibrary

        (tmp_path / "lib" / "L").mkdir(parents=True)
        make_font(tmp_path / "lib" / "L" / "l.ttf", "Free Latin", LATIN + "你好")
        lib = FontLibrary([FontEntry(id="L", file="L/l.ttf", family="Free Latin", display="L")], tmp_path / "lib")
    res = dv.compose(src, voice, sents, tmp_path / "out", ComposeConfig(separation="dsp", subtitles=subs, spec=spec),
                     stems_dir=tmp_path / "stems", font_library=lib)
    out = res["files"]["video"]
    d = _streams(out)
    kinds = [s["codec_type"] for s in d["streams"]]
    assert kinds.count("video") == 1 and kinds.count("audio") == 1
    assert ("subtitle" in kinds) == (subs in ("soft", "both"))
    assert abs(float(d["format"]["duration"]) - 4.0) < 0.15
    o = res["report"]["output"]
    assert o["video_copied"] is copied
    if spec.mode == "custom":
        assert out.suffix == ".mkv" and (o["width"], o["height"]) == (214, 120)


@needs_ffmpeg
def test_dub_longer_than_the_video_holds_the_last_frame(tmp_path):
    src = _video(tmp_path / "src.mp4", seconds=3.0)
    voice = tmp_path / "voice.wav"
    sr = 48000
    t = np.arange(int(4.5 * sr)) / sr
    sf.write(voice, (0.3 * np.sin(2 * np.pi * 330 * t) * (t > 0.5)).astype(np.float32), sr)
    res = dv.compose(src, voice, [], tmp_path / "out", ComposeConfig(separation="none", subtitles="none"))
    assert res["report"]["mix"]["extended_s"] == pytest.approx(1.8, abs=0.1)
    assert abs(float(_streams(res["files"]["video"])["format"]["duration"]) - 4.8) < 0.15
    res = dv.compose(src, voice, [], tmp_path / "out2", ComposeConfig(separation="none", subtitles="none", extend=False))
    assert res["report"]["mix"]["truncated_s"] > 1.5 and res["report"]["output"]["video_copied"]


def test_subtitle_punctuation():
    assert dv.clean_punct("Now, since Opus 5.5 has been released, people love it.") == \
        "Now  since Opus 5.5 has been released  people love it"
    assert dv.clean_punct("这款手机的价格是5,999元，比去年贵了不少。") == "这款手机的价格是5,999元  比去年贵了不少"
    line = r"Dialogue: 0,0:00:01.00,0:00:02.00,Default,,0,0,0,,{\fad(150,150)}Hello, world.\N你好，世界。"
    assert dv.clean_ass(line + "\n") == r"Dialogue: 0,0:00:01.00,0:00:02.00,Default,,0,0,0,,{\fad(150,150)}Hello  world\N你好  世界" + "\n"
    srt = "1\n00:00:01,000 --> 00:00:02,000\nHello, world.\n"
    assert dv.clean_srt(srt) == "1\n00:00:01,000 --> 00:00:02,000\nHello  world\n"
