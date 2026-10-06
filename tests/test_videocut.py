import shutil
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import asdict

import numpy as np
import pytest
import soundfile as sf

from subalign import videocut as vc
from subalign.roughcut import Item
from subalign.videocut import FrameStore, Removal, Shot, VideoCutConfig

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not available")


def _rem(a, b, reasons=("filler",), mutable=True, edge=False, value=None):
    r = Removal(start=a, end=b, reasons=list(reasons), mutable=mutable, edge=edge)
    r.value = value if value is not None else vc.VALUE.get(reasons[0], 1.0) + 0.8 * (b - a)
    return r


# ------------------------------------------------------------------ deciding
def test_short_shots_give_up_the_cheapest_cut():
    # three fillers 0.8 s apart: cutting all of them would leave 0.6 s shots
    rems = [_rem(2.0, 2.2), _rem(3.0, 3.3, mutable=False), _rem(3.9, 4.1), _rem(9.0, 12.0, ("retake",), False)]
    vc.choose_modes(rems, VideoCutConfig(min_shot=1.5, max_cuts=10), duration=20.0)
    modes = [r.mode for r in rems]
    assert modes[3] == "cut"                              # retakes are protected
    assert modes.count("cut") <= 2
    # shots between the remaining cuts are long enough
    cuts = [r for r in rems if r.mode == "cut"]
    for a, b in zip(cuts, cuts[1:]):
        assert b.start - a.end >= 1.5
    # a mutable filler is silenced, a non-mutable one kept
    for r in rems:
        if r.mode != "cut":
            assert r.mode == ("mute" if r.mutable else "keep") and r.why == "short_shot"


def test_density_limit():
    rems = [_rem(t, t + 0.4, ("repeat",), False) for t in (1, 3, 5, 7, 9)]
    vc.choose_modes(rems, VideoCutConfig(min_shot=0.5, max_cuts=3), duration=12.0)
    assert sum(r.mode == "cut" for r in rems) == 3
    assert any(r.why == "density" for r in rems)


def test_screen_recordings_cut_everything_and_locked_modes_stay():
    rems = [_rem(1.0, 1.2), _rem(1.6, 1.8)]
    vc.choose_modes(rems, VideoCutConfig(), "screen", duration=5.0)
    assert [r.mode for r in rems] == ["cut", "cut"]
    rems = [_rem(1.0, 1.2), _rem(1.6, 1.8)]
    rems[0].mode, rems[0].locked = "keep", True
    rems[1].mode, rems[1].locked = "cut", True
    vc.choose_modes(rems, VideoCutConfig(), duration=5.0)
    assert [r.mode for r in rems] == ["keep", "cut"]


def test_removals_from_keeps_reasons_and_values():
    items = [Item("大", 0.0, 0.3, 0, "大"), Item("嗯", 0.6, 0.9, 0, "嗯", cut="filler", auto="filler"),
             Item("家", 1.2, 1.5, 0, "家")]
    rems = vc.removals_from_keeps([(0.2, 0.5), (1.1, 2.0)], [0.12, 0.0], items, 2.5)
    assert [r.edge for r in rems] == [True, False, True]
    mid = rems[1]
    assert mid.reasons == ["filler"] and mid.text == "嗯" and mid.mutable and mid.speech == pytest.approx(0.3)
    assert rems[2].reasons == ["pause"] and not rems[2].mutable


def test_final_keeps():
    rems = [_rem(0.0, 0.3, edge=True), _rem(2.0, 2.4), _rem(5.0, 6.0), _rem(9.5, 10.0, edge=True)]
    rems[1].mode, rems[1].fill = "mute", 0.1
    rems[2].fill = 0.12
    keeps, fills, joints = vc.final_keeps(rems, 10.0)
    assert keeps == [(0.3, 5.0), (6.0, 9.5)] and fills == [0.12, 0.0] and joints == [rems[2]]


# ------------------------------------------------------------------ picture
def test_frame_plan_dissolve_is_centred():
    shots = [Shot(0, 10, 0), Shot(10, 20, 30, fade_in=4)]
    plan = vc.frame_plan(shots)
    assert len(plan) == 20
    assert plan[7] == [(7, 1.0, 0)] and plan[12] == [(42, 1.0, 1)]
    blend = plan[8:12]
    assert [len(p) for p in blend] == [2, 2, 2, 2]
    assert blend[0][0] == (8, 0.875, 0) and blend[0][1] == (38, 0.125, 1)
    assert sum(w for _, w, _ in blend[1]) == pytest.approx(1.0)


def test_store_jump_and_face_consistency():
    st = FrameStore(25.0, (16, 9))
    st.frames = {0: np.zeros((9, 16), np.uint8), 1: np.full((9, 16), 10, np.uint8), 2: np.full((9, 16), 200, np.uint8)}
    st.motion = np.array([0, 0.04, 0.04], np.float32)
    assert st.jump(0, 1) < st.jump(0, 2)
    st.faces = {0: (0.4, 0.2, 0.2, 0.3), 10: None, 20: None, 30: None, 40: None}
    assert st.face_near(0.8) is None                       # one hit in five samples: texture
    st.faces = {i: (0.4, 0.2, 0.2, 0.3) for i in range(0, 50, 10)}
    assert st.face_near(0.8) == pytest.approx((0.4, 0.2, 0.2, 0.3))


def test_transform_zoom_keeps_size():
    pytest.importorskip("cv2")
    img = np.zeros((90, 160, 3), np.uint8)
    img[40:50, 75:85] = 255
    out = vc._transform(img, 1.5, (0.5, 0.5))
    assert out.shape == img.shape and out[45, 80, 0] == 255 and out[39, 74, 0] > 0 and out[45, 70, 0] == 0


# ------------------------------------------------------------------ end to end: sync to the frame
def _encoded_video(path, fps=25, seconds=8.0, words=()):
    """Every frame shows its own index as its brightness; the sound has a 220 Hz
    tone for every word (start, end)."""
    n = int(seconds * fps)
    w, h = 64, 48
    sr = 48000
    t = np.arange(int(seconds * sr)) / sr
    audio = np.zeros_like(t, dtype=np.float32)
    for a, b in words:
        m = (t >= a) & (t < b)
        audio[m] = 0.4 * np.sin(2 * np.pi * 220 * t[m])
    wav = path.with_suffix(".wav")
    sf.write(wav, audio, sr)
    frames = np.repeat(np.arange(n, dtype=np.uint8)[:, None, None], h * w, axis=1).reshape(n, h, w)
    p = subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{w}x{h}", "-r", str(fps),
                        "-i", "-", "-i", str(wav), "-c:v", "libx264", "-qp", "0", "-pix_fmt", "yuv444p", "-c:a", "aac",
                        "-shortest", str(path)], input=frames.tobytes())
    assert p.returncode == 0
    return path


def _decode_gray(path):
    out = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "gray", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(out, np.uint8).reshape(-1, 48, 64)


@needs_ffmpeg
@pytest.mark.parametrize("transitions,min_shot", [("cut", 0.5), ("fade", 0.5), ("auto", 3.0)])
def test_end_to_end_cuts_stay_in_sync(tmp_path, transitions, min_shot):
    fps = 25
    # words with a filler (嗯) and a retake in between
    words = [("我", 0.5, 0.9), ("们", 1.0, 1.4), ("嗯", 1.8, 2.2), ("今", 2.6, 3.0), ("天", 3.1, 3.5),
             ("讲", 3.6, 4.0), ("错", 4.4, 4.8), ("讲", 5.6, 6.0), ("话", 6.1, 6.5), ("了", 6.6, 7.0)]
    src = _encoded_video(tmp_path / "src.mp4", fps, 8.0, [(a, b) for _, a, b in words])
    cuts = {"嗯": "filler", "错": "retake"}
    items = [Item(w, a, b, 0, w, cut=cuts.get(w), auto=cuts.get(w)) for w, a, b in words]
    plan = {"items": [asdict(it) for it in items], "settings": {"pauses": False}}
    cfg = VideoCutConfig(min_shot=min_shot, transitions=transitions, fcpxml=True, mute=False)
    from subalign.roughcut import RoughCutConfig

    res = vc.video_cut(src, tmp_path / "out", RoughCutConfig(pauses=False), cfg, plan=plan)
    out_v = next(f for f in res["files"] if f.suffix == ".mp4")
    out_a = next(f for f in res["files"] if f.suffix == ".wav")
    frames = _decode_gray(out_v)
    shown = np.round(frames[:, 16:32, 24:40].mean(axis=(1, 2))).astype(int)    # source frame on screen
    y, sr = sf.read(out_a)
    env = np.convolve(np.abs(y), np.ones(240) / 240, mode="same")
    on = np.flatnonzero((env[1:] > 0.1) & (env[:-1] <= 0.1)) / sr              # word onsets in the output
    given_up = {r.text for r in res["removals"] if r.mode == "keep"}
    kept = [(w, a) for w, a, _ in words if w not in cuts or w in given_up]
    assert len(on) == len(kept)
    for (w, a), t_out in zip(kept, on):
        n = int(round(t_out * fps))
        assert abs(shown[n] - a * fps) <= 1, (w, t_out, shown[n], a * fps)
    if min_shot > 2:          # the filler is too close to the retake: given up, heard again
        assert given_up == {"嗯"} and res["stats"]["cuts"] == 1
    else:
        assert res["stats"]["cuts"] == 2
        if transitions == "fade":
            assert res["stats"]["transitions"].get("fade", 0) >= 1
    # the timeline file is valid XML whose clips cover the whole output
    root = ET.parse(next(f for f in res["files"] if f.suffix == ".fcpxml")).getroot()
    clips = root.findall(".//spine/asset-clip")
    from fractions import Fraction

    total = sum(Fraction(c.get("duration").rstrip("s")) for c in clips)
    assert abs(float(total) - len(frames) / fps) <= 2 / fps
