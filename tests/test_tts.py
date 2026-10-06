import shutil

import numpy as np
import pytest
import soundfile as sf

from subalign import studio
from subalign.tts import dubbing, textprep
from subalign.tts.dubbing import DubConfig

SR = 48000
needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None and not studio.BUNDLED_FFMPEG.exists(),
                                  reason="ffmpeg not available")


# ------------------------------------------------------------------ text
@pytest.mark.parametrize("src,want", [
    ("共1,234,567人", "共一百二十三万四千五百六十七人"),
    ("2026-10-06", "二零二六年十月六日"),
    ("2026年10月6日10:30", "二零二六年十月六日十点三十分"),
    ("第3章有1/3", "第三章有三分之一"),
    ("粉丝1w+", "粉丝一万多"),
    ("涨了3.5%", "涨了百分之三点五"),
    ("3-5个", "三到五个"),
    ("售价¥99", "售价九十九元"),
    ("电话13812345678", "电话一三八一二三四五六七八"),
    ("气温-5℃", "气温零下五摄氏度"),
    ("2000元", "两千元"),
    ("12个", "十二个"),
])
def test_numbers(src, want):
    assert textprep.normalize(src).rstrip("。") == want


def test_clean_removes_notes_and_emoji():
    t = textprep.normalize("大家好（这里停一下）😀【画面：产品特写】今天聊聊AI哈哈哈哈哈 https://x.com/a")
    assert "停一下" not in t and "画面" not in t and "😀" not in t and "http" not in t
    assert t.startswith("大家好今天聊聊AI")


def test_markup_survives_normalisation():
    t = textprep.normalize("银<行|hang2>有3家，<停|0.5>记住<重|关键>")
    assert "<行|hang2>" in t and "<停|0.5>" in t and "<重|关键>" in t and "三家" in t


def test_to_engine():
    tts, emph = textprep.to_engine("去银<行|hang2>办<重|业务>。<停|0.3>")
    assert tts == "去银<行|HANG2>办业务。"
    assert emph == ["业务"]


def test_segments_split_merge_and_pauses():
    text = textprep.normalize("第一句话在这里。第二句<停|1.2>继续说。好。\n\n新的一段开始了，后面还有很多内容。")
    segs = textprep.split_segments(text)
    texts = [s.text for s in segs]
    # an explicit pause inside a sentence ends the piece with a comma; a 3-char piece joins
    # the sentence before it, but nothing is ever merged across the explicit pause
    assert texts[0] == "第一句话在这里。第二句，"
    assert segs[0].pause_after == pytest.approx(1.2)
    assert texts[1] == "继续说。好。"                     # 1-char sentence merged into its neighbour
    assert segs[1].pause_after >= 0.8                   # paragraph break
    assert segs[-1].paragraph_end


def test_long_sentence_split():
    s = "，".join(["这是一个比较长的分句内容"] * 8) + "。"
    segs = textprep.split_segments(s, max_chars=30)
    assert len(segs) > 1 and all(textprep._plain_len(x.text) <= 30 for x in segs)
    assert segs[-1].text.endswith("。")


def test_polyphone_positions_skip_markup():
    s = "我们<行|hang2>业的银行很重要"
    found = {p["pos"]: p for p in textprep.polyphones(s)}
    assert all(s[pos] == p["char"] for pos, p in found.items())
    assert 2 not in found                                # already annotated
    assert any(p["char"] == "行" and p["pos"] == s.rindex("行") for p in found.values())


# ------------------------------------------------------------------ audio helpers
def _tone(seconds, f0=200.0, amp=0.3, lead=0.0, tail=0.0):
    t = np.arange(int(seconds * SR)) / SR
    y = sum(np.sin(2 * np.pi * k * f0 * t) / k for k in range(1, 12)) * amp / 2
    return np.concatenate([np.zeros(int(lead * SR)), y, np.zeros(int(tail * SR))]).astype(np.float32)


def test_trim_take():
    y = _tone(1.0, lead=0.5, tail=0.7)
    z = dubbing.trim_take(y)
    assert 1.0 <= len(z) / SR <= 1.2
    assert abs(z[0]) < 1e-3 and abs(z[-1]) < 1e-3          # faded edges


def test_speaking_rate_ignores_pauses():
    y = np.concatenate([_tone(1.0), np.zeros(SR), _tone(1.0)])
    assert dubbing.speaking_rate(y, "一二三四五六七八九十") == pytest.approx(5.0, rel=0.1)


@needs_ffmpeg
def test_naturalize_keeps_level_and_adds_air():
    y = _tone(2.0)
    z = dubbing.naturalize(y, DubConfig(air=1.0, warmth=0.3, deharsh_db=2.0))
    assert np.isfinite(z).all()
    assert abs(20 * np.log10(np.std(z) / np.std(y))) < 1.5
    spec = lambda x: np.abs(np.fft.rfft(x))
    f = np.fft.rfftfreq(len(y), 1 / SR)
    hi = f > 9000
    assert spec(z)[hi].sum() > spec(y)[hi].sum()


# ------------------------------------------------------------------ project
class FakeWorker:
    def __init__(self):
        self.calls = []

    def request(self, **req):
        self.calls.append(req)
        n = len(req["text"])
        sf.write(req["out"], _tone(0.2 * n, lead=0.2, tail=0.3), SR)
        return {"duration": 0.2 * n + 0.5, "seconds": 0.1}


def _project(tmp_path, text="第一句话在这里。第二句话也在这里。第三句话说完了。"):
    spk = tmp_path / "a.wav"
    sf.write(spk, _tone(3.0), SR)
    cfg = DubConfig(qa=False, breaths=False, rate_tolerance=0.0)
    return dubbing.create_project(tmp_path / "p", text, spk, None, cfg)


def test_synth_edit_and_takes(tmp_path):
    proj = _project(tmp_path)
    pdir = tmp_path / "p"
    w = FakeWorker()
    assert len(proj["segments"]) == 3
    seg = dubbing.synth_segment(pdir, 1, w)
    assert seg["status"] == "done" and (pdir / seg["audio"]).exists()
    assert w.calls[0]["duration_factor"] == 1.0 and w.calls[0]["emo"] is None
    first = seg["audio"]
    seg = dubbing.edit_segment(pdir, 1, text="改过的第1句<停|0.2>")
    assert seg["status"] == "edited" and "第一句" in seg["text"]
    seg = dubbing.synth_segment(pdir, 1, w)
    assert seg["status"] == "done" and len(seg["takes"]) == 2 and seg["audio"] != first
    seg = dubbing.use_take(pdir, 1, first)
    assert seg["audio"] == first
    seg = dubbing.edit_segment(pdir, 2, pause_after=1.5)
    assert seg["pause_after"] == 1.5


@needs_ffmpeg
def test_assemble(tmp_path):
    _project(tmp_path)
    pdir = tmp_path / "p"
    w = FakeWorker()
    for sid in (1, 2, 3):
        dubbing.synth_segment(pdir, sid, w)
    mix = dubbing.assemble(pdir, dubbing.load(pdir))
    y, sr = sf.read(pdir / mix["file"])
    assert sr == SR and len(mix["timing"]) == 3
    t = mix["timing"]
    assert t[0]["start"] < t[0]["end"] < t[1]["start"] < t[2]["start"]
    gap = y[int((t[0]["end"] + 0.1) * SR):int((t[1]["start"] - 0.1) * SR)]
    assert 0 < np.sqrt(np.mean(gap ** 2)) < 10 ** (-50 / 20)       # room tone: not digital silence, but quiet
    assert dubbing.load(pdir)["mix"]["duration"] == pytest.approx(len(y) / SR, abs=0.02)
    p = dubbing.export_mix(pdir, "mp3")
    assert p.exists() and p.suffix == ".mp3"


def test_worker_restarts_after_a_cuda_error():
    class Fake(dubbing.TTSWorker):
        def __init__(self):
            super().__init__()
            self.calls, self.killed = [], 0

        def _request(self, req):
            self.calls.append(req)
            if len(self.calls) == 1:
                raise RuntimeError("AcceleratorError: CUDA error: unknown error")
            return {"ok": True, "out": "x.wav"}

        def _kill(self):
            self.killed += 1

    w = Fake()
    assert w.request(cmd="synth", text="你好")["ok"] and w.killed == 1 and len(w.calls) == 2

    class Other(Fake):
        def _request(self, req):
            raise RuntimeError("text too long")
    with pytest.raises(RuntimeError, match="too long"):
        Other().request(cmd="synth", text="x")
