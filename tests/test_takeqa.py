import numpy as np
import pytest
import soundfile as sf

from subalign.tts import dubbing, takeqa
from subalign.tts.dubbing import DubConfig

from test_expressive import _sil, _voice

SR = 48000


def test_score_terms():
    seg = {"src_spread": 10.0, "src_f0": 3.0}
    ok = takeqa.score({"voice": 0.8, "spread": 9.0, "f0": 2.0, "length": 3.0, "slot": 3.0}, seg, 0.0, 0.12, 1.15)
    assert ok["err"] == 0 and ok["voice"] == pytest.approx(0.4) and ok["perf"] == pytest.approx(1 / 6, abs=1e-3)
    assert ok["fit"] == 0 and ok["total"] == pytest.approx(0.4 + 1 / 6, abs=1e-3)
    flat = takeqa.score({"voice": 0.8, "spread": 4.0, "f0": -1.0}, seg, 0.0, 0.12, 1.15)
    assert flat["total"] > ok["total"]                                   # a flat reading of a lively line
    misread = takeqa.score({"voice": 0.9, "spread": 10.0, "f0": 3.0}, seg, 0.2, 0.12, 1.15)
    assert misread["total"] > 10                                          # failing QA only wins when all fail
    long_ = takeqa.score({"length": 4.0, "slot": 3.0}, {}, None, 0.12, 1.15)
    assert long_ == {"fit": pytest.approx(5 * (4 / 3 - 1.15), abs=1e-3), "total": pytest.approx(5 * (4 / 3 - 1.15), abs=1e-3)}
    assert takeqa.score({}, {}, None, 0.12, 1.15) == {"total": 0}


def test_slot():
    assert takeqa.slot_for({"src_start": 1.0, "src_end": 2.0}, {"src_start": 3.0}) == pytest.approx(1.92)
    assert takeqa.slot_for({"src_start": 1.0, "src_end": 2.0}, None) == pytest.approx(2.0)
    assert takeqa.slot_for({"src_start": None}, None) is None


class FakeWorker:
    """Takes with more and more pitch movement (glide 0, 2, 4, ... semitones)."""

    def __init__(self):
        self.calls = []

    def request(self, **req):
        self.calls.append(req)
        g = 2.0 * (len(self.calls) - 1)
        sf.write(req["out"], np.concatenate([_sil(0.2), _voice(2.0, f0=150, glide=g), _sil(0.2)]), SR)
        return {"duration": 2.4, "seconds": 0.1}


def _project(tmp_path, **cfg):
    ref = tmp_path / "ref.wav"
    sf.write(ref, _voice(4.0, f0=150), SR)
    sents = [{"text": "第一句话。", "start": 0.0, "end": 2.5, "src_spread": 5.5, "src_f0": 0.0}]
    c = DubConfig(qa=False, breaths=False, timeline=True, pick_best=True, line_ref=True, **cfg)
    dubbing.create_project(tmp_path / "p", "", ref, None, c, sentences=sents)
    return tmp_path / "p"


def test_pick_best_takes_the_closest_performance(tmp_path, monkeypatch):
    monkeypatch.setattr(takeqa, "voice_similarity", lambda take, ref: 0.8)
    pdir = _project(tmp_path, candidates=3)
    w = FakeWorker()
    seg = dubbing.synth_segment(pdir, 1, w)
    assert len(w.calls) == 3 and len(seg["takes"]) == 3
    spreads = [t["measure"]["spread"] for t in seg["takes"]]
    assert spreads[0] < spreads[1] < spreads[2]
    want = min(seg["takes"], key=lambda t: abs(t["measure"]["spread"] - 5.5))
    assert seg["audio"] == want["file"] and seg["score"] == want["score"] and seg["measure"]["voice"] == 0.8
    # picking another take carries its measurements
    other = next(t for t in seg["takes"] if t["file"] != seg["audio"])
    assert dubbing.use_take(pdir, 1, other["file"])["score"] == other["score"]


def test_pick_best_keeps_trying_while_every_take_misreads(tmp_path, monkeypatch):
    monkeypatch.setattr(takeqa, "voice_similarity", lambda take, ref: 0.8)
    errs = iter([0.5, 0.4, 0.0, 0.0])
    monkeypatch.setattr(dubbing, "check_take", lambda path, text, lang: {"error": next(errs), "heard": ""})
    pdir = _project(tmp_path, candidates=2, max_tries=4)
    dubbing.set_config(pdir, qa=True)
    seg = dubbing.synth_segment(pdir, 1, FakeWorker())
    assert len(seg["takes"]) == 3                       # two candidates failed, the third passed
    assert seg["qa"]["error"] == 0.0 and seg["status"] == "done"


def test_without_pick_best_first_good_take_wins(tmp_path, monkeypatch):
    errs = iter([0.0, 0.0])
    monkeypatch.setattr(dubbing, "check_take", lambda path, text, lang: {"error": next(errs), "heard": ""})
    pdir = _project(tmp_path, max_tries=3)
    dubbing.set_config(pdir, qa=True, pick_best=False)
    seg = dubbing.synth_segment(pdir, 1, FakeWorker())
    assert len(seg["takes"]) == 1 and "score" not in seg["takes"][0]


def test_short_take_gets_no_voice_score(tmp_path, monkeypatch):
    called = []
    monkeypatch.setattr(takeqa, "voice_similarity", lambda take, ref: called.append(take) or 0.8)
    short = tmp_path / "short.wav"
    sf.write(short, np.concatenate([_sil(0.1), _voice(1.0), _sil(0.1)]), SR)
    m = takeqa.measure(short, {}, str(short), slot=None)
    assert m["voice"] is None and not called and m["length"] < takeqa.MIN_VOICE_S
