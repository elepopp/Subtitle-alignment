import shutil

import numpy as np
import pytest
import soundfile as sf

from subalign import studio
from subalign.tts import dubbing, expressive
from subalign.tts.dubbing import DubConfig

SR = 48000
needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None and not studio.BUNDLED_FFMPEG.exists(),
                                  reason="ffmpeg not available")


def _voice(dur, sr=SR, f0=140.0, db=-20.0, glide=0.0, seed=0):
    """Speech-like: a harmonic tone with syllable-rate amplitude modulation."""
    t = np.arange(int(dur * sr)) / sr
    f = f0 * 2 ** (glide * np.sin(2 * np.pi * 0.7 * t) / 12)
    ph = 2 * np.pi * np.cumsum(f) / sr
    y = sum(np.sin(k * ph) / k for k in range(1, 6))
    y *= 0.6 + 0.4 * np.sin(2 * np.pi * 4.5 * t) ** 2
    y = y / np.sqrt(np.mean(y ** 2)) * 10 ** (db / 20)
    return y.astype(np.float32)


def _sil(dur, sr=SR):
    return np.zeros(int(dur * sr), np.float32)


# ------------------------------------------------------------------ measurements
def test_level_ignores_pauses():
    a = _voice(2.0, db=-20)
    b = np.concatenate([_voice(1.0, db=-20), _sil(1.0), _voice(1.0, db=-20)])
    assert abs(expressive.level_db(a, SR) - expressive.level_db(b, SR)) < 1.0
    assert expressive.level_db(_voice(2.0, db=-30), SR) < expressive.level_db(a, SR) - 8


def test_silent_gaps_inside_only():
    y = np.concatenate([_sil(0.3), _voice(1.0), _sil(0.5), _voice(1.0), _sil(0.1), _voice(0.5), _sil(0.3)])
    gaps = expressive.silent_gaps(y, SR, 0.3)
    assert len(gaps) == 1
    a, b = gaps[0]
    assert a == pytest.approx(1.3, abs=0.06) and b == pytest.approx(1.8, abs=0.06)


def test_pitch_stats():
    med, spread = expressive.pitch_stats(_voice(1.5, sr=16000, f0=220.0))
    assert med == pytest.approx(24.0, abs=0.5) and spread < 1.0          # 220 Hz = 2 octaves over 55 Hz
    _, wide = expressive.pitch_stats(_voice(1.5, sr=16000, f0=220.0, glide=4.0))
    assert wide > 4.0
    assert expressive.pitch_stats(_sil(1.0, 16000)) == (None, None)


def test_expressiveness():
    ref = {"level": (-20.0, 3.0), "f0": (20.0, 1.0), "spread": (4.0, 1.0)}
    calm = expressive.expressiveness({"level": -20.5, "f0": 20.2, "spread": 3.0}, ref)
    loud = expressive.expressiveness({"level": -10.0, "f0": 24.0, "spread": 8.0}, ref)
    whisper = expressive.expressiveness({"level": -32.0, "f0": None, "spread": None}, ref)
    assert calm == 0.0 and loud > 0.8 and whisper > 0.8
    assert expressive.expressiveness({}, ref) == 0.0


# ------------------------------------------------------------------ the dub side
def test_clause_positions():
    assert expressive.clause_positions("我说的是明天，不是今天。", "zh") == [0.6]
    pos = expressive.clause_positions("Well, I said tomorrow, not today.", "en")
    assert len(pos) == 2 and 0 < pos[0] < pos[1] < 1
    assert expressive.clause_positions("No pause here.", "en") == []


def test_place_pauses_lengthens_the_matching_comma():
    take = np.concatenate([_voice(1.0), _sil(0.15), _voice(1.0)])
    text = "I said tomorrow, not today."
    y, n = expressive.place_pauses(take, text, [{"at": 0.5, "dur": 0.7}], "en", SR)
    assert n == 1 and (len(y) - len(take)) / SR == pytest.approx(0.55, abs=0.03)
    gaps = expressive.silent_gaps(y, SR, 0.3)
    assert len(gaps) == 1 and gaps[0][1] - gaps[0][0] == pytest.approx(0.7, abs=0.05)
    # no punctuation near the pause / pause shorter than the model's: unchanged
    assert expressive.place_pauses(take, text, [{"at": 0.95, "dur": 0.7}], "en", SR)[1] == 0
    assert expressive.place_pauses(take, text, [{"at": 0.5, "dur": 0.1}], "en", SR)[1] == 0
    assert expressive.place_pauses(take, "No comma at all.", [{"at": 0.5, "dur": 0.7}], "en", SR)[1] == 0


def test_dynamics_gains_follow_the_original():
    g = expressive.dynamics_gains([-24, -20, -22], [-26, -20, -14], range_db=8)
    out = np.array([-24, -20, -22]) + np.array(g)
    assert out[1] == pytest.approx(-22.0)                   # the median line sits at the median
    assert out[2] - out[1] == pytest.approx(6.0)             # 6 dB louder in the original
    assert out[1] - out[0] == pytest.approx(6.0)
    g = expressive.dynamics_gains([-22, -22, -22], [-40, -20, -20], range_db=8)
    assert g[0] == pytest.approx(-8.0)                       # capped
    assert expressive.dynamics_gains([-70, -22], [-20, -20], 8)[0] == 0.0      # silent take untouched


# ------------------------------------------------------------------ analysis
def _recording(tmp_path):
    """S1 calm, S2 calm, S1 shouting with a pause in the middle, S1 calm, a short S2 line."""
    parts = [_sil(0.5), _voice(3.0, f0=120, db=-24), _sil(0.6), _voice(2.5, f0=210, db=-24), _sil(0.6),
             _voice(1.2, f0=170, db=-12, glide=5), _sil(0.6), _voice(1.2, f0=170, db=-12, glide=5), _sil(0.6),
             _voice(3.0, f0=120, db=-24), _sil(0.6), _voice(0.6, f0=210, db=-24), _sil(0.5)]
    y = np.concatenate(parts)
    p = tmp_path / "orig.wav"
    sf.write(p, y, SR)
    t, spans = 0.0, []
    for k, x in enumerate(parts):
        d = len(x) / SR
        if k % 2:
            spans.append([t, t + d])
        t += d
    # line 3 = the two shouted pieces with the 0.6 s pause between them
    s = [spans[0], spans[1], [spans[2][0], spans[3][1]], spans[4], spans[5]]
    who = ["A", "B", "A", "A", "B"]
    return p, [{"start": a, "end": b, "speaker": w} for (a, b), w in zip(s, who)]


def test_analyze_source(tmp_path):
    p, sents = _recording(tmp_path)
    sents.append({"start": None, "end": None})              # an untimed line
    res = expressive.analyze_source(sents, p, tmp_path / "ref", expressive.ExprConfig(diarize=False))
    r = res["sentences"]
    assert [x.get("speaker") for x in r[:5]] == ["S1", "S2", "S1", "S1", "S2"] and r[5] == {}
    assert set(res["speakers"]) == {"S1", "S2"}
    for k, v in res["speakers"].items():
        y, sr = sf.read(v["ref"])
        assert sr == expressive.REF_SR and 1.0 < len(y) / sr <= 15.0
    # the voice reference of S1 is made of the calm lines, not the shouted one
    s1, _ = sf.read(res["speakers"]["S1"]["ref"])
    assert len(s1) / expressive.REF_SR == pytest.approx(3.15 * 2 + 0.2, abs=0.1)
    assert r[2]["expr"] > 0.5 > r[0]["expr"]
    assert r[2]["src_level"] > r[0]["src_level"] + 8
    assert len(r[2]["src_pauses"]) == 1 and r[2]["src_pauses"][0]["at"] == pytest.approx(0.5, abs=0.05)
    assert r[0]["src_pauses"] == []
    # every timed line has its own emotion reference; the 0.6 s line was widened
    # (to the left only up to the end of S1's line before it)
    for x in r[:5]:
        assert x["emo"] and (tmp_path / "ref" / "emo").exists()
    short, _ = sf.read(r[4]["emo"])
    assert len(short) / expressive.REF_SR >= 1.45
    assert r[0]["spk"] == res["speakers"]["S1"]["ref"] and r[1]["spk"] == res["speakers"]["S2"]["ref"]


def test_analyze_source_one_speaker_without_labels(tmp_path):
    p, sents = _recording(tmp_path)
    for s in sents:
        s.pop("speaker")
    res = expressive.analyze_source(sents, p, tmp_path / "ref", expressive.ExprConfig(diarize=False),
                                    voice_refs=False)
    assert {x["speaker"] for x in res["sentences"]} == {"S1"} and res["speakers"] == {}
    assert all(x["spk"] is None and x["emo"] for x in res["sentences"])


# ------------------------------------------------------------------ dubbing project
class FakeWorker:
    def __init__(self):
        self.calls = []

    def request(self, **req):
        self.calls.append(req)
        sf.write(req["out"], np.concatenate([_sil(0.2), _voice(0.9), _sil(0.12), _voice(0.9), _sil(0.3)]), SR)
        return {"duration": 2.4, "seconds": 0.1}


def _dub(tmp_path, **cfg_kw):
    p, sents = _recording(tmp_path)
    sents = sents[:4]
    ana = expressive.analyze_source(sents, p, tmp_path / "p" / "ref", expressive.ExprConfig(diarize=False))
    texts = ["First line here, calm.", "Second speaker, also calm.", "Now I shout, really loud!", "Calm again, at the end."]
    for s, a, t in zip(sents, ana["sentences"], texts):
        s.update({k: v for k, v in a.items() if v is not None}, text=t)
    cfg = DubConfig(qa=False, breaths=False, lang="EN", timeline=True, source_emo=True, follow_dynamics=True,
                    source_pauses=True, emo_alpha=0.9, emo_floor=0.4, **cfg_kw)
    spk = ana["speakers"]["S1"]["ref"]
    dubbing.create_project(tmp_path / "p", "", spk, None, cfg, sentences=sents, speakers=ana["speakers"])
    return tmp_path / "p", ana


def test_per_sentence_references(tmp_path):
    pdir, ana = _dub(tmp_path)
    proj = dubbing.load(pdir)
    assert proj["speakers"] == ana["speakers"]
    w = FakeWorker()
    for sid in (1, 2, 3):
        dubbing.synth_segment(pdir, sid, w)
    c1, c2, c3 = w.calls
    assert c1["spk"] == ana["speakers"]["S1"]["ref"] and c2["spk"] == ana["speakers"]["S2"]["ref"]
    assert c1["emo"] == ana["sentences"][0]["emo"] and c3["emo"] == ana["sentences"][2]["emo"]
    assert 0.4 <= c1["emo_alpha"] < c3["emo_alpha"] <= 0.9
    # turned off: the project's single emotion reference (none here) and the global weight
    dubbing.set_config(pdir, source_emo=False)
    dubbing.synth_segment(pdir, 3, w)
    assert w.calls[-1]["emo"] is None and w.calls[-1]["emo_alpha"] == 0.9


@needs_ffmpeg
def test_assemble_follows_original_dynamics_and_pauses(tmp_path):
    pdir, ana = _dub(tmp_path)
    w = FakeWorker()
    for sid in (1, 2, 3, 4):
        dubbing.synth_segment(pdir, sid, w)
    mix = dubbing.assemble(pdir, dubbing.load(pdir))
    t = {x["id"]: x for x in mix["timing"]}
    assert t[3]["gain_db"] - t[1]["gain_db"] == pytest.approx(8.0, abs=1.0)    # shouted line ~12 dB up, capped at 8
    assert t[3]["pauses"] == 1 and t[1]["pauses"] == 0
    # off: evened out again
    dubbing.set_config(pdir, follow_dynamics=False, source_pauses=False)
    mix = dubbing.assemble(pdir, dubbing.load(pdir))
    t = {x["id"]: x for x in mix["timing"]}
    assert abs(t[3]["gain_db"] - t[1]["gain_db"]) < 1.0 and t[3]["pauses"] == 0


def test_sound_events_and_small_clusters():
    assert expressive.is_event("[music]") and expressive.is_event(">> (applause) ♪") and expressive.is_event("（笑）")
    assert not expressive.is_event("I said [music] no") and not expressive.is_event("")
    E = np.array([[1, 0], [0.98, 0.2], [1, 0.1], [0, 1], [0.9, 0.4]], float)
    # cluster 2 (one 1.5 s line) is too small: it joins cluster 0, its nearest
    assert expressive.merge_small([0, 0, 1, 1, 2], [3, 3, 3, 3, 1.5], E, 4.0) == [0, 0, 1, 1, 0]
    assert expressive.merge_small([0, 1], [5, 5], E[:2], 4.0) == [0, 1]


def test_event_lines_are_not_analysed(tmp_path):
    p, sents = _recording(tmp_path)
    sents[1]["source_text"] = "[music]"
    sents[1].pop("speaker")
    res = expressive.analyze_source(sents, p, tmp_path / "ref", expressive.ExprConfig(diarize=False))
    r = res["sentences"]
    assert r[1]["emo"] is None and r[1]["src_level"] is None and r[1]["expr"] == 0.0
    assert r[1]["speaker"] in ("S1", "S2")


def test_line_over_silence_is_skipped(tmp_path):
    p, sents = _recording(tmp_path)
    sents.append({"start": sents[-1]["end"] + 0.1, "end": sents[-1]["end"] + 0.45, "speaker": "A"})   # trailing silence
    r = expressive.analyze_source(sents, p, tmp_path / "ref", expressive.ExprConfig(diarize=False))["sentences"]
    assert r[-1]["emo"] is None and r[-1]["src_level"] is None and r[-1]["expr"] == 0.0
