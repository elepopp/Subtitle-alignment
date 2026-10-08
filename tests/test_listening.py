import json

import numpy as np
import pytest
import soundfile as sf

from subalign import listening
from subalign.tts.expressive import level_db

from test_expressive import _sil, _voice

SR = 24000


def test_sign_test():
    assert listening.sign_test_p(0, 0) == 1.0
    assert listening.sign_test_p(5, 5) == 1.0
    assert listening.sign_test_p(9, 1) == pytest.approx(0.0215, abs=1e-4)      # 2 * 11/1024
    assert listening.sign_test_p(1, 9) == listening.sign_test_p(9, 1)
    assert listening.sign_test_p(15, 5) == pytest.approx(0.0414, abs=1e-4)


def test_pairing_by_translation_sentence_or_time_and_text():
    a = [{"id": 1, "dt_id": 7, "audio": "a1"}, {"id": 2, "dt_id": 8, "audio": None}, {"id": 3, "dt_id": 9, "audio": "a3"}]
    b = [{"id": 5, "dt_id": 9, "audio": "b9"}, {"id": 6, "dt_id": 7, "audio": "b7"}, {"id": 7, "dt_id": 8, "audio": "b8"}]
    assert [(x["id"], y["id"]) for x, y in listening.pair_segments(a, b)] == [(1, 6), (3, 5)]
    a = [{"id": 1, "src_start": 1.04, "text": "你好。", "audio": "x"}]
    b = [{"id": 4, "src_start": 0.98, "text": "你好。", "audio": "y"}, {"id": 5, "src_start": 1.0, "text": "别的。", "audio": "z"}]
    assert [(x["id"], y["id"]) for x, y in listening.pair_segments(a, b)] == [(1, 4)]


def test_level_match():
    for db in (-35, -12):
        y = listening.level_match(_voice(1.5, SR, db=db))
        assert level_db(y, SR) == pytest.approx(listening.LEVEL_DB, abs=0.5)
    loud = listening.level_match(_voice(1.0, SR, db=-3), target=0.0)
    assert np.max(np.abs(loud)) <= 0.95 + 1e-6


def _dub(tmp_path, name, n, db, f0):
    d = tmp_path / name
    (d / "seg").mkdir(parents=True)
    segs = []
    for i in range(1, n + 1):
        sf.write(d / "seg" / f"{i}.wav", np.concatenate([_sil(0.2, SR), _voice(1.0, SR, f0=f0, db=db), _sil(0.2, SR)]), SR)
        segs.append({"id": i, "dt_id": 100 + i, "audio": f"seg/{i}.wav", "text": f"第{i}句。", "src_start": 2.0 * i,
                     "src_end": 2.0 * i + 1.2, "source_text": f"line {i}"})
    return {"title": name, "segments": segs}, d


def test_create_answer_reveal(tmp_path):
    pa, da = _dub(tmp_path, "A", 12, -30, 120)      # A quiet, B loud: both must come out at the same level
    pb, db = _dub(tmp_path, "B", 12, -10, 200)
    src = tmp_path / "orig.wav"
    sf.write(src, _voice(30.0, SR, db=-25), SR)
    tdir = tmp_path / "t"
    t = listening.create(tdir, pa, da, pb, db, "quiet", "loud", n=8, source=src, seed=3)
    assert len(t["items"]) == 8 and t["pairs"] == 12
    flips = [it["flip"] for it in t["items"]]
    assert 0 < sum(flips) < 8                          # X is sometimes A, sometimes B
    for it in t["items"]:
        k = f"{it['id']:03d}"
        x, sr = sf.read(tdir / f"{k}_x.wav")
        y, _ = sf.read(tdir / f"{k}_y.wav")
        o, _ = sf.read(tdir / f"{k}_o.wav")
        assert sr == listening.SR
        assert abs(level_db(x, sr) - level_db(y, sr)) < 1.0 and abs(level_db(o, sr) - level_db(x, sr)) < 1.0
        assert it["original"] and len(o) / sr == pytest.approx(1.2 + 0.35, abs=0.02)
    # before the reveal the page sees nothing that tells A from B
    v = listening.view(t)
    assert "tally" not in v and "a" not in v and all(set(i) == {"id", "text", "source_text", "original"} for i in v["items"])
    assert "flip" not in json.dumps(v)
    # answer: always pick B ("loud") on 6 items, "same" on 1, leave 1 open
    for it in t["items"][:6]:
        listening.answer(tdir, it["id"], "x" if it["flip"] else "y")
    listening.answer(tdir, t["items"][6]["id"], "same")
    with pytest.raises(ValueError):
        listening.answer(tdir, 1, "z")
    with pytest.raises(KeyError):
        listening.answer(tdir, 99, "x")
    v = listening.view(listening.reveal(tdir))
    assert v["tally"] == {"a": 0, "b": 6, "same": 1, "answered": 7, "n": 8, "p": pytest.approx(0.0312, abs=1e-4)}
    assert all(i["x_is"] in ("a", "b") for i in v["items"]) and v["a"]["label"] == "quiet"


def test_create_without_common_sentences(tmp_path):
    pa, da = _dub(tmp_path, "A", 3, -20, 120)
    pb = {"segments": [{"id": 1, "dt_id": 999, "audio": "x"}]}
    with pytest.raises(ValueError):
        listening.create(tmp_path / "t", pa, da, pb, tmp_path, "a", "b")
