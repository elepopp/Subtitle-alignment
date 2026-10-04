import numpy as np
import pytest

from subalign.align.ctc import romanize, viterbi_align
from subalign.align.syllable_dp import SONG_PARAMS, SPEECH_PARAMS, Unit, align_units
from subalign.audio.features import analyze, detect_content_type
from synth import synth_song

LINES = [8, 6, 10, 7, 9, 5]


def _units(truth, anchor_noise=None, seed=0):
    rng = np.random.default_rng(seed)
    units, tr = [], []
    for ln in truth:
        for k, (s, e) in enumerate(ln):
            u = Unit(weight=1.0, line_start=k == 0, line_end=k == len(ln) - 1)
            if anchor_noise:
                u.anchor, u.anchor_conf, u.anchor_sigma = s + rng.normal(0, anchor_noise), 0.8, anchor_noise
            units.append(u)
            tr.append(s)
    return units, np.array(tr)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_song_acoustic_only(seed):
    y, truth = synth_song(LINES, seed=seed)
    units, tr = _units(truth)
    out = align_units(units, analyze(y), SONG_PARAMS)
    err = np.abs(out[:, 0] - tr)
    assert np.mean(err < 0.05) > 0.9
    assert np.all(np.diff(out[:, 0]) > 0)
    assert np.all(out[:, 1] > out[:, 0])


def test_song_legato_with_noisy_anchors():
    y, truth = synth_song(LINES, seed=4, legato_prob=0.6)
    units, tr = _units(truth, anchor_noise=0.25, seed=4)
    out = align_units(units, analyze(y), SONG_PARAMS)
    assert np.mean(np.abs(out[:, 0] - tr) < 0.08) > 0.85


def test_speech():
    y, truth = synth_song(LINES, seed=5, speech=True, gap=0.6)
    units, tr = _units(truth, anchor_noise=0.08, seed=5)
    out = align_units(units, analyze(y), SPEECH_PARAMS)
    assert np.mean(np.abs(out[:, 0] - tr) < 0.08) > 0.85


def test_line_ends_stop_at_silence():
    y, truth = synth_song([5, 5], seed=6, gap=2.0)
    units, _ = _units(truth)
    out = align_units(units, analyze(y), SONG_PARAMS)
    true_end = truth[0][-1][1]
    assert abs(out[4, 1] - true_end) < 0.15


def test_content_type():
    y, _ = synth_song(LINES, seed=0)
    assert detect_content_type(analyze(y)) == "song"


def test_ctc_viterbi_with_star():
    seq = [0] * 5 + [1] * 4 + [0] * 3 + [2] * 3 + [0] * 10 + [3] * 5 + [0] * 3
    lp = np.full((len(seq), 4), np.log(0.02))
    lp[np.arange(len(seq)), seq] = np.log(0.94)
    spans = viterbi_align(lp, [[1], [2], [3]], blank=0, frame_s=0.02, line_breaks=[0, 2])
    assert [round(s.start, 2) for s in spans] == [0.1, 0.24, 0.5]


def test_romanize():
    assert romanize("한") == "han" and romanize("か") == "ka" and romanize("é") == "e"
