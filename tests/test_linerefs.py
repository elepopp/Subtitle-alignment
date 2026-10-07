import numpy as np
import pytest
import soundfile as sf

from subalign.tts import linerefs
from subalign.tts.linerefs import LineRefConfig

from test_expressive import _sil, _voice

SR = 24000
A16 = 16000


def test_spoken_and_keys():
    assert linerefs.spoken("Look, we know [music] some of you >> (laughs)") == "Look, we know some of you"
    assert linerefs.keys("It's 5.5, right?") == ["its", "5", "5", "right"]
    assert linerefs.keys("[music]") == []


def test_cut_bounds_stay_in_the_gaps():
    spans = [(1.0, 2.0), (2.4, 3.0), (3.0, 4.0)]
    assert linerefs.cut_bounds(spans, 0, 0.12, 10) == (pytest.approx(0.88), pytest.approx(2.12))
    lo, hi = linerefs.cut_bounds(spans, 1, 0.5, 10)
    assert lo == pytest.approx(2.2) and hi == pytest.approx(3.0)      # middle of the gap / touching words
    lo, hi = linerefs.cut_bounds([(1.0, 2.5), (2.2, 3.0)], 1, 0.12, 10)
    assert lo == pytest.approx(2.2)                                    # overlap: never into the other line


def test_check_clip_finds_words_of_other_lines():
    text = "we have been listening"
    heard = [("so", 0.0, 0.2), (" we", 0.35, 0.5), (" have", 0.5, 0.7), (" been", 0.7, 0.9), (" listening", 0.9, 1.4),
             (" this", 1.6, 1.8)]
    r = linerefs.check_clip(heard, text)
    assert r["extra_head"] == 1 and r["extra_tail"] == 1 and r["missing"] == 0
    assert r["trim"] == [pytest.approx(0.275), pytest.approx(1.5)]
    clean = linerefs.check_clip(heard[1:5], text)
    assert clean["trim"] == [None, None] and clean["missing"] == 0
    half = linerefs.check_clip(heard[3:5], text)
    assert half["missing"] == pytest.approx(6 / 19, abs=0.001)     # "wehave" of "wehavebeenlistening" not heard
    assert linerefs.check_clip([], text)["missing"] == 1.0


class FakeASR:
    """Hears the words of the recording that fall inside each clip; ``starts``: where
    the clips it will be given begin, in order."""

    def __init__(self, words, starts):
        self.words, self.starts = words, iter(starts)

    def transcribe(self, path, language=None, vad=False):
        from subalign.asr.base import Segment, Transcript, Word

        clip, sr = sf.read(path)
        t0 = next(self.starts)
        t1 = t0 + len(clip) / sr
        ws = [Word(w, s - t0, e - t0) for w, s, e in self.words if s >= t0 - 0.05 and e <= t1 + 0.05]
        return Transcript([Segment(0, t1 - t0, " ".join(w.text for w in ws), ws)])


def test_build_trims_a_neighbour_that_slipped_in(tmp_path):
    rng = np.random.default_rng(0)
    # two lines, the subtitle of line 2 starts 0.5 s early (inside line 1's last word)
    parts = [_sil(0.5, A16), _voice(2.0, A16, f0=140), _sil(0.3, A16), _voice(2.0, A16, f0=150), _sil(0.5, A16)]
    y16 = np.concatenate(parts) + rng.normal(0, 1e-4, sum(len(p) for p in parts)).astype(np.float32)
    words = [("one", 0.5, 1.0), ("two", 1.0, 1.5), ("three", 1.5, 2.0), ("four", 2.0, 2.5),
             ("five", 2.8, 3.3), ("six", 3.3, 3.8), ("seven", 3.8, 4.3), ("eight", 4.3, 4.8)]
    # clips as cut from the subtitle spans: line 1 0.38 .. 2.5, line 2 2.0 .. 4.92 (overlap)
    asr = FakeASR(words, [0.38, 2.0])
    from scipy.signal import resample_poly

    y = resample_poly(y16, 3, 2).astype(np.float32)
    spans = [(0.5, 2.5), (2.0, 4.8)]                      # line 2's subtitle starts too early
    res = linerefs.build(spans, [False, False], ["one two three four", "five six seven eight"], ["S1", "S1"],
                         [True, True], y, SR, y16, "en", tmp_path, ["0001", "0002"],
                         LineRefConfig(min_ref_s=0), asr)
    assert res[0]["line_check"]["trimmed"] == [0, 0] and res[0]["line_check"]["ok"]
    c2 = res[1]["line_check"]
    assert c2["ok"] and c2["trimmed"] == [1, 0] and c2["heard"].startswith("four")
    clip, sr = sf.read(res[1]["line"])
    # the clip now starts in the gap before "five" (between 2.5 and 2.8 s), not inside "four"
    assert len(clip) / sr == pytest.approx(4.92 - 2.65, abs=0.02)


def test_refine_spans_uses_alignment_with_context(monkeypatch):
    from subalign.align import ctc
    from subalign.align.ctc import CTCUnitSpan

    calls = []

    class Emitter:
        def __init__(self, *a, **k):
            self.model = None

    def fake_align(emitter, y, lines, sr=16000):
        calls.append([list(l) for l in lines])
        # every line's words are 0.3 s into its window position (window starts at w0)
        out = []
        for k, l in enumerate(lines):
            out.append([CTCUnitSpan(0.5 + k * 2.0 + j * 0.3, 0.5 + k * 2.0 + j * 0.3 + 0.25, -1.0) for j in range(len(l))])
        return out

    monkeypatch.setattr(ctc, "HFCTCEmitter", Emitter)
    monkeypatch.setattr(ctc, "ctc_align_tokens", fake_align)
    spans = [(2.0, 3.0), (4.0, 5.0), (6.0, 7.0)]
    y16 = np.zeros(10 * A16, np.float32)
    out = linerefs.refine_spans(spans, ["a b", "", "c d e"], y16, "en", LineRefConfig(group=2, margin=1.5))
    # first window: lines 0, 1 (+1 context = 2), from 0.5 s
    assert calls[0] == [["a", "b"], [], ["c", "d", "e"]]
    assert out[0] == (pytest.approx(1.0), pytest.approx(1.55))
    assert out[1] is None                                 # nothing to align
    assert out[2] is not None                             # second window: lines 1 (context), 2
    assert len(calls) == 2


def test_refine_spans_rejects_implausible_moves(monkeypatch):
    from subalign.align import ctc
    from subalign.align.ctc import CTCUnitSpan

    class Emitter:
        def __init__(self, *a, **k):
            self.model = None

    monkeypatch.setattr(ctc, "HFCTCEmitter", Emitter)
    monkeypatch.setattr(ctc, "ctc_align_tokens",
                        lambda e, y, lines, sr=16000: [[CTCUnitSpan(30.0, 31.0, -1.0) for _ in l] for l in lines])
    out = linerefs.refine_spans([(2.0, 3.0)], ["a b"], np.zeros(40 * A16, np.float32), "en", LineRefConfig())
    assert out == [None]


def test_check_clip_is_not_fooled_by_word_splits():
    text = "I can ask Codeex to check the other tasks or send them a follow-up."
    heard = [(w, k * 0.3, k * 0.3 + 0.25) for k, w in enumerate(
        "I can ask Codex to check the other tasks or send them a follow -up. And".split())]
    r = linerefs.check_clip(heard, text)
    assert r["extra_head"] == 0 and r["extra_tail"] == 1          # only "And" is another line's
    assert r["missing"] <= 0.03
