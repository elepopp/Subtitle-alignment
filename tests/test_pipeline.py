import json

import numpy as np
import pytest
import soundfile as sf

from subalign.align.aligner import AlignConfig, align_audio
from subalign.asr import Segment, Transcript, Word
from subalign.asr import backends
from subalign.formats import read
from subalign.pipeline import ExportConfig, LLMConfig, PipelineConfig, run
from synth import synth_song

LYRICS = ["我们一起走过的路", "风吹过了山岗", "你说那是最美的时光", "不要忘记"]


@pytest.fixture
def song(tmp_path):
    y, truth = synth_song([len(l) for l in LYRICS], seed=3, accompaniment=0.0)
    # stereo: centred voice + side-panned accompaniment
    t = np.arange(len(y)) / 16000
    acc = 0.05 * np.sin(2 * np.pi * 330 * t)
    st = np.vstack([y + acc, y - acc]).T
    p = tmp_path / "song.wav"
    sf.write(p, st, 16000)
    return p, truth


class FakeASR:
    """Returns the truth with an ASR-like error (homophone) and jitter."""
    truth = None
    text = None

    def __init__(self, **kw):
        pass

    def transcribe(self, path, language=None, prompt=None, **kw):
        segs = []
        rng = np.random.default_rng(0)
        for line, tl in zip(self.text, self.truth):
            words = [Word(ch, s + rng.normal(0, 0.1), e, 0.9) for ch, (s, e) in zip(line, tl)]
            segs.append(Segment(tl[0][0], tl[-1][1], line, words))
        return Transcript(segs, "zh")


def test_song_with_lyrics_acoustic(song, tmp_path):
    path, truth = song
    cfg = AlignConfig(mode="song", asr_backend="none", ctc="off", separation="dsp")
    res = align_audio(path, "\n".join(LYRICS), cfg, workdir=tmp_path / "w")
    assert res.mode == "song" and set(res.stems) == {"vocals", "instrumental"}
    err = [abs(t.start - s) for ln, tl in zip(res.document.lines, truth) for t, (s, _) in zip(ln.tokens, tl)]
    assert np.mean(np.array(err) < 0.06) > 0.9


def test_song_with_asr_anchors_and_report(song, tmp_path, monkeypatch):
    path, truth = song
    FakeASR.truth = truth
    FakeASR.text = [LYRICS[0], "风吹过了山刚", LYRICS[2], "不要忘"]
    monkeypatch.setitem(backends.BACKENDS, "fake", FakeASR)
    cfg = AlignConfig(mode="song", asr_backend="fake", ctc="off", separation="none")
    res = align_audio(path, "\n".join(LYRICS), cfg, workdir=tmp_path / "w")
    assert res.anchor_source == "asr"
    kinds = {(i["script"], i["heard"]) for i in res.report["issues"]}
    assert ("岗", "刚") in kinds and ("记", "") in kinds
    err = [abs(t.start - s) for ln, tl in zip(res.document.lines, truth) for t, (s, _) in zip(ln.tokens, tl)]
    assert np.mean(np.array(err) < 0.06) > 0.9


def test_recognition_mode_and_full_run(song, tmp_path, monkeypatch):
    path, truth = song
    FakeASR.truth = truth
    FakeASR.text = LYRICS
    monkeypatch.setitem(backends.BACKENDS, "fake", FakeASR)

    class FakeLLM:
        def complete_json(self, system, user, schema=None):
            data = json.loads(user.split("\n", 1)[1]) if "\n" in user else json.loads(user)
            return {"translations": [{"id": l["id"], "text": f"EN{l['id']}"} for l in data["lines"]]}

    monkeypatch.setattr(LLMConfig, "client", lambda self: FakeLLM())
    cfg = PipelineConfig(align=AlignConfig(mode="song", asr_backend="fake", ctc="off", separation="none"),
                         export=ExportConfig(formats=["lrc", "ass", "json"], layouts=["landscape", "portrait"]),
                         translate_to="en")
    res = run(path, None, tmp_path / "out", cfg)
    assert res["lines"] == 4
    doc = read(tmp_path / "out" / "song.portrait.json")
    assert [l.text for l in doc.lines] == LYRICS
    assert doc.lines[1].translation == "EN1"
    assert "EN0" in (tmp_path / "out" / "song.landscape.lrc").read_text(encoding="utf-8")


class FakeEmitter:
    """CTC emitter whose emissions are synthesised from the ground truth."""

    def __init__(self, truth, text):
        self.truth, self.text = truth, text
        chars = sorted({c for line in text for c in line})
        self.vocab = {"<pad>": 0, **{c: i + 1 for i, c in enumerate(chars)}}
        self.blank, self.frame_s = 0, 0.02

    def emissions(self, y, sr=16000):
        T = int(len(y) / sr / self.frame_s) + 1
        lp = np.full((T, len(self.vocab)), np.log(0.01))
        lp[:, 0] = np.log(0.9)
        for line, tl in zip(self.text, self.truth):
            for ch, (s, e) in zip(line, tl):
                f = int(round((s + 0.03) / self.frame_s))   # CTC peaks are slightly late
                lp[f:f + 2, :] = np.log(0.01)
                lp[f:f + 2, self.vocab[ch]] = np.log(0.95)
        return lp

    def unit_ids(self, keys):
        return [[self.vocab[c] for c in k if c in self.vocab] for k in keys]


def test_ctc_anchor_path(song, tmp_path, monkeypatch):
    from subalign.align import aligner

    path, truth = song
    monkeypatch.setattr(aligner, "_try_ctc", lambda cfg: FakeEmitter(truth, LYRICS))
    cfg = AlignConfig(mode="song", asr_backend="none", ctc="on", separation="none")
    res = align_audio(path, "\n".join(LYRICS), cfg, workdir=tmp_path / "w")
    assert res.anchor_source == "ctc"
    err = [abs(t.start - s) for ln, tl in zip(res.document.lines, truth) for t, (s, _) in zip(ln.tokens, tl)]
    assert np.mean(np.array(err) < 0.05) > 0.9


def test_cli_convert_restyle(tmp_path):
    from subalign.cli import main

    src = tmp_path / "in.lrc"
    src.write_text("[00:01.00]<00:01.00>你<00:01.50>好<00:02.00>\n[00:03.00]<00:03.00>世<00:03.40>界<00:04.00>\n",
                   encoding="utf-8")
    assert main(["convert", str(src), "-f", "ass,qrc,srt", "--style", "neon", "--layout", "portrait",
                 "-o", str(tmp_path / "o")]) == 0
    ass = (tmp_path / "o" / "in.ass").read_text(encoding="utf-8")
    assert "PlayResX: 1080" in ass and "\\blur" in ass
    assert "(1000,500)" in (tmp_path / "o" / "in.qrc").read_text(encoding="utf-8")
