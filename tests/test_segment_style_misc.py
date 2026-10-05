import json

import numpy as np

from subalign.align.timing import _interp_run
from subalign.llm import parse_json_loose
from subalign.models import Document
from subalign.proofread import _rewrite_lines, apply_glossary, remove_fillers
from subalign.segment.layout import get_layout
from subalign.segment.linebreak import segment_document
from subalign.separate import dsp_separate
from subalign.style import ass_color, load_style, make_style
from subalign.text.tokenize import make_line, text_width
from subalign.translate import translate_texts


def _doc(text, a=0.0, b=10.0):
    ln = make_line(text)
    _interp_run(ln.tokens, a, b)
    ln.update_bounds()
    return Document([ln])


def test_portrait_rows_fit_and_keep_timing():
    d = _doc("今天我们来聊一聊人工智能的发展历史，以及它对未来社会可能产生的深远影响。首先我们要了解什么是机器学习。")
    lay = get_layout("portrait")
    out = segment_document(d, lay)
    toks = [t for l in out.lines for t in l.tokens]
    assert [t.text for t in toks] == [t.text for t in d.lines[0].tokens]
    for l in out.lines:
        assert all(text_width(r) <= lay.max_units for r in l.rows())
        assert len(l.rows()) <= lay.max_lines
    assert any(l.rows()[-1].endswith("，") or l.rows()[-1].endswith("。") for l in out.lines)


def test_latin_words_not_split():
    d = _doc("This is a fairly long English sentence that should definitely be broken into rows")
    out = segment_document(d, get_layout("portrait"))
    words = " ".join(" ".join(l.rows()) for l in out.lines).split()
    assert words == d.lines[0].text.split()


def test_custom_resolution_layout():
    lay = get_layout("720x1280")
    assert lay.orientation == "portrait" and lay.play_res_x == 720


def test_style_presets_and_colors(tmp_path):
    assert ass_color("#FF0000") == "&H000000FF"
    assert ass_color("#00FF0080") == "&H7F00FF00"
    p = tmp_path / "s.json"
    p.write_text(json.dumps({"preset": "neon", "main": {"fontsize": 80}, "effect": {"fade_in_ms": 0}}))
    cfg = load_style(p)
    assert cfg.main.fontsize == 80 and cfg.effect.syllable == "glow"
    for name in ("default", "box", "karaoke", "karaoke-pop", "neon", "typewriter", "bounce", "shortvideo"):
        make_style(name)


def test_glossary_and_fillers_keep_times():
    d = _doc("嗯 我们用派森写代码", 0, 5)
    remove_fillers(d, "zh")
    assert d.lines[0].text.startswith("我们")
    apply_glossary(d, {"派森": "Python"})
    ln = d.lines[0]
    assert "Python" in ln.text and all(t.timed for t in ln.tokens)
    assert ln.start >= 0.5


def test_rewrite_line_level_cues_keeps_cue_times():
    from subalign.formats import read_text

    doc = read_text("1\n00:00:05,000 --> 00:00:07,000\n字幕对其很好\n")
    apply_glossary(doc, {"对其": "对齐"})
    ln = doc.lines[0]
    assert ln.text == "字幕对齐很好"
    assert (ln.start, ln.end) == (5.0, 7.0)
    assert all(5.0 <= t.start <= t.end <= 7.0 for t in ln.tokens)


def test_parse_json_loose():
    assert parse_json_loose('Sure!\n```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_loose('result: {"a": [1]} done') == {"a": [1]}


def test_translate_retries_missing_ids():
    class LLM:
        calls = 0

        def complete_json(self, system, user, schema=None):
            LLM.calls += 1
            data = json.loads(user[user.index("{"):])
            ids = [l["id"] for l in data["lines"]]
            if len(ids) > 1:
                ids = ids[:-1]          # drop one -> must be retried
            return {"translations": [{"id": i, "text": f"t{i}"} for i in ids]}

    out = translate_texts(["a", "b", "c"], LLM(), "en")
    assert out == ["t0", "t1", "t2"] and LLM.calls == 2


def test_dsp_separation_stereo():
    sr = 16000
    t = np.arange(sr * 3) / sr
    voice = 0.3 * np.sin(2 * np.pi * 300 * t) * (np.sin(2 * np.pi * 1.5 * t) > 0)
    side = 0.3 * np.sin(2 * np.pi * 700 * t)
    y = np.vstack([voice + side, voice - side]).astype(np.float32)
    v, inst = dsp_separate(y, sr, n_fft=1024)
    sdr = 10 * np.log10(np.sum(voice ** 2) / np.sum((voice - v[0]) ** 2))
    assert sdr > 6
    assert np.allclose(v + inst, y, atol=1e-4)
