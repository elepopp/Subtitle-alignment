import json

import numpy as np
import pytest

from subalign.llm import LLMClient
from subalign.models import Document, Line, Token
from subalign.translate import isochrony as iso
from subalign.translate.isochrony import IsoConfig
from subalign.tts import dubbing, textprep
from subalign.tts.dubbing import DubConfig


# ------------------------------------------------------------------ counting
def test_counts_chinese_and_english():
    assert iso.count("我今天买了三个苹果。", "zh")["syllables"] == 9
    assert iso.count("I bought three apples today.", "en")["syllables"] == 7
    # numbers are counted as they are read out, acronyms letter by letter
    assert iso.count("2026年", "zh")["syllables"] == 5                  # 二零二六年
    assert iso.count("In 2026", "en")["syllables"] == 6                 # in twen-ty twen-ty six
    assert iso.count("AI", "en")["syllables"] == 2
    assert iso.count("3.5%", "en")["syllables"] == iso.count("three point five percent", "en")["syllables"]


@pytest.mark.parametrize("word,n", [("the", 1), ("table", 2), ("wanted", 2), ("boxes", 2), ("make", 1),
                                    ("computer", 3), ("beautiful", 3 + 1), ("walked", 1)])
def test_en_syllables(word, n):
    assert iso.en_syllables(word) == n


def test_number_to_en():
    assert iso.number_to_en("2026") == "twenty twenty-six"
    assert iso.number_to_en("1,234") == "one thousand two hundred thirty-four"
    assert iso.number_to_en("$5") == "five dollars"
    assert iso.number_to_en("2000") == "two thousand"


def test_position_is_by_syllables():
    t = "我买了苹果"
    assert iso.position(t, t.index("苹果"), "zh") == pytest.approx(0.6)


# ------------------------------------------------------------------ sentences / anchors
def _line(text, start, step=0.25, **kw):
    toks = [Token(c, start + i * step, start + (i + 1) * step) for i, c in enumerate(text)]
    return Line(tokens=toks, start=start, end=start + len(text) * step, **kw)


def test_sentences_from_document_merges_fragments():
    doc = Document(lines=[_line("我今天在超市", 0.0), _line("买了一个苹果。", 1.6), _line("然后回家", 5.0)])
    ss = iso.sentences_from_document(doc)
    assert [s["text"] for s in ss] == ["我今天在超市买了一个苹果。", "然后回家"]
    assert ss[0]["start"] == 0.0 and ss[0]["end"] == pytest.approx(1.6 + 7 * 0.25)
    # character offsets of the second line are shifted into the merged text
    k = ss[0]["text"].index("苹果")
    assert dict((o, t) for o, t in ss[0]["times"])[k] == pytest.approx(1.6 + 4 * 0.25)


def test_anchor_position_uses_word_timing():
    # the speaker pauses before 苹果: by time it comes at 75 %, by syllables at 50 %
    toks = [Token("我", 0.0, 0.3), Token("买", 0.3, 0.6), Token("苹", 3.0, 3.3), Token("果", 3.3, 4.0)]
    s = {"id": 1, "text": "我买苹果", "start": 0.0, "end": 4.0, "times": [[0, 0.0], [1, 0.3], [2, 3.0], [3, 3.3]]}
    assert toks
    frac, sec = iso.anchor_position(s, 2, "zh")
    assert frac == pytest.approx(0.75) and sec == pytest.approx(3.0)
    s2 = dict(s, times=[])
    assert iso.anchor_position(s2, 2, "zh")[0] == pytest.approx(0.5)


def test_zh_anchors():
    a = [x["text"] for x in iso.zh_anchors("上周张三在北京买了一台新手机")]
    assert "张三" in a and "北京" in a


def test_heuristic_anchors_english():
    a = [x["text"] for x in iso.heuristic_anchors("Last week John bought a phone at the Apple Store for 999 dollars.")]
    assert a == ["John", "Apple Store", "999"]


def test_sentences_from_text():
    ss = iso.sentences_from_text("第一句。第二句！\n第三句", "zh")
    assert [s["text"] for s in ss] == ["第一句。", "第二句！", "第三句"]
    ss = iso.sentences_from_text("It costs 3.5 dollars. Really? Yes.", "en")
    assert [s["text"] for s in ss] == ["It costs 3.5 dollars.", "Really?", "Yes."]


# ------------------------------------------------------------------ measuring
def _sent(cfg, text="我今天买了一个苹果，回家给妈妈做了一个蛋糕。", anchors=("苹果", "蛋糕")):
    s = {"id": 1, "text": text, "start": 0.0, "end": 5.0, "times": []}
    return iso.analyze(s, cfg, [{"text": a, "offset": text.index(a)} for a in anchors])


def test_measure_flags_anchor_moved_to_other_half():
    cfg = IsoConfig(source="zh", target="en")
    s = _sent(cfg)
    assert s["anchors"][0]["pos"] < 0.5 and s["anchors"][1]["pos"] > 0.5
    terms = [{"src": "苹果", "tgt": "apple"}, {"src": "蛋糕", "tgt": "cake"}]
    good = iso.measure(s, "Today I bought an apple, went home, and baked my mom a cake.", terms, cfg)
    bad = iso.measure(s, "I baked my mom a cake at home today after I bought an apple.", terms, cfg)
    assert all(a["ok"] for a in good["anchors"])
    assert any(a["cross"] for a in bad["anchors"]) and not bad["ok"]
    assert good["cost"] < bad["cost"]


def test_budget_uses_speaking_rates():
    cfg = IsoConfig(source="zh", target="en")
    s = _sent(cfg)
    assert s["target"]["syllables"] == round(s["syllables"] * 6.19 / 5.18)
    assert s["target"]["lo"] < s["target"]["syllables"] < s["target"]["hi"]


def test_missing_anchor_and_kept_names():
    cfg = IsoConfig(source="zh", target="en")
    s = _sent(cfg, "我在iPhone上看到了张三。", anchors=("iPhone", "张三"))
    m = iso.measure(s, "On my iPhone I saw him.", [], cfg)
    by = {a["src"]: a for a in m["anchors"]}
    assert not by["iPhone"]["missing"]                  # kept as is, found without a term
    assert by["张三"]["missing"] and not m["ok"]


def test_set_custom_and_remeasure():
    cfg = IsoConfig(source="zh", target="en")
    s = _sent(cfg)
    s["candidates"] = [iso.measure(s, "I bought an apple and made a cake.", [{"src": "苹果", "tgt": "apple"}, {"src": "蛋糕", "tgt": "cake"}], cfg)]
    iso.rank(s)
    iso.set_custom(s, "Today I bought an apple, then went home and baked a cake for my mom.", cfg)
    assert s["locked"] and s["translation"].startswith("Today") and s["candidates"][s["choice"]]["custom"]
    # a wider tolerance re-scores locally; the user's choice stays
    iso.remeasure(s, IsoConfig(source="zh", target="en", tolerance=0.5))
    assert s["translation"].startswith("Today") and s["target"]["hi"] > s["target"]["syllables"] + 3


# ------------------------------------------------------------------ the LLM loop
class FakeLLM(LLMClient):
    """Round 0: one candidate too long and one with the noun moved to the end.
    Feedback round: a good one.  Judge: prefers the first finalist."""

    def __init__(self):
        self.calls = []

    def complete(self, system, user, schema=None):
        self.calls.append((system, user))
        payload = json.loads(user[user.index("{"):])
        if "anchor words" in system:
            return json.dumps({"lines": [{"id": l["id"], "anchors": ["苹果"]} for l in payload["lines"]]})
        if "review" in system:
            return json.dumps({"lines": [{"id": l["id"], "scores": [9] + [5] * (len(l["candidates"]) - 1)} for l in payload["lines"]]})
        out = []
        for l in payload["lines"]:
            if "previous_attempts" in l:
                cands = [{"text": "Apples, I went out and bought some today.", "terms": [{"src": "苹果", "tgt": "Apples"}]}]
            else:
                cands = [{"text": "Today I went to the big supermarket down the road and finally bought some fresh apples.",
                          "terms": [{"src": "苹果", "tgt": "apples"}]},
                         {"text": "I bought, today, some apples.", "terms": [{"src": "苹果", "tgt": "apples"}]}]
            out.append({"id": l["id"], "candidates": cands})
        return json.dumps({"lines": out})


def test_translate_loop_feedback_and_judge():
    cfg = IsoConfig(source="zh", target="en", candidates=2, rounds=2, tolerance=0.15)
    sents = iso.sentences_from_text("苹果我今天买了一些。", "zh")
    llm = FakeLLM()
    stages = []
    iso.translate(sents, llm, cfg, progress=lambda st, d, t: stages.append(st))
    s = sents[0]
    assert s["anchors"][0]["text"] == "苹果" and s["anchors"][0]["pos"] == 0
    # the 2x-too-long first candidate is discarded outright
    assert len(s["candidates"]) == 2 and s["discarded"] == 1
    assert s["translation"] == "Apples, I went out and bought some today."
    assert s["candidates"][s["choice"]]["ok"]
    assert stages[:2] == ["anchors", "translate"] and "refine1" in stages and stages[-1] == "judge"
    # the feedback round told the model what was wrong, with numbers
    refine = next(u for sy, u in llm.calls if "previous_attempts" in u)
    assert "too short" in refine and "heard at" in refine


def test_translate_survives_bad_json():
    class Broken(LLMClient):
        def complete(self, system, user, schema=None):
            return "sorry"
    sents = iso.sentences_from_text("我买了苹果。", "zh")
    iso.translate(sents, Broken(), IsoConfig(source="zh", target="en", rounds=0, judge=False))
    assert sents[0]["error"] and sents[0].get("translation") is None
    assert sents[0]["anchors"]                           # jieba fallback


def test_to_srt():
    ss = [{"id": 1, "text": "你好", "start": 1.0, "end": 2.5, "translation": "Hello"}]
    assert iso.to_srt(ss) == "1\n00:00:01,000 --> 00:00:02,500\nHello\n"
    assert "Hello\n你好" in iso.to_srt(ss, bilingual=True)


# ------------------------------------------------------------------ dubbing integration
def test_textprep_english():
    t = textprep.normalize("Hello，world! It costs $5 & more...Really? (note) It’s 3.5% off", lang="en")
    assert t == "Hello, world! It costs $5 and more... Really? It's 3.5% off"
    segs = textprep.split_segments("First sentence here. Second one is here too. Ok.", lang="en")
    assert [s.text for s in segs] == ["First sentence here.", "Second one is here too. Ok."]
    assert textprep.prepare("Read 3 books.", lang="en")["polyphones"] == []


def test_qa_units_english_words_and_numbers():
    assert dubbing._units("I have 3 apples.", "en") == ["i", "have", "three", "apples"]
    assert dubbing._units("我有三个", "zh") == ["我", "有", "三", "个"]


def test_create_project_from_translation(tmp_path):
    sents = [{"text": "Hello there", "start": 1.0, "end": 2.0, "source_text": "你好"},
             {"text": "I have 3 apples", "start": 2.5, "end": 4.0, "source_text": "我有三个苹果"}]
    cfg = DubConfig(lang="EN", timeline=True)
    p = dubbing.create_project(tmp_path / "p", "", tmp_path / "a.wav", None, cfg, sentences=sents)
    segs = p["segments"]
    assert [s["text"] for s in segs] == ["Hello there.", "I have 3 apples."]
    assert segs[0]["src_start"] == 1.0 and segs[0]["pause_after"] == 0.5 and segs[1]["source_text"] == "我有三个苹果"


def test_fit_factor():
    cfg = DubConfig(timeline=True)
    assert dubbing.fit_factor(2.0, 1.6, 1.5, cfg) == pytest.approx(cfg.max_stretch)    # capped
    assert dubbing.fit_factor(1.7, 1.6, 1.5, cfg) == pytest.approx(1.7 / 1.6)
    assert dubbing.fit_factor(1.0, 3.0, 1.5, cfg) == pytest.approx(cfg.min_stretch)
    assert dubbing.fit_factor(1.4, 3.0, 1.5, cfg) == 1.0


def test_timeline_placement_keeps_original_starts():
    sr = dubbing.SR
    clips = [np.full(int(0.8 * sr), 0.1, np.float32), np.full(int(1.0 * sr), 0.2, np.float32)]
    segs = [{"id": 1, "src_start": 1.0, "src_end": 1.9}, {"id": 2, "src_start": 3.0, "src_end": 4.0}]
    y, timing = dubbing._place_on_timeline(clips, segs, DubConfig(timeline=True), [])
    assert [t["start"] for t in timing] == [1.0, 3.0] and all(t["late"] == 0 for t in timing)
    assert y[int(1.2 * sr)] == pytest.approx(0.1) and y[int(2.5 * sr)] == 0 and y[int(3.5 * sr)] == pytest.approx(0.2)


def test_budget_capped_by_time_for_fast_speakers():
    cfg = IsoConfig(source="zh", target="en")
    text = "这款手机的价格是五千九百九十九元，比去年贵了不少。"           # 23 syllables
    fast = iso.analyze({"id": 1, "text": text, "start": 0.0, "end": 3.2, "times": []}, cfg, [])
    assert fast["target"]["by"] == "time" and fast["target"]["syllables"] == round(3.2 * 6.19 * 0.9)
    slow = iso.analyze({"id": 1, "text": text, "start": 0.0, "end": 6.0, "times": []}, cfg, [])
    assert slow["target"]["by"] == "ratio" and slow["target"]["syllables"] == round(23 * 6.19 / 5.18)
    untimed = iso.analyze({"id": 1, "text": text, "start": None, "end": None, "times": []}, cfg, [])
    assert untimed["target"]["by"] == "ratio"
