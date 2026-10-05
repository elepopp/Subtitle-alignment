import numpy as np

from subalign.roughcut import (Item, RoughCutConfig, _rms_env, detect, edited_document, plan_cuts, render,
                               time_map)
from subalign.text.tokenize import normalize_key


def _items(spec):
    """spec: list of lines, each a list of (text, start, end)."""
    out = []
    for li, line in enumerate(spec):
        for text, a, b in line:
            out.append(Item(text=text, start=a, end=b, line=li, key=normalize_key(text)))
    return out


def _chars(text, t0, dur=0.2, gaps=None):
    """CJK characters spoken back to back from t0 (gaps: {index: pause before it})."""
    out, t = [], t0
    for i, c in enumerate(text):
        t += (gaps or {}).get(i, 0.0)
        out.append((c, t, t + dur))
        t += dur
    return out


def _cut_text(items):
    return "".join(it.text for it in items if it.cut and not it.island)


def test_fillers_but_not_inside_words():
    # 嗯 between pauses, 额度 / 额外 are words, 好啊 ends with a particle
    line = _chars("大家好嗯今天讲额度和额外好啊", 0.0, gaps={3: 0.4, 4: 0.4})
    items = detect(_items([line]), RoughCutConfig())
    assert _cut_text(items) == "嗯"


def test_paused_filler_and_phrase_rules():
    # 那个 followed by a pause is a filler; 那个人 is a demonstrative
    a = _chars("那个其实呢", 0.0, gaps={2: 0.4})
    b = _chars("那个人很好", 3.0)
    c = _chars("啊我想想", 6.0)              # 啊 at the start of a line
    items = detect(_items([a, b, c]), RoughCutConfig())
    assert _cut_text(items) == "那个啊"


def test_stutter_repeat_and_reduplication():
    a = _chars("我我我觉得", 0.0, gaps={1: 0.15, 2: 0.15})
    b = _chars("我们今天我们今天要讲", 3.0, gaps={4: 0.3})
    c = _chars("谢谢大家看看这个", 7.0)       # reduplicated words are not stutters
    items = detect(_items([a, b, c]), RoughCutConfig())
    assert _cut_text(items) == "我我" + "我们今天"


def test_retake_keeps_last_take():
    a = _chars("我们先看第一个例子", 0.0)
    b = _chars("我们先看第一个例子吧", 2.6)
    items = detect(_items([a, b]), RoughCutConfig())
    assert _cut_text(items) == "我们先看第一个例子"
    assert not any(it.cut for it in items if it.line == 1)


def test_keep_words_and_levels():
    line = _chars("嗯那个好", 0.0, gaps={1: 0.3, 3: 0.3})
    items = detect(_items([line]), RoughCutConfig(keep_words=["嗯"]))
    assert _cut_text(items) == "那个"
    items = detect(_items([line]), RoughCutConfig(level="conservative"))
    assert _cut_text(items) == "嗯"


def _speech(spec, sr=16000, total=None):
    """Synthetic 'speech': a harmonic tone burst per item, silence elsewhere."""
    n = int((total or max(b for _, _, b in spec) + 0.5) * sr)
    y = np.zeros(n, dtype=np.float32)
    for k, (_, a, b) in enumerate(spec):
        t = np.arange(int((b - a) * sr)) / sr
        f = 160 + 20 * (k % 5)
        y[int(a * sr):int(a * sr) + len(t)] += 0.3 * np.sin(2 * np.pi * f * t) * np.hanning(len(t))
    return y[None]


def test_plan_render_smooth_joins_and_natural_pause():
    sr = 16000
    # 你好[0.3]嗯[0.4]世界 + a 2 s dead pause + 再见
    spec = [("你", 0.5, 0.75), ("好", 0.75, 1.0), ("嗯", 1.3, 1.6), ("世", 2.0, 2.25), ("界", 2.25, 2.5),
            ("再", 4.5, 4.75), ("见", 4.75, 5.0)]
    items = _items([spec])
    items[2].cut = "filler"
    y = _speech(spec, sr, total=5.5)
    env, hop = _rms_env(y, sr)
    cfg = RoughCutConfig()
    keeps, fills, pauses = plan_cuts(items, 5.5, cfg, env, hop)
    assert len(pauses) == 1 and pauses[0]["to"] <= 0.45
    r = render(y, sr, keeps, fills, np.zeros((1, 3200), np.float32), env, hop, cfg)
    out = r.audio[0]
    # the filler is gone, the pause that replaces it is natural (0.12 .. 0.45 s) and the
    # 2 s silence shrank to ~0.4 s
    f = time_map(r)
    assert f(1.45) is None
    gap1 = f(2.0) - f(1.0)
    gap2 = f(4.5) - f(2.5)
    assert 0.12 <= gap1 <= 0.45 and 0.3 <= gap2 <= 0.45
    # no clicks: the second difference at every join stays below the signal's own peaks
    d2 = np.abs(np.diff(out, 2))
    for b in r.seg_bounds[1:-1]:
        i = int(b * sr)
        assert d2[max(0, i - 48):i + 48].max() <= np.percentile(d2, 99.9) + 1e-6
    # subtitles of the result follow the new timeline
    doc = edited_document(items, r)
    assert [t.text for t in doc.lines[0].tokens] == ["你", "好", "世", "界", "再", "见"]
    assert abs(doc.lines[0].tokens[2].start - f(2.0)) < 1e-6


def test_room_tone_fills_a_cut_between_touching_words():
    sr = 16000
    spec = [("我", 0.5, 0.8), ("嗯", 0.8, 1.1), ("觉", 1.1, 1.4), ("得", 1.4, 1.7)]   # 我嗯觉得 without pauses
    items = _items([spec])
    items[1].cut = "filler"
    y = _speech(spec, sr, total=2.2)
    env, hop = _rms_env(y, sr)
    keeps, fills, _ = plan_cuts(items, 2.2, RoughCutConfig(min_gap=0.12), env, hop)
    assert any(abs(x - 0.12) < 1e-6 for x in fills)        # a short pause is inserted, words are not butted
