from subalign.align.sequence import align_keys, align_tokens, transfer_times
from subalign.text.tokenize import make_line, strip_punct, syllable_weight, text_width, tokenize


def test_tokenize_cjk_and_latin():
    toks = tokenize("你好，世界！Hello, world.")
    assert [t.text for t in toks] == ["你", "好，", "世", "界！", "Hello,", "world."]
    assert toks[4].space_after and not toks[5].space_after


def test_opening_punct_attaches_forward():
    toks = tokenize("「我说」 OK吗？")
    assert toks[0].text == "「我"
    assert [t.text for t in toks][-2:] == ["OK", "吗？"]


def test_width_and_weight():
    assert text_width("中文ab") == 6
    assert syllable_weight(tokenize("beautiful")[0]) == 3
    assert syllable_weight(tokenize("中")[0]) == 1


def test_strip_punct_modes():
    assert strip_punct("你好，世界！我爱你。", "space") == "你好 世界！我爱你"
    assert strip_punct("你好，世界。", "strip") == "你好世界"


def test_align_keys_handles_substitution_and_gaps():
    ref = list("今天天气很好我们出去玩")
    hyp = list("今天天汽很好出去玩吧")
    ops = align_keys(ref, hyp)
    kinds = [o.op for o in ops]
    assert kinds.count("match") == 8
    assert "del" in kinds and "ins" in kinds


def test_transfer_times_interpolates_unmatched():
    ref = make_line("我们一起去公园").tokens
    hyp = make_line("我们一起公园").tokens
    for i, t in enumerate(hyp):
        t.start, t.end = i * 0.5, i * 0.5 + 0.4
    transfer_times(ref, hyp)
    assert all(t.timed for t in ref)
    starts = [t.start for t in ref]
    assert starts == sorted(starts)
    assert ref[4].start >= ref[3].end - 1e-9  # interpolated "去"


def test_repeated_chorus_aligns_to_the_right_repeat():
    # songs repeat whole sections; when ASR garbles the first pass and gets the repeat right,
    # the longest exact block is "section (script, 1st) == section (heard, 2nd)" - a global
    # alignment must still pair 1st with 1st and 2nd with 2nd
    section = list("暖暖的午后闪过一片片粉红的衣裳淡淡相思都写在脸上沉沉离别背在肩上")
    garbled = [("晨" if i % 4 == 0 or i == len(section) - 1 else c) for i, c in enumerate(section)]
    ref = section + section
    hyp = garbled + section
    ops = align_keys(ref, hyp)
    assert not [o for o in ops if o.op in ("del", "ins")]
    assert all(o.ref == o.hyp for o in ops)


def test_guess_language_and_credit_lines():
    from subalign.text.tokenize import guess_language, is_credit_line

    assert guess_language("春天的黄昏 请你陪我到") == "zh"
    assert guess_language("きみの名前を呼んだ") == "ja"
    assert guess_language("사랑해요 정말로") == "ko"
    assert guess_language("I walk this empty street on the boulevard") == "en"
    md = {"artist": "江珊", "title": "梦里水乡"}
    assert is_credit_line("作词：洛兵") and is_credit_line("Composed by: Someone")
    assert is_credit_line("江珊 -梦里水乡", md, 0)
    assert not is_credit_line("春天的黄昏", md, 3) and not is_credit_line("一 - 二", md, 0)
