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
