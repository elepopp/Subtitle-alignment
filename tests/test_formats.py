import pytest

from subalign.align.timing import _interp_run
from subalign.formats import FORMATS, detect_text_format, read_text, write
from subalign.formats.lyrics import krc_decrypt, krc_encrypt, read_krc
from subalign.models import Document
from subalign.text.tokenize import make_line


@pytest.fixture
def doc():
    l1 = make_line("你好，世界")
    _interp_run(l1.tokens, 1.0, 2.5)
    l1.update_bounds()
    l1.translation = "Hello world"
    l2 = make_line("Hello my friend")
    _interp_run(l2.tokens, 3.0, 4.2)
    l2.update_bounds()
    return Document([l1, l2], kind="song", metadata={"title": "T", "artist": "A"})


@pytest.mark.parametrize("fmt", [f for f in FORMATS if FORMATS[f].reader and f not in ("txt",)])
def test_roundtrip(doc, fmt):
    out = write(doc, fmt)
    if FORMATS[fmt].binary:
        back = read_krc(out)
    else:
        back = read_text(out, "ttml" if fmt == "ttml-line" else ("krc" if fmt == "krc" else None))
    assert [l.text for l in back.lines] == [l.text for l in doc.lines]
    for la, lb in zip(doc.lines, back.lines):
        assert abs(la.start - lb.start) < 0.011
        if FORMATS[fmt].level == "word":
            for a, b in zip(la.tokens, lb.tokens):
                assert abs(a.start - b.start) < 0.011
    if fmt not in ("qrc", "yrc"):  # formats without a translation slot
        assert back.lines[0].translation == "Hello world"


@pytest.mark.parametrize("fmt", ["srt", "vtt", "ass", "lrc", "qrc", "krc", "yrc", "ttml", "sbv", "json"])
def test_detect(doc, fmt):
    assert detect_text_format(write(doc, fmt)) == fmt


def test_krc_crypto():
    txt = "[0,100]<0,100,0>啊"
    assert krc_decrypt(krc_encrypt(txt)) == txt


def test_lrc_compressed_and_offset():
    d = read_text("[offset:500]\n[00:01.00][00:05.00]副歌\n[00:03.00]主歌\n")
    assert [l.text for l in d.lines] == ["副歌", "主歌", "副歌"]
    assert abs(d.lines[0].start - 0.5) < 1e-6


def test_ass_karaoke_effects(doc):
    from subalign.style import make_style

    out = write(doc, "ass", style=make_style("karaoke-pop"))
    assert "\\k" in out and "\\t(" in out and "\\fscx100" in out
    assert "Style: Translation" in out
    back = read_text(out)
    assert [round(t.start, 2) for t in back.lines[0].tokens] == [round(t.start, 2) for t in doc.lines[0].tokens]


def test_srt_karaoke(doc):
    out = write(doc, "srt-karaoke")
    assert out.count("-->") == sum(len(l.tokens) for l in doc.lines)
    assert '<font color="#FFD700">你</font>' in out
