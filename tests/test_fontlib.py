import json
import zipfile
from pathlib import Path

import pytest

pytest.importorskip("fontTools")

from subalign import fontlib  # noqa: E402
from subalign.fontlib import FontLibrary  # noqa: E402


def make_font(path: Path, family: str, chars: str, zh_name: str = "") -> Path:
    """A tiny TrueType font with a square glyph for every character of ``chars``."""
    from fontTools.fontBuilder import FontBuilder
    from fontTools.pens.ttGlyphPen import TTGlyphPen

    glyphs = [".notdef"] + [f"u{ord(c):04X}" for c in dict.fromkeys(chars)]
    fb = FontBuilder(1000, isTTF=True)
    fb.setupGlyphOrder(glyphs)
    fb.setupCharacterMap({ord(c): f"u{ord(c):04X}" for c in chars})
    pen = TTGlyphPen(None)
    pen.moveTo((100, 0)), pen.lineTo((100, 700)), pen.lineTo((600, 700)), pen.lineTo((600, 0)), pen.closePath()
    sq = pen.glyph()
    fb.setupGlyf({g: sq for g in glyphs})
    fb.setupHorizontalMetrics({g: (700, 100) for g in glyphs})
    fb.setupHorizontalHeader(ascent=800, descent=-200)
    names = {"familyName": family, "styleName": "Regular"}
    if zh_name:
        names = {"familyName": {"en": family, "zh-CN": zh_name}, "styleName": "Regular"}
    fb.setupNameTable(names)
    fb.setupOS2()
    fb.setupPost()
    fb.save(str(path))
    return path


LATIN = fontlib.LATIN + " "
HANS = fontlib._gb2312_level1()


@pytest.fixture(scope="module")
def lib(tmp_path_factory):
    root = tmp_path_factory.mktemp("free")
    entries = []
    for fid, fam, chars in (("Latin", "Test Latin", LATIN), ("Hans", "Test Hans", HANS + LATIN)):
        (root / fid).mkdir()
        p = make_font(root / fid / f"{fid}.ttf", fam, chars)
        cov = fontlib.coverage(fontlib._cmap(fontlib._faces(p)[0]))
        entries.append(fontlib.FontEntry(id=fid, file=f"{fid}/{fid}.ttf", family=fam, display=fam,
                                         names=[fam], scripts=fontlib.scripts_of(cov), coverage=cov, license="SIL OFL 1.1"))
    return FontLibrary(entries, root)


def test_coverage_and_scripts(lib):
    lat, hans = lib.entries
    assert lat.scripts == ["latin"] and set(hans.scripts) >= {"zh-Hans", "latin"}
    assert lib.missing(lat, "Hi 你好") == {"你", "好"} and not lib.missing(hans, "Hi 你好")


def test_defaults_and_best_for(lib):
    assert lib.default_for("zh").family == "Test Hans"            # the only one covering simplified Chinese
    assert lib.best_for("hello", [lib.resolve("Test Latin")]).family == "Test Latin"
    assert lib.best_for("hello 世界").family == "Test Hans"


def test_enforce_replaces_unfree_and_uncovering_fonts(lib):
    ass = ("[V4+ Styles]\nStyle: Default,Arial,60,&H00FFFFFF\nStyle: Translation,Test Latin,40,&H00FFFFFF\n"
           "[Events]\nDialogue: 0,0:00:01.00,0:00:02.00,Default,,0,0,0,,{\\fad(100,100)}Hello world\n"
           "Dialogue: 0,0:00:01.00,0:00:02.00,Translation,,0,0,0,,你好世界\n")
    text, subs, used = lib.enforce(ass, "en", "zh")
    assert "Style: Default,Test Latin,60" in text                  # Arial is not in the library
    assert "Style: Translation,Test Hans,40" in text              # Test Latin has no Chinese glyphs
    assert {s["reason"][:2] for s in subs} == {"不在", "缺少"}
    assert {e.family for e in used} == {"Test Latin", "Test Hans"}


def test_install_package_picks_regular_and_licence(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    reg = make_font(src / "Demo-Regular.ttf", "Demo Sans", HANS[:3000] + HANS[3000:] + LATIN, zh_name="演示黑体")
    bold = make_font(src / "Demo-Bold.ttf", "Demo Sans Bold", LATIN)
    z = tmp_path / "演示黑体1.0_猫啃网.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.write(reg, "演示黑体1.0/Demo-Regular.ttf")
        zf.write(bold, "演示黑体1.0/Demo-Bold.ttf")
        zf.writestr("演示黑体1.0/OFL.txt", "This Font Software is licensed under the SIL Open Font License, Version 1.1.")
        zf.writestr("字体授权说明.txt", "这款字体无论是个人还是企业都是可以免费商用的。")
    e = fontlib.install_package(z, tmp_path / "free")
    assert Path(e.file).name == "Demo-Regular.ttf" and e.family == "Demo Sans" and e.display == "演示黑体"
    assert e.license == "SIL OFL 1.1" and e.verified and "OFL.txt" in e.license_files
    assert "zh-Hans" in e.scripts and (tmp_path / "free" / e.file).exists()
    assert not (tmp_path / "free" / e.id / "_cand").exists()


def test_install_package_without_licence_is_flagged(tmp_path):
    f = make_font(tmp_path / "x.ttf", "X Font", HANS + LATIN)
    z = tmp_path / "某字体_猫啃网.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.write(f, "某字体/x.ttf")
        zf.writestr("字体授权说明.txt", "这款字体无论是个人还是企业都是可以免费商用的。")
    e = fontlib.install_package(z, tmp_path / "free")
    assert not e.verified and "下载站" in e.license


def test_library_load_skips_missing_files(tmp_path):
    (tmp_path / "A").mkdir()
    make_font(tmp_path / "A" / "a.ttf", "A Font", LATIN)
    cat = [{"id": "A", "file": "A/a.ttf", "family": "A Font", "display": "A"},
           {"id": "B", "file": "B/b.ttf", "family": "B Font", "display": "B"}]
    (tmp_path / "catalog.json").write_text(json.dumps(cat), encoding="utf-8")
    FontLibrary.load.cache_clear()
    lib = FontLibrary.load(tmp_path)
    assert [e.id for e in lib.entries] == ["A"] and lib.resolve("a font").id == "A"
    FontLibrary.load.cache_clear()
