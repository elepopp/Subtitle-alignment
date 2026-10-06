"""Free-for-commercial-use fonts for translated subtitles (可免费商用字体库).

``fonts/*.zip`` (font packages put there by the user) -> one common weight of
every family is extracted to ``fonts/free/<id>/`` together with its licence
files; English fonts (SIL OFL) are downloaded from Google Fonts.  Every font is
measured - which scripts it really covers (simplified / traditional Chinese,
Japanese, Korean, Latin) - and listed in ``fonts/free/catalog.json``.

Burned-in translation subtitles only ever use fonts of this catalogue:
:meth:`FontLibrary.enforce` replaces any other font (or a font lacking glyphs
for the text) by a catalogued one that covers it, so libass never falls back to
an arbitrary system font.
"""
from __future__ import annotations

import json
import logging
import re
import shutil
import urllib.request
import zipfile
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

log = logging.getLogger("subalign")
ROOT = Path(__file__).resolve().parents[1]
FONTS_DIR = ROOT / "fonts"
FREE_DIR = FONTS_DIR / "free"
FONT_EXT = (".ttf", ".otf", ".ttc", ".otc")

# English fonts from Google Fonts (all SIL Open Font License 1.1); (family, google/fonts directory)
ENGLISH = [("Roboto", "roboto"), ("Open Sans", "opensans"), ("Lato", "lato"), ("Montserrat", "montserrat"),
           ("Poppins", "poppins"), ("Inter", "inter"), ("Source Sans 3", "sourcesans3"), ("Noto Sans", "notosans"),
           ("Nunito", "nunito"), ("Oswald", "oswald"), ("Merriweather", "merriweather"), ("Bebas Neue", "bebasneue")]
# default font per language (catalogue family names, first available wins)
DEFAULTS = {"zh": ["Alibaba PuHuiTi 3.0", "HarmonyOS Sans SC"], "en": ["Roboto", "Inter", "Open Sans"],
            "ja": ["KURIYAMAKOUCHIFONT_N"], "ko": ["Judou Sans Hans"]}

# licence restrictions worth knowing (by package name)
NOTES = {"鸿雷小纸条青春体": "可商用；禁止修改字体、转售或向第三方传播字体文件",
         "卓特清雅体": "可商用；禁止修改 / 转换格式、禁止注册商标",
         "鸿蒙黑体": "HarmonyOS Sans 字体许可：可商用，禁止修改后再分发",
         "阿里巴巴普惠体": "阿里巴巴普惠体许可：免费商用"}


# ------------------------------------------------------------------ coverage
@lru_cache(maxsize=1)
def _gb2312_level1() -> str:
    """The 3755 most common simplified characters (GB2312 level 1)."""
    out = []
    for hi in range(0xB0, 0xD8):
        for lo in range(0xA1, 0xFF):
            try:
                out.append(bytes([hi, lo]).decode("gb2312"))
            except UnicodeDecodeError:
                pass
    return "".join(out)


TRADITIONAL = "這個們說時會與來對國過還麼開學發會實際頭動問題點應該從經東書買賣電話員錢歡樂聽"
KANA = "あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをんアイウエオカキクケコンー"
HANGUL = "가나다라마바사아자차카타파하한국어입니다안녕하세요"
LATIN = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,!?'\"-()"


def _faces(path: Path):
    import io

    from fontTools.ttLib import TTCollection, TTFont

    data = io.BytesIO(path.read_bytes())                  # in memory: Windows cannot move an open file
    if path.suffix.lower() in (".ttc", ".otc"):
        return list(TTCollection(data, lazy=True).fonts)
    return [TTFont(data, lazy=True)]


def _names(font) -> Tuple[List[str], str, str]:
    """(all family names, English family name, Chinese display name)."""
    names, en, zh = [], "", ""
    for rec in font["name"].names:
        if rec.nameID not in (1, 16):
            continue
        try:
            s = rec.toUnicode().strip()
        except Exception:
            continue
        if not s:
            continue
        names.append(s)
        lang = rec.langID
        if rec.platformID == 3 and lang in (0x0804, 0x0404, 0x0C04, 0x1004, 0x0411):     # zh-CN / zh-TW / zh-HK / zh-SG / ja
            zh = zh if (zh and rec.nameID == 1) else s
        elif rec.platformID == 3 and lang == 0x0409 or rec.platformID == 1 and lang == 0:
            en = s if (not en or rec.nameID == 16) else en
    names = list(dict.fromkeys(names))
    return names, en or (names[0] if names else ""), zh


def _cmap(font) -> Set[int]:
    return set(font.getBestCmap() or {})


def coverage(cmap: Set[int]) -> Dict[str, float]:
    def share(s):
        return round(sum(1 for c in s if ord(c) in cmap) / len(s), 3)
    return {"zh-Hans": share(_gb2312_level1()), "zh-Hant": share(TRADITIONAL), "ja": share(KANA), "ko": share(HANGUL),
            "latin": share(LATIN)}


def scripts_of(cov: Dict[str, float]) -> List[str]:
    out = []
    if cov["zh-Hans"] >= 0.98:
        out.append("zh-Hans")
    if cov["zh-Hant"] >= 0.95:
        out.append("zh-Hant")
    if cov["ja"] >= 0.95:
        out.append("ja")
    if cov["ko"] >= 0.95:
        out.append("ko")
    if cov["latin"] >= 0.98:
        out.append("latin")
    return out


# ------------------------------------------------------------------ choosing one weight per package
_WEIGHTS = [("regular", 0), ("book", 1), ("normal", 1), ("medium", 2), ("-r.", 2), ("_n.", 2), ("text", 4)]
_AVOID = ("italic", "oblique", "slant", "condensed", "thin", "extralight", "light", "semibold", "demi", "bold",
          "heavy", "black", "extrabold", "ultra", "_s.", "-el.", "-l.", "-b.", "-h.", "-sb.", "-m.", "l3")


def _weight_score(name: str) -> int:
    n = name.lower()
    base = next((s for k, s in _WEIGHTS if k in n), 3)
    bad = sum(1 for k in _AVOID if k in n and not (k == "slant" and "yixie" in n))
    return base + 5 * bad


def _zip_name(i: zipfile.ZipInfo) -> str:
    n = i.filename
    if not (i.flag_bits & 0x800):
        try:
            n = n.encode("cp437").decode("gbk")
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass
    return n


def _package(zip_path: Path) -> str:
    return re.sub(r"[_\s]*猫啃网$", "", zip_path.stem)


@dataclass
class FontEntry:
    id: str
    file: str                       # relative to FREE_DIR
    family: str                     # the name written into subtitles (English family name)
    display: str                    # name shown in the UI
    names: List[str] = field(default_factory=list)
    face: int = 0
    scripts: List[str] = field(default_factory=list)
    coverage: Dict[str, float] = field(default_factory=dict)
    license: str = ""
    license_files: List[str] = field(default_factory=list)
    verified: bool = True           # an original licence text is included
    source: str = ""
    note: str = ""


def _license_kind(texts: Sequence[str]) -> Tuple[str, bool]:
    t = "\n".join(texts)
    for key, name in (("Open Font License", "SIL OFL 1.1"), ("commercially or non-commercially", "作者许可：可商用（保留版权声明）"), ("HarmonyOS Sans Fonts License", "HarmonyOS Sans 字体许可"),
                      ("Eclipse Public Licen", "Eclipse Public License 1.0"), ("ARPHIC PUBLIC LICENSE", "Arphic PL + IPA"),
                      ("Apache License", "Apache 2.0"), ("商用免费", "厂商声明：免费商用"), ("商业使用授权", "作者声明：可商用"),
                      ("商业用途", "作者声明：可商用"), ("Permission is hereby granted", "OFL / MIT 类许可")):
        if key.lower() in t.lower():
            return name, True
    return "免费商用（仅下载站声明，未附原始授权）", False


def install_package(zip_path: Path, dest_root: Path = FREE_DIR) -> Optional[FontEntry]:
    """Pick the common weight of a font package and extract it with its licence files."""
    pkg = _package(zip_path)
    zf = zipfile.ZipFile(zip_path)
    infos = [(i, _zip_name(i)) for i in zf.infolist() if not i.is_dir()]
    fonts = [(i, n) for i, n in infos if n.lower().endswith(FONT_EXT) and not Path(n).name.startswith("._")
             and i.file_size > 100_000]
    if not fonts:
        return None
    best = min(_weight_score(Path(n).name) for _, n in fonts)
    cands = [(i, n) for i, n in fonts if _weight_score(Path(n).name) == best]
    latin_words = re.findall(r"[A-Za-z]{3,}", pkg)
    dest = dest_root / re.sub(r"[^\w\-.]+", "_", pkg)
    dest.mkdir(parents=True, exist_ok=True)
    tmp = dest / "_cand"
    tmp.mkdir(exist_ok=True)
    scored = []
    for i, n in cands:                                   # several "regular" files: the one covering most
        p = tmp / Path(n).name
        p.write_bytes(zf.read(i))
        for fi, face in enumerate(_faces(p)):
            cov = coverage(_cmap(face))
            fn = p.name.lower()
            score = (cov["zh-Hans"] * 3 + cov["zh-Hant"] + cov["latin"] + cov["ja"] * 0.5
                     + (0.2 if p.suffix.lower() == ".ttf" else 0) - p.stat().st_size / 1e10
                     + (1.0 if any(w.lower() in fn for w in latin_words) else 0)     # the package's own font
                     - (0.8 if re.search(r"p?jp[-_.]", fn) else 0))                  # not the Japanese variant
            scored.append((score, p, fi, face, cov))
    score, p, fi, face, cov = max(scored, key=lambda x: x[0])
    final = dest / p.name
    for old in dest.glob("*"):
        if old.is_file() and old.suffix.lower() in FONT_EXT:
            old.unlink()
    shutil.move(str(p), final)
    shutil.rmtree(tmp, ignore_errors=True)
    lic_files, texts = [], []
    for i, n in infos:
        base = Path(n).name
        if re.search(r"(licen[cs]e|ofl|apl|ipa|readme|授权|使用须知|声明|说明).*\.(txt|md)$", base, re.I) \
                or base.lower().endswith(".docx") and "声明" in base:
            out = dest / base
            out.write_bytes(zf.read(i))
            lic_files.append(base)
            try:
                if base.lower().endswith(".docx"):
                    import io

                    x = zipfile.ZipFile(io.BytesIO(zf.read(i))).read("word/document.xml").decode("utf-8", "replace")
                    texts.append(re.sub(r"<[^>]+>", "", x))
                elif "猫啃" not in base and "授权说明" not in base and base != "字体授权.txt":
                    raw = zf.read(i)
                    texts.append(next((raw.decode(e) for e in ("utf-8-sig", "gbk", "latin-1") if _ok(raw, e)), ""))
            except Exception:
                pass
    lic, verified = _license_kind(texts)
    names, en, zh = _names(face)
    note = next((v for k, v in NOTES.items() if k in pkg), "")
    if "阿里巴巴普惠体" in pkg:
        verified = True
        lic = "阿里巴巴普惠体许可：免费商用"
    display = re.sub(r"[\d.]+$", "", re.sub(r"\(.*?\)", "", pkg)).strip(" _") or zh or en
    return FontEntry(id=dest.name, file=f"{dest.name}/{final.name}", family=en or pkg, display=display,
                     names=names, face=fi, scripts=scripts_of(cov), coverage=cov, license=lic, license_files=lic_files,
                     verified=verified, source=zip_path.name, note=note)


def _ok(raw: bytes, enc: str) -> bool:
    try:
        raw.decode(enc)
        return True
    except UnicodeDecodeError:
        return False


def _get(url: str, ua: str = "Mozilla/4.0", timeout: float = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": ua})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def install_english(family: str, gdir: str, dest_root: Path = FREE_DIR, weight: int = 400) -> FontEntry:
    """A static TTF of ``family`` at ``weight`` from Google Fonts, with its OFL."""
    css = _get(f"https://fonts.googleapis.com/css2?family={family.replace(' ', '+')}:wght@{weight}").decode()
    url = re.search(r"url\((https://[^)]+\.ttf)\)", css).group(1)
    dest = dest_root / re.sub(r"\W+", "", family)
    dest.mkdir(parents=True, exist_ok=True)
    p = dest / f"{family.replace(' ', '')}-Regular.ttf"
    p.write_bytes(_get(url))
    (dest / "OFL.txt").write_bytes(_get(f"https://raw.githubusercontent.com/google/fonts/main/ofl/{gdir}/OFL.txt"))
    face = _faces(p)[0]
    cov = coverage(_cmap(face))
    names, en, _ = _names(face)
    return FontEntry(id=dest.name, file=f"{dest.name}/{p.name}", family=en or family, display=family, names=names,
                     scripts=scripts_of(cov), coverage=cov, license="SIL OFL 1.1", license_files=["OFL.txt"],
                     source="Google Fonts")


def install(fonts_dir: Path = FONTS_DIR, english: bool = True, progress=print) -> List[FontEntry]:
    """Extract every package in ``fonts_dir`` and download the English fonts -> catalogue."""
    dest = fonts_dir / "free"
    dest.mkdir(parents=True, exist_ok=True)
    old = {e["id"]: e for e in json.loads((dest / "catalog.json").read_text(encoding="utf-8"))} \
        if (dest / "catalog.json").exists() else {}
    entries: List[FontEntry] = []
    if not english:                                      # keep the English fonts downloaded before
        entries += [FontEntry(**{k: v for k, v in e.items() if k in FontEntry.__dataclass_fields__})
                    for e in old.values() if e.get("source") == "Google Fonts" and (dest / e["file"]).exists()]
    for z in sorted(fonts_dir.glob("*.zip")):
        try:
            e = install_package(z, dest)
            if e:
                entries.append(e)
                progress(f"  {e.display:<16} {Path(e.file).name:<40} {','.join(e.scripts) or '-':<28} {e.license}")
        except Exception as ex:                          # a broken package must not stop the others
            progress(f"  ! {z.name}: {ex}")
    if english:
        for fam, gdir in ENGLISH:
            try:
                e = install_english(fam, gdir, dest)
                entries.append(e)
                progress(f"  {e.display:<16} {Path(e.file).name:<40} {','.join(e.scripts):<28} {e.license}")
            except Exception as ex:
                if fam.replace(" ", "") in old:          # keep what was downloaded before
                    entries.append(FontEntry(**old[fam.replace(" ", "")]))
                progress(f"  ! {fam}: {ex}")
    (dest / "catalog.json").write_text(json.dumps([asdict(e) for e in entries], ensure_ascii=False, indent=1),
                                       encoding="utf-8")
    FontLibrary.load.cache_clear()
    return entries


# ------------------------------------------------------------------ using the catalogue
class FontLibrary:
    def __init__(self, entries: Sequence[FontEntry], root: Path = FREE_DIR):
        self.entries = list(entries)
        self.root = root
        self._alias = {}
        for e in self.entries:
            for n in [e.family, e.display, e.id] + list(e.names):
                if n:
                    self._alias.setdefault(n.strip().lower(), e)
        self._cmaps: Dict[str, Set[int]] = {}

    @classmethod
    @lru_cache(maxsize=1)
    def load(cls, root: Path = FREE_DIR) -> "FontLibrary":
        f = root / "catalog.json"
        entries = [FontEntry(**{k: v for k, v in d.items() if k in FontEntry.__dataclass_fields__})
                   for d in json.loads(f.read_text(encoding="utf-8"))] if f.exists() else []
        return cls([e for e in entries if (root / e.file).exists()], root)

    def path(self, e: FontEntry) -> Path:
        return self.root / e.file

    def resolve(self, name: str) -> Optional[FontEntry]:
        return self._alias.get((name or "").strip().lower())

    def cmap(self, e: FontEntry) -> Set[int]:
        if e.id not in self._cmaps:
            self._cmaps[e.id] = _cmap(_faces(self.path(e))[e.face])
        return self._cmaps[e.id]

    def missing(self, e: FontEntry, chars: Iterable[str]) -> Set[str]:
        cm = self.cmap(e)
        return {c for c in chars if not c.isspace() and ord(c) not in cm and ord(c) > 0x20}

    def default_for(self, lang: str) -> Optional[FontEntry]:
        lang = (lang or "").lower()
        key = "zh" if lang.startswith(("zh", "yue")) else lang.split("-")[0]
        for n in DEFAULTS.get(key, []):
            e = self.resolve(n)
            if e:
                return e
        want = {"zh": "zh-Hans", "ja": "ja", "ko": "ko"}.get(key, "latin")
        for e in self.entries:                           # any font that covers the script
            if want in e.scripts:
                return e
        for n in DEFAULTS["zh"]:
            e = self.resolve(n)
            if e:
                return e
        return self.entries[0] if self.entries else None

    def best_for(self, chars: Iterable[str], prefer: Sequence[Optional[FontEntry]] = ()) -> Optional[FontEntry]:
        chars = set(chars)
        order = [e for e in prefer if e] + [e for e in self.entries if e not in prefer]
        best, best_miss = None, None
        for e in order:
            m = len(self.missing(e, chars))
            if m == 0:
                return e
            if best is None or m < best_miss:
                best, best_miss = e, m
        return best

    def enforce(self, ass_text: str, lang_main: str = "", lang_second: str = "") -> Tuple[str, List[Dict[str, str]], List[FontEntry]]:
        """Every style of the ASS uses a catalogued font that covers its text.
        Returns (text, substitutions, fonts used)."""
        styles = re.findall(r"^Style:\s*([^,]+),([^,]*),", ass_text, re.M)
        chars: Dict[str, Set[str]] = {s: set() for s, _ in styles}
        for m in re.finditer(r"^Dialogue:\s*(?:[^,]*,){3}([^,]*),(?:[^,]*,){5}(.*)$", ass_text, re.M):
            txt = re.sub(r"\{[^}]*\}", "", m.group(2)).replace("\\N", "").replace("\\n", "").replace("\\h", " ")
            chars.setdefault(m.group(1).strip(), set()).update(txt)
        subs, used, repl = [], [], {}
        for st, font in styles:
            st = st.strip()
            e = self.resolve(font)
            need = chars.get(st, set())
            lang = lang_second if st.lower().startswith("trans") else lang_main
            if e is None or self.missing(e, need):
                e2 = self.best_for(need, [e if e else None, self.default_for(lang)])
                if e2 is None:
                    continue
                subs.append({"style": st, "from": font.strip(), "to": e2.family,
                             "reason": "不在可商用字体库" if e is None else f"缺少 {len(self.missing(e, need))} 个字"})
                e = e2
            repl[st] = e.family
            if e not in used:
                used.append(e)

        def fix(m):
            name = m.group(1).strip()
            return f"Style: {m.group(1)},{repl.get(name, m.group(2))}," if name in repl else m.group(0)
        return re.sub(r"^Style:\s*([^,]+),([^,]*),", fix, ass_text, flags=re.M), subs, used


# ------------------------------------------------------------------ previews (font picker)
def sample_text(scripts: Sequence[str]) -> str:
    if "zh-Hans" in scripts:
        return "字幕预览 Subtitle 123"
    if "zh-Hant" in scripts:
        return "字幕預覽 Subtitle 123"
    if "ja" in scripts:
        return "字幕プレビュー Aa 123"
    if "ko" in scripts:
        return "자막 미리보기 Aa 123"
    return "Subtitle Preview 123"


def render_preview(font_path: Path, out: Path, text: str, face: int = 0, size: int = 44) -> Path:
    """The sample text set in the font itself (black on transparent PNG), for the font picker."""
    from PIL import Image, ImageDraw, ImageFont

    font = ImageFont.truetype(str(font_path), size, index=face)
    box = font.getbbox(text)
    w, h = box[2] - box[0] + 12, box[3] - box[1] + 12
    img = Image.new("LA", (max(w, 10), max(h, 10)), (0, 0))
    ImageDraw.Draw(img).text((6 - box[0], 6 - box[1]), text, font=font, fill=(0, 255))
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    return out


def preview_png(e: "FontEntry", root: Path = FREE_DIR) -> Path:
    out = root / e.id / "preview.png"
    if not out.exists():
        render_preview(root / e.file, out, sample_text(e.scripts), e.face)
    return out
