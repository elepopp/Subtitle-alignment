"""Script preparation for TTS (配音稿预处理).

Markup (angle brackets, so the bracket-note removal never touches it)::

    <行|hang2>      reading of a polyphonic character (Pinyin + tone 1-5)
    <停|0.5>        a pause of 0.5 s
    <重|关键词>     emphasis (applied after synthesis: the word is aligned and lifted)

``prepare(text)`` returns the cleaned, normalised script split into TTS
segments, each with the text the engine receives and the pause that follows.
Rules that raise the success rate of autoregressive TTS:

* notes in brackets, emoji / [表情] tags, Markdown, URLs removed
* numbers, dates, times, percentages, money, units, ranges written out the
  way they are read (1w -> 一万, 2026年 -> 二零二六年, 10:30 -> 十点三十分)
* punctuation normalised (full width, no runs), every segment ends with one
* long sentences split at clause boundaries, very short ones merged
  (AR models are least stable on very long and on 1-3 character inputs)
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

DIGITS = "零一二三四五六七八九"
UNITS = ["", "十", "百", "千"]
BIG = ["", "万", "亿", "万亿"]


# ------------------------------------------------------------------ numbers
def _four(n: int, zero_pad: bool) -> str:
    """0..9999 -> Chinese, ``zero_pad``: a leading 零 when < 1000 inside a bigger number."""
    s, out, zero = str(n), [], False
    if zero_pad and n < 1000:
        out.append("零")
    for i, ch in enumerate(s):
        d = int(ch)
        pos = len(s) - 1 - i
        if d == 0:
            zero = True
            continue
        if zero and out and out[-1] != "零":
            out.append("零")
        zero = False
        out.append(DIGITS[d] + UNITS[pos])
    return "".join(out)


def int_to_cn(n: int, liang: bool = True) -> str:
    """12345 -> 一万二千三百四十五; 10 -> 十; 2000 -> 两千 (``liang``)."""
    if n == 0:
        return "零"
    if n < 0:
        return "负" + int_to_cn(-n, liang)
    groups = []
    while n:
        groups.append(n % 10000)
        n //= 10000
    out = []
    for gi in range(len(groups) - 1, -1, -1):
        g = groups[gi]
        if g == 0:
            if out and out[-1] != "零":
                out.append("零")
            continue
        out.append(_four(g, zero_pad=bool(out)) + BIG[gi])
    s = "".join(out).rstrip("零")
    s = re.sub("零+", "零", s)
    if s.startswith("一十"):
        s = s[1:]                                    # 十二, not 一十二
    if liang:
        s = re.sub(r"^二(?=[千万亿])", "两", s)        # 两千 / 两万 at the start
        s = re.sub(r"(?<=[万亿零])二(?=[千万亿])", "两", s)
    return s


def digits_cn(s: str) -> str:
    return "".join(DIGITS[int(c)] for c in s if c.isdigit())


def num_to_cn(s: str) -> str:
    """'3.14' -> 三点一四, '1,234' -> 一千二百三十四, '-5' -> 负五."""
    s = s.replace(",", "")
    neg = s.startswith("-")
    s = s.lstrip("+-")
    if "." in s:
        a, b = s.split(".", 1)
        out = int_to_cn(int(a or 0), liang=False) + "点" + digits_cn(b)
    else:
        if len(s) > 1 and s.startswith("0"):
            out = digits_cn(s)                       # 007 -> 零零七
        else:
            out = int_to_cn(int(s))
    return ("负" if neg else "") + out


UNIT_WORDS = {
    "km": "公里", "kg": "公斤", "cm": "厘米", "mm": "毫米", "ml": "毫升", "mg": "毫克", "kw": "千瓦",
    "m²": "平方米", "㎡": "平方米", "m2": "平方米", "℃": "摄氏度", "°c": "摄氏度", "°": "度",
    "kb": "KB", "mb": "MB", "gb": "GB", "tb": "TB", "ms": "毫秒", "km/h": "公里每小时", "h": "小时",
}
CURRENCY = {"¥": "元", "￥": "元", "$": "美元", "€": "欧元", "£": "英镑"}


def normalize_numbers(t: str) -> str:
    # 1,234,567 -> 1234567
    t = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", t)
    # dates 2026-10-06 / 2026/10/6 / 2026.10.6
    t = re.sub(r"(?<![A-Za-z0-9])(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})(?![A-Za-z0-9])",
               lambda m: f"{digits_cn(m[1])}年{int_to_cn(int(m[2]), False)}月{int_to_cn(int(m[3]), False)}日", t)
    # years: 2026年 -> 二零二六年 (digit by digit)
    t = re.sub(r"(\d{4})(?=年)", lambda m: digits_cn(m[1]), t)
    # times 10:30 / 8:05 / 10:30:15
    t = re.sub(r"(?<![A-Za-z0-9])(\d{1,2}):(\d{2})(?::(\d{2}))?(?![A-Za-z0-9])", lambda m: int_to_cn(int(m[1]), False) + "点" + (
        ("零" + DIGITS[int(m[2])] if m[2].startswith("0") and m[2] != "00" else int_to_cn(int(m[2]), False)) + "分"
        if m[2] != "00" else "") + (int_to_cn(int(m[3]), False) + "秒" if m[3] else ""), t)
    # money: ¥100 / $5.5
    t = re.sub(r"([¥￥$€£])\s*(\d+(?:\.\d+)?)", lambda m: num_to_cn(m[2]) + CURRENCY[m[1]], t)
    # percentages / permille
    t = re.sub(r"(-?\d+(?:\.\d+)?)\s*%", lambda m: "百分之" + num_to_cn(m[1]), t)
    t = re.sub(r"(-?\d+(?:\.\d+)?)\s*‰", lambda m: "千分之" + num_to_cn(m[1]), t)
    # 1w / 1.5W / 3k / 10w+  (万 / 千 suffixes used in social media)
    t = re.sub(r"(\d+(?:\.\d+)?)\s*([wW万kK千])(\+)?(?![a-zA-Z])", lambda m: num_to_cn(m[1]) + (
        "万" if m[2] in "wW万" else "千") + ("多" if m[3] else ""), t)
    # temperatures, incl. negative: -5℃ -> 零下五摄氏度 / 零下五度
    t = re.sub(r"-(\d+(?:\.\d+)?)\s*(℃|°C|°c|度)", lambda m: "零下" + num_to_cn(m[1]) + ("度" if m[2] == "度" else "摄氏度"), t)
    # landlines 010-12345678 / 0755-1234567: digit by digit (before ranges)
    t = re.sub(r"(?<![\d.])(0\d{2,3})[-—](\d{7,8})(?![\d.])", lambda m: digits_cn(m[1]) + "，" + digits_cn(m[2]), t)
    # ranges 3-5 / 3~5 / 3—5 -> 三到五
    t = re.sub(r"(\d+(?:\.\d+)?)\s*[-~～—–]\s*(\d+(?:\.\d+)?)", lambda m: num_to_cn(m[1]) + "到" + num_to_cn(m[2]), t)
    # fractions 1/3 -> 三分之一
    t = re.sub(r"(?<![A-Za-z0-9/])(\d+)\s*/\s*(\d+)(?![A-Za-z0-9/])", lambda m: num_to_cn(m[2]) + "分之" + num_to_cn(m[1]), t)
    # units after a number
    def unit(m):
        u = m[2].lower()
        return num_to_cn(m[1]) + UNIT_WORDS.get(u, m[2])
    t = re.sub(r"(\d+(?:\.\d+)?)\s*(km/h|km|kg|cm|mm|ml|mg|kw|m²|㎡|℃|°C|°c|°|KB|MB|GB|TB|kb|mb|gb|tb|ms)(?![a-zA-Z])", unit, t)
    # read digit by digit: mobile numbers, numbers with a leading 0 (landlines, codes), long IDs
    t = re.sub(r"(?<![\d.])(1[3-9]\d{9}|0\d{6,}|\d{13,})(?![\d.])", lambda m: digits_cn(m[1]), t)
    # x.y version-like / decimals and plain integers
    t = re.sub(r"(?<![A-Za-z0-9])[vV](\d+(?:\.\d+)*)(?![A-Za-z0-9])",
               lambda m: "V" + "点".join(num_to_cn(x) for x in m[1].split(".")), t)      # v2.5 -> V二点五
    # plain numbers, but not inside alphanumeric tokens such as MP3 / GPT4o
    t = re.sub(r"(?<![A-Za-z0-9.])(-?\d+(?:\.\d+)?)(?![A-Za-z0-9])", lambda m: num_to_cn(m[1]), t)
    return t


# ------------------------------------------------------------------ cleaning
_MARK = re.compile(r"<(?P<tag>[^<>|]{1,8})\|(?P<val>[^<>]{1,40})>")
_EMOJI = re.compile("[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U0001F900-\U0001F9FF️‍⭐⭕⏩-⏺]")
SYMBOLS = {"&": "和", "×": "乘", "÷": "除以", "±": "正负", "≈": "约", "→": "，", "←": "，", "+": "加", "=": "等于",
           "@": "艾特", "#": "", "*": "", "~": "，", "～": "，", "|": "，", "_": " ", "^": ""}
PUNCT_FW = {",": "，", ".": "。", "?": "？", "!": "！", ";": "；", ":": "：", "(": "（", ")": "）"}
SENT_END = "。！？；…"
PH_OPEN, PH_CLOSE = chr(0xE000), chr(0xE001)
CLAUSE = "，、：—"


def _protect(t: str) -> Tuple[str, List[str]]:
    """Replace markup with placeholders that contain no digits (numbers are rewritten
    while it is protected)."""
    keep: List[str] = []

    def sub(m):
        keep.append(m[0])
        return PH_OPEN + chr(0xE100 + len(keep) - 1) + PH_CLOSE
    return _MARK.sub(sub, t), keep


def _restore(t: str, keep: List[str]) -> str:
    return re.sub(PH_OPEN + "(.)" + PH_CLOSE, lambda m: keep[ord(m[1]) - 0xE100], t)


def clean(text: str, remove_notes: bool = True) -> str:
    t = unicodedata.normalize("NFKC", text)
    t, keep = _protect(t)
    t = re.sub(r"https?://\S+|www\.\S+", "", t)                       # links
    t = re.sub(r"\[([^\[\]]+)\]\([^()]+\)", r"\1", t)                   # markdown links
    t = re.sub(r"^\s{0,3}(#{1,6}|>|[-*+]|\d+[.)])\s+", "", t, flags=re.M)   # md headers / lists / quotes
    t = re.sub(r"(\*\*|__|`)", "", t)
    if remove_notes:
        for o, c in (("（", "）"), ("(", ")"), ("【", "】"), ("[", "]"), ("{", "}"), ("〔", "〕"), ("「注", "」")):
            t = re.sub(re.escape(o) + r"[^" + re.escape(o + c) + r"]*" + re.escape(c), "", t)
    t = _EMOJI.sub("", t)
    t = re.sub(r"([哈呵嘿嘻啊呜])\1{3,}", r"\1\1\1", t)                  # 哈哈哈哈哈 -> 哈哈哈
    return _restore(t, keep)


def normalize_punct(t: str) -> str:
    t = t.replace("...", "…").replace("……", "…").replace("。。", "。")
    for a, b in PUNCT_FW.items():
        # full width unless it sits between two ASCII letters / digits (3.5, a,b)
        t = re.sub(rf"(?<![A-Za-z0-9]){re.escape(a)}|{re.escape(a)}(?![A-Za-z0-9])", b, t)
    t = t.replace("“", "").replace("”", "").replace("\"", "").replace("‘", "").replace("’", "")
    t = t.replace("《", "").replace("》", "").replace("「", "").replace("」", "").replace("『", "").replace("』", "")
    t = re.sub(r"[，、]{2,}", "，", t)
    t = re.sub(r"([。！？；…])[。！？；…，、]+", r"\1", t)
    t = re.sub(r"([，：])[，：]+", r"\1", t)
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"(?<=[一-鿿，。！？；：、]) (?=[一-鿿])", "", t)    # no spaces inside Chinese
    return t.strip()


def normalize_symbols(t: str) -> str:
    t, keep = _protect(t)
    for a, b in SYMBOLS.items():
        t = t.replace(a, b)
    return _restore(t, keep)


# ------------------------------------------------------------------ polyphones
# common polyphonic characters whose reading changes the meaning (常见多音字)
COMMON_POLY = set(
    "行重长乐还得地着了发会种传曾调都差便处弹当倒恶分干好号和横几假间将角觉校卷看空量露落埋没难宁片"
    "屏强切曲塞散少舍识数似宿提通为系相兴血要应正只中转钻薄背奔藏称冲创答担度读朝参尽解据削晕扎载"
    "咽熟泊模降圈占折缝给更供划华济教结禁劲卡壳拉累凉论率蒙秘奇骑铺朴期挑帖投吓鲜省盛属说厦吭拗"
    "那哪晃挣症殖仔作柏扒膀刨堡辟剥炮漂仆曝苔叨荷喝吁蔓脉闷眯泌缪抹摩粘拧泡喷撇瀑戚纤翘亲区雀任"
    "撒色煞扇蛇谁什拾刷仇臭畜揣幢椎攒脏择炸涨轴")


def polyphones(text: str) -> List[Dict]:
    """Polyphonic characters with their candidate readings and the reading chosen
    from context (pypinyin + jieba phrases).  Annotated ones are skipped."""
    try:
        from pypinyin import Style, pinyin
        from pypinyin.contrib.tone_convert import to_tone3
    except ImportError:  # pragma: no cover
        return []
    t, keep = _protect(text)
    # index in ``t`` -> index in ``text`` (a placeholder stands for a whole markup)
    omap, o, k = [], 0, 0
    while k < len(t):
        if t[k] == PH_OPEN and k + 2 < len(t) and t[k + 2] == PH_CLOSE:
            omap += [o, o, o]
            o += len(keep[ord(t[k + 1]) - 0xE100])
            k += 3
        else:
            omap.append(o)
            o += 1
            k += 1
    # pinyin() works on words: per-character readings for the protected text
    ctx = pinyin(t, style=Style.TONE3, neutral_tone_with_five=True, errors=lambda x: [[""] for _ in x])
    if len(ctx) != len(t):
        ctx = [[""]] * len(t)
    out = []
    for i, ch in enumerate(t):
        if ch not in COMMON_POLY:
            continue
        cands = pinyin(ch, style=Style.TONE3, heteronym=True, neutral_tone_with_five=True)[0]
        cands = list(dict.fromkeys(cands))[:4]
        if len(cands) > 1:
            out.append({"pos": omap[i], "char": ch, "candidates": cands, "default": ctx[i][0] or cands[0],
                        "context": t[max(0, i - 4):i + 5].replace(PH_OPEN, "").replace(PH_CLOSE, "")})
    return out


# ------------------------------------------------------------------ segments
@dataclass
class Segment:
    text: str                      # display text (with markup)
    tts_text: str                  # what the engine receives
    pause_after: float = 0.45      # seconds of silence after this segment
    emphasis: List[str] = field(default_factory=list)
    paragraph_end: bool = False


def to_engine(t: str, engine: str = "indextts2.5") -> Tuple[str, List[str]]:
    """Markup -> engine text.  Pinyin: <行|hang2> -> <行|HANG2> (IndexTTS-2.5);
    emphasis: the word is kept (it is lifted after synthesis)."""
    emph: List[str] = []

    def sub(m):
        tag, val = m["tag"], m["val"]
        if tag in ("重", "强调", "emph"):
            emph.append(val)
            return val
        if tag in ("停", "pause"):
            return ""
        if re.fullmatch(r"[a-zA-ZüÜv]+[1-5]", val.strip()):
            py = val.strip().upper().replace("Ü", "V")
            return f"<{tag}|{py}>" if engine.startswith("indextts") else tag
        return tag
    return _MARK.sub(sub, t).strip(), emph


def split_segments(t: str, max_chars: int = 60, min_chars: int = 4, comma_pause: float = 0.25,
                   sentence_pause: float = 0.45, paragraph_pause: float = 0.8, lang: str = "zh") -> List[Segment]:
    if not is_cjk_lang(lang):
        return _split_latin(t, int(max_chars * 2.5), comma_pause, sentence_pause, paragraph_pause)
    segs: List[Segment] = []
    held = set()                      # segments followed by an explicit pause: never merged across it
    paras = [p.strip() for p in re.split(r"\n\s*\n|\n", t) if p.strip()]
    for pi, para in enumerate(paras):
        # explicit pauses split the text
        parts = re.split(r"(<(?:停|pause)\|[\d.]+>)", para)
        sentences: List[Tuple[str, Optional[float]]] = []
        for part in parts:
            m = re.fullmatch(r"<(?:停|pause)\|([\d.]+)>", part)
            if m:
                if sentences:
                    txt = sentences[-1][0]
                    if txt[-1] not in SENT_END + CLAUSE:
                        txt += "，"            # a pause inside a sentence is not a full stop
                    sentences[-1] = (txt, float(m[1]))
                continue
            for s in re.findall(rf"[^{SENT_END}]+[{SENT_END}]*", part):
                if s.strip():
                    sentences.append((s.strip(), None))
        for s, explicit in sentences:
            pieces = [s]
            if _plain_len(s) > max_chars:
                pieces = _split_long(s, max_chars)
            for k, piece in enumerate(pieces):
                last = k == len(pieces) - 1
                if piece[-1] not in SENT_END + CLAUSE:
                    piece += "。" if last else "，"
                tts, emph = to_engine(piece)
                pause = comma_pause if not last else (explicit if explicit is not None else sentence_pause)
                segs.append(Segment(piece, tts, pause, emph))
                if last and explicit is not None:
                    held.add(id(segs[-1]))
        if segs:
            segs[-1].paragraph_end = True
            if pi < len(paras) - 1 and segs[-1].pause_after < paragraph_pause:
                segs[-1].pause_after = paragraph_pause
    # merge very short segments into a neighbour (AR TTS is unstable on 1-3 chars)
    merged: List[Segment] = []
    for sg in segs:
        if (merged and _plain_len(sg.text) < min_chars and not merged[-1].paragraph_end
                and id(merged[-1]) not in held):
            prev = merged[-1]
            joiner = "" if prev.text[-1] in SENT_END + CLAUSE else "，"
            prev.text = prev.text + joiner + sg.text
            prev.tts_text, prev.emphasis = to_engine(prev.text)
            prev.pause_after, prev.paragraph_end = sg.pause_after, sg.paragraph_end
            if id(sg) in held:
                held.add(id(prev))
        else:
            merged.append(sg)
    if len(merged) > 1 and _plain_len(merged[0].text) < min_chars and id(merged[0]) not in held:
        a, b = merged[0], merged[1]
        b.text = a.text + b.text
        b.tts_text, b.emphasis = to_engine(b.text)
        merged = merged[1:]
    return merged


def _plain_len(s: str) -> int:
    return len(re.sub(r"[\s，。！？；：、…—]", "", _MARK.sub(lambda m: m["val"] if m["tag"] in ("重",) else m["tag"], s)))


def _split_latin(t: str, max_chars: int, comma_pause: float, sentence_pause: float,
                 paragraph_pause: float) -> List[Segment]:
    """Sentences end at . ! ? ; followed by a space (not 3.5 / e.g. inside a word);
    long ones are split at commas, one- or two-word sentences join their neighbour."""
    segs: List[Segment] = []
    paras = [p.strip() for p in t.split("\n") if p.strip()]
    for pi, para in enumerate(paras):
        parts = re.split(r"(<(?:停|pause)\|[\d.]+>)", para)
        sentences: List[Tuple[str, Optional[float]]] = []
        for part in parts:
            m = re.fullmatch(r"<(?:停|pause)\|([\d.]+)>", part)
            if m:
                if sentences:
                    txt = sentences[-1][0]
                    if txt[-1] not in LATIN_END + LATIN_CLAUSE:
                        txt += ","
                    sentences[-1] = (txt, float(m[1]))
                continue
            for x in re.split(r"(?<=[.!?;])\s+(?=\S)", part):
                if x.strip():
                    sentences.append((x.strip(), None))
        for x, explicit in sentences:
            pieces = [x]
            if len(x) > max_chars:
                pieces, cur = [], ""
                for c in re.findall(r"[^,:;—]+[,:;—]?\s*", x):
                    if cur and len(cur + c) > max_chars:
                        pieces.append(cur.strip())
                        cur = c
                    else:
                        cur += c
                if cur.strip():
                    pieces.append(cur.strip())
            for k, piece in enumerate(pieces):
                last = k == len(pieces) - 1
                if piece[-1] not in LATIN_END + LATIN_CLAUSE:
                    piece += "." if last else ","
                tts, emph = to_engine(piece)
                pause = comma_pause if not last else (explicit if explicit is not None else sentence_pause)
                segs.append(Segment(piece, tts, pause, emph))
        if segs:
            segs[-1].paragraph_end = True
            if pi < len(paras) - 1 and segs[-1].pause_after < paragraph_pause:
                segs[-1].pause_after = paragraph_pause
    merged: List[Segment] = []
    for sg in segs:
        if merged and len(sg.text.split()) < 3 and not merged[-1].paragraph_end:
            prev = merged[-1]
            prev.text = prev.text + " " + sg.text
            prev.tts_text, prev.emphasis = to_engine(prev.text)
            prev.pause_after, prev.paragraph_end = sg.pause_after, sg.paragraph_end
        else:
            merged.append(sg)
    return merged


def _split_long(s: str, max_chars: int) -> List[str]:
    """Split at clause punctuation into pieces close to max_chars."""
    clauses = re.findall(rf"[^{CLAUSE}]+[{CLAUSE}]?", s)
    out, cur = [], ""
    for c in clauses:
        if cur and _plain_len(cur + c) > max_chars:
            out.append(cur)
            cur = c
        else:
            cur += c
    if cur:
        out.append(cur)
    return out


def is_cjk_lang(lang: Optional[str]) -> bool:
    return (lang or "zh").lower().split("-")[0] in ("zh", "yue", "ja", "ko")


# ------------------------------------------------------------------ alphabetic languages
SYMBOLS_LATIN = {"&": " and ", "@": " at ", "#": "", "*": "", "~": ", ", "～": ", ", "|": ", ", "_": " ", "^": "",
                 "→": ", ", "←": ", ", "=": " equals ", "×": " times ", "≈": " about "}
PUNCT_HW = {"，": ",", "。": ".", "？": "?", "！": "!", "；": ";", "：": ":", "（": "(", "）": ")", "、": ",",
            "“": "", "”": "", "‘": "'", "’": "'", "\"": "", "《": "", "》": "", "「": "", "」": ""}
LATIN_END = ".!?;"
LATIN_CLAUSE = ",:;—"


def normalize_latin(t: str) -> str:
    """English / European scripts: half-width punctuation, spaced properly.  Numbers are
    left as digits - the engine reads them out in the right language."""
    t, keep = _protect(t)
    for a, b in SYMBOLS_LATIN.items():
        t = t.replace(a, b)
    for a, b in PUNCT_HW.items():
        t = t.replace(a, b)
    t = re.sub(r"\.{3,}|…+", "…", t)                     # kept apart from full stops until the end
    t = re.sub(r"\s+([,.!?;:])", r"\1", t)
    t = re.sub(r"([,!?;:])(?=[^\s\d])", r"\1 ", t)
    t = re.sub(r"(?<![\d.])\.(?=[A-Za-z])", ". ", t)
    t = re.sub(r"([,;:])[,;:]+", r"\1", t)
    t = re.sub(r"([.!?])[,.;:]+", r"\1", t)
    t = re.sub(r"…\s*", "... ", t)
    t = re.sub(r"[ \t]+", " ", t)
    t = "\n".join(x.strip() for x in t.split("\n"))
    return _restore(t, keep).strip()


def normalize(text: str, remove_notes: bool = True, lang: str = "zh") -> str:
    """clean -> numbers -> symbols -> punctuation (markup is kept)."""
    t = clean(text, remove_notes)
    if not is_cjk_lang(lang):
        return normalize_latin(t)
    t, keep = _protect(t)
    t = normalize_numbers(t)
    t = _restore(t, keep)
    t = normalize_symbols(t)
    return normalize_punct(t)


def prepare(text: str, remove_notes: bool = True, max_chars: int = 60, lang: str = "zh", **pauses) -> Dict:
    """Full pipeline: clean -> symbols -> numbers -> punctuation -> segments."""
    t = normalize(text, remove_notes, lang)
    segs = split_segments(t, max_chars=max_chars, lang=lang, **pauses)
    poly = polyphones(t) if (lang or "zh").lower().startswith(("zh", "yue")) else []
    return {"text": t, "segments": [asdict(s) for s in segs], "polyphones": poly}
