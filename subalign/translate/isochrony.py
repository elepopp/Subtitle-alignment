"""Dubbing translation (配音翻译): translations that fit the original timing.

A dubbed line is spoken over the original picture, so besides meaning it has to
satisfy two constraints that plain subtitle translation ignores:

* **isochrony** - it takes about as long to say as the original.  Speaking time
  follows the number of syllables, and every language has its own typical rate
  (Mandarin ~5.2 syllables/s, English ~6.2, Japanese ~7.8 morae/s; Pellegrino et
  al. 2011), so the budget is the source syllable count scaled by the two rates.
  Phonemes are counted too and shown, but syllables decide the length: an English
  syllable carries more phonemes than a Chinese one and is spoken just as fast.
* **anchor order** - nouns, names and numbers are heard at about the same moment as
  in the original (cuts, gestures and on-screen objects line up with them).  A noun
  heard at 3 s of a 10 s sentence must come in the first half of the translation,
  even where the target grammar would put it at the end, so the sentence is rewritten
  (fronting, passive, apposition, two short clauses ...).

Flow per sentence: measure the source (syllables, phonemes, anchors and where they
are heard - from word timings when the source was recognised) -> the LLM writes
several structurally different candidates -> each is measured here (syllables,
where every anchor lands) and scored -> sentences whose best candidate is still off
go back with concrete feedback -> optionally the LLM picks the most natural among
the best-measured finalists.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ..llm import LLMClient, LLMError
from . import LANG_NAMES

log = logging.getLogger("subalign")

# syllables (morae for Japanese) per second in ordinary speech.  Pellegrino, Coupe &
# Marsico (2011) for en / fr / de / it / ja / zh / es / vi; the others are estimates.
SYLLABLE_RATE = {"zh": 5.18, "yue": 5.4, "en": 6.19, "fr": 7.18, "de": 5.97, "it": 6.99, "ja": 7.84,
                 "es": 7.82, "vi": 5.22, "ko": 6.6, "pt": 7.3, "ru": 6.4, "th": 5.6, "id": 6.8, "ar": 6.0}
# syllables per word, to give the model a word count it can actually aim for
SYLLABLES_PER_WORD = {"en": 1.45, "fr": 1.5, "de": 1.75, "es": 2.0, "it": 2.1, "pt": 2.0, "ru": 2.3,
                      "vi": 1.0, "id": 2.3}
CJK_LANGS = ("zh", "yue", "ja", "ko")
SENT_END = "。！？!?；;…"


def base_lang(lang: Optional[str]) -> str:
    return (lang or "").lower().replace("_", "-").split("-")[0]


# ------------------------------------------------------------------ counting
_ONES = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
         "sixteen seventeen eighteen nineteen").split()
_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()


def int_to_en(n: int) -> str:
    if n < 0:
        return "minus " + int_to_en(-n)
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10] + ("-" + _ONES[n % 10] if n % 10 else "")
    if n < 1000:
        return _ONES[n // 100] + " hundred" + (" " + int_to_en(n % 100) if n % 100 else "")
    for div, name in ((10 ** 12, "trillion"), (10 ** 9, "billion"), (10 ** 6, "million"), (1000, "thousand")):
        if n >= div:
            return int_to_en(n // div) + " " + name + (" " + int_to_en(n % div) if n % div else "")
    return str(n)


_CURRENCY_EN = {"$": "dollars", "¥": "yuan", "￥": "yuan", "€": "euros", "£": "pounds"}


def number_to_en(tok: str) -> str:
    """'2026' -> twenty twenty-six, '3.5%' -> three point five percent, '$5' -> five dollars."""
    cur = _CURRENCY_EN.get(tok[0], "") if tok and not tok[0].isdigit() else ""
    t = tok[1:] if cur else tok
    pct = t.endswith("%")
    t = t.rstrip("%")
    if ":" in t:                                            # 10:30
        words = " ".join(int_to_en(int(x)) for x in t.split(":") if x.isdigit())
    else:
        t = t.replace(",", "")
        a, _, b = t.partition(".")
        if not a.isdigit():
            return tok
        n = int(a)
        if len(a) == 4 and "," not in tok and 1100 <= n <= 2099 and not b and n % 100:   # years: twenty twenty-six
            words = int_to_en(n // 100) + " " + (int_to_en(n % 100) if n % 100 >= 10 else "oh " + int_to_en(n % 100))
        else:
            words = int_to_en(n)
        if b.isdigit():
            words += " point " + " ".join(_ONES[int(c)] for c in b)
    return words + (" percent" if pct else "") + (" " + cur if cur else "")


def en_syllables(word: str) -> int:
    """Heuristic English syllable count (vowel groups with the usual silent-e rules)."""
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return 0
    if len(w) <= 3:
        return 1
    extra = 1 if re.search(r"[^aeiouy]le$", w) else 0        # ta-ble, lit-tle
    w = re.sub(r"(?:[^laeiouysxzcgh]es|[^laeiouytd]ed|[^aeiouy]e)$", "", w)
    w = re.sub(r"^y", "", w)
    n = len(re.findall(r"[aeiouy]{1,2}", w))
    # vowel pairs that are two syllables (cre-ate, ri-ot, pi-ano, i-dea)
    n += len(re.findall(r"(?:ia|io|ea(?=t)|eo|ua|uo|ie(?=t))", w))
    return max(1, n + extra)


def latin_syllables(word: str, lang: str) -> int:
    if lang == "vi":
        return 1
    if lang in ("es", "it", "pt", "id", "ru", "de", "fr"):
        w = word.lower()
        if lang == "fr":
            w = re.sub(r"(?:es|e|ent)$", "", w) or w           # silent endings
        return max(1, len(re.findall(r"[aeiouyàáâãäåèéêëìíîïòóôõöùúûüýæœ]+|[аеёиоуыэюя]+", w)))
    return en_syllables(word)


_EN_GRAPH = re.compile(r"tch|dge|igh|ough|augh|th|sh|ch|ph|wh|ng|ck|ee|oo|ea|ai|ay|oa|ou|ow|oi|oy|au|aw|ie|ei|ey|ue|ew"
                       r"|([b-df-hj-np-tv-z])\1")


def latin_phonemes(word: str) -> int:
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return 0
    if len(w) > 3 and w.endswith("e") and not w.endswith(("ee", "ie", "ye", "oe")):
        w = w[:-1]                                           # silent e
    n = len(_EN_GRAPH.sub("#", w))
    return n + w.count("x") + w.count("qu")                 # x = /ks/, qu = /kw/


def _pinyin_phonemes(ch: str) -> int:
    from ..text.phonetic import _split_pinyin, pinyin_of

    py = pinyin_of(ch)
    if not py:
        return 2
    ini, fin = _split_pinyin(py)
    if ini in ("y", "w") and fin[:1] in ("i", "u"):
        ini = ""                                             # yi / wu / yin: the glide is the vowel
    return (1 if ini else 0) + len(re.sub(r"ng|er|ai|ei|ao|ou", "X", fin))


_UNIT = re.compile(
    r"(?P<num>[$¥￥€£]?\d+(?:[.,:]\d+)*%?)"
    r"|(?P<han>[㐀-䶿一-鿿豈-﫿])"
    r"|(?P<kana>[぀-ヿㇰ-ㇿ])"
    r"|(?P<hangul>[가-힯])"
    r"|(?P<thai>[฀-๿]+)"
    r"|(?P<word>[A-Za-zÀ-ɏḀ-ỿͰ-ϿЀ-ӿ]+"
    r"(?:['’][A-Za-zÀ-ɏḀ-ỿ]+)*)")
_SMALL_KANA = set("ゃゅょぁぃぅぇぉゎャュョァィゥェォヮ")


@dataclass
class Unit:
    start: int                 # character offsets in the text
    end: int
    syllables: float
    phonemes: float


def profile(text: str, lang: str) -> List[Unit]:
    """Spoken units of ``text`` with their syllable / phoneme weight; numbers are
    weighted as they are read out, acronyms (AI, GPT) letter by letter."""
    lang = base_lang(lang)
    out: List[Unit] = []
    for m in _UNIT.finditer(text):
        kind, s = m.lastgroup, m.group(0)
        if kind == "num":
            if lang in ("zh", "yue", "ja", "ko"):
                from ..tts.textprep import normalize_numbers

                nxt = text[m.end():m.end() + 1]             # 2026年 is read digit by digit
                spoken = normalize_numbers(s + nxt)
                if nxt and spoken.endswith(nxt):
                    spoken = spoken[:-1]
                n = sum(1 for c in spoken if "一" <= c <= "鿿")
                syl = n * (1.6 if lang == "ja" else 1.0)
                ph = sum(_pinyin_phonemes(c) for c in spoken if "一" <= c <= "鿿")
            else:
                words = number_to_en(s).replace("-", " ").split()
                syl = sum(en_syllables(w) for w in words)
                ph = sum(latin_phonemes(w) for w in words)
        elif kind == "han":
            if lang == "ja":
                syl, ph = 1.7, 3.0
            else:
                syl, ph = 1.0, float(_pinyin_phonemes(s)) if lang in ("zh", "yue", "") else 2.5
        elif kind == "kana":
            syl = 0.0 if s in _SMALL_KANA else 1.0
            ph = 0.0 if s in _SMALL_KANA else (1.0 if s in "あいうえおアイウエオんンっッー" else 2.0)
        elif kind == "hangul":
            k = ord(s) - 0xAC00
            syl = 1.0
            ph = 1.0 + (0 if k // 588 == 11 else 1) + (1 if k % 28 else 0)     # ㅇ initial is silent
        elif kind == "thai":
            syl, ph = max(1.0, len(s) / 2.6), len(s) * 0.8
        else:
            if s.isupper() and 2 <= len(s) <= 5 and s.isascii():   # acronym, spelled out
                syl = float(sum(3 if c == "W" else 1 for c in s))
                ph = 2.0 * len(s)
            else:
                syl = float(latin_syllables(s, lang if lang not in CJK_LANGS else "en"))
                ph = float(latin_phonemes(s))
        out.append(Unit(m.start(), m.end(), syl, ph))
    return out


def count(text: str, lang: str) -> Dict[str, float]:
    p = profile(text, lang)
    return {"syllables": round(sum(u.syllables for u in p), 1), "phonemes": round(sum(u.phonemes for u in p))}


def position(text: str, offset: int, lang: str, prof: Optional[List[Unit]] = None) -> float:
    """Where in the spoken sentence (0..1, by syllables) the character ``offset`` starts."""
    prof = prof if prof is not None else profile(text, lang)
    total = sum(u.syllables for u in prof)
    if total <= 0:
        return 0.0
    before = sum(u.syllables for u in prof if u.end <= offset)
    return round(min(1.0, before / total), 3)


# ------------------------------------------------------------------ sentences
def _joiner(a: str, b: str) -> str:
    return "" if (a and b and _is_cjk(a[-1]) and _is_cjk(b[0])) else " "


def _is_cjk(ch: str) -> bool:
    return bool(re.match(r"[぀-ヿ㐀-䶿一-鿿가-힯＀-￯。，、！？；：]", ch))


def sentences_from_document(doc, merge: bool = True, max_seconds: float = 10.0, max_gap: float = 0.35) -> List[Dict]:
    """Recognised / subtitle lines -> translation units.  Lines that continue a sentence
    (no final punctuation, short gap, same speaker) are merged so that word order can be
    rewritten across them; ``times`` keeps (char offset, start time) of every timed token."""
    units: List[Dict] = []
    for ln in doc.lines:
        text, times, off = "", [], 0
        for i, t in enumerate(ln.tokens):
            if t.start is not None:
                times.append([len(text), round(t.start, 3)])
            text += t.text
            if t.space_after and i < len(ln.tokens) - 1:
                text += " "
        lead = len(text) - len(text.lstrip())
        text = text.strip()
        if not text:
            continue
        times = [[max(0, o - lead), s] for o, s in times]
        u = {"text": text, "start": ln.start, "end": ln.end, "times": times, "speaker": ln.speaker}
        prev = units[-1] if units else None
        if (merge and prev and prev["end"] is not None and u["start"] is not None and prev["text"][-1] not in SENT_END
                and u["start"] - prev["end"] < max_gap and u["end"] - prev["start"] <= max_seconds
                and prev.get("speaker") == u.get("speaker")):
            j = _joiner(prev["text"], text)
            shift = len(prev["text"]) + len(j)
            prev["text"] += j + text
            prev["times"] += [[o + shift, s] for o, s in times]
            prev["end"] = u["end"]
        else:
            units.append(u)
    for i, u in enumerate(units):
        u["id"] = i + 1
    return units


def sentences_from_text(text: str, lang: str) -> List[Dict]:
    """Plain text -> sentences (no timing)."""
    lang = base_lang(lang)
    out = []
    for para in [p.strip() for p in text.splitlines() if p.strip()]:
        if lang in CJK_LANGS:
            parts = re.findall(rf"[^{SENT_END}]+[{SENT_END}]*", para)
        else:
            parts = re.split(r"(?<=[.!?…;])\s+(?=[\"'“‘(]?[A-Z0-9À-ɏ])", para)
        out += [p.strip() for p in parts if p.strip()]
    return [{"id": i + 1, "text": t, "start": None, "end": None, "times": []} for i, t in enumerate(out)]


# ------------------------------------------------------------------ anchors
_NOUN_FLAGS = {"n", "nr", "nrt", "nrfg", "ns", "nt", "nz", "eng", "nw"}
_PROPER = {"nr", "nrt", "nrfg", "ns", "nt", "nz", "eng"}
_SURNAMES = set("王李张刘陈杨黄赵吴周徐孙马朱胡郭何高林罗郑梁谢宋唐许韩冯邓曹彭曾肖田董袁潘于蒋蔡余杜叶程苏魏吕丁任沈姚卢姜崔"
                "钟谭陆汪范金石廖贾夏韦付方白邹孟熊秦邱江尹薛闫段雷侯龙史陶黎贺顾毛郝龚邵万钱严覃武戴莫孔向汤欧司上诸")
_ZH_GENERIC = set("时候 事情 东西 问题 方面 部分 地方 情况 样子 感觉 大家 自己 回家 时间 办法 意思 内容 过程 结果 原因 "
                  "方式 东西 人们 朋友们 什么 这样 那样".split())


def zh_anchors(text: str, max_n: int = 5) -> List[Dict]:
    """Nouns / names / numbers of a Chinese sentence (jieba POS), adjacent nouns joined
    (苹果 + 手机 -> 苹果手机).  Single-character common nouns are too generic to anchor."""
    try:
        import jieba
        import jieba.posseg as pseg

        jieba.setLogLevel(60)
    except ImportError:  # pragma: no cover - optional
        return []
    spans: List[List[Any]] = []          # [start, end, word, priority]
    off = 0
    for w in pseg.lcut(text):
        word, flag = w.word, w.flag
        a = text.find(word, off)
        if a < 0:
            continue
        off = a + len(word)
        if flag in ("nr", "nrt", "nrfg") and not (2 <= len(word) <= 4 and word[0] in _SURNAMES):
            flag = "x"                                   # the HMM invents names (太贵/nr)
        noun = flag in _NOUN_FLAGS and word not in _ZH_GENERIC
        num = flag == "m" and bool(re.search(r"\d", word))
        if not (noun or num):
            continue
        prio = 3 if flag in _PROPER else (2 if num else 1)
        if spans and spans[-1][1] == a and noun and spans[-1][3] != 2:
            spans[-1][1], spans[-1][2] = off, spans[-1][2] + word
            spans[-1][3] = max(spans[-1][3], prio)
        else:
            spans.append([a, off, word, prio])
    spans = [s for s in spans if len(s[2]) >= 2 or s[3] >= 2]
    keep = sorted(sorted(spans, key=lambda s: (-s[3], -len(s[2])))[:max_n], key=lambda s: s[0])
    return [{"text": s[2], "offset": s[0]} for s in keep]


_STOP_EN = set("""a an the and or but if then so of to in on at by for with from as is are was were be been being
this that these those it its i you he she we they me him her us them my your his our their what which who whom
whose when where why how not no yes do does did have has had will would can could should may might must shall
there here very just also too only about into over under than more most some any all each every one""".split())


def heuristic_anchors(text: str, max_n: int = 5) -> List[Dict]:
    """Without a tagger: numbers and capitalised words that do not start the sentence."""
    out = []
    for m in re.finditer(r"\d+(?:[.,:]\d+)*%?|[A-Z][\w'’-]*(?:\s+[A-Z][\w'’-]*)*", text):
        w = m.group(0)
        if m.start() == 0 and not w[0].isdigit() and " " not in w:
            continue
        if w.lower() in _STOP_EN:
            continue
        out.append({"text": w, "offset": m.start()})
    return out[:max_n]


ANCHOR_SCHEMA = {"type": "object", "properties": {"lines": {"type": "array", "items": {
    "type": "object", "properties": {"id": {"type": "integer"}, "anchors": {"type": "array", "items": {"type": "string"}}},
    "required": ["id", "anchors"], "additionalProperties": False}}}, "required": ["lines"], "additionalProperties": False}


def llm_anchors(sents: Sequence[Dict], client: LLMClient, lang: str, max_n: int = 5) -> Dict[int, List[Dict]]:
    system = ("You mark the anchor words of spoken sentences for dubbing: concrete nouns, names, places, brands, "
              f"products and numbers that a viewer connects with the picture (at most {max_n} per sentence, the most "
              "important ones). Copy each anchor exactly as written in the sentence (an exact substring). Skip pronouns "
              "and abstract filler nouns.\n"
              'Reply with JSON only: {"lines": [{"id": <id>, "anchors": ["...", "..."]}]}')
    user = json.dumps({"language": LANG_NAMES.get(lang, lang), "lines": [{"id": s["id"], "text": s["text"]} for s in sents]},
                      ensure_ascii=False)
    out: Dict[int, List[Dict]] = {}
    data = client.complete_json(system, user, ANCHOR_SCHEMA)
    items = data.get("lines", []) if isinstance(data, dict) else data
    texts = {s["id"]: s["text"] for s in sents}
    for it in items or []:
        try:
            sid = int(it.get("id"))
        except (TypeError, ValueError, AttributeError):
            continue
        if sid not in texts:
            continue
        found, used = [], set()
        for a in it.get("anchors") or []:
            a = str(a).strip()
            k = _find_term(texts[sid], a)
            if a and k >= 0 and k not in used:
                used.add(k)
                found.append({"text": texts[sid][k:k + len(a)], "offset": k})
        out[sid] = sorted(found, key=lambda x: x["offset"])[:max_n]
    return out


def _find_term(text: str, term: str, start: int = 0) -> int:
    """Case-insensitive search; alphabetic terms only at word boundaries."""
    if not term:
        return -1
    if re.match(r"[A-Za-z0-9]", term[0]):
        m = re.search(r"(?<![A-Za-z0-9])" + re.escape(term) + r"(?![A-Za-z0-9])", text[start:], re.I)
        return start + m.start() if m else -1
    return text.lower().find(term.lower(), start)


def anchor_position(sent: Dict, offset: int, lang: str) -> Tuple[float, Optional[float]]:
    """(fraction 0..1, seconds into the sentence) where the anchor is heard: from the
    word timings when the sentence was recognised, else by syllables."""
    times, start, end = sent.get("times") or [], sent.get("start"), sent.get("end")
    if times and start is not None and end is not None and end > start:
        t = None
        for o, s in times:
            if o <= offset:
                t = s
            else:
                break
        if t is not None:
            rel = max(0.0, t - start)
            return round(min(1.0, rel / (end - start)), 3), round(rel, 2)
    frac = position(sent["text"], offset, lang)
    dur = (end - start) if (start is not None and end is not None) else None
    return frac, (round(frac * dur, 2) if dur else None)


# ------------------------------------------------------------------ config / analysis
@dataclass
class IsoConfig:
    source: str = "zh"
    target: str = "en"
    candidates: int = 4               # candidates per sentence and round
    rounds: int = 2                   # feedback rounds for sentences whose best candidate is off
    tolerance: float = 0.10           # syllable budget +-10 %
    judge: bool = True                # LLM picks the most natural among the best-measured finalists
    finalists: int = 3
    batch_size: int = 6
    context: int = 2
    ratio: Optional[float] = None     # target / source syllables; default from the speaking rates
    max_anchors: int = 5
    # timed sentences: never budget more than the slot holds at this share of the target
    # language's normal rate (IndexTTS speaks English at ~5.1 syllables/s, ~0.85 of normal);
    # 0 = budget by syllable count only
    voice_rate: float = 0.9
    instructions: str = ""
    glossary: Dict[str, str] = field(default_factory=dict)

    @property
    def syllable_ratio(self) -> float:
        if self.ratio:
            return float(self.ratio)
        s, t = SYLLABLE_RATE.get(base_lang(self.source), 6.0), SYLLABLE_RATE.get(base_lang(self.target), 6.0)
        return t / s


def analyze(sent: Dict, cfg: IsoConfig, anchors: Optional[List[Dict]] = None) -> Dict:
    """Source measurements, anchors with their positions and the syllable budget."""
    src = base_lang(cfg.source)
    c = count(sent["text"], src)
    sent["syllables"], sent["phonemes"] = c["syllables"], c["phonemes"]
    if sent.get("start") is not None and sent.get("end") is not None:
        sent["duration"] = round(sent["end"] - sent["start"], 2)
        sent["timed"] = True
    else:
        sent["duration"] = round(c["syllables"] / SYLLABLE_RATE.get(src, 6.0), 2)
        sent["timed"] = False
    if anchors is not None:
        sent["anchors"] = []
        for a in anchors:
            frac, sec = anchor_position(sent, a["offset"], src)
            sent["anchors"].append({"text": a["text"], "offset": a["offset"], "pos": frac, "time": sec})
    n, by = c["syllables"] * cfg.syllable_ratio, "ratio"
    if sent["timed"] and cfg.voice_rate > 0 and sent["duration"] > 0:
        # the scaled count assumes the voice speaks as fast as the original speaker; a fast
        # speaker's line would not fit when the synthetic voice speaks at a normal pace
        cap = sent["duration"] * SYLLABLE_RATE.get(base_lang(cfg.target), 6.0) * cfg.voice_rate
        if cap < n:
            n, by = cap, "time"
    tgt = max(1, round(n))
    tol = max(1, round(tgt * cfg.tolerance))
    sent["target"] = {"syllables": tgt, "lo": max(1, tgt - tol), "hi": tgt + tol, "by": by}
    sent.setdefault("candidates", [])
    sent.setdefault("choice", None)
    return sent


# ------------------------------------------------------------------ measuring candidates
def _zone(p: float) -> str:
    return "first half" if p < 0.4 else ("second half" if p > 0.6 else "middle")


def measure(sent: Dict, text: str, terms: Sequence[Dict], cfg: IsoConfig) -> Dict:
    """Length and anchor placement of a candidate translation, with its cost (lower is better)."""
    tgt_lang = base_lang(cfg.target)
    prof = profile(text, tgt_lang)
    syl = round(sum(u.syllables for u in prof), 1)
    ph = round(sum(u.phonemes for u in prof))
    target = sent["target"]["syllables"]
    dev = (syl - target) / target if target else 0.0
    placed = []
    for a in sent.get("anchors") or []:
        term = _match_term(a["text"], terms)
        k = -1
        if term:
            k = _find_term(text, term)
            if k < 0:                                       # the model inflected / reworded it: any word of it
                for w in sorted(re.findall(r"\w+", term), key=len, reverse=True):
                    if len(w) >= 2 and (k := _find_term(text, w)) >= 0:
                        break
        elif _find_term(text, a["text"]) >= 0:              # kept as is (names, brands, numbers)
            term, k = a["text"], _find_term(text, a["text"])
        if k < 0:
            placed.append({"src": a["text"], "tgt": term or "", "src_pos": a["pos"], "tgt_pos": None, "missing": True,
                           "cross": False, "ok": False})
            continue
        p = position(text, k, tgt_lang, prof)
        cross = not (0.4 <= a["pos"] <= 0.6) and (p - 0.5) * (a["pos"] - 0.5) < 0
        placed.append({"src": a["text"], "tgt": text[k:k + len(term)] if term else "", "src_pos": a["pos"], "tgt_pos": p,
                       "missing": False, "cross": cross, "ok": not cross and abs(p - a["pos"]) <= 0.25})
    # quadratic: a few percent inside the tolerance costs little (naturalness decides there),
    # the edge of the tolerance costs 1, twice the tolerance 4
    c_len = (abs(dev) / max(cfg.tolerance, 0.02)) ** 2
    c_pos = 0.0
    if placed:
        c_pos = sum(1.5 if x["missing"] else (abs(x["tgt_pos"] - x["src_pos"]) / 0.25) ** 2 + (2.0 if x["cross"] else 0.0)
                    for x in placed) / len(placed)
    ok = abs(dev) <= cfg.tolerance + 1e-9 and not any(x["cross"] or x["missing"] for x in placed)
    return {"text": text, "terms": list(terms), "syllables": syl, "phonemes": ph, "len_dev": round(dev, 3),
            "anchors": placed, "cost": round(c_len + 0.8 * c_pos, 3), "ok": ok}


_CJK_RE = re.compile(r"[぀-ヿ㐀-䶿一-鿿가-힯]")


def in_language(text: str, lang: str) -> bool:
    """Small models sometimes answer in the source language: reject those candidates."""
    lang = base_lang(lang)
    cjk = len(_CJK_RE.findall(text))
    letters = len(re.findall(r"[^\W\d_]", text))
    if not letters:
        return False
    if lang in CJK_LANGS:
        if lang == "ko":
            return len(re.findall(r"[가-힯]", text)) / letters > 0.3
        if lang == "ja":
            return len(re.findall(r"[぀-ヿ]", text)) > 0 or cjk / letters > 0.5
        return cjk / letters > 0.3
    return cjk / letters < 0.1


def _match_term(src: str, terms: Sequence[Dict]) -> Optional[str]:
    for t in terms or []:
        s, g = str(t.get("src", "")).strip(), str(t.get("tgt", "")).strip()
        if s and g and (s == src or s in src or src in s):
            return g
    return None


def final_cost(c: Dict) -> float:
    j = c.get("judge")
    return c["cost"] + (0.25 * (10 - j) if isinstance(j, (int, float)) else 0.0)


def rank(sent: Dict) -> None:
    """Pick the best candidate (unless the user chose one / wrote their own)."""
    cands = sent.get("candidates") or []
    for c in cands:
        c["final"] = round(final_cost(c), 3)
    if not cands or sent.get("locked"):
        return
    sent["choice"] = min(range(len(cands)), key=lambda i: cands[i]["final"])
    sent["translation"] = cands[sent["choice"]]["text"]


def best(sent: Dict) -> Optional[Dict]:
    c = sent.get("candidates") or []
    i = sent.get("choice")
    return c[i] if i is not None and 0 <= i < len(c) else None


def remeasure(sent: Dict, cfg: IsoConfig) -> Dict:
    """Budget and candidate scores again (after the tolerance / ratio changed) - no LLM."""
    anchors = sent.get("anchors")
    analyze(sent, cfg, [{"text": a["text"], "offset": a["offset"]} for a in anchors] if anchors is not None else None)
    keep = ("round", "judge", "custom")
    sent["candidates"] = [dict(measure(sent, c["text"], c.get("terms", []), cfg), **{k: c[k] for k in keep if k in c})
                          for c in sent.get("candidates", [])]
    rank(sent)
    return sent


def set_custom(sent: Dict, text: str, cfg: IsoConfig) -> Dict:
    """The user's own translation: measured with the term mapping of the chosen candidate."""
    b = best(sent)
    m = measure(sent, text.strip(), (b or {}).get("terms", []), cfg)
    m.update(custom=True, round=-1)
    sent["candidates"] = [c for c in sent.get("candidates", []) if not c.get("custom")] + [m]
    sent["choice"], sent["locked"], sent["translation"] = len(sent["candidates"]) - 1, True, m["text"]
    return m


# ------------------------------------------------------------------ LLM
GEN_SCHEMA = {"type": "object", "properties": {"lines": {"type": "array", "items": {
    "type": "object", "properties": {
        "id": {"type": "integer"},
        "candidates": {"type": "array", "items": {"type": "object", "properties": {
            "text": {"type": "string"},
            "terms": {"type": "array", "items": {"type": "object", "properties": {
                "src": {"type": "string"}, "tgt": {"type": "string"}}, "required": ["src", "tgt"], "additionalProperties": False}}},
            "required": ["text", "terms"], "additionalProperties": False}}},
    "required": ["id", "candidates"], "additionalProperties": False}}}, "required": ["lines"], "additionalProperties": False}

JUDGE_SCHEMA = {"type": "object", "properties": {"lines": {"type": "array", "items": {
    "type": "object", "properties": {"id": {"type": "integer"}, "scores": {"type": "array", "items": {"type": "number"}}},
    "required": ["id", "scores"], "additionalProperties": False}}}, "required": ["lines"], "additionalProperties": False}


def _length_hint(n: int, lo: int, hi: int, lang: str) -> str:
    if lang in ("zh", "yue"):
        return f"{n} Chinese characters ({lo}-{hi}; every character is one syllable, numbers as read out)"
    if lang == "ja":
        return f"{n} morae ({lo}-{hi}; count kana as read)"
    if lang == "ko":
        return f"{n} Hangul syllable blocks ({lo}-{hi})"
    spw = SYLLABLES_PER_WORD.get(lang, 1.6)
    return f"{n} spoken syllables ({lo}-{hi}), roughly {max(1, round(n / spw))} words"


def gen_system(cfg: IsoConfig) -> str:
    src, tgt = LANG_NAMES.get(base_lang(cfg.source), cfg.source), LANG_NAMES.get(cfg.target.lower(), cfg.target)
    parts = [
        f"You are a dubbing translator. You translate spoken {src} into {tgt} that a voice actor will speak over "
        "the original video, replacing the original voice. Every line has to fit the picture:",
        "1. LENGTH - the line must take as long to say as the original: hit the syllable budget given for each line "
        "(count what is spoken: numbers as read out). Shorten by dropping fillers and redundancy; lengthen only by saying "
        "the same thing more fully (fuller phrasing, repeating the subject, a natural discourse marker such as 'well' or "
        "'you know'). Never drop meaning-bearing content and never invent facts, opinions or details that are not in the "
        "original.",
        "2. ANCHOR ORDER - each listed anchor (a noun, name or number) must be heard at about the same point of the line "
        "as in the original (`heard_at`, as a share of the line). Rewrite the grammar to get there instead of letting "
        f"{tgt} default word order move the anchor into the other half: front or delay the phrase, use passive or "
        "topic-comment structure, apposition, a short lead-in, or split into two short clauses.",
        "3. NATURAL SPEECH - idiomatic, spoken, faithful in meaning, tone and register; no notes, brackets or alternatives "
        "inside a line. Keep names, brands and numbers correct.",
        f"Write {cfg.candidates} clearly different candidates per line (different sentence structures, not synonym swaps). "
        "For every candidate list `terms`: each anchor (`src`, exactly as given) and its translation (`tgt`) exactly as it "
        "appears in your text.",
        "Context lines belong to other lines (already dubbed separately): use them only to understand this line, never "
        "move their content into it. If a line cannot reach its budget without inventing content, stay a little short "
        "- the voice is slowed down slightly to fit.",
    ]
    if cfg.instructions:
        parts.append("Style requirements: " + cfg.instructions)
    if cfg.glossary:
        parts.append("Glossary (source => target), use consistently:\n" + "\n".join(f"- {k} => {v}" for k, v in cfg.glossary.items()))
    parts.append('Reply with JSON only: {"lines": [{"id": <id>, "candidates": [{"text": "...", '
                 '"terms": [{"src": "...", "tgt": "..."}]}]}]}')
    return "\n\n".join(parts)


def _problems(sent: Dict, c: Dict) -> List[str]:
    out = []
    t = sent["target"]
    if c["syllables"] > t["hi"]:
        out.append(f"too long: {c['syllables']:g} syllables, budget {t['syllables']} ({t['lo']}-{t['hi']}) - cut "
                   f"about {c['syllables'] - t['syllables']:g}")
    elif c["syllables"] < t["lo"]:
        out.append(f"too short: {c['syllables']:g} syllables, budget {t['syllables']} ({t['lo']}-{t['hi']}) - add "
                   f"about {t['syllables'] - c['syllables']:g}")
    for a in c["anchors"]:
        if a["missing"]:
            out.append(f"anchor '{a['src']}' is missing (or not listed in terms)")
        elif not a["ok"]:
            out.append(f"'{a['tgt']}' is heard at {round(a['tgt_pos'] * 100)}% but '{a['src']}' is at "
                       f"{round(a['src_pos'] * 100)}% in the original - move it into the {_zone(a['src_pos'])}")
    return out


def _line_payload(sent: Dict, cfg: IsoConfig, hint: str = "", feedback: bool = False) -> Dict:
    t = sent["target"]
    d: Dict[str, Any] = {"id": sent["id"], "text": sent["text"], "seconds": sent.get("duration"),
                         "budget": _length_hint(t["syllables"], t["lo"], t["hi"], base_lang(cfg.target))}
    if sent.get("anchors"):
        d["anchors"] = [{"src": a["text"], "heard_at": f"{round(a['pos'] * 100)}%",
                         "place": f"{_zone(a['pos'])}, around {round(a['pos'] * 100)}%"} for a in sent["anchors"]]
    if hint:
        d["note"] = hint
    if feedback:
        prev = sorted(sent.get("candidates") or [], key=final_cost)[:3]
        d["previous_attempts"] = [{"text": c["text"], "problems": _problems(sent, c) or ["ok"]} for c in prev]
        d["task"] = ("The previous attempts miss the constraints - write new candidates that fix the problems listed, "
                     "translating only this line's own content.")
    return d


def _parse_candidates(data: Any) -> Dict[int, List[Dict]]:
    items = data.get("lines", data) if isinstance(data, dict) else data
    out: Dict[int, List[Dict]] = {}
    for it in items or []:
        try:
            sid = int(it.get("id"))
        except (TypeError, ValueError, AttributeError):
            continue
        cands = []
        for c in it.get("candidates") or []:
            if isinstance(c, str):
                c = {"text": c, "terms": []}
            text = str(c.get("text", "")).strip() if isinstance(c, dict) else ""
            if text:
                terms = [t for t in c.get("terms") or [] if isinstance(t, dict)]
                cands.append({"text": text, "terms": terms})
        out[sid] = cands
    return out


def _generate(batch: List[Dict], all_sents: List[Dict], client: LLMClient, cfg: IsoConfig, rnd: int,
              hints: Dict[int, str]) -> None:
    pos = {s["id"]: i for i, s in enumerate(all_sents)}
    i0, i1 = pos[batch[0]["id"]], pos[batch[-1]["id"]]
    payload = {"context_before": [s["text"] for s in all_sents[max(0, i0 - cfg.context):i0]],
               "lines": [_line_payload(s, cfg, hints.get(s["id"], ""), feedback=rnd > 0) for s in batch],
               "context_after": [s["text"] for s in all_sents[i1 + 1:i1 + 1 + cfg.context]]}
    user = "Translate the `lines`:\n" + json.dumps(payload, ensure_ascii=False)
    system = gen_system(cfg)
    got: Dict[int, List[Dict]] = {}
    try:
        got = _parse_candidates(client.complete_json(system, user, GEN_SCHEMA))
    except (LLMError, ValueError, TypeError, AttributeError) as e:
        log.warning("dub translation batch failed: %s", e)
    for s in batch:
        cands = got.get(s["id"])
        if not cands:                                       # retry this line alone
            try:
                one = {"lines": [_line_payload(s, cfg, hints.get(s["id"], ""), feedback=rnd > 0)]}
                cands = _parse_candidates(client.complete_json(system, "Translate the `lines`:\n" + json.dumps(
                    one, ensure_ascii=False), GEN_SCHEMA)).get(s["id"], [])
            except Exception as e:
                s["error"] = f"{type(e).__name__}: {e}"[:300]
                continue
        seen = {c["text"] for c in s.get("candidates", [])}
        for c in cands:
            if c["text"] in seen or not in_language(c["text"], cfg.target):
                continue
            seen.add(c["text"])
            m = measure(s, c["text"], c["terms"], cfg)
            if not -0.5 <= m["len_dev"] <= 0.6:             # merged a neighbouring line / dropped half of it
                s["discarded"] = s.get("discarded", 0) + 1
                continue
            m["round"] = rnd
            s.setdefault("candidates", []).append(m)
        if cands:
            s.pop("error", None)
        rank(s)


def _judge(batch: List[Dict], client: LLMClient, cfg: IsoConfig, retry: bool = True) -> None:
    lines, picks = [], {}
    for s in batch:
        cands = s.get("candidates") or []
        if len(cands) < 2:
            continue
        top = sorted(range(len(cands)), key=lambda i: cands[i]["cost"])[:cfg.finalists]
        if len(top) < 2:
            continue
        picks[s["id"]] = top
        lines.append({"id": s["id"], "original": s["text"], "candidates": [cands[i]["text"] for i in top]})
    if not lines:
        return
    tgt = LANG_NAMES.get(cfg.target.lower(), cfg.target)
    system = (f"You review {tgt} dubbing lines. All candidates already fit the timing; score each from 1 to 10 for "
              "faithfulness to the original and how natural it sounds when spoken aloud by a native speaker "
              "(10 = faithful and completely natural). Be strict about awkward word order, unnatural phrasing and "
              "lost meaning; score 1 when a candidate adds information that is not in the original.\n"
              'Reply with JSON only: {"lines": [{"id": <id>, "scores": [<score per candidate, same order>]}]}')
    data: Any = {}
    try:
        data = client.complete_json(system, json.dumps({"lines": lines}, ensure_ascii=False), JUDGE_SCHEMA)
    except (LLMError, ValueError, TypeError) as e:
        log.warning("judging failed: %s", e)
    by_id = {s["id"]: s for s in batch}
    scored = set()
    for it in (data.get("lines", data) if isinstance(data, dict) else data) or []:
        try:
            sid, scores = int(it.get("id")), list(it.get("scores") or [])
        except (TypeError, ValueError, AttributeError):
            continue
        if sid not in picks or len(scores) != len(picks[sid]):
            continue
        for i, sc in zip(picks[sid], scores):
            try:
                by_id[sid]["candidates"][i]["judge"] = max(1.0, min(10.0, float(sc)))
                scored.add(sid)
            except (TypeError, ValueError):
                pass
        rank(by_id[sid])
    if retry:                                               # small models skip ids in long batches
        for sid in picks:
            if sid not in scored:
                _judge([by_id[sid]], client, cfg, retry=False)


def detect_anchors(sents: List[Dict], client: Optional[LLMClient], cfg: IsoConfig) -> None:
    """Anchors of every sentence, then :func:`analyze`.  The LLM marks them (POS taggers
    are unreliable on speech: jieba tags 超市 as a verb); jieba / a heuristic is the fallback."""
    src = base_lang(cfg.source)
    for b0 in range(0, len(sents), 12):
        batch = sents[b0:b0 + 12]
        found: Dict[int, List[Dict]] = {}
        if client is not None:
            try:
                found = llm_anchors(batch, client, src, cfg.max_anchors)
            except Exception as e:
                log.warning("anchor detection failed: %s", e)
        for s in batch:
            if s["id"] in found:
                analyze(s, cfg, found[s["id"]])
            elif src in ("zh", "yue"):
                analyze(s, cfg, zh_anchors(s["text"], cfg.max_anchors))
            else:
                analyze(s, cfg, heuristic_anchors(s["text"], cfg.max_anchors))


def translate(sents: List[Dict], client: LLMClient, cfg: IsoConfig, ids: Optional[Sequence[int]] = None,
              hints: Optional[Dict[int, str]] = None,
              progress: Optional[Callable[[str, int, int], None]] = None) -> List[Dict]:
    """Translate (``ids``: only these sentences, adding to their candidates).  ``progress(stage, done, total)``
    is called after every batch; the sentence dicts are updated in place."""
    hints = hints or {}
    todo = [s for s in sents if ids is None or s["id"] in set(ids)]
    note = progress or (lambda *_: None)
    pending = [s for s in todo if s.get("anchors") is None]
    if pending:
        note("anchors", 0, len(todo))
        detect_anchors(pending, client, cfg)
    bs = max(1, cfg.batch_size)
    for b0 in range(0, len(todo), bs):
        _generate(todo[b0:b0 + bs], sents, client, cfg, 0, hints)
        note("translate", min(len(todo), b0 + bs), len(todo))
    for r in range(1, cfg.rounds + 1):
        bad = [s for s in todo if not (best(s) or {}).get("ok")]
        if not bad:
            break
        for b0 in range(0, len(bad), bs):
            _generate(bad[b0:b0 + bs], sents, client, cfg, r, hints)
            note(f"refine{r}", min(len(bad), b0 + bs), len(bad))
    if cfg.judge:
        for b0 in range(0, len(todo), bs * 2):
            _judge(todo[b0:b0 + bs * 2], client, cfg)
            note("judge", min(len(todo), b0 + bs * 2), len(todo))
    return sents


# ------------------------------------------------------------------ export
def _srt_time(t: float) -> str:
    ms = int(round(max(0.0, t) * 1000))
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


def to_srt(sents: Sequence[Dict], bilingual: bool = False) -> str:
    out = []
    for i, s in enumerate(x for x in sents if x.get("start") is not None):
        tr = s.get("translation") or ""
        body = (tr + "\n" + s["text"]) if bilingual else tr
        out.append(f"{i + 1}\n{_srt_time(s['start'])} --> {_srt_time(s['end'])}\n{body}\n")
    return "\n".join(out)
