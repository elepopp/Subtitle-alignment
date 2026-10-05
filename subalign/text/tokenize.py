"""Script-aware tokenisation, normalisation and display-width helpers."""
from __future__ import annotations

import re
import unicodedata
from typing import List, Optional

from ..models import Line, Token

# Characters that are timed one-per-unit.
_CJK_RANGES = (
    (0x3040, 0x30FF),   # hiragana + katakana
    (0x31F0, 0x31FF),   # katakana phonetic ext
    (0x3400, 0x4DBF),   # CJK ext A
    (0x4E00, 0x9FFF),   # CJK unified
    (0xF900, 0xFAFF),   # CJK compat
    (0xAC00, 0xD7AF),   # hangul syllables
    (0x20000, 0x2FA1F),  # CJK ext B..
)

# Punctuation that attaches to the *following* token (opening brackets / quotes).
OPENING_PUNCT = set("([{（［｛〔【《〈「『“‘«‹¿¡")
# Sentence-final punctuation (strong break points).
SENTENCE_END = set(".!?。！？…‼⁇⁈⁉")
# Clause punctuation (weak break points).
CLAUSE_END = set(",;:，、；：—～~")

_WORD_RE = re.compile(r"[^\W_]+(?:['’\-][^\W_]+)*", re.UNICODE)


def is_cjk(ch: str) -> bool:
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in _CJK_RANGES)


def is_punct(ch: str) -> bool:
    cat = unicodedata.category(ch)
    return cat.startswith("P") or cat.startswith("S") or ch in "…～~"


def char_width(ch: str) -> int:
    """Display width in half-width units (CJK / full-width = 2)."""
    if unicodedata.combining(ch):
        return 0
    return 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1


def text_width(s: str) -> int:
    return sum(char_width(c) for c in s)


def normalize_key(s: str) -> str:
    """Matching key: NFKC, lowercase, without punctuation or whitespace."""
    s = unicodedata.normalize("NFKC", s).lower()
    return "".join(c for c in s if not (is_punct(c) or c.isspace()))


def tokenize(text: str) -> List[Token]:
    """Split text into timing units.

    * CJK ideographs, kana and hangul -> one token per character
    * alphabetic / numeric runs -> one token per word
    * punctuation is attached to the neighbouring token (opening punctuation to
      the next token, everything else to the previous token)
    """
    text = unicodedata.normalize("NFC", text)
    tokens: List[Token] = []
    pending_prefix = ""
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch.isspace():
            if tokens:
                tokens[-1].space_after = True
            i += 1
            continue
        if is_cjk(ch):
            tokens.append(Token(pending_prefix + ch))
            pending_prefix = ""
            i += 1
            continue
        m = _WORD_RE.match(text, i)
        if m and not is_cjk(m.group(0)[0]):
            word = m.group(0)
            # stop the word at the first CJK char (mixed runs like "abc中文")
            cut = next((k for k, c in enumerate(word) if is_cjk(c)), len(word))
            word = word[:cut]
            tokens.append(Token(pending_prefix + word))
            pending_prefix = ""
            i += len(word)
            continue
        # punctuation / symbol
        if ch in OPENING_PUNCT or not tokens or tokens[-1].space_after:
            pending_prefix += ch
        else:
            tokens[-1].text += ch
        i += 1
    if pending_prefix:
        if tokens:
            tokens[-1].text += pending_prefix
        else:
            tokens.append(Token(pending_prefix))
    # tokens consisting solely of punctuation (e.g. a lone "♪") have no key;
    # they are kept but never timed independently.
    if tokens:
        tokens[-1].space_after = False
    return tokens


def token_key(tok: Token) -> str:
    return normalize_key(tok.text)


def syllable_weight(tok: Token) -> float:
    """Rough expected relative duration of a token (1.0 == one CJK syllable)."""
    key = token_key(tok)
    if not key:
        return 0.3
    if all(is_cjk(c) for c in key):
        return float(len(key))
    if key.isdigit():
        return max(1.0, 0.9 * len(key))
    # syllables ~ vowel groups for alphabetic words
    groups = re.findall(r"[aeiouyàáâãäåèéêëìíîïòóôõöùúûüý]+", key)
    n = len(groups)
    if key.endswith("e") and n > 1 and not key.endswith(("le", "ee")):
        n -= 1
    return float(max(1, n))


def make_line(text: str, **kw) -> Line:
    return Line(tokens=tokenize(text), **kw)


def lines_from_text(text: str) -> List[Line]:
    """Plain text (one subtitle / lyric line per text line) -> untimed lines."""
    out = []
    for raw in text.splitlines():
        raw = raw.strip()
        if not raw:
            continue
        ln = make_line(raw)
        if ln.tokens:
            out.append(ln)
    return out


def ends_sentence(tok: Optional[Token]) -> bool:
    return bool(tok and tok.text and tok.text[-1] in SENTENCE_END)


def ends_clause(tok: Optional[Token]) -> bool:
    return bool(tok and tok.text and tok.text[-1] in CLAUSE_END)


def strip_punct(text: str, mode: str = "keep") -> str:
    """Subtitle punctuation policy: keep | strip | space (CJK style)."""
    if mode == "keep":
        return text
    out = []
    for c in text:
        if c in SENTENCE_END or c in CLAUSE_END:
            if mode == "space" and c not in "…" and out and not out[-1].isspace():
                out.append(" ")
            # strip: drop it; but keep '?' '!' which carry meaning
            if c in "?？!！":
                if out and out[-1] == " ":
                    out.pop()
                out.append(c)
            continue
        out.append(c)
    return "".join(out).strip()
