"""Phonetic similarity used when matching ASR output against a script.

ASR errors in Chinese are dominated by homophones / near-homophones
(e.g. 在/再, 他/她, zh/z, n/l, -in/-ing), so a pinyin-aware substitution cost
gives much better alignments (and proofreading diffs) than plain character
equality.  ``pypinyin`` is optional; without it we fall back to string
similarity.
"""
from __future__ import annotations

from difflib import SequenceMatcher
from functools import lru_cache
from typing import Optional, Tuple

from .tokenize import is_cjk, normalize_key

try:  # optional dependency
    from pypinyin import Style, lazy_pinyin  # type: ignore

    _HAS_PINYIN = True
except Exception:  # pragma: no cover - optional
    _HAS_PINYIN = False

_INITIALS = ("zh", "ch", "sh", "b", "p", "m", "f", "d", "t", "n", "l", "g", "k", "h",
             "j", "q", "x", "r", "z", "c", "s", "y", "w")
# fuzzy pinyin pairs commonly confused by speakers and recognisers
_FUZZY_INITIALS = {frozenset(p) for p in (("zh", "z"), ("ch", "c"), ("sh", "s"), ("n", "l"),
                                          ("f", "h"), ("r", "l"), ("j", "z"), ("q", "c"), ("x", "s"))}
_FUZZY_FINALS = {frozenset(p) for p in (("in", "ing"), ("en", "eng"), ("an", "ang"),
                                        ("ian", "iang"), ("uan", "uang"), ("on", "ong"))}


def has_pinyin() -> bool:
    return _HAS_PINYIN


@lru_cache(maxsize=65536)
def pinyin_of(ch: str) -> Optional[str]:
    if not _HAS_PINYIN or not ch or not is_cjk(ch[0]):
        return None
    try:
        return lazy_pinyin(ch, style=Style.NORMAL, errors="ignore")[0]
    except Exception:
        return None


def _split_pinyin(py: str) -> Tuple[str, str]:
    for ini in _INITIALS:
        if py.startswith(ini):
            return ini, py[len(ini):]
    return "", py


def pinyin_similarity(a: str, b: str) -> float:
    if a == b:
        return 1.0
    ia, fa = _split_pinyin(a)
    ib, fb = _split_pinyin(b)
    ini = 1.0 if ia == ib else (0.7 if frozenset((ia, ib)) in _FUZZY_INITIALS else 0.0)
    fin = 1.0 if fa == fb else (0.7 if frozenset((fa, fb)) in _FUZZY_FINALS else 0.0)
    return 0.4 * ini + 0.6 * fin


@lru_cache(maxsize=262144)
def unit_similarity(a: str, b: str) -> float:
    """Similarity in [0, 1] between two token keys (already normalised)."""
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0
    if len(a) == 1 and len(b) == 1 and is_cjk(a) and is_cjk(b):
        pa, pb = pinyin_of(a), pinyin_of(b)
        if pa and pb:
            # identical pinyin => homophone; scale so that it never beats an exact match
            return 0.9 * pinyin_similarity(pa, pb)
        return 0.0
    return SequenceMatcher(None, a, b, autojunk=False).ratio() * 0.9


def key_similarity(a: str, b: str) -> float:
    return unit_similarity(normalize_key(a), normalize_key(b))
