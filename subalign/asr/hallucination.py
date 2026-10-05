"""Screening of ASR hallucinations (Whisper writing text that was never said).

Typical cases: subtitle credits / outro phrases learnt from training data
("字幕由…提供", "请不吝点赞 订阅", "优优独播剧场——YoYo Television Series
Exclusive", "Thanks for watching") over music or silence, looped repetitions,
and text placed where there is no voice at all.

Every segment collects evidence; no single signal removes anything:

=====================================  =====
credit / channel phrase (never spoken)  +3 (removed on its own)
outro phrase a person might also say    +2
no / little voice in its time span      +3 (< 5 %) / +2 (< 20 %) / +1 (< 35 %)
looped repetition inside the segment    +2 (speech only - songs repeat)
same text as the previous segment       +1 (speech only)
implausible speaking rate               +1
very low decoder confidence             +1
=====================================  =====

score >= 3: removed; score == 2: kept but marked for review.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..text.tokenize import normalize_key
from .base import Segment, Transcript

# subtitle credits / channel promos from the training data: never actually spoken in the audio
STRONG_PATTERNS = [
    r"字幕(由|提供|制作|组|校对)", r"(中文|双语)?字幕\s*(by|:|：)", r"不吝点赞", r"点赞.{0,6}(订阅|转发|打赏)",
    r"订阅.{0,6}(频道|转发|打赏)", r"明镜(与|和)点点", r"优优独播剧场", r"yoyo\s*television", r"amara\.org",
    r"subtitles?\s+by", r"please\s+subscribe", r"请订阅",
]
# outro phrases a speaker might really say: need a second signal
WEAK_PATTERNS = [
    r"(thanks|thank you)\s+for\s+watching", r"谢谢(大家)?(的)?(观看|收看)", r"感谢(您的)?(观看|收看)",
    r"下(期|集)(再见|见)", r"本(期)?视频(到此结束|就到这里)",
]
_STRONG = re.compile("|".join(STRONG_PATTERNS), re.IGNORECASE)
_WEAK = re.compile("|".join(WEAK_PATTERNS), re.IGNORECASE)


def _voice_ratio(feats, start: float, end: float) -> Optional[float]:
    if feats is None or end <= start:
        return None
    a, b = int(start / feats.hop_s), int(np.ceil(end / feats.hop_s))
    seg = feats.active[max(0, a):min(feats.n, b)]
    return float(np.mean(seg > 0.5)) if len(seg) else None


def _looped(key: str) -> bool:
    """A 2..8 unit pattern repeated 4+ times back to back."""
    for n in range(2, 9):
        if re.search(r"(.{%d})\1{3,}" % n, key):
            return True
    return False


def screen(transcript: Transcript, feats=None, song: bool = False) -> Tuple[Transcript, List[Dict[str, Any]]]:
    """Return the transcript without likely hallucinations, plus notes on every
    removed / suspicious segment (start, end, text, score, reasons, action)."""
    kept: List[Segment] = []
    notes: List[Dict[str, Any]] = []
    prev_key = None
    for seg in transcript.segments:
        text = seg.text.strip()
        key = normalize_key(text)
        if not key:
            continue
        score, why = 0, []
        if _STRONG.search(text):
            score += 3
            why.append("字幕署名 / 频道宣传语（训练数据残留）")
        elif _WEAK.search(text):
            score += 2
            why.append("常见片尾套话")
        vr = _voice_ratio(feats, seg.start, seg.end)
        if vr is not None:
            if vr < 0.05:
                score += 3
                why.append(f"该时段没有人声 ({vr:.0%})")
            elif vr < 0.2:
                score += 2
                why.append(f"该时段几乎没有人声 ({vr:.0%})")
            elif vr < 0.35:
                score += 1
                why.append(f"该时段人声很少 ({vr:.0%})")
        if not song:
            if len(key) >= 8 and _looped(key):
                score += 2
                why.append("循环重复")
            if prev_key is not None and key == prev_key and len(key) >= 4:
                score += 1
                why.append("与上一句完全相同")
        dur = max(1e-3, seg.end - seg.start)
        rate = len(key) / dur
        if (rate > 12 and len(key) > 6) or (dur > 8 and rate < 0.4):
            score += 1
            why.append(f"语速异常 ({rate:.1f} 字/秒)")
        if seg.avg_logprob is not None and seg.avg_logprob < -1.0 and (seg.no_speech_prob or 0) > 0.5:
            score += 1
            why.append("识别置信度很低")
        prev_key = key
        if score >= 3:
            notes.append({"start": seg.start, "end": seg.end, "text": text, "score": score, "reasons": why,
                          "action": "removed"})
            continue
        if score == 2:
            notes.append({"start": seg.start, "end": seg.end, "text": text, "score": score, "reasons": why,
                          "action": "flagged"})
            for w in seg.words:
                w.prob = min(w.prob, 0.3)
        kept.append(seg)
    return Transcript(kept, transcript.language), notes
