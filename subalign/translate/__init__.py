"""LLM subtitle / lyric translation.

Lines are sent in numbered batches together with a few lines of surrounding
context; the model must return exactly one translation per id, so timing is
never disturbed.  Missing ids are retried individually.
"""
from __future__ import annotations

import json
from typing import Callable, Dict, List, Optional, Sequence

from ..llm import LLMClient, LLMError
from ..models import Document

LANG_NAMES = {
    "zh": "Simplified Chinese", "zh-cn": "Simplified Chinese", "zh-tw": "Traditional Chinese",
    "zh-hant": "Traditional Chinese", "en": "English", "ja": "Japanese", "ko": "Korean", "fr": "French",
    "de": "German", "es": "Spanish", "pt": "Portuguese", "ru": "Russian", "it": "Italian", "vi": "Vietnamese",
    "th": "Thai", "id": "Indonesian", "ar": "Arabic", "yue": "Cantonese (written)",
}

SCHEMA = {
    "type": "object",
    "properties": {"translations": {"type": "array", "items": {
        "type": "object",
        "properties": {"id": {"type": "integer"}, "text": {"type": "string"}},
        "required": ["id", "text"], "additionalProperties": False}}},
    "required": ["translations"],
    "additionalProperties": False,
}


def _system_prompt(target: str, kind: str, glossary: Optional[Dict[str, str]], style: str, max_units: Optional[int]) -> str:
    lang = LANG_NAMES.get(target.lower(), target)
    what = "song lyrics" if kind == "song" else "video subtitles"
    parts = [
        f"You translate {what} into {lang}.",
        "Each input line has an integer id and is displayed on screen at its own time, so translate line by "
        "line: return exactly one translation for every id, in the same order, never merging or splitting ids. "
        "A sentence may continue across several ids - keep the meaning distributed the same way.",
        "Keep translations concise enough to read at subtitle speed. Preserve names, numbers and tone.",
    ]
    if kind == "song":
        parts.append("For lyrics, favour natural, singable phrasing and keep imagery and emotion; do not add "
                     "explanations. Repeated lines (choruses) should be translated consistently.")
    if max_units:
        parts.append(f"Each translation should fit in about {max_units} half-width characters "
                     f"(a CJK character counts as 2) when possible.")
    if style:
        parts.append(f"Style requirements: {style}")
    if glossary:
        parts.append("Use this glossary (source => target) consistently:\n" +
                     "\n".join(f"- {k} => {v}" for k, v in glossary.items()))
    parts.append('Reply with JSON only: {"translations": [{"id": <id>, "text": "<translation>"}, ...]}')
    return "\n\n".join(parts)


def translate_texts(texts: Sequence[str], client: LLMClient, target: str, kind: str = "speech",
                    batch_size: int = 40, context: int = 3, glossary: Optional[Dict[str, str]] = None,
                    style: str = "", max_units: Optional[int] = None,
                    progress: Optional[Callable[[int, int], None]] = None) -> List[str]:
    system = _system_prompt(target, kind, glossary, style, max_units)
    out: List[Optional[str]] = [None] * len(texts)
    for b0 in range(0, len(texts), batch_size):
        b1 = min(len(texts), b0 + batch_size)
        payload = {
            "context_before": [texts[i] for i in range(max(0, b0 - context), b0)],
            "lines": [{"id": i, "text": texts[i]} for i in range(b0, b1)],
            "context_after": [texts[i] for i in range(b1, min(len(texts), b1 + context))],
        }
        user = ("Translate the `lines` (context lines are for reference only, do not translate them):\n"
                + json.dumps(payload, ensure_ascii=False))
        try:
            data = client.complete_json(system, user, SCHEMA)
            items = data.get("translations", data) if isinstance(data, dict) else data
            for it in items:
                i = int(it.get("id", -1))
                if b0 <= i < b1 and isinstance(it.get("text"), str):
                    out[i] = it["text"].strip()
        except (LLMError, ValueError, TypeError, AttributeError):
            pass
        # retry anything missing one-by-one
        for i in range(b0, b1):
            if out[i] is None and texts[i].strip():
                try:
                    data = client.complete_json(system, user=json.dumps(
                        {"lines": [{"id": i, "text": texts[i]}]}, ensure_ascii=False), schema=SCHEMA)
                    items = data.get("translations", data) if isinstance(data, dict) else data
                    out[i] = str(items[0]["text"]).strip()
                except Exception:
                    out[i] = ""
        if progress:
            progress(b1, len(texts))
    return [o or "" for o in out]


def translate_document(doc: Document, client: LLMClient, target: str, **kw) -> Document:
    texts = [ln.text for ln in doc.lines]
    res = translate_texts(texts, client, target, kind=doc.kind, **kw)
    for ln, tr in zip(doc.lines, res):
        ln.translation = tr or None
    doc.metadata["translation_language"] = target
    return doc
