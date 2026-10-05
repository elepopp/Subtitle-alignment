"""Automatic proofreading (自动校对).

* :func:`diff_report` - script vs. what was actually said/sung.  Uses the
  phonetic aligner, so homophone ASR errors are reported as low-severity
  substitutions while real deviations (skipped / added / changed words) are
  flagged with their timestamps.
* :func:`llm_proofread` - no script: an LLM fixes recognition errors
  (homophones, terms, punctuation) line by line; corrected text is then
  re-aligned onto the original token timings, so timestamps survive.
* :func:`apply_glossary`, :func:`remove_fillers` - deterministic rules.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence

from ..align.sequence import AlignOp, align_tokens, error_rate, transfer_times
from ..align.timing import fill_missing_times
from ..llm import LLMClient, LLMError
from ..models import Document, Line, Token
from ..text.tokenize import tokenize


@dataclass
class Issue:
    kind: str            # substitution | missing (in audio) | extra (in audio)
    script: str
    heard: str
    start: Optional[float]
    end: Optional[float]
    line: int
    severity: str        # low (homophone-like) | high


def diff_report(script_lines: Sequence[Line], hyp_tokens: Sequence[Token],
                ops: Optional[List[AlignOp]] = None) -> Dict:
    ref = [t for ln in script_lines for t in ln.tokens]
    line_of = [i for i, ln in enumerate(script_lines) for _ in ln.tokens]
    ops = ops if ops is not None else align_tokens(ref, hyp_tokens)
    issues: List[Issue] = []
    run: List[AlignOp] = []

    def flush():
        if not run:
            return
        s_txt = "".join(ref[o.ref].text for o in run if o.ref is not None)
        h_txt = "".join(hyp_tokens[o.hyp].text for o in run if o.hyp is not None)
        times = [(hyp_tokens[o.hyp].start, hyp_tokens[o.hyp].end) for o in run if o.hyp is not None]
        if not times:
            times = [(ref[o.ref].start, ref[o.ref].end) for o in run if o.ref is not None]
        kinds = {o.op for o in run}
        kind = "substitution" if "sub" in kinds or kinds == {"del", "ins"} else (
            "missing" if kinds == {"del"} else "extra")
        sev = "low" if kinds == {"sub"} and min(o.sim for o in run) >= 0.5 else "high"
        lref = [line_of[o.ref] for o in run if o.ref is not None]
        line = lref[0] if lref else (line_of[_nearest_ref(ops, run[0])] if ref else 0)
        issues.append(Issue(kind, s_txt, h_txt, times[0][0], times[-1][1], line, sev))
        run.clear()

    for o in ops:
        if o.op == "match":
            flush()
        else:
            run.append(o)
    flush()
    return {"error_rate": round(error_rate(ops), 4),
            "issues": [asdict(i) for i in issues],
            "n_high": sum(1 for i in issues if i.severity == "high")}


def _nearest_ref(ops: Sequence[AlignOp], op: AlignOp) -> int:
    idx = ops.index(op)
    for k in range(idx, -1, -1):
        if ops[k].ref is not None:
            return ops[k].ref
    for k in range(idx, len(ops)):
        if ops[k].ref is not None:
            return ops[k].ref
    return 0


def format_report(rep: Dict) -> str:
    from ..formats.common import ts_vtt

    rows = [f"# Proofreading report\n\nerror rate: {rep['error_rate']:.2%}, "
            f"{len(rep['issues'])} differences ({rep['n_high']} significant)\n"]
    for it in rep["issues"]:
        t = ts_vtt(it["start"]) if it["start"] is not None else "--:--:--"
        mark = "!" if it["severity"] == "high" else "~"
        rows.append(f"{mark} [{t}] line {it['line'] + 1} {it['kind']}: script「{it['script']}」 heard「{it['heard']}」")
    return "\n".join(rows) + "\n"


# --------------------------------------------------------------------------- rules
def apply_glossary(doc: Document, glossary: Dict[str, str]) -> Document:
    """Replace terms (exact text, or /regex/ keys) and keep timing via realignment."""
    def fix(text: str) -> str:
        for k, v in glossary.items():
            if len(k) > 2 and k.startswith("/") and k.endswith("/"):
                text = re.sub(k[1:-1], v, text)
            else:
                text = text.replace(k, v)
        return text

    return _rewrite_lines(doc, [fix(ln.text) for ln in doc.lines])


FILLERS = {
    "zh": ["嗯", "呃", "额", "啊", "那个", "就是说"],
    "en": ["um", "uh", "erm", "uhm", "you know"],
}


def remove_fillers(doc: Document, language: str = "zh", extra: Sequence[str] = ()) -> Document:
    words = set(FILLERS.get((language or "zh")[:2], []) + list(extra))
    for ln in doc.lines:
        keep = []
        for t in ln.tokens:
            k = re.sub(r"[^\w]", "", t.text.lower())
            if k in words:
                continue
            keep.append(t)
        ln.tokens = keep
    doc.lines = [ln for ln in doc.lines if ln.tokens]
    for ln in doc.lines:
        ln.update_bounds()
    return doc


# --------------------------------------------------------------------------- LLM
PROOF_SCHEMA = {
    "type": "object",
    "properties": {"lines": {"type": "array", "items": {
        "type": "object", "properties": {"id": {"type": "integer"}, "text": {"type": "string"}},
        "required": ["id", "text"], "additionalProperties": False}}},
    "required": ["lines"], "additionalProperties": False,
}


def llm_proofread(doc: Document, client: LLMClient, context: str = "", glossary: Optional[Dict[str, str]] = None,
                  batch_size: int = 60, punctuate: bool = True, verbatim: bool = False) -> Document:
    what = "song lyrics recognised from singing" if doc.kind == "song" else "speech transcribed by ASR"
    system = (
        f"You proofread {what}. Fix recognition errors only: wrong homophones or near-homophones, "
        "misheard words, wrong terms/names, and " + ("punctuation" if punctuate else "nothing else") + ". "
        "Do NOT paraphrase, summarise, reorder, translate or censor; keep the speaker's wording and language. "
        "Return one line per id, same ids, same order. "
        'Reply with JSON only: {"lines": [{"id": <id>, "text": "<corrected>"}]}'
    )
    if verbatim:
        system += ("\n\nThis is a verbatim transcript: keep fillers (嗯, 呃, um, uh), repetitions, false starts "
                   "and unfinished words exactly as they are.")
    if context:
        system += f"\n\nTopic / background: {context}"
    if glossary:
        system += "\n\nCorrect spellings of key terms:\n" + "\n".join(f"- {v}" for v in dict.fromkeys(glossary.values()))
    new_texts = [ln.text for ln in doc.lines]
    for b0 in range(0, len(doc.lines), batch_size):
        b1 = min(len(doc.lines), b0 + batch_size)
        user = json.dumps({"lines": [{"id": i, "text": doc.lines[i].text} for i in range(b0, b1)]}, ensure_ascii=False)
        try:
            data = client.complete_json(system, user, PROOF_SCHEMA)
        except LLMError:
            continue
        for it in (data.get("lines", []) if isinstance(data, dict) else []):
            try:
                i = int(it["id"])
            except (KeyError, ValueError, TypeError):
                continue
            if b0 <= i < b1 and isinstance(it.get("text"), str) and it["text"].strip():
                new_texts[i] = it["text"].strip()
    return _rewrite_lines(doc, new_texts)


def _rewrite_lines(doc: Document, texts: Sequence[str]) -> Document:
    """Replace each line's text, transferring timings from the old tokens."""
    for ln, txt in zip(doc.lines, texts):
        if txt == ln.text:
            continue
        new_toks = tokenize(txt)
        if not new_toks:
            continue
        start, end = ln.start, ln.end
        transfer_times(new_toks, ln.tokens, interpolate=False)
        # interpolate untimed tokens inside the original cue (line-level input such as
        # SRT has no token times at all), and never move the cue itself
        bounds = []
        if start is not None:
            bounds.append(Token("", start, start))
        bounds += new_toks
        if end is not None:
            bounds.append(Token("", end, end))
        fill_missing_times(bounds)
        ln.tokens = new_toks
        ln.update_bounds()
        if start is not None and end is not None:
            ln.start, ln.end = start, end
    return doc
