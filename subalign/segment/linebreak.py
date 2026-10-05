"""Layout-aware line breaking (横屏 / 竖屏).

Global optimum over all break positions (Knuth-Plass style DP), so a long
sentence is split where it reads best instead of greedily filling rows:

* hard limit: row width (display units) <= layout.max_units
* rewards breaking after sentence / clause punctuation and at audible pauses
* penalises breaking inside a word (jieba, when installed, gives Chinese
  word boundaries), orphans (1-2 unit rows), uneven rows, too-short / too-long
  cue durations and excessive reading speed (CPS)

Rows are then grouped into cues of up to ``layout.max_lines`` rows.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Set

from ..models import Document, Line, Token
from ..text.tokenize import (CLAUSE_END, SENTENCE_END, is_cjk, normalize_key, strip_punct, text_width)
from .layout import Layout

try:  # optional Chinese word segmentation
    import jieba  # type: ignore

    jieba.setLogLevel(60)
    _HAS_JIEBA = True
except Exception:  # pragma: no cover
    _HAS_JIEBA = False


def _word_internal_breaks(tokens: Sequence[Token]) -> Set[int]:
    """Indices i such that a break *after* token i would split a CJK word."""
    if not _HAS_JIEBA:
        return set()
    text = "".join(t.text for t in tokens)
    # map char offsets -> token index
    owner = []
    for i, t in enumerate(tokens):
        owner.extend([i] * len(t.text))
    bad = set()
    pos = 0
    for w in jieba.cut(text, HMM=False):
        a, b = pos, pos + len(w)
        if b - a > 1:
            toks = sorted({owner[k] for k in range(a, b) if k < len(owner)})
            for k in toks[:-1]:
                bad.add(k)
        pos = b
    return bad


def _row_width(tokens: Sequence[Token], i: int, j: int, punct: str) -> int:
    parts = []
    for k in range(i, j):
        parts.append(tokens[k].text)
        if tokens[k].space_after and k < j - 1:
            parts.append(" ")
    return text_width(strip_punct("".join(parts), punct))


def break_rows(tokens: Sequence[Token], layout: Layout, punct: str = "keep",
               speech: bool = True) -> List[List[Token]]:
    """Split a token sequence into display rows."""
    n = len(tokens)
    if n == 0:
        return []
    W = layout.max_units
    total = _row_width(tokens, 0, n, punct)
    if total <= W:
        return [list(tokens)]
    inner = _word_internal_breaks(tokens)
    # break-quality bonus/penalty after each token
    brk = [0.0] * n
    for i, t in enumerate(tokens):
        last = t.text[-1] if t.text else ""
        nxt = tokens[i + 1] if i + 1 < n else None
        if last in SENTENCE_END:
            brk[i] -= 4.0
        elif last in CLAUSE_END:
            brk[i] -= 2.5
        if nxt is not None and t.end is not None and nxt.start is not None:
            gap = nxt.start - t.end
            brk[i] -= min(3.0, 6.0 * max(0.0, gap - 0.08))
        if i in inner:
            brk[i] += 3.0
        if nxt is not None and not t.space_after and not (is_cjk(normalize_key(t.text)[-1:] or "a")
                                                          or is_cjk(normalize_key(nxt.text)[:1] or "a")):
            brk[i] += 50.0  # never split glued alphanumerics
    n_rows_min = -(-total // W)
    ideal = total / n_rows_min
    INF = float("inf")
    best = [INF] * (n + 1)
    back = [0] * (n + 1)
    best[0] = 0.0
    for j in range(1, n + 1):
        for i in range(j - 1, -1, -1):
            w = _row_width(tokens, i, j, punct)
            if w > W:
                break
            if best[i] == INF:
                continue
            c = 10.0                                   # per-row cost -> fewer rows
            c += 6.0 * ((w - ideal) / W) ** 2          # balanced rows
            units = j - i
            if units <= 2 and n > 4:
                c += 6.0                               # orphan
            if speech and tokens[i].start is not None and tokens[j - 1].end is not None:
                dur = tokens[j - 1].end - tokens[i].start
                if dur < layout.min_duration:
                    c += 3.0 * (layout.min_duration - dur)
                if dur > layout.max_duration:
                    c += 2.0 * (dur - layout.max_duration)
                cps = (w / 2.0) / max(dur, 0.2)
                if cps > layout.max_cps:
                    c += 0.5 * (cps - layout.max_cps)
            if j < n:
                c += brk[j - 1]
            v = best[i] + c
            if v < best[j]:
                best[j], back[j] = v, i
    if best[n] == INF:  # a single token wider than the screen: hard split by count
        rows, cur, cw = [], [], 0
        for t in tokens:
            tw = text_width(t.text)
            if cur and cw + tw > W:
                rows.append(cur)
                cur, cw = [], 0
            cur.append(t)
            cw += tw + 1
        return rows + ([cur] if cur else [])
    cuts = []
    j = n
    while j > 0:
        cuts.append((back[j], j))
        j = back[j]
    cuts.reverse()
    return [list(tokens[a:b]) for a, b in cuts]


def _group_rows(rows: List[List[Token]], max_lines: int) -> List[List[List[Token]]]:
    """Pack rows into cues; a sentence end or a long pause always starts a new cue."""
    groups: List[List[List[Token]]] = []
    for r in rows:
        if groups and len(groups[-1]) < max_lines:
            prev = groups[-1][-1][-1]
            gap = (r[0].start - prev.end) if (r[0].start is not None and prev.end is not None) else 0.0
            if not (prev.text and prev.text[-1] in SENTENCE_END) and gap < 0.6:
                groups[-1].append(r)
                continue
        groups.append([r])
    return groups


def _line_from_rows(rows: List[List[Token]], src: Line) -> Line:
    toks: List[Token] = []
    for r in rows:
        if toks:
            toks[-1].space_after = toks[-1].space_after or not (is_cjk(normalize_key(toks[-1].text)[-1:] or "a"))
        toks.extend(r)
    ln = Line(tokens=toks, translation=None, style=src.style, speaker=src.speaker)
    ln.update_bounds()
    return ln


def segment_document(doc: Document, layout: Layout, punct: str = "keep", max_lines: Optional[int] = None,
                     split_translation: bool = True) -> Document:
    """Re-flow every line so no displayed row exceeds the layout width.

    Lines that fit are untouched.  Overlong lines become several consecutive
    lines (each keeps its exact token timing).  For speech, up to
    ``max_lines`` rows share one cue (``Line.breaks`` marks the row breaks,
    rendered by SRT / VTT / ASS writers); songs always get one row per line so
    karaoke highlighting stays readable.
    """
    max_lines = max_lines or layout.max_lines
    out: List[Line] = []
    speech = doc.kind != "song"
    for ln in doc.lines:
        rows = break_rows(ln.tokens, layout, punct, speech=speech)
        if len(rows) <= 1:
            out.append(ln)
            continue
        groups = _group_rows(rows, max_lines) if speech else [[r] for r in rows]
        new_lines = []
        for g in groups:
            nl = _line_from_rows(g, ln)
            acc, br = 0, []
            for r in g[:-1]:
                acc += len(r)
                br.append(acc - 1)
            nl.breaks = br
            new_lines.append(nl)
        if ln.translation:
            if split_translation:
                _distribute_translation(ln.translation, new_lines)
            else:
                new_lines[0].translation = ln.translation
        out.extend(new_lines)
    res = Document(lines=out, language=doc.language, kind=doc.kind, metadata=dict(doc.metadata))
    return res


def _distribute_translation(text: str, lines: List[Line]) -> None:
    """Split a translation across the pieces of a split line, proportionally to
    duration, preferring punctuation / spaces as cut points."""
    if len(lines) == 1:
        lines[0].translation = text
        return
    durs = [max(ln.duration, 0.1) for ln in lines]
    total = sum(durs)
    cuts = []
    acc = 0.0
    n = len(text)
    for d in durs[:-1]:
        acc += d
        target = int(round(n * acc / total))
        best, best_score = target, 1e9
        for k in range(max(1, target - 12), min(n - 1, target + 12) + 1):
            ch = text[k - 1]
            score = abs(k - target) - (8 if ch in "，。,.;；!?！？" else 4 if ch == " " else 0)
            if score < best_score:
                best, best_score = k, score
        cuts.append(best)
    pieces, prev = [], 0
    for c in cuts + [n]:
        c = max(c, prev)
        pieces.append(text[prev:c].strip())
        prev = c
    for ln, p in zip(lines, pieces):
        ln.translation = p or None
