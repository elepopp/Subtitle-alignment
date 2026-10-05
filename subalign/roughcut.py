"""Speech rough cut (语音粗剪): remove fillers, stutters / false starts, retaken
sentences and dead air, with joins that do not sound cut.

Pipeline
--------
1. transcript with per-unit timing: ASR prompted to keep disfluencies (Whisper
   normally drops 嗯/呃) + the acoustic DP refinement of :mod:`align.aligner`
2. detection (every decision is recorded in a *plan* that can be reviewed and
   edited, then rendered again with ``--plan``):

   * fillers   嗯 呃 额 / um uh; 啊 哦 only after a pause (else they are sentence
               particles: 好啊); 那个 / 就是说 only before a pause (那个人 stays);
               never inside a word (额度, 额外 - jieba)
   * repeats   我我我觉得 / 我们今天，我们今天要讲 -> keep the last copy
   * retakes   a sentence said again right away -> keep the last take
   * unrecognised voicing: voiced sounds no word covers (fillers the recogniser
               did not write down)
   * pauses    silences longer than ``max_pause`` shrink to ``keep_pause``
   * optional LLM pass for redundant phrases (废话)

3. planning: between two kept words the removed material and the surrounding
   silence are replaced by a *natural pause*: built from the original pauses on
   both sides when there are any, topped up with room tone (the recording's own
   background noise) when the removed filler sat between two words with no
   pause - butting two words together is what makes a cut audible.  Cut points
   snap to the quietest instant nearby.
4. rendering: equal-power crossfades at every join (longer when a cut lands in
   voiced audio); video is cut on a frame-aligned grid so it stays in sync; the
   transcript is re-timed onto the edited timeline (subtitles of the result).
"""
from __future__ import annotations

import json
import logging
import math
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .models import Document, Line, Token
from .text.tokenize import is_cjk, normalize_key

log = logging.getLogger("subalign")

# makes Whisper write disfluencies down instead of cleaning them up
FILLER_PROMPT = "嗯，呃，那个，我们我们，就是说，额，啊，这个这个。"

_ALWAYS = ["嗯", "呃", "额", "唔", "嗯嗯", "呃呃", "额额", "um", "uh", "erm", "uhm", "hmm", "mm", "er", "ah"]
LEVELS: Dict[str, Dict[str, Any]] = {
    "conservative": dict(always=["嗯", "呃", "额", "嗯嗯", "呃呃", "um", "uh", "erm", "uhm"], paused=[], phrases=[],
                         max_pause=1.2, keep_pause=0.5, island_max=0.8),
    "standard": dict(always=_ALWAYS, paused=["啊", "哦", "诶", "欸", "哎", "唉"], phrases=["那个", "就是说"],
                     max_pause=0.8, keep_pause=0.4, island_max=1.2),
    "aggressive": dict(always=_ALWAYS, paused=["啊", "哦", "诶", "欸", "哎", "唉", "呀", "哈"],
                       phrases=["那个", "就是说", "然后", "这个", "就是", "you know", "i mean", "like"],
                       max_pause=0.6, keep_pause=0.3, island_max=1.5),
}
REASONS = {"filler": "语气词", "repeat": "重复 / 口吃", "retake": "重录句", "unrecognized": "未识别发声",
           "llm": "大模型标记", "manual": "手动", "pause": "长停顿"}


@dataclass
class RoughCutConfig:
    level: str = "standard"
    fillers: bool = True
    repeats: bool = True
    retakes: bool = True
    unrecognized: bool = True
    pauses: bool = True
    max_pause: Optional[float] = None        # level default
    keep_pause: Optional[float] = None       # level default
    min_gap: float = 0.12                    # pause left where a cut joins two words with no pause
    lead_pause: float = 0.3                  # silence kept at the start / end of the file
    crossfade_ms: float = 25.0
    voiced_crossfade_ms: float = 40.0
    extra_fillers: Sequence[str] = ()
    keep_words: Sequence[str] = ()
    llm: bool = False

    def resolved(self) -> Dict[str, Any]:
        lv = dict(LEVELS.get(self.level, LEVELS["standard"]))
        if self.max_pause is not None:
            lv["max_pause"] = self.max_pause
        if self.keep_pause is not None:
            lv["keep_pause"] = self.keep_pause
        lv["always"] = list(lv["always"]) + [w for w in self.extra_fillers if w]
        return lv


# ------------------------------------------------------------------ plan model
@dataclass
class Item:
    """One unit of the plan: a transcript token or an unrecognised voiced island."""
    text: str
    start: float
    end: float
    line: int = -1                  # line index (tokens); -1 for islands
    key: str = ""
    space_after: bool = False
    auto: Optional[str] = None      # reason proposed by the detector
    cut: Optional[str] = None       # current decision (None = keep)
    island: bool = False


def _units(text: str) -> List[str]:
    """Phrase -> comparison units (CJK per character, other scripts per word)."""
    t = text.strip().lower()
    if any(is_cjk(c) for c in t):
        return [normalize_key(c) for c in t if normalize_key(c)]
    return [normalize_key(w) for w in t.split() if normalize_key(w)]


def items_from_document(doc: Document) -> List[Item]:
    out = []
    for li, ln in enumerate(doc.lines):
        for t in ln.tokens:
            if t.start is None or t.end is None:
                continue
            out.append(Item(text=t.text, start=float(t.start), end=float(t.end), line=li,
                            key=normalize_key(t.text), space_after=t.space_after))
    return out


# ------------------------------------------------------------------ detection
def _word_spans(items: List[Item]) -> List[Tuple[int, int]]:
    """For every item, the (first, last) item index of the jieba word it belongs to."""
    spans = [(i, i) for i in range(len(items))]
    try:
        import jieba  # type: ignore

        jieba.setLogLevel(logging.WARNING)
    except Exception:
        return spans
    by_line: Dict[int, List[int]] = {}
    for i, it in enumerate(items):
        if not it.island and it.key:
            by_line.setdefault(it.line, []).append(i)
    for idxs in by_line.values():
        chars, owner = [], []
        for i in idxs:
            for c in items[i].key:
                chars.append(c)
                owner.append(i)
        text = "".join(chars)
        for w, a, b in jieba.tokenize(text, HMM=False):   # dictionary words only (HMM invents 一聊额)
            if b - a > 1:
                lo, hi = owner[a], owner[b - 1]
                for k in range(a, b):
                    spans[owner[k]] = (lo, hi)
    return spans


def _keyed(items: List[Item]) -> List[int]:
    return [i for i, it in enumerate(items) if it.key and not it.island]


def detect(items: List[Item], cfg: RoughCutConfig, feats=None, llm_client=None) -> List[Item]:
    """Fill ``auto``/``cut`` on ``items`` (and append unrecognised islands)."""
    lv = cfg.resolved()
    keep = {normalize_key(w) for w in cfg.keep_words if w}
    K = _keyed(items)
    spans = _word_spans(items)
    line_first = {}
    for i in K:
        line_first.setdefault(items[i].line, i)

    def gap_before(p: int) -> float:
        return items[K[p]].start - items[K[p - 1]].end if p > 0 else 9.0

    def gap_after(p: int) -> float:
        return items[K[p + 1]].start - items[K[p]].end if p + 1 < len(K) else 9.0

    def mark(idxs, reason):
        for i in idxs:
            if items[i].auto is None and items[i].key not in keep:
                items[i].auto = reason

    # ---- fillers (longest phrases first)
    if cfg.fillers:
        cands = [(p, "always") for p in lv["always"]] + [(p, "paused") for p in lv["paused"]] + \
                [(p, "phrase") for p in lv["phrases"]]
        cands = sorted(((u, kind) for p, kind in cands if (u := _units(p))), key=lambda x: -len(x[0]))
        filler_keys = {tuple(u) for u, _ in cands}
        for p in range(len(K)):
            if items[K[p]].auto:
                continue
            for units, kind in cands:
                n = len(units)
                if p + n > len(K) or [items[K[p + j]].key for j in range(n)] != units:
                    continue
                if any(items[K[p + j]].auto for j in range(n)):
                    continue
                # never inside a longer word (额度, 额外, 那个人 is handled by the pause rule)
                lo, hi = spans[K[p]][0], spans[K[p + n - 1]][1]
                if lo < K[p] or hi > K[p + n - 1]:
                    word = tuple(items[i].key for i in range(lo, hi + 1) if items[i].key)
                    if word not in filler_keys and not all((c,) in filler_keys for c in word):
                        continue
                gb, ga = gap_before(p), gap_after(p + n - 1)
                first = line_first.get(items[K[p]].line) == K[p]
                if kind == "paused" and not (gb >= 0.12 or first):
                    continue
                if kind == "phrase":
                    nxt_filler = p + n < len(K) and any(
                        [items[K[p + n + j]].key for j in range(len(u)) if p + n + j < len(K)] == u for u, _ in cands)
                    if not ((ga >= 0.18 or nxt_filler) and (gb >= 0.08 or first)):
                        continue
                mark([K[p + j] for j in range(n)], "filler")
                break

    # ---- repeats / stutters / false starts: keep the last copy
    if cfg.repeats:
        pos = [i for i in K if items[i].auto is None]
        last_removed: Optional[str] = None
        k = 0
        while k < len(pos):
            hit = False
            for n in range(min(24, (len(pos) - k) // 2), 0, -1):
                a, b = pos[k:k + n], pos[k + n:k + 2 * n]
                if [items[i].key for i in a] != [items[i].key for i in b]:
                    continue
                gap = items[b[0]].start - items[a[-1]].end
                if gap > (3.0 if n >= 6 else 1.5):
                    continue
                if n == 1:
                    triple = k + 2 < len(pos) and items[pos[k + 2]].key == items[a[0]].key
                    lo, hi = spans[a[0]]
                    redup = lo <= a[0] and hi >= b[0] and hi > lo      # 看看 / 谢谢 are words
                    ok = triple or (not redup and (last_removed == items[a[0]].key or gap >= 0.12))
                elif n == 2:
                    ok = gap >= 0.08                                   # 研究研究 has no pause
                else:
                    ok = gap >= 0.04 or n >= 4
                if not ok:
                    continue
                between = [i for i in range(a[0], b[0]) if i not in b]   # copy + fillers / punctuation in between
                mark(between, "repeat")
                last_removed = items[a[-1]].key if n == 1 else None
                k += n
                hit = True
                break
            if not hit:
                last_removed = None
                k += 1

    # ---- retakes: a whole line said again right after (fuzzy)
    if cfg.retakes:
        lines: Dict[int, List[int]] = {}
        for i in K:
            if items[i].auto is None:
                lines.setdefault(items[i].line, []).append(i)
        order = sorted(lines)
        for x, y in zip(order, order[1:]):
            A, B = lines[x], lines[y]
            sa, sb = "".join(items[i].key for i in A), "".join(items[i].key for i in B)
            if len(sa) < 5 or len(sb) < 5 or items[B[0]].start - items[A[-1]].end > 4.0:
                continue
            if SequenceMatcher(None, sa, sb, autojunk=False).ratio() >= 0.85:
                mark([i for i in range(A[0], A[-1] + 1)], "retake")

    # ---- LLM: redundant phrases (废话)
    if llm_client is not None:
        try:
            _detect_llm(items, llm_client)
        except Exception as e:  # never fail the cut because of the LLM
            log.warning("LLM rough-cut pass failed: %s", e)

    # attached punctuation-only tokens follow their neighbour
    for i, it in enumerate(items):
        if not it.key and not it.island and it.auto is None and i > 0 and items[i - 1].auto:
            it.auto = items[i - 1].auto

    # ---- unrecognised voiced islands
    if feats is not None:
        items.extend(_islands(items, feats, lv["island_max"], cfg.unrecognized))
    for it in items:
        it.cut = it.auto
    return items


def _islands(items: List[Item], feats, max_len: float, enabled: bool) -> List[Item]:
    hop = feats.hop_s
    act = feats.active > 0.5
    cover = np.zeros(feats.n, dtype=bool)
    for it in items:
        if it.key:
            a, b = max(0, int((it.start - 0.06) / hop)), min(feats.n, int(math.ceil((it.end + 0.06) / hop)))
            cover[a:b] = True
    free = act & ~cover
    floor = float(np.percentile(feats.db, 10))
    out = []
    i = 0
    while i < feats.n:
        if not free[i]:
            i += 1
            continue
        j = i
        while j < feats.n and (free[j] or (j + 8 < feats.n and free[j:j + 8].any() and not cover[j])):
            j += 1
        dur = (j - i) * hop
        if 0.12 <= dur <= max_len and float(np.mean(feats.db[i:j])) > floor + 8:
            out.append(Item(text=f"[发声 {dur:.1f}s]", start=i * hop, end=j * hop, island=True,
                            auto="unrecognized" if enabled else None))
        i = j
    return out


_LLM_SYSTEM = (
    "You are a podcast / video editor making a rough cut of a spoken transcript. Mark only words that can be "
    "deleted without changing the meaning or breaking the grammar of what remains: verbal tics, filler phrases, "
    "false starts, words or clauses that are immediately repeated, and sentences that are said again (delete the "
    "earlier take). Never delete content words, names, numbers or anything that carries information. Quote the "
    "deleted text exactly as it appears in the line. "
    'Reply with JSON only: {"cuts": [{"id": <line id>, "text": "<exact text to delete>"}]}')


def _detect_llm(items: List[Item], client) -> None:
    by_line: Dict[int, List[int]] = {}
    for i, it in enumerate(items):
        if not it.island:
            by_line.setdefault(it.line, []).append(i)
    lines = [{"id": li, "text": "".join(items[i].text + (" " if items[i].space_after else "") for i in idxs).strip()}
             for li, idxs in sorted(by_line.items())]
    for b0 in range(0, len(lines), 40):
        batch = lines[b0:b0 + 40]
        data = client.complete_json(_LLM_SYSTEM, json.dumps({"lines": batch}, ensure_ascii=False))
        for c in (data.get("cuts", []) if isinstance(data, dict) else []):
            try:
                li, text = int(c["id"]), str(c["text"])
            except (KeyError, ValueError, TypeError):
                continue
            idxs = [i for i in by_line.get(li, []) if items[i].key]
            want = _units(text)
            if not want:
                continue
            keys = [items[i].key for i in idxs]
            for s in range(len(keys) - len(want) + 1):
                if keys[s:s + len(want)] == want:
                    for i in idxs[s:s + len(want)]:
                        if items[i].auto is None:
                            items[i].auto = "llm"
                    break


# ------------------------------------------------------------------ planning
@dataclass
class Joint:
    fill: float = 0.0          # room tone inserted after this keep segment (s)
    voiced: bool = False


def _rms_env(y: np.ndarray, sr: int, hop_s: float = 0.0025) -> Tuple[np.ndarray, float]:
    hop = max(1, int(sr * hop_s))
    m = y.mean(axis=0) if y.ndim == 2 else y
    n = len(m) // hop
    e = np.sqrt(np.mean(m[:n * hop].reshape(n, hop) ** 2, axis=1) + 1e-12)
    e = np.convolve(e, np.ones(4) / 4, mode="same")
    return e, hop / sr


def plan_cuts(items: List[Item], duration: float, cfg: RoughCutConfig, env: np.ndarray, env_hop: float
              ) -> Tuple[List[Tuple[float, float]], List[float], List[Dict[str, Any]]]:
    """Return keep segments, the room-tone fill after each segment, and pause edits."""
    lv = cfg.resolved()
    max_pause, keep_pause = lv["max_pause"], lv["keep_pause"]
    kept = sorted((it for it in items if it.cut is None and it.key and not it.island), key=lambda it: it.start)
    drops = sorted((it for it in items if it.cut is not None), key=lambda it: it.start)
    removals: List[Tuple[float, float, float]] = []    # (from, to, fill)
    pause_edits: List[Dict[str, Any]] = []

    def snap(t: float, lo: float, hi: float) -> float:
        lo, hi = max(lo, 0.0), min(hi, duration)
        if hi - lo < env_hop * 2:
            return min(max(t, lo), hi)
        a, b = int(lo / env_hop), int(math.ceil(hi / env_hop))
        seg = env[a:b]
        if not len(seg):
            return t
        # quietest instant, mildly preferring the target position
        dist = np.abs(np.arange(a, a + len(seg)) * env_hop - t)
        score = seg / (np.median(env) + 1e-9) + dist * 2.0
        return (a + int(np.argmin(score))) * env_hop

    if not kept:
        return [(0.0, duration)], [0.0], pause_edits

    def drops_between(t0: float, t1: float) -> List[Item]:
        return [d for d in drops if t0 - 1e-3 <= (d.start + d.end) / 2 <= t1 + 1e-3]

    # leading part of the file
    fs = kept[0].start
    D = drops_between(0.0, fs)
    lead = min(cfg.lead_pause, fs - (max(d.end for d in D) if D else 0.0))
    if (cfg.pauses and fs > cfg.lead_pause + 0.05) or D:
        to = snap(fs - max(lead, 0.0), max(D[-1].end if D else 0.0, fs - max(lead, 0.0) - 0.03), fs - 0.02)
        if to > 0.02:
            removals.append((0.0, to, 0.0))

    for a, b in zip(kept, kept[1:]):
        pe, ns = a.end, b.start
        if ns <= pe:
            continue
        D = drops_between(pe, ns)
        if not D:
            gap = ns - pe
            if not (cfg.pauses and gap > max_pause):
                continue
            g1 = g2 = keep_pause / 2
            x = snap(pe + g1, pe + min(g1, 0.04), pe + g1 + 0.05)
            y = snap(ns - g2, ns - g2 - 0.05, ns - min(g2, 0.03))
            if y - x > 0.05:
                removals.append((x, y, 0.0))
                pause_edits.append({"start": pe, "end": ns, "from": round(gap, 2), "to": round(gap - (y - x), 2)})
            continue
        d0, d1 = min(d.start for d in D), max(d.end for d in D)
        pb, pa = max(0.0, d0 - pe), max(0.0, ns - d1)
        natural = max(pb, pa)
        G = max(cfg.min_gap, natural)
        if cfg.pauses:
            G = min(G, max(keep_pause, cfg.min_gap))
        g1 = min(pb, G / 2)
        g2 = min(pa, G - g1)
        g1 = min(pb, G - g2)
        fill = max(0.0, G - g1 - g2)
        # cut points: inside the pauses when there are any, at the word boundary otherwise
        x = snap(pe + g1, pe + min(g1, 0.03), max(pe + min(g1, 0.03), min(d0, pe + g1 + 0.03)))
        y = snap(ns - g2, min(ns - min(g2, 0.03), max(d1, ns - g2 - 0.03)), ns - min(g2, 0.03))
        if y > x:
            removals.append((x, y, fill))

    # trailing part
    le = kept[-1].end
    D = drops_between(le, duration)
    tail = min(cfg.lead_pause, (min(d.start for d in D) if D else duration) - le)
    if (cfg.pauses and duration - le > cfg.lead_pause + 0.05) or D:
        x = snap(le + max(tail, 0.0), le + 0.02, min(D[0].start if D else duration, le + max(tail, 0.0) + 0.03))
        if duration - x > 0.02:
            removals.append((x, duration, 0.0))

    removals.sort()
    merged: List[List[float]] = []
    for r in removals:
        if merged and r[0] <= merged[-1][1] + 1e-4:
            merged[-1][1] = max(merged[-1][1], r[1])
            merged[-1][2] = max(merged[-1][2], r[2])
        else:
            merged.append(list(r))
    keeps, fills = [], []
    t = 0.0
    for x, y, fill in merged:
        if x - t > 1e-3:
            keeps.append((t, x))
            fills.append(fill)
        elif fills:
            fills[-1] = max(fills[-1], fill)
        t = y
    if duration - t > 1e-3:
        keeps.append((t, duration))
        fills.append(0.0)
    if fills:
        fills[-1] = 0.0
    return keeps, fills, pause_edits


# ------------------------------------------------------------------ rendering
def _room_tone(y: np.ndarray, sr: int, env: np.ndarray, env_hop: float, items: List[Item]) -> np.ndarray:
    """Up to 1 s of the recording's own background noise (quietest region away from speech)."""
    n = len(env)
    speech = np.zeros(n, dtype=bool)
    for it in items:
        speech[max(0, int((it.start - 0.15) / env_hop)):min(n, int((it.end + 0.15) / env_hop))] = True
    quiet = (~speech) & (env <= np.percentile(env, 25))
    best, cur, best_end = 0, 0, 0
    for i, q in enumerate(quiet):
        cur = cur + 1 if q else 0
        if cur > best:
            best, best_end = cur, i + 1
    ch = y.shape[0]
    if best * env_hop < 0.15:
        return np.zeros((ch, int(0.2 * sr)), dtype=np.float32)
    a, b = int((best_end - best) * env_hop * sr), int(best_end * env_hop * sr)
    b = min(b, a + sr)
    return y[:, a:b].astype(np.float32)


def _tone(room: np.ndarray, length: int) -> np.ndarray:
    """Loop the room tone to ``length`` samples with crossfaded seams."""
    ch, m = room.shape
    if length <= m:
        return room[:, :length].copy()
    ov = max(1, min(m // 4, 2048))
    out = room.copy()
    while out.shape[1] < length:
        t = np.linspace(0, 1, ov, dtype=np.float32)
        out[:, -ov:] = out[:, -ov:] * np.cos(t * np.pi / 2) + room[:, :ov] * np.sin(t * np.pi / 2)
        out = np.concatenate([out, room[:, ov:]], axis=1)
    return out[:, :length]


@dataclass
class Rendered:
    audio: np.ndarray
    sr: int
    seg_out: List[float]           # output start time of every keep segment
    seg_bounds: List[float]        # output instant where each keep segment begins (crossfade centre), + total
    keeps: List[Tuple[float, float]]


def render(y: np.ndarray, sr: int, keeps: List[Tuple[float, float]], fills: List[float], room: np.ndarray,
           env: np.ndarray, env_hop: float, cfg: RoughCutConfig) -> Rendered:
    level = float(np.percentile(env, 90)) + 1e-9

    def voiced_at(t: float) -> bool:
        i = min(len(env) - 1, max(0, int(t / env_hop)))
        return float(env[max(0, i - 2):i + 3].max()) > 0.12 * level

    pieces: List[Tuple[str, int, np.ndarray, float]] = []      # (kind, seg index, samples, crossfade before)
    for k, (s, e) in enumerate(keeps):
        seg = y[:, int(round(s * sr)):int(round(e * sr))]
        cf = cfg.crossfade_ms
        if k > 0:
            if voiced_at(s) or voiced_at(keeps[k - 1][1]):
                cf = cfg.voiced_crossfade_ms
        pieces.append(("seg", k, seg, cf if k > 0 else 0.0))
        if k < len(keeps) - 1 and fills[k] > 0:
            ov = int(cfg.crossfade_ms / 1000 * sr)
            pieces.append(("tone", k, _tone(room, int(fills[k] * sr) + 2 * ov), cfg.crossfade_ms))
    total = sum(p[2].shape[1] for p in pieces)
    out = np.zeros((y.shape[0], total), dtype=np.float32)
    cur = 0
    seg_out, seg_bounds = [0.0] * len(keeps), [0.0] * len(keeps)
    for kind, k, x, cf in pieces:
        n = x.shape[1]
        ov = min(int(cf / 1000 * sr), n // 2, cur // 2 if cur else 0)
        start = cur - ov
        if ov > 0:
            t = np.linspace(0, 1, ov, dtype=np.float32)
            out[:, start:cur] = out[:, start:cur] * np.cos(t * np.pi / 2) + x[:, :ov] * np.sin(t * np.pi / 2)
            out[:, cur:start + n] = x[:, ov:]
        else:
            out[:, start:start + n] = x
        if kind == "seg":
            seg_out[k] = start / sr
            seg_bounds[k] = (start + ov / 2) / sr
        cur = start + n
    out = out[:, :cur]
    f = min(int(0.005 * sr), cur // 2)          # tiny fades at the file edges
    if f:
        r = np.linspace(0, 1, f, dtype=np.float32)
        out[:, :f] *= r
        out[:, -f:] *= r[::-1]
    return Rendered(audio=out, sr=sr, seg_out=seg_out, seg_bounds=seg_bounds + [cur / sr], keeps=keeps)


def time_map(r: Rendered):
    """Original time -> edited time (None inside removed material)."""
    def f(t: float) -> Optional[float]:
        for k, (s, e) in enumerate(r.keeps):
            if s - 1e-3 <= t <= e + 1e-3:
                return max(0.0, r.seg_out[k] + (min(max(t, s), e) - s))
        return None
    return f


def edited_document(items: List[Item], r: Rendered, kind: str = "speech", language: Optional[str] = None) -> Document:
    f = time_map(r)
    lines: Dict[int, List[Token]] = {}
    for it in items:
        if it.island or it.cut is not None:
            continue
        a, b = f(it.start), f(it.end)
        if a is None and b is None:
            continue
        a = b if a is None else a
        b = a if b is None else b
        lines.setdefault(it.line, []).append(Token(it.text, a, max(a, b), 1.0, it.space_after))
    doc = Document(lines=[], language=language, kind=kind)
    for li in sorted(lines):
        toks = lines[li]
        if toks:
            toks[-1].space_after = False
            ln = Line(tokens=toks)
            ln.update_bounds()
            doc.lines.append(ln)
    return doc


# ------------------------------------------------------------------ media helpers
def probe(path: Path) -> Dict[str, Any]:
    info = {"sr": 44100, "channels": 2, "video": False, "fps": None}
    if not shutil.which("ffprobe"):
        return info
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(path)],
                             capture_output=True, check=True, text=True, encoding="utf-8").stdout
        for st in json.loads(out).get("streams", []):
            if st.get("codec_type") == "audio" and "sr_set" not in info:
                info["sr"] = int(st.get("sample_rate") or 44100)
                info["channels"] = int(st.get("channels") or 2)
                info["sr_set"] = True
            elif st.get("codec_type") == "video" and st.get("disposition", {}).get("attached_pic", 0) == 0:
                num, den = (st.get("avg_frame_rate") or st.get("r_frame_rate") or "0/1").split("/")
                fps = float(num) / float(den or 1) if float(den or 1) else 0.0
                if fps > 0:
                    info["video"], info["fps"] = True, fps
    except Exception as e:  # pragma: no cover
        log.info("ffprobe failed: %s", e)
    info.pop("sr_set", None)
    return info


def cut_video(src: Path, audio: Path, dst: Path, r: Rendered, fps: float) -> Path:
    """Frame-aligned video cut (jump cuts) muxed with the rendered audio."""
    frame = 1.0 / fps
    bounds = [round(b / frame) * frame for b in r.seg_bounds]
    ranges = []
    for k, (s, _e) in enumerate(r.keeps):
        o0, o1 = bounds[k], bounds[k + 1]
        if o1 - o0 < frame / 2:
            continue
        a = s + (o0 - r.seg_out[k])
        a = max(0.0, round(a / frame) * frame)
        ranges.append((a, a + (o1 - o0)))
    # half-frame margins on both sides: exactly round((b - a) * fps) frames per range, so the
    # edited video never drifts against the audio
    expr = "+".join(f"between(t,{a - frame / 2:.6f},{b - frame / 2:.6f})" for a, b in ranges)
    script = dst.with_suffix(".vf.txt")
    script.write_text(f"select='{expr}',setpts=N/FRAME_RATE/TB", encoding="utf-8")
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(src), "-i", str(audio),
           "-/filter:v", str(script), "-map", "0:v:0", "-map", "1:a:0", "-r", f"{fps:.6f}",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "192k", "-shortest", str(dst)]
    try:
        subprocess.run(cmd, check=True)
    finally:
        script.unlink(missing_ok=True)
    return dst


# ------------------------------------------------------------------ entry point
def transcribe_for_cut(audio: Path, script: Optional[str], language: Optional[str], asr_backend: str,
                       asr_model: Optional[str], device: Optional[str], ctc: str, workdir: Path) -> Document:
    """Units with accurate start times.  Recognition writes the text (prompted to keep
    disfluencies); CTC forced alignment of that text then gives the per-unit onsets -
    ASR word times alone put the pauses in the wrong places."""
    from .align.aligner import AlignConfig, align_audio

    asr_cfg = AlignConfig(mode="speech", language=language, asr_backend=asr_backend, asr_model=asr_model,
                          asr_options={"prompt": FILLER_PROMPT, "vad": False}, ctc="off", separation="none",
                          refine=True, proofread=False, device=device)
    if script is None:
        from .asr import get_backend

        kw: Dict[str, Any] = {"model": asr_model} if asr_model else {}
        if device and asr_backend in ("faster-whisper", "whisper", "funasr"):
            kw["device"] = device
        tr = get_backend(asr_backend, **kw).transcribe(str(audio), language=language, prompt=FILLER_PROMPT, vad=False)
        language = language or tr.language
        script = "\n".join(s.text.strip() for s in tr.segments if s.text.strip())
        if not script:
            return Document(lines=[], language=language)
    if ctc != "off":
        cfg = AlignConfig(mode="speech", language=language, asr_backend="none", ctc="on" if ctc == "on" else "auto",
                          separation="none", refine=True, proofread=False, device=device)
        try:
            res = align_audio(audio, script, cfg, workdir=workdir)
            if res.anchor_source == "ctc":
                res.document.language = res.document.language or language
                return res.document
            log.warning("CTC alignment unavailable for the rough cut; using recogniser timings")
        except Exception as e:
            if ctc == "on":
                raise
            log.warning("CTC alignment failed (%s); using recogniser timings", e)
    return align_audio(audio, script if asr_backend == "none" else None, asr_cfg, workdir=workdir).document


def acoustic_ends(items: List[Item], env: np.ndarray, env_hop: float) -> None:
    """A unit ends where the signal falls to the noise floor and stays there (>=60 ms),
    not where the next unit starts - pauses are measured, not guessed.  Energy (not
    voicing) is used so unvoiced endings (s, sh, t) are kept."""
    db = 20 * np.log10(env + 1e-9)
    floor, loud = np.percentile(db, 10), np.percentile(db, 95)
    thr = floor + max(6.0, 0.18 * (loud - floor))
    quiet = db < thr
    k = max(1, int(0.06 / env_hop))
    run = np.convolve(quiet.astype(np.int32), np.ones(k, dtype=np.int32), mode="valid") >= k  # run starting at i
    units = sorted((it for it in items if it.key and not it.island), key=lambda it: it.start)
    for a, b in zip(units, units[1:] + [None]):
        lim = b.start if b is not None else min(a.end + 2.0, len(env) * env_hop)
        i0, i1 = int((a.start + 0.04) / env_hop), int(lim / env_hop)
        hits = np.flatnonzero(run[i0:max(i0, min(i1, len(run)))])
        end = (i0 + hits[0]) * env_hop + 0.01 if len(hits) else lim
        a.end = float(min(max(end, a.start + 0.04), lim))


def report_md(stats: Dict[str, Any], items: List[Item], pause_edits: List[Dict[str, Any]]) -> str:
    def ts(t: float) -> str:
        m = int(t // 60)
        return f"{m:02d}:{t - 60 * m:05.2f}"

    rows = [f"# 语音粗剪报告\n",
            f"原时长 {stats['duration']:.1f}s → 剪后 {stats['output_duration']:.1f}s"
            f"（减少 {stats['duration'] - stats['output_duration']:.1f}s，{stats['removed_ratio']:.0%}）\n"]
    for k, v in stats["counts"].items():
        rows.append(f"- {REASONS.get(k, k)}：{v} 处，{stats['seconds'].get(k, 0):.1f}s")
    rows.append("\n## 删除明细\n")
    for g in groups(items):
        if g["cut"]:
            rows.append(f"- [{ts(g['start'])}] {REASONS.get(g['cut'], g['cut'])}：{g['text']}")
    if pause_edits:
        rows.append("\n## 缩短的停顿\n")
        rows += [f"- [{ts(p['start'])}] {p['from']:.2f}s → {p['to']:.2f}s" for p in pause_edits]
    return "\n".join(rows) + "\n"


def groups(items: List[Item]) -> List[Dict[str, Any]]:
    """Consecutive items with the same decision (for reports / review)."""
    out: List[Dict[str, Any]] = []
    for it in sorted(items, key=lambda x: x.start):
        if out and out[-1]["cut"] == it.cut and it.cut is not None and not it.island and not out[-1]["island"]:
            out[-1]["text"] += it.text
            out[-1]["end"] = it.end
            continue
        out.append({"cut": it.cut, "text": it.text, "start": it.start, "end": it.end, "island": it.island})
    return out


def rough_cut(audio: Path, out_dir: Path, cfg: RoughCutConfig, script: Optional[str] = None,
              plan: Optional[Dict[str, Any]] = None, language: Optional[str] = None, asr_backend: str = "auto",
              asr_model: Optional[str] = None, device: Optional[str] = None, ctc: str = "auto",
              audio_format: Optional[str] = None, video: bool = True, llm_client=None) -> Dict[str, Any]:
    from .audio.features import analyze
    from .audio.io import load_audio, save_audio

    audio, out_dir = Path(audio), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = audio.stem
    info = probe(audio)
    sr = info["sr"]
    y = load_audio(audio, sr, mono=info["channels"] == 1)
    if y.ndim == 1:
        y = y[None]
    duration = y.shape[1] / sr
    env, env_hop = _rms_env(y, sr)
    if plan is not None:
        items = [Item(**{k: v for k, v in d.items() if k in Item.__dataclass_fields__}) for d in plan["items"]]
        language = language or plan.get("language")
        s = plan.get("settings") or {}
        for k in ("max_pause", "keep_pause", "pauses", "min_gap", "crossfade_ms", "level"):
            if k in s and getattr(cfg, k) == getattr(RoughCutConfig(), k):
                setattr(cfg, k, s[k])
    else:
        doc = transcribe_for_cut(audio, script, language, asr_backend, asr_model, device, ctc, out_dir / "work")
        language = language or doc.language
        items = items_from_document(doc)
        acoustic_ends(items, env, env_hop)
        items = detect(items, cfg, analyze(load_audio(audio, 16000)), llm_client)
    keeps, fills, pause_edits = plan_cuts(items, duration, cfg, env, env_hop)
    room = _room_tone(y, sr, env, env_hop, items)
    r = render(y, sr, keeps, fills, room, env, env_hop, cfg)

    ext = (audio_format or ("wav" if info["video"] or audio.suffix.lower() not in (".mp3", ".flac", ".m4a", ".wav")
                            else audio.suffix.lower().lstrip("."))).lstrip(".")
    out_audio = save_audio(out_dir / f"{base}.cut.{ext}", r.audio if r.audio.shape[0] > 1 else r.audio[0], sr)
    files = [out_audio]
    if video and info["video"] and shutil.which("ffmpeg"):
        tmp = out_audio if ext == "wav" else save_audio(out_dir / f"{base}.cut.tmp.wav", r.audio, sr)
        try:
            files.append(cut_video(audio, tmp, out_dir / f"{base}.cut.mp4", r, info["fps"]))
        except subprocess.CalledProcessError as e:
            log.warning("video cut failed: %s", e)
        finally:
            if tmp != out_audio:
                tmp.unlink(missing_ok=True)

    counts: Dict[str, int] = {}
    seconds: Dict[str, float] = {}
    for g in groups(items):
        if g["cut"]:
            counts[g["cut"]] = counts.get(g["cut"], 0) + 1
            seconds[g["cut"]] = seconds.get(g["cut"], 0.0) + g["end"] - g["start"]
    if pause_edits:
        counts["pause"] = len(pause_edits)
        seconds["pause"] = sum(p["from"] - p["to"] for p in pause_edits)
    out_dur = r.audio.shape[1] / sr
    stats = {"duration": round(duration, 3), "output_duration": round(out_dur, 3),
             "removed_ratio": round(1 - out_dur / max(duration, 1e-9), 4), "counts": counts,
             "seconds": {k: round(v, 2) for k, v in seconds.items()}, "joints": len(keeps) - 1}
    plan_out = {"format": "subalign-roughcut", "version": 1, "source": audio.name, "language": language,
                "settings": {"level": cfg.level, "max_pause": cfg.resolved()["max_pause"],
                             "keep_pause": cfg.resolved()["keep_pause"], "pauses": cfg.pauses,
                             "min_gap": cfg.min_gap, "crossfade_ms": cfg.crossfade_ms},
                "stats": stats, "pause_edits": pause_edits, "items": [asdict(it) for it in items]}
    pp = out_dir / f"{base}.roughcut.json"
    pp.write_text(json.dumps(plan_out, ensure_ascii=False, indent=1), encoding="utf-8")
    rp = out_dir / f"{base}.roughcut.md"
    rp.write_text(report_md(stats, items, pause_edits), encoding="utf-8")
    edited = edited_document(items, r, language=language)
    return {"audio": out_audio, "files": files + [pp, rp], "stats": stats, "document": edited, "base": base}
