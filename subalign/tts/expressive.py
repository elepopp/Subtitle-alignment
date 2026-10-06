"""Expressive dubbing (表演迁移): carry the original performance into a translated
voice-over instead of reading every line in one tone.

From the original recording (its vocals stem when the recognition separated one)
and the sentence timing of a dubbing translation:

* **speakers** - the recognition's speaker labels, or a quick sentence-level
  diarisation (CAM++ embeddings + clustering); every speaker gets a voice
  reference of their own, cut from their calmest lines (an emotional line as the
  *identity* reference drags its emotion into every sentence)
* **per-sentence emotion reference** - the original line itself (widened to at
  least ``min_emo_s`` with the same speaker's surroundings): IndexTTS-2.5 takes
  identity and emotion from separate references, so the line's energy, tension and
  pace carry over without a text label
* **expressiveness** (0-1) - how far the line departs from the speaker's usual
  level, pitch and pitch movement.  Calm lines get a weak emotion weight (a source
  language reference applied at full strength leaks its intonation), strong ones a
  high weight
* **level** - the line's active speech level, so the dub can follow the original's
  loud / quiet lines instead of flattening them
* **pauses** - silences inside the line (hesitation, a beat before the point), to
  be re-placed at the matching clause boundary of the translation
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import textprep

log = logging.getLogger("subalign")

REF_SR = 24000          # what IndexTTS reads references at
AN_SR = 16000           # analysis


@dataclass
class ExprConfig:
    diarize: bool = True            # tell speakers apart when the recognition did not
    n_speakers: Optional[int] = None
    min_emo_s: float = 1.5          # shorter lines are widened with their surroundings
    max_emo_s: float = 12.0
    spk_target_s: float = 12.0      # voice reference length per speaker (IndexTTS reads ~15 s)
    min_pause: float = 0.3          # silences inside a line that count as a pause
    min_speaker_s: float = 4.0      # a "speaker" with less speech is merged into the nearest one


# lines that are only sound-event tags ([music], (applause), ♪): no speech to analyse
_EVENT = re.compile(r"^(?:\s*(?:>>|-)?\s*(?:[\[(（【][^\])）】]*[\])）】]|♪+))+\s*$")


def is_event(text: Optional[str]) -> bool:
    return bool(text) and bool(_EVENT.match(text))


# ------------------------------------------------------------------ measurements
def _frame_db(y: np.ndarray, sr: int, hop_s: float = 0.01) -> np.ndarray:
    hop = max(1, int(hop_s * sr))
    n = len(y) // hop
    if n == 0:
        return np.zeros(0)
    return 20 * np.log10(np.sqrt(np.mean(y[:n * hop].reshape(n, hop) ** 2, axis=1)) + 1e-9)


def level_db(y: np.ndarray, sr: int) -> float:
    """Active speech level: mean power of the 20 ms frames within 25 dB of the loud
    end (pauses between words do not pull a sparse line down)."""
    db = _frame_db(y, sr, 0.02)
    if len(db) == 0:
        return -90.0
    top = np.percentile(db, 95)
    act = db[db > top - 25]
    return float(10 * np.log10(np.mean(10 ** (act / 10)) + 1e-12))


def pitch_stats(y16: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
    """(median f0 in semitones re 55 Hz, spread = 10-90 % range in semitones) of the
    voiced frames, or (None, None) when there is too little voicing."""
    from ..audio.features import yin

    if len(y16) < AN_SR * 0.3:
        return None, None
    f0, per = yin(y16, AN_SR)
    v = f0[(f0 > 0) & (per > 0.6)]
    if len(v) < 15:
        return None, None
    st = 12 * np.log2(v / 55.0)
    lo, med, hi = np.percentile(st, [10, 50, 90])
    return float(med), float(hi - lo)


def silent_gaps(y: np.ndarray, sr: int, min_len: float, depth: float = 30.0, edge: float = 0.05
                ) -> List[Tuple[float, float]]:
    """(start, end) seconds of the silences inside ``y`` (not touching its edges):
    10 ms frames more than ``depth`` dB under the loud end, at least ``min_len`` long."""
    db = _frame_db(y, sr)
    if len(db) < 5:
        return []
    quiet = db < np.percentile(db, 95) - depth
    out, i, n = [], 0, len(db)
    while i < n:
        if not quiet[i]:
            i += 1
            continue
        j = i
        while j < n and quiet[j]:
            j += 1
        a, b = i * 0.01, j * 0.01
        if b - a >= min_len and a > edge and b < n * 0.01 - edge:
            out.append((round(a, 3), round(b, 3)))
        i = j
    return out


def _robust(xs: Sequence[float], floor: float) -> Tuple[float, float]:
    a = np.asarray([x for x in xs if x is not None], dtype=float)
    if len(a) == 0:
        return 0.0, floor
    med = float(np.median(a))
    return med, max(floor, 1.4826 * float(np.median(np.abs(a - med))))


def expressiveness(stats: Dict, ref: Dict) -> float:
    """0 (the speaker's ordinary delivery) .. 1 (far from it: shouting, whispering,
    raised pitch, a lot of melody) - distance in robust z-scores."""
    zs = []
    for k in ("level", "f0", "spread"):
        if stats.get(k) is None or k not in ref:
            continue
        med, scale = ref[k]
        z = (stats[k] - med) / scale
        if k == "spread":
            z = max(0.0, z)             # a flat line is not more expressive than usual
        zs.append(z)
    if not zs:
        return 0.0
    e = float(np.sqrt(np.mean(np.square(zs))))
    return round(float(np.clip((e - 0.5) / 2.0, 0.0, 1.0)), 3)


# ------------------------------------------------------------------ speakers
def _label_order(labels: Sequence) -> Dict:
    m: Dict = {}
    for lab in labels:
        if lab is not None and lab not in m:
            m[lab] = f"S{len(m) + 1}"
    return m


def merge_small(labels: List[int], durs: Sequence[float], E: np.ndarray, min_s: float) -> List[int]:
    """Clusters with less than ``min_s`` of speech (music, a cough, a mis-split line)
    join the nearest cluster that has enough, by centroid cosine similarity."""
    labels = list(labels)
    tot: Dict[int, float] = {}
    for l, d in zip(labels, durs):
        tot[l] = tot.get(l, 0.0) + d
    big = [l for l, t in tot.items() if t >= min_s]
    if not big or len(big) == len(tot):
        return labels
    cent = {l: E[[k for k, x in enumerate(labels) if x == l]].mean(axis=0) for l in tot}
    for l in tot:
        if l in big:
            continue
        c = cent[l] / (np.linalg.norm(cent[l]) + 1e-9)
        to = max(big, key=lambda b: float(c @ cent[b]) / (np.linalg.norm(cent[b]) + 1e-9))
        labels = [to if x == l else x for x in labels]
    return labels


def assign_speakers(spans: List[Tuple[float, float]], given: Sequence[Optional[str]], y16: np.ndarray,
                    cfg: ExprConfig, skip: Sequence[bool] = ()) -> List[str]:
    """One speaker label per sentence: the recognition's, else a sentence-level
    diarisation, else everyone is S1.  ``skip``: lines left out of the clustering
    (sound events); they, like lines too short to embed, take the nearest speaker."""
    n = len(spans)
    skip = list(skip) or [False] * n
    if any(g for g in given):
        labels = [g if not sk else None for g, sk in zip(given, skip)]
    elif cfg.diarize and cfg.n_speakers != 1 and n >= 4:
        labels = [None] * n
        try:
            from ..diarize import cluster, embed

            idx = [i for i, (a, b) in enumerate(spans) if b - a >= 0.8 and not skip[i]]
            if len(idx) >= 2:
                clips = []
                for i in idx:
                    a, b = spans[i]
                    if b - a > 12:
                        m = (a + b) / 2
                        a, b = m - 6, m + 6
                    clips.append(y16[int(a * AN_SR):int(b * AN_SR)])
                E = embed(clips)
                lab = [int(x) for x in cluster(E, cfg.n_speakers)]
                if not cfg.n_speakers:
                    lab = merge_small(lab, [spans[i][1] - spans[i][0] for i in idx], E, cfg.min_speaker_s)
                for i, l in zip(idx, lab):
                    labels[i] = l
        except Exception as e:          # optional model / dependency
            log.warning("speaker separation skipped: %s", e)
            labels = [None] * n
    else:
        labels = [None] * n
    # unlabelled (short) lines: the speaker of the nearest labelled line in time
    known = [i for i, l in enumerate(labels) if l is not None]
    if not known:
        return ["S1"] * n
    for i in range(n):
        if labels[i] is None:
            j = min(known, key=lambda k: abs(spans[k][0] - spans[i][0]))
            labels[i] = labels[j]
    names = _label_order(labels)
    return [names[l] for l in labels]


# ------------------------------------------------------------------ plan
def _save(path: Path, y: np.ndarray) -> str:
    from ..audio.io import save_audio

    path.parent.mkdir(parents=True, exist_ok=True)
    f = min(int(0.01 * REF_SR), len(y) // 4)
    y = y.astype(np.float32).copy()
    if f > 1:
        y[:f] *= np.linspace(0, 1, f)
        y[-f:] *= np.linspace(1, 0, f)
    save_audio(path, y, REF_SR)
    return str(path)


def _emo_window(i: int, spans: List[Tuple[float, float]], speakers: List[str], total: float,
                cfg: ExprConfig) -> Tuple[float, float]:
    """The line itself with a little air; a short line grows into its surroundings,
    never into another speaker's line."""
    a, b = spans[i]
    a, b = max(0.0, a - 0.08), min(total, b + 0.12)
    if b - a < cfg.min_emo_s:
        lo, hi = 0.0, total
        for j, (c, d) in enumerate(spans):
            if j == i or speakers[j] == speakers[i]:
                continue
            if d <= spans[i][0]:
                lo = max(lo, d + 0.05)
            elif c >= spans[i][1]:
                hi = min(hi, c - 0.05)
        need = cfg.min_emo_s - (b - a)
        a2 = max(lo, a - need / 2)
        b2 = min(hi, b + need - (a - a2))           # what the left side could not take goes right
        if b2 - a2 < cfg.min_emo_s:
            a2 = max(lo, b2 - cfg.min_emo_s)
        a, b = min(a, a2), max(b, b2)
    return a, min(b, a + cfg.max_emo_s)


def _pick_voice(idx: List[int], spans, expr: List[float], levels: List[float], target: float,
                calm: float = 0.35, enough: float = 6.0) -> List[int]:
    """The calmest usable lines of one speaker (1.5-12 s first), up to ``target`` s;
    an expressive line (> ``calm``) only while there is less than ``enough`` s."""
    def dur(i):
        return spans[i][1] - spans[i][0]
    good = [i for i in idx if 1.5 <= dur(i) <= 12 and levels[i] > -60]
    pool = good or sorted(idx, key=dur, reverse=True)[:4]
    pool = sorted(pool, key=lambda i: (expr[i], -dur(i)))
    out, tot = [], 0.0
    for i in pool:
        if tot >= target or (expr[i] > calm and tot >= enough):
            break
        if tot + dur(i) > 15 and out:
            continue
        out.append(i)
        tot += dur(i)
    return sorted(out, key=lambda i: spans[i][0])


def analyze_source(sentences: List[Dict], audio: Path, out_dir: Path, cfg: Optional[ExprConfig] = None,
                   voice_refs: bool = True, emo_refs: bool = True) -> Dict:
    """Per-sentence performance data + reference files under ``out_dir``.

    ``sentences``: ``[{"start", "end", "speaker"?}]`` (lines without timing get an
    empty entry).  Returns ``{"sentences": [{"speaker", "spk", "emo", "expr",
    "src_level", "src_pauses"}], "speakers": {"S1": {"ref", "lines"}}}``; paths are
    absolute strings, ``spk`` / ``emo`` are None when not produced."""
    from ..audio.io import load_audio

    cfg = cfg or ExprConfig()
    timed = [i for i, s in enumerate(sentences) if s.get("start") is not None and s.get("end") is not None
             and s["end"] > s["start"]]
    res: List[Dict] = [{} for _ in sentences]
    if not timed:
        return {"sentences": res, "speakers": {}}
    y = load_audio(audio, REF_SR)
    y16 = load_audio(audio, AN_SR)
    total = len(y) / REF_SR
    spans = [(float(sentences[i]["start"]), float(min(sentences[i]["end"], total))) for i in timed]
    events = [is_event(sentences[i].get("source_text") or sentences[i].get("text")) for i in timed]
    speakers = assign_speakers(spans, [sentences[i].get("speaker") for i in timed], y16, cfg, events)

    stats = []
    for (a, b), ev in zip(spans, events):
        if ev:
            stats.append({"level": None, "f0": None, "spread": None, "pauses": [], "event": True})
            continue
        seg16 = y16[int(a * AN_SR):int(b * AN_SR)]
        lvl = level_db(seg16, AN_SR)
        if lvl < -60:                   # no speech where the timing says (timing off / stem empty)
            stats.append({"level": None, "f0": None, "spread": None, "pauses": [], "event": True})
            continue
        f0, spread = pitch_stats(seg16)
        st = {"level": lvl, "f0": f0, "spread": spread}
        st["pauses"] = []
        if b - a > 0.8:
            for ga, gb in silent_gaps(seg16, AN_SR, cfg.min_pause):
                st["pauses"].append({"at": round((ga + gb) / 2 / (b - a), 3), "dur": round(gb - ga, 2)})
        stats.append(st)
    # per-speaker reference values (the whole recording when a speaker has few lines)
    refs: Dict[str, Dict] = {}
    speech = [s for s in stats if not s.get("event")]
    everyone = {"level": _robust([s["level"] for s in speech], 3.0), "f0": _robust([s["f0"] for s in speech], 1.0),
                "spread": _robust([s["spread"] for s in speech], 1.0)}
    for spk in set(speakers):
        mine = [s for s, k in zip(stats, speakers) if k == spk and not s.get("event")]
        refs[spk] = everyone if len(mine) < 4 else {
            "level": _robust([s["level"] for s in mine], 3.0), "f0": _robust([s["f0"] for s in mine], 1.0),
            "spread": _robust([s["spread"] for s in mine], 1.0)}
    expr = [expressiveness(s, refs[k]) for s, k in zip(stats, speakers)]
    levels = [s["level"] if s["level"] is not None else -90.0 for s in stats]

    spk_info: Dict[str, Dict] = {}
    if voice_refs:
        for spk in sorted(set(speakers), key=lambda k: int(k[1:])):
            idx = [j for j, k in enumerate(speakers) if k == spk and not stats[j].get("event")]
            if not idx:
                continue
            pick = _pick_voice(idx, spans, expr, levels, cfg.spk_target_s)
            if not pick:
                continue
            med = float(np.median([levels[j] for j in pick]))
            parts = []
            for j in pick:
                a, b = spans[j]
                c = y[int(max(0, a - 0.05) * REF_SR):int((b + 0.1) * REF_SR)]
                parts += [c * 10 ** (np.clip(med - levels[j], -6, 6) / 20), np.zeros(int(0.2 * REF_SR), np.float32)]
            ref = np.concatenate(parts[:-1])[:int(15 * REF_SR)]
            if len(ref) < REF_SR * 1.0:
                continue
            spk_info[spk] = {"ref": _save(out_dir / f"voice_{spk}.wav", ref), "lines": len(idx),
                             "seconds": round(len(ref) / REF_SR, 1)}
    for j, i in enumerate(timed):
        r = {"speaker": speakers[j], "expr": expr[j], "src_level": None if stats[j].get("event") else round(levels[j], 2),
             "src_pauses": stats[j]["pauses"], "spk": (spk_info.get(speakers[j]) or {}).get("ref"), "emo": None}
        if emo_refs and not stats[j].get("event"):
            a, b = _emo_window(j, spans, speakers, total, cfg)
            if b - a >= 0.4:
                r["emo"] = _save(out_dir / "emo" / f"{i + 1:04d}.wav", y[int(a * REF_SR):int(b * REF_SR)])
        res[i] = r
    log.info("expressive dub: %d line(s) (%d sound events), %d speaker(s), mean expressiveness %.2f",
             len(timed), sum(events), len(set(speakers)), float(np.mean(expr)))
    return {"sentences": res, "speakers": spk_info}


# ------------------------------------------------------------------ pauses in the dub
def clause_positions(text: str, lang: str) -> List[float]:
    """Where the translation can pause: clause punctuation inside the sentence, as the
    fraction of its syllables (characters for CJK) spoken before it."""
    from ..translate.isochrony import latin_syllables

    plain = re.sub(r"<([^<>|]+)\|[^<>]+>", r"\1", text).strip()
    cjk = textprep.is_cjk_lang(lang)
    marks = textprep.SENT_END + textprep.CLAUSE + "，,;；:：…—" if cjk else ",;:—…" + ".!?"
    base = lang.lower().split("-")[0]
    units, cuts = 0.0, []
    if cjk:
        for k, ch in enumerate(plain):
            if ch in marks:
                if k < len(plain) - 1:
                    cuts.append(units)
            elif not ch.isspace():
                units += 1
    else:
        for m in re.finditer(r"[^\s]+", plain):
            w = m.group(0)
            core = re.sub(r"[^\w']", "", w)
            if core:
                units += latin_syllables(core, base)
            if w[-1] in marks and m.end() < len(plain.rstrip()):
                cuts.append(units)
    if units <= 0:
        return []
    return [round(c / units, 3) for c in cuts if 0 < c < units]


def _voiced_time(y: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray]:
    """Frame times and the cumulative fraction of voiced (speech) frames at each."""
    db = _frame_db(y, sr)
    if len(db) == 0:
        return np.zeros(1), np.zeros(1)
    v = (db > np.percentile(db, 95) - 30).astype(float)
    cum = np.cumsum(v) / max(1.0, v.sum())
    return np.arange(len(db)) * 0.01, cum


def place_pauses(y: np.ndarray, text: str, pauses: Sequence[Dict], lang: str, sr: int,
                 max_pause: float = 1.2, tol: float = 0.2) -> Tuple[np.ndarray, int]:
    """Lengthen the take's silence at the clause boundary matching each of the
    original line's pauses (by position in the sentence) to the original's length.
    Only lengthens, only at punctuation the model already paused at.  Returns
    (audio, number of pauses placed)."""
    if not pauses:
        return y, 0
    bounds = clause_positions(text, lang)
    gaps = silent_gaps(y, sr, 0.04, depth=32.0)
    if not bounds or not gaps:
        return y, 0
    times, cum = _voiced_time(y, sr)
    used_b, used_g, inserts = set(), set(), []
    for p in sorted(pauses, key=lambda p: -p["dur"]):
        cand = [k for k in range(len(bounds)) if k not in used_b and abs(bounds[k] - p["at"]) <= tol]
        if not cand:
            continue
        k = min(cand, key=lambda k: abs(bounds[k] - p["at"]))
        t_exp = float(times[min(len(times) - 1, int(np.searchsorted(cum, bounds[k])))])
        gc = [g for g in range(len(gaps)) if g not in used_g
              and abs((gaps[g][0] + gaps[g][1]) / 2 - t_exp) <= max(0.25, 0.15 * len(y) / sr)]
        if not gc:
            continue
        g = min(gc, key=lambda g: abs((gaps[g][0] + gaps[g][1]) / 2 - t_exp))
        used_b.add(k)
        used_g.add(g)
        add = min(p["dur"], max_pause) - (gaps[g][1] - gaps[g][0])
        if add > 0.05:
            inserts.append((int((gaps[g][0] + gaps[g][1]) / 2 * sr), int(add * sr)))
    if not inserts:
        return y, 0
    parts, prev = [], 0
    for at, n in sorted(inserts):
        parts += [y[prev:at], np.zeros(n, dtype=y.dtype)]
        prev = at
    parts.append(y[prev:])
    return np.concatenate(parts), len(inserts)


def dynamics_gains(take_levels: Sequence[float], src_levels: Sequence[Optional[float]],
                   range_db: float) -> List[float]:
    """dB gain per take so that every take sits at the takes' median level plus the
    original line's offset from the originals' median (capped at ``range_db``)."""
    tl = np.asarray(take_levels, dtype=float)
    ok = tl > -60
    target = float(np.median(tl[ok])) if ok.any() else -23.0
    src = [s for s in src_levels if s is not None]
    smed = float(np.median(src)) if src else 0.0
    out = []
    for t, s in zip(take_levels, src_levels):
        if t <= -60:
            out.append(0.0)
            continue
        off = 0.0 if s is None else float(np.clip(s - smed, -range_db, range_db))
        out.append(float(np.clip(target + off - t, -6 - range_db, 6 + range_db)))
    return out
