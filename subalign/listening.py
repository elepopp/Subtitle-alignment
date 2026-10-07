"""Blind listening tests (盲听对比) between two dubs of the same translation.

Numbers such as pitch error or voice similarity only stand in for what a listener
hears; a change to the dubbing is judged here by ear:

* two dubbing projects (conditions A and B) are paired sentence by sentence (same
  translated sentence) and ``n`` pairs are drawn at random
* every item plays the **original line** and two versions **X** and **Y**; which of
  A / B is X is random per item and kept on the server (the page only ever sees
  item numbers), so the listener cannot tell
* X, Y and the original are loudness-matched: a louder version is otherwise heard
  as the better one
* the listener answers X / Y / about the same; the reveal counts A and B wins and
  gives a two-sided sign test (ties left out): the chance of a split at least this
  uneven if A and B were equally good
"""
from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

SR = 24000
LEVEL_DB = -20.0            # active speech level every clip is brought to
CHOICES = ("x", "y", "same")


# ------------------------------------------------------------------ pairing
def _key(seg: Dict) -> Tuple:
    return (seg.get("dt_id"),) if seg.get("dt_id") is not None else \
        (round(seg.get("src_start") or -1, 1), seg.get("text"))


def pair_segments(segs_a: Sequence[Dict], segs_b: Sequence[Dict]) -> List[Tuple[Dict, Dict]]:
    """Sentences generated in both projects: by translation sentence id, else by
    original start time + text."""
    b = {_key(s): s for s in segs_b if s.get("audio")}
    return [(s, b[_key(s)]) for s in segs_a if s.get("audio") and _key(s) in b]


# ------------------------------------------------------------------ audio
def level_match(y: np.ndarray, sr: int = SR, target: float = LEVEL_DB) -> np.ndarray:
    from .tts.expressive import level_db

    g = 10 ** ((target - level_db(y, sr)) / 20)
    y = (y * g).astype(np.float32)
    peak = float(np.max(np.abs(y))) if len(y) else 0.0
    return y * (0.95 / peak) if peak > 0.95 else y


def _take(path: Path) -> np.ndarray:
    from .audio.io import load_audio
    from .tts.dubbing import trim_take

    return trim_take(load_audio(path, SR).astype(np.float32), SR)


def _write(path: Path, y: np.ndarray) -> None:
    from .audio.io import save_audio

    save_audio(path, level_match(y), SR)


# ------------------------------------------------------------------ statistics
def sign_test_p(wins_a: int, wins_b: int) -> float:
    """Two-sided exact binomial (sign) test, ties excluded."""
    n = wins_a + wins_b
    if n == 0:
        return 1.0
    k = max(wins_a, wins_b)
    tail = sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n
    return round(min(1.0, 2 * tail), 4)


def tally(test: Dict) -> Dict:
    a = b = same = 0
    for it in test["items"]:
        c = (test.get("answers") or {}).get(str(it["id"]), {}).get("choice")
        if c == "same":
            same += 1
        elif c in ("x", "y"):
            if (c == "x") != it["flip"]:
                a += 1
            else:
                b += 1
    return {"a": a, "b": b, "same": same, "answered": a + b + same, "n": len(test["items"]),
            "p": sign_test_p(a, b)}


# ------------------------------------------------------------------ tests
def create(tdir: Path, proj_a: Dict, dir_a: Path, proj_b: Dict, dir_b: Path, label_a: str, label_b: str,
           n: int = 20, source: Optional[Path] = None, title: str = "", seed: Optional[int] = None) -> Dict:
    """A new test in ``tdir``: up to ``n`` random paired sentences, clips written
    (original from ``source`` at the sentence's original time, when given)."""
    from .audio.io import load_audio

    pairs = pair_segments(proj_a["segments"], proj_b["segments"])
    if not pairs:
        raise ValueError("两个配音项目没有共同生成过的句子")
    rng = random.Random(seed)
    pick = sorted(rng.sample(range(len(pairs)), min(n, len(pairs))))
    tdir.mkdir(parents=True, exist_ok=True)
    src = load_audio(source, SR).astype(np.float32) if source and Path(source).exists() else None
    items = []
    for k, i in enumerate(pick, 1):
        sa, sb = pairs[i]
        flip = rng.random() < 0.5                         # True: X is B
        xa, xb = _take(dir_a / sa["audio"]), _take(dir_b / sb["audio"])
        _write(tdir / f"{k:03d}_x.wav", xb if flip else xa)
        _write(tdir / f"{k:03d}_y.wav", xa if flip else xb)
        orig = False
        if src is not None and sa.get("src_start") is not None and sa.get("src_end") is not None:
            a, b = max(0.0, sa["src_start"] - 0.15), sa["src_end"] + 0.2
            clip = src[int(a * SR):int(b * SR)]
            if len(clip) > 0.3 * SR:
                _write(tdir / f"{k:03d}_o.wav", clip)
                orig = True
        items.append({"id": k, "flip": flip, "a": {"sid": sa["id"], "file": sa["audio"]},
                      "b": {"sid": sb["id"], "file": sb["audio"]}, "text": sa.get("text"),
                      "source_text": sa.get("source_text"), "original": orig})
    test = {"version": 1, "title": title or f"{label_a} vs {label_b}", "created": round(time.time(), 3),
            "a": {"label": label_a, "project": dir_a.name, "title": proj_a.get("title")},
            "b": {"label": label_b, "project": dir_b.name, "title": proj_b.get("title")},
            "pairs": len(pairs), "items": items, "answers": {}, "revealed": False}
    save(tdir, test)
    return test


def load(tdir: Path) -> Dict:
    return json.loads((tdir / "test.json").read_text(encoding="utf-8"))


def save(tdir: Path, test: Dict) -> None:
    tmp = tdir / "test.json.tmp"
    tmp.write_text(json.dumps(test, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(tdir / "test.json")


def answer(tdir: Path, item: int, choice: str, note: str = "") -> Dict:
    if choice not in CHOICES:
        raise ValueError(f"choice must be one of {CHOICES}")
    test = load(tdir)
    if not any(it["id"] == item for it in test["items"]):
        raise KeyError(item)
    test.setdefault("answers", {})[str(item)] = {"choice": choice, "note": note[:500], "t": round(time.time(), 3)}
    save(tdir, test)
    return test


def reveal(tdir: Path) -> Dict:
    test = load(tdir)
    test["revealed"] = True
    save(tdir, test)
    return test


def view(test: Dict) -> Dict:
    """What the page may see: before the reveal, nothing that tells A from B."""
    out = {k: test[k] for k in ("title", "created", "pairs", "revealed")}
    out["labels"] = {"a": test["a"]["label"], "b": test["b"]["label"]}
    out["answers"] = {k: v["choice"] for k, v in (test.get("answers") or {}).items()}
    out["items"] = [{"id": it["id"], "text": it.get("text"), "source_text": it.get("source_text"),
                     "original": it.get("original", False)} for it in test["items"]]
    if test["revealed"]:
        out["tally"] = tally(test)
        out["a"], out["b"] = test["a"], test["b"]
        for it, o in zip(test["items"], out["items"]):
            o["x_is"] = "b" if it["flip"] else "a"
    return out
