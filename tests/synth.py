"""Synthetic 'singing' generator used by the tests.

Each syllable is a harmonic tone with vibrato; most syllables start with a
short noise burst (consonant), some are legato (pitch change only).  Line-final
syllables are sustained, lines are separated by silence (or accompaniment).
"""
from __future__ import annotations

import numpy as np

SR = 16000


def synth_song(lines, seed: int = 0, legato_prob: float = 0.3, gap: float = 1.2,
               accompaniment: float = 0.0, lead_in: float = 1.0, speech: bool = False):
    """lines: list of syllable counts per line.

    Returns (audio, truth) where truth is a list of lines of (start, end).
    """
    rng = np.random.default_rng(seed)
    out = [np.zeros(int(lead_in * SR), dtype=np.float32)]
    t = lead_in
    truth = []
    scale = [0, 2, 4, 5, 7, 9, 11, 12]
    for n_syl in lines:
        line = []
        base = 196.0 * 2 ** (rng.integers(0, 5) / 12)
        prev_deg = None
        for k in range(n_syl):
            last = k == n_syl - 1
            if speech:
                d = rng.uniform(0.12, 0.28)
            else:
                d = rng.uniform(1.0, 1.8) if last else rng.choice([0.18, 0.25, 0.35, 0.5, 0.7])
            deg = int(rng.choice(scale))
            if prev_deg is not None and deg == prev_deg:
                deg = scale[(scale.index(deg) + 2) % len(scale)]
            prev_deg = deg
            f = base * 2 ** (deg / 12)
            n = int(d * SR)
            tt = np.arange(n) / SR
            vib = 0.0 if speech else 0.25 * np.sin(2 * np.pi * 5.5 * tt) * np.clip(tt / 0.3, 0, 1)
            if speech:
                f = f * (1 + 0.15 * np.linspace(0.5, -0.5, n))   # gliding intonation
            phase = 2 * np.pi * np.cumsum(f * 2 ** (vib / 12)) / SR
            tone = sum(np.sin(h * phase) / h for h in range(1, 6)).astype(np.float32)
            env = np.minimum(1, np.minimum(tt / 0.015, (d - tt) / 0.03 + 0.2)).clip(0, 1)
            sig = 0.25 * tone * env
            legato = (not speech) and k > 0 and rng.random() < legato_prob
            if not legato:
                b = int(0.03 * SR)
                sig[:b] = sig[:b] * 0.3 + 0.15 * rng.standard_normal(b) * np.hanning(2 * b)[b:]
            out.append(sig.astype(np.float32))
            line.append((t, t + d))
            t += d
        truth.append(line)
        g = gap * rng.uniform(0.7, 1.4)
        out.append(np.zeros(int(g * SR), dtype=np.float32))
        t += g
    y = np.concatenate(out)
    # keep sample-accurate timeline consistent with float accumulation
    if accompaniment > 0:
        tt = np.arange(len(y)) / SR
        acc = accompaniment * (np.sin(2 * np.pi * 110 * tt) + 0.5 * np.sin(2 * np.pi * 164.8 * tt))
        y = y + acc.astype(np.float32)
    return y.astype(np.float32), truth
