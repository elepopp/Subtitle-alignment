import math
import shutil
import subprocess

import numpy as np
import pytest
from scipy.signal import butter, sosfilt, stft

from subalign import studio
from subalign.studio import (StudioConfig, declick, deess, export, measure_loudness, mix_bgm, normalize_loudness,
                             reduce_plosives)

SR = 48000
needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None and not studio.BUNDLED_FFMPEG.exists(),
                                  reason="ffmpeg not available")


def _voice(seconds=6.0, f0=180.0, seed=0, gaps=((2.0, 2.6), (4.0, 4.5))):
    """Harmonic 'voice' (formant-ish spectrum, vibrato) with silent gaps."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SR)) / SR
    f = f0 * (1 + 0.02 * np.sin(2 * np.pi * 5 * t))
    ph = 2 * np.pi * np.cumsum(f) / SR
    y = sum(np.sin(k * ph) / k ** 0.7 for k in range(1, 30))
    y = y / np.abs(y).max() * 0.3 * (1 + 0.3 * np.sin(2 * np.pi * 3 * t))
    env = np.ones(len(t))
    r = int(0.01 * SR)                                       # 10 ms fades: a hard cut is itself a pop
    for a, b in gaps:
        a, b = int(a * SR), int(b * SR)
        env[a:b] = 0
        env[a - r:a] = np.linspace(1, 0, r)
        env[b:b + r] = np.linspace(0, 1, r)
    return (y * env + rng.standard_normal(len(t)) * 1e-4).astype(np.float32)


def _band_db(y, lo, hi, a=None, b=None):
    seg = y[int((a or 0) * SR):int(b * SR) if b else None]
    f, _, Z = stft(seg, SR, nperseg=2048, noverlap=1536)
    return 10 * np.log10(np.mean(np.abs(Z[(f >= lo) & (f < hi)]) ** 2) + 1e-20)


def test_deess_turns_down_sibilants_only():
    y = _voice()
    rng = np.random.default_rng(1)
    s = sosfilt(butter(4, [5000 / (SR / 2), 9000 / (SR / 2)], "band", output="sos"), rng.standard_normal(len(y)))
    sib = [(0.5, 0.65), (1.2, 1.35), (3.0, 3.15)]
    for a, b in sib:
        y[int(a * SR):int(b * SR)] += 0.4 * s[int(a * SR):int(b * SR)]
    out, info = deess(y, max_db=8)
    for a, b in sib:
        assert _band_db(y, 4500, 10000, a, b) - _band_db(out, 4500, 10000, a, b) > 3
    # the voice body between the "s" sounds is untouched
    assert abs(_band_db(y, 200, 4000, 1.5, 1.9) - _band_db(out, 200, 4000, 1.5, 1.9)) < 0.3


def test_declick_removes_impulses_and_leaves_voice():
    y = _voice(gaps=((2.0, 2.6), (4.0, 4.5)))
    clean = y.copy()
    rng = np.random.default_rng(2)
    in_pauses = [2.3, 4.2]                                   # mouth clicks between phrases
    for t in in_pauses + [0.31, 3.6]:                        # (and two buried in a very spiky voice)
        i = int(t * SR)
        y[i:i + 24] += 0.3 * np.hanning(24) * np.sign(rng.standard_normal(24))
    out, n = declick(y)
    assert n >= len(in_pauses)
    for t in in_pauses:
        i = int(t * SR)
        before = np.mean((y[i - 48:i + 72] - clean[i - 48:i + 72]) ** 2)
        after = np.mean((out[i - 48:i + 72] - clean[i - 48:i + 72]) ** 2)
        assert 10 * math.log10(before / after) > 10
    # glottal pulses of a voice are periodic, not isolated: no false clicks
    _, n_clean = declick(clean)
    assert n_clean == 0


def test_plosives_pulled_down_voice_kept():
    y = _voice(f0=110)                                       # deep voice: strong low band
    clean = y.copy()
    voice_lf = _band_db(clean, 40, 160, 0, 6)                # the speaker's usual low-band level
    rng = np.random.default_rng(3)
    sos = butter(4, 150 / (SR / 2), output="sos")
    pops = [0.8, 3.2, 5.0]
    for t in pops:
        m = int(0.09 * SR)
        x = sosfilt(sos, rng.standard_normal(m))
        x = x / np.abs(x).max() * np.minimum(1, np.arange(m) / (0.005 * SR)) * np.exp(-np.arange(m) / (0.03 * SR))
        a = int(t * SR)
        y[a:a + m] += 2.5 * x
        # make every pop a real blast relative to the speaker's usual low band (+20 dB)
        k = 10 ** ((voice_lf + 20 - _band_db(y, 40, 160, t, t + 0.08)) / 20)
        y[a:a + m] = clean[a:a + m] + (y[a:a + m] - clean[a:a + m]) * k
    out, n = reduce_plosives(y)
    assert n >= len(pops)
    for t in pops:
        assert _band_db(y, 40, 160, t, t + 0.08) - _band_db(out, 40, 160, t, t + 0.08) > 4
    # clean voices (deep and high) keep their low end
    for f0 in (90, 110, 150, 220):
        v = _voice(f0=f0)
        o, _ = reduce_plosives(v)
        assert abs(_band_db(v, 40, 160, 0, 6) - _band_db(o, 40, 160, 0, 6)) < 0.5


@needs_ffmpeg
def test_loudness_target_and_true_peak():
    y = _voice() * 2.5
    out, rep = normalize_loudness(y, -16.0, -1.5)
    m = measure_loudness(out)
    assert abs(m["lufs"] + 16.0) < 0.5
    assert m["true_peak"] <= -1.4


@needs_ffmpeg
def test_bgm_sits_at_the_requested_ratio():
    v = _voice(seconds=8, gaps=())
    t = np.arange(int(3 * SR)) / SR                         # shorter than the voice: looped
    music = np.vstack([np.sin(2 * np.pi * 220 * t), np.sin(2 * np.pi * 277 * t)]).astype(np.float32) * 0.5
    cfg = StudioConfig(bgm_ratio=0.2, bgm_duck=False, bgm_fade_in=0, bgm_fade_out=0)
    mix, info = mix_bgm(v, music, cfg)
    assert mix.shape == (2, len(v))
    bed = mix - np.vstack([v, v])
    rel = measure_loudness(bed)["lufs"] - measure_loudness(v)["lufs"]
    assert abs(rel - 20 * math.log10(0.2)) < 1.0
    assert abs(info["bgm_relative_db"] + 14.0) < 0.1


@needs_ffmpeg
@pytest.mark.parametrize("fmt,codec", [("wav", "pcm_s24le"), ("mp3", "mp3"), ("aac", "aac"), ("flac", "flac")])
def test_export_formats(tmp_path, fmt, codec):
    y = _voice(seconds=1, gaps=())
    p = export(y, tmp_path / f"x{studio.EXT[fmt]}", StudioConfig(format=fmt, bitrate="192k", channels=2))
    probe = shutil.which("ffprobe") or str(studio.BUNDLED_FFMPEG.with_name(studio.BUNDLED_FFMPEG.name.replace("ffmpeg", "ffprobe")))
    out = subprocess.run([probe, "-v", "error", "-show_entries", "stream=codec_name,channels,sample_rate", "-of", "csv=p=0",
                          str(p)], capture_output=True, text=True).stdout
    assert codec in out and ",2" in out.replace("48000,2", ",2") and "48000" in out


def test_breaths_and_coughs_found_in_pauses_only():
    from subalign.audio.features import analyze
    from subalign.audio.io import to_mono_16k
    from subalign.roughcut import RoughCutConfig, breaths_and_coughs, voiced_units

    y = _voice(seconds=8, gaps=((2.0, 3.0), (5.0, 6.0)))
    rng = np.random.default_rng(4)
    nb = sosfilt(butter(2, [400 / (SR / 2), 4000 / (SR / 2)], "band", output="sos"), rng.standard_normal(len(y)))
    a, b = int(2.3 * SR), int(2.65 * SR)                       # breath in the first pause
    y[a:b] += (0.012 * nb[a:b] / np.std(nb[a:b]) * np.hanning(b - a)).astype(np.float32)
    a = int(5.3 * SR)                                         # cough in the second pause
    m = int(0.4 * SR)
    cough = sosfilt(butter(2, [150 / (SR / 2), 6000 / (SR / 2)], "band", output="sos"), rng.standard_normal(m))
    y[a:a + m] += (0.3 * cough / np.std(cough) * np.exp(-np.linspace(0, 6, m)) * np.minimum(1, np.linspace(0, 40, m))).astype(np.float32)
    # an "s" inside speech (no pause around it) must not be taken for a breath
    a, b = int(1.0 * SR), int(1.12 * SR)
    y[a:b] = (0.05 * nb[a:b] / np.std(nb[a:b])).astype(np.float32)
    y16, _ = to_mono_16k(y, SR)
    feats = analyze(y16)
    found = breaths_and_coughs(voiced_units(feats), feats, RoughCutConfig(breaths="reduce", coughs=True))
    kinds = sorted((it.auto, round(it.start, 1)) for it in found)
    assert [k for k, _ in kinds] == ["breath", "cough"]
    assert 2.2 <= kinds[0][1] <= 2.5 and 5.2 <= kinds[1][1] <= 5.4
