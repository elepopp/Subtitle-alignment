import numpy as np
import pytest

from subalign.tts import acoustics as ac

SR = 48000


def _pink(sec, seed=0):
    from scipy.signal import lfilter
    w = np.random.default_rng(seed).standard_normal(int(sec * SR))
    return lfilter([0.049922035, -0.095993537, 0.050612699, -0.004408786], [1, -2.494956002, 2.017265875, -0.522189400], w).astype(np.float32) * 0.1


def test_band_levels_normalised_at_1k():
    b = ac.band_levels(_pink(3), SR)
    assert len(b) == len(ac.CENTRES) and b[np.argmin(abs(ac.CENTRES - 1000))] == pytest.approx(0)
    # pink noise: roughly flat per 1/3 octave
    m = (ac.CENTRES >= 200) & (ac.CENTRES <= 8000)
    assert np.ptp(b[m]) < 4


def test_eq_gains_and_apply():
    target = np.zeros(len(ac.CENTRES))
    target[(ac.CENTRES >= 2000) & (ac.CENTRES <= 5000)] = 4.0
    target[ac.CENTRES > 10000] = 20.0                 # missing top octave: never lifted
    dub = np.zeros(len(ac.CENTRES))
    g = ac.eq_gains(target, dub, max_db=6, smooth=1)
    assert np.all(g[ac.CENTRES > 10000] == 0)
    assert np.mean(g[(ac.CENTRES >= 300) & (ac.CENTRES <= 3000)]) == pytest.approx(0, abs=1e-9)    # level kept
    assert g[np.argmin(abs(ac.CENTRES - 3175))] - g[np.argmin(abs(ac.CENTRES - 500))] == pytest.approx(4, abs=0.01)
    assert np.all(np.abs(ac.eq_gains(target * 10, dub, max_db=6, smooth=1)) <= 6 + 1e-9)
    y = _pink(4)
    z = ac.apply_eq(y, SR, g)
    d = ac.band_levels(z, SR) - ac.band_levels(y, SR)
    k = np.argmin(abs(ac.CENTRES - 3175)), np.argmin(abs(ac.CENTRES - 500))
    assert d[k[0]] - d[k[1]] == pytest.approx(4, abs=1.0)
    assert len(z) == len(y)
    assert ac.apply_eq(y, SR, np.zeros(len(ac.CENTRES))) is y


def _bursts(n=8, rev=None):
    t = np.arange(int(0.5 * SR)) / SR
    tone = (np.sin(2 * np.pi * 220 * t) * 0.3).astype(np.float32)
    y = np.concatenate([np.concatenate([tone, np.zeros(int(0.6 * SR), np.float32)]) for _ in range(n)])
    return ac.add_room(y, SR, rev, 0.4) if rev else y


def test_decay_rate_dry_vs_room():
    dry = ac.decay_rate(_bursts(), SR)
    wet = ac.decay_rate(_bursts(rev=150.0), SR)
    assert dry is not None and wet is not None and dry > 3 * wet
    assert 60 < wet < 400


def test_fit_room_finds_the_wet_level():
    target = ac.decay_rate(_bursts(rev=150.0), SR)
    wet, got = ac.fit_room(_bursts(), SR, target)
    assert wet > 0 and got == pytest.approx(target, rel=0.5)
    assert ac.fit_room(_bursts(), SR, ac.decay_rate(_bursts(), SR))[0] == 0.0
