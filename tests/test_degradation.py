"""Tests of analysis/public/degradation.py: the original and tablet-like signal conditions
(article section 6.3.2, Appendix A).

    python -m tests.test_degradation
"""

import numpy as np

from analysis.public import degradation as D

PITCH, W, CX, CY = 80.0, 1288.0, 960.0, 540.0
SEED, KEY = 20261015, 12345


def synth(rate=1000.0, T=60.0, lines=10):
    """Noise-free ideal trace: sawtooth x (left to right on each line) and staircase y."""
    t = np.arange(0, T, 1.0 / rate)
    line = np.minimum((t / (T / lines)).astype(int), lines - 1)
    frac = (t - line * (T / lines)) / (T / lines)
    x = CX - W / 2 + frac * W
    y = CY - PITCH * (lines - 1) / 2 + line * PITCH
    return t, x, y


def resampled_30hz(t, x, y):
    """The input resampled to 30 Hz with missing samples removed, used as a noise-free reference."""
    t2, x2, y2 = D.resample(t, x, y, 30.0)
    m = np.isfinite(x2) & np.isfinite(y2)
    return t2[m], x2[m], y2[m]


def test_resample_is_interpolation():
    t, x, y = resampled_30hz(*synth())
    tb, xb, yb = synth()
    assert abs(np.median(np.diff(t)) - 1 / 30) < 1e-9
    xi = np.interp(t, tb, xb)
    yi = np.interp(t, tb, yb)
    assert np.abs(x - xi).max() < 1e-6 and np.abs(y - yi).max() < 1e-6
    print("ok resampling = linear interpolation, values unchanged")


def test_original():
    """original = no resampling (the samples as published, missing samples removed only)."""
    tb, xb, yb = synth()
    xb2 = xb.copy()
    xb2[100:110] = np.nan
    t, x, y = D.make_signal(tb, xb2, yb, "original", PITCH, W, CX, CY, SEED, KEY)
    assert (
        len(t) == len(tb) - 10
        and np.array_equal(t, tb[np.isfinite(xb2)])
        and np.array_equal(x, xb2[np.isfinite(xb2)])
    )
    assert D.SIGNALS == ("original", "tablet-like")
    print("ok original keeps the samples as published")


def test_ar1_correlation():
    """ar1: stationary SD and lag-1 correlation exp(-dt/tau) on the 29.4 Hz grid."""
    t = np.arange(0, 600.0, 1 / D.RATE_HZ)
    tau = 0.3
    d = D.ar1(t, 2.0, tau, np.random.default_rng(0))
    r1 = np.corrcoef(d[:-1], d[1:])[0, 1]
    expect = np.exp(-(1 / D.RATE_HZ) / tau)
    assert abs(r1 - expect) < 0.03 and abs(d.std() / 2.0 - 1) < 0.1, (r1, expect, d.std())
    print("ok AR(1) SD and correlation = exp(-dt/tau)")


def test_determinism_and_independence():
    """Same seed and trial give the same signal, and another trial gives another."""
    t0, x0, y0, ev = _staircase(PITCH, W, CX, CY)
    a = D.make_signal(t0, x0, y0, "tablet-like", PITCH, W, CX, CY, SEED, KEY, ev=ev)
    b = D.make_signal(t0, x0, y0, "tablet-like", PITCH, W, CX, CY, SEED, KEY, ev=ev)
    assert all(np.array_equal(u, v) for u, v in zip(a, b))
    c = D.make_signal(t0, x0, y0, "tablet-like", PITCH, W, CX, CY, SEED, KEY + 1, ev=ev)
    assert not np.array_equal(a[2], c[2])
    print("ok reproducible (same seed, same output), independent across trials")


def _staircase(pitch, w, cx, cy, rate=1000.0, T=60.0, lines=10):
    """Noise-free staircase trace with its line transition times."""
    t = np.arange(0, T, 1.0 / rate)
    line = np.minimum((t / (T / lines)).astype(int), lines - 1)
    frac = (t - line * (T / lines)) / (T / lines)
    return (
        t,
        cx - w / 2 + frac * w,
        cy - pitch * (lines - 1) / 2 + line * pitch,
        [k * (T / lines) for k in range(lines)],
    )


def _within_line_resid(t, y, ev, line_y):
    """y minus the interval mean for each line interval (the within-line residual)."""
    seg = np.searchsorted(np.asarray(ev), t, side="right") - 1
    res = []
    for L in range(len(line_y)):
        m = seg == L
        if m.sum() >= 15:
            res.append(y[m] - y[m].mean())
    return np.concatenate(res)


def test_tablet_like():
    """Five checks: sampling rate, exact session gains and offsets, session-table
    resampling, within-line correlation and invariance to the stimulus geometry."""
    t0, x0, y0, ev = _staircase(PITCH, W, CX, CY)
    a = D.make_signal(t0, x0, y0, "tablet-like", PITCH, W, CX, CY, SEED, KEY, ev=ev)
    # (1) Sampling rate 29.4 Hz
    assert abs(np.median(np.diff(a[0])) - 1 / D.RATE_HZ) < 1e-9
    # (2) With only the gain/offset layers on and a fixed session row, exactly that row is realized
    affine_only = dict(D.TABLET, y_line_sd=0, x_line_sd=0, y_slow_sd=0, x_slow_sd=0, tilt_mean=0, tilt_sd=0)
    sess = np.array([[0.70, 40.0, 0.80, -30.0]])
    rng = D.trial_rng(SEED, KEY)
    t, x, y = D.tablet_like(t0, x0, y0, ev, PITCH, W, CX, CY, rng, params=affine_only, sessions=sess)
    tb, xb, yb = resampled_30hz(t0, x0, y0)
    xi = np.interp(t, tb, xb)
    yi = np.interp(t, tb, yb)
    # Staircase and sawtooth signals give interpolation errors at boundary samples when the grids differ,
    # so the check uses the median residual, which is dominated by samples away from the boundaries
    ey = CY + 0.70 * (yi - CY) + 40.0
    ex = CX + 0.80 * (xi - CX) - 30.0
    assert np.median(np.abs(y - ey)) < 1e-6 and np.median(np.abs(x - ex)) < 1e-6, (
        np.median(np.abs(y - ey)),
        np.median(np.abs(x - ex)),
    )
    assert np.mean(np.abs(y - ey) < 1e-6) > 0.9 and np.mean(np.abs(x - ex) < 1e-6) > 0.9
    # (3) Gain and offset are drawn as a whole row of the measured session table: with noise off, the
    #     realized (gain, offset) values must match one row exactly, and 300 draws must hit many distinct
    #     rows (sampling with replacement)
    S = D.tablet_sessions()
    hits = set()
    for k in range(300):
        rng = D.trial_rng(SEED, KEY + k)
        t, x, y = D.tablet_like(t0, x0, y0, ev, PITCH, W, CX, CY, rng, params=affine_only, sessions=S)
        yi = np.interp(t, tb, yb)
        xi = np.interp(t, tb, xb)
        ok = False
        for ridx, row in enumerate(S):
            ey = CY + row[0] * (yi - CY) + row[1]
            ex = CX + row[2] * (xi - CX) + row[3]
            if np.median(np.abs(y - ey)) < 1e-6 and np.median(np.abs(x - ex)) < 1e-6:
                ok = True
                hits.add(ridx)
                break
        assert ok, f"draw {k}: realized affine matches no session row"
    assert len(hits) >= 100, len(hits)
    # (4) All components: the within-line y residual is strongly correlated between successive
    #     samples, as in the tablet recordings
    t, x, y = a
    r = _within_line_resid(t, y, ev, np.unique(y0))
    r1 = float(np.corrcoef(r[:-1], r[1:])[0, 1])
    assert r1 > 0.94, r1

    # (5) Units: with another geometry (pitch 40, width 600) the ratio within-line y SD / line pitch is
    #     the same (within 15%)
    def ratio(pitch, w, key):
        tt, xx, yy, evv = _staircase(pitch, w, CX, CY)
        sds = []
        for kk in range(20):
            t, x, y = D.make_signal(
                tt, xx, yy, "tablet-like", pitch, w, CX, CY, SEED, key + kk, ev=evv
            )
            sds.append(_within_line_resid(t, y, evv, np.unique(yy)).std() / pitch)
        return float(np.median(sds))

    r80, r40 = ratio(PITCH, W, 700), ratio(40.0, 600.0, 900)
    assert abs(r40 / r80 - 1) < 0.15, (r80, r40)
    print(
        f"ok tablet-like: 29.4 Hz, exact gain/offset, session-table resampling, lag-1 {r1:.3f}, geometry invariant ({r80:.3f} vs {r40:.3f})"
    )


if __name__ == "__main__":
    for f in (
        test_resample_is_interpolation,
        test_original,
        test_ar1_correlation,
        test_determinism_and_independence,
        test_tablet_like,
    ):
        f()
    print("ALL OK")
