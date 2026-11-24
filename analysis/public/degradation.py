"""Signal conditions for the public oral-reading datasets (article section 6.3.2, Appendix A).

  original     the recording as published (Table 3 and the left side of Figure 9).
  tablet-like  the five-component degradation of Appendix A (low sampling rate, range
               compression, displacement, slow vertical drift, within-line slope, Eq. A1) with
               the constants TABLET and the measured session table session_affine.csv (right side
               of Figure 9).

Spatial magnitudes are ratios so that they transfer across stimuli (Appendix A.6): vertical values
in multiples of the line pitch, horizontal values in multiples of the text width. Random numbers
are seeded per trial, so trials are independent and a re-run is identical. Tested by
tests/test_degradation.py.
"""

from pathlib import Path

import numpy as np
import pandas as pd

RATE_HZ = 29.4  # sampling rate of the tablet recordings (component 1)
SIGNALS = ("original", "tablet-like")
_STREAM_ID = 549629346  # part of the seed of the degradation's random stream

# Clinical stimulus geometry. Pixel values of TABLET are transferred to other stimuli as ratios of
# TABLET_PITCH and TABLET_TEXT_W.
TABLET_PITCH = 80.0  # stimulus line pitch (px)
TABLET_TEXT_W = 1288.0  # stimulus text width (px)

# Tablet-like degradation constants (applied by tablet_like)
# Five components:
#   1. Low sampling rate: linear resampling to 29.4 Hz.
#   2. Range compression and 3. displacement: the session values (g_y, o_y, g_x, o_x) are drawn
#      together, with replacement, from the measured session table (session_affine.csv, sessions
#      with at least min_lines_affine lines). Drawing whole rows keeps the correlations and tails
#      of the measured distribution. Displacement also includes a per-line horizontal landing
#      shift (x_line_sd).
#   4. Slow vertical drift: AR(1) (y_slow_sd, y_slow_tau), with a small horizontal AR(1) component.
#   5. Within-line slope: per-session slope of y on x drawn from N(tilt_mean, tilt_sd), in
#      clinical-stimulus pixels (vertical per horizontal), rescaled to the dataset geometry.
# Pixel values refer to the clinical stimulus and are rescaled by line pitch and text width.
TABLET = dict(
    x_line_sd=65.2,  # per-line horizontal landing shift: SD (px)
    y_line_sd=0.0,  # no per-line vertical shift
    y_slow_sd=52.0,  # slow vertical drift: SD (px)
    y_slow_tau=1.5,  # slow vertical drift: time constant (s)
    x_slow_sd=15.0,  # slow horizontal component: SD (px)
    x_slow_tau=0.34,  # slow horizontal component: time constant (s)
    tilt_mean=0.043,  # within-line slope: mean across sessions
    tilt_sd=0.065,  # within-line slope: SD across sessions
    min_lines_affine=5,  # minimum lines per session in the session table
)
SESSION_TABLE = Path(__file__).resolve().parent / "session_affine.csv"
_SESSIONS = None


def tablet_sessions(min_lines=None):
    """Measured session table as a matrix with columns (y_gain, y_offset_center_px, x_gain_ols,
    x_offset_center_px), restricted to sessions with at least min_lines lines."""
    global _SESSIONS
    if _SESSIONS is None:
        _SESSIONS = pd.read_csv(SESSION_TABLE)
    k = TABLET["min_lines_affine"] if min_lines is None else min_lines
    d = _SESSIONS[_SESSIONS.n_lines_used >= k]
    cols = ["y_gain", "y_offset_center_px", "x_gain_ols", "x_offset_center_px"]
    a = d[cols].to_numpy(float)
    assert len(a) >= 100 and np.isfinite(a).all(), (len(a), SESSION_TABLE)
    return a


def trial_rng(seed, trial_key):
    """Random generator of the tablet-like degradation of one trial."""
    return np.random.default_rng([seed, trial_key, _STREAM_ID, 0])


def resample(t, x, y, rate_hz, max_gap=0.2):
    """Resample onto a rate_hz grid by linear interpolation. Spans where the original samples have a gap above max_gap are NaN."""
    m = np.isfinite(x) & np.isfinite(y)
    t, x, y = t[m], x[m], y[m]
    if len(t) < 2:
        return t, x, y
    dt = 1.0 / rate_hz
    g = np.arange(t[0], t[-1] + dt / 2, dt)
    xi = np.interp(g, t, x)
    yi = np.interp(g, t, y)
    idx = np.searchsorted(t, g).clip(1, len(t) - 1)
    bad = (t[idx] - t[idx - 1]) > max_gap
    xi[bad] = np.nan
    yi[bad] = np.nan
    return g, xi, yi


def ar1(t, sd, tau, rng):
    """AR(1) noise (stationary variance sd^2, correlation time tau). Uneven spacing allowed."""
    n = len(t)
    if n == 0:
        return np.zeros(0)
    out = np.empty(n)
    out[0] = rng.normal(0, sd)
    a = np.exp(-np.diff(t) / tau)
    innov = rng.normal(0, 1, n - 1) * sd * np.sqrt(1 - a**2)
    for i in range(1, n):
        out[i] = a[i - 1] * out[i - 1] + innov[i - 1]
    return out


def tablet_like(t, x, y, ev, pitch, text_w, text_cx, text_cy, rng, params=None, sessions=None):
    """Tablet-like degradation (section 6.3.2, Appendix A, Eq. A1).

    Five components: 1. low sampling rate (29.4 Hz resampling), 2. range compression (session
    gains), 3. displacement (session offsets, plus a per-line horizontal landing shift), 4. slow
    vertical drift (AR(1), with a small horizontal component), 5. within-line slope (drawn per
    session). Gains and offsets are taken together as one row of the measured session table.
    Pixel values of TABLET are rescaled to the dataset line pitch and text width. Without ev the
    per-line horizontal shift is skipped.
    params and sessions override TABLET and the session table (used by the tests). Returns (t, x, y) with missing samples
    removed."""
    V = TABLET if params is None else params
    S = tablet_sessions() if sessions is None else np.asarray(sessions, float)
    py = pitch / TABLET_PITCH
    px = text_w / TABLET_TEXT_W
    t2, x2, y2 = resample(t, x, y, RATE_HZ)
    row = S[rng.integers(len(S))]
    gy, oy, gx, ox = float(row[0]), float(row[1]) * py, float(row[2]), float(row[3]) * px
    y2 = text_cy + gy * (y2 - text_cy) + oy
    x2 = text_cx + gx * (x2 - text_cx) + ox
    if ev is not None and len(ev):
        evs = np.asarray(ev, float)
        seg = np.searchsorted(evs, t2, side="right")
        # per-line shifts: vertical (SD y_line_sd) and horizontal (SD x_line_sd)
        yl = rng.normal(0, V["y_line_sd"] * py, len(evs) + 1)
        xl = rng.normal(0, V["x_line_sd"] * px, len(evs) + 1)
        y2 = y2 + yl[seg]
        x2 = x2 + xl[seg]
    tilt = rng.normal(V["tilt_mean"], V["tilt_sd"])
    y2 = y2 + ar1(t2, V["y_slow_sd"] * py, V["y_slow_tau"], rng) + tilt * (x2 - text_cx) * (py / px)
    x2 = x2 + ar1(t2, V["x_slow_sd"] * px, V["x_slow_tau"], rng)
    m = np.isfinite(x2) & np.isfinite(y2)
    return t2[m], x2[m], y2[m]


def make_signal(t, x, y, signal, pitch, text_w, text_cx, text_cy, seed, trial_key, ev=None):
    """Signal of one trial: "original" = the samples as published, "tablet-like" = tablet_like with
    the trial's random stream. Returns (t, x, y) with missing samples removed."""
    if signal == "tablet-like":
        return tablet_like(t, x, y, ev, pitch, text_w, text_cx, text_cy, trial_rng(seed, trial_key))
    if signal != "original":
        raise ValueError(signal)
    t2, x2, y2 = np.asarray(t, float), np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x2) & np.isfinite(y2)
    return t2[m], x2[m], y2[m]
