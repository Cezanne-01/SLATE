"""Adaptive line-transition detection and line assignment (article sections 5.1 and 5.3,
constants in section 6.1).

A return sweep moves back across most of a line in a short time, so it appears as a leftward
move larger than any movement within a line. Line transitions are found in the smoothed
horizontal gaze coordinate x~ (section 5.1.1) in four steps:

    1. candidates (5.1.1): the leftward displacement d_i = x~_i - x~_{i+w} over T_W exceeds
       theta_l = KAPPA * r_l * S, where S is the 5th-95th percentile range of x~
    2. minimum interval (5.1.2): after an accepted transition, new candidates are ignored for
       g0 = clip(BETA * S / v, G_MIN, G_MAX), with v the trial's rightward progress speed
    3. shape check (5.1.3, Figure 5): candidates that look like an out-and-back excursion or a
       within-line regression are rejected
    4. first-line entry (5.1.4): within the first ENTRY_WIN seconds, the lower threshold
       KAPPA0 * S is used

The output is the sequence of transition times B = (b_1, ..., b_n). Each transition is the start
of one line, so the number of lines read equals the number of detections, and a time point
belongs to the line whose transition precedes it (5.3). The same constants are used for all data.
"""

import numpy as np
import pandas as pd

KAPPA = 0.30  # threshold coefficient: theta = KAPPA * S

# Minimum interval (section 5.1.2)
BETA = 0.7  # ratio of the minimum interval to the time needed to read one line
G_MIN = 1.0  # lower bound of g0 (s)
G_MAX = 4.5  # upper bound of g0 (s), which limits the interval for slow readers
G_FALLBACK = 3.0  # interval (s) for records whose speed cannot be measured

# Time windows (sections 5.1.1 and 6.1). Windows are defined in seconds and converted to samples
# with the median sampling interval of each record, so the same settings apply to data recorded
# at other sampling rates.
SMOOTH_S = 0.17  # moving-median window for x~ (s), 5 samples at 30 Hz
T_W = 0.30  # displacement window for d_i (s), 9 samples at 30 Hz

# First-line entry (section 5.1.4). After calibration, gaze rests near the horizontal center of
# the screen, so the first move goes from the center to the start of line 1, about half the
# longest line width. The entry is time-stamped at the landing on line 1 (the lowest x within
# LANDING_WIN after the move). Later transitions are time-stamped at the center of the
# displacement window.
ENTRY_WIN = 4.0  # search window for the first-line entry (s after the passage is shown)
KAPPA0 = 0.10  # first-line-entry coefficient: threshold KAPPA0 * S
LANDING_WIN = 1.0  # window (s) after the move in which the landing on line 1 is located
G_FIRST = 1.0  # minimum interval (s) after the first-line entry when no line widths are given

# Line-width factor r_l (section 5.1.1). While line l is being read, the next
# return sweep and the time needed to read the line are proportional to the width of line l:
# r_l = clip(W_l / median width, R_MIN, R_MAX). The minimum interval is
# g_l = max(G_LINE_MIN, g0 * r_l - delta_l), with delta_1 = LANDING_WIN, and the threshold uses
# min(r_l, 1) from the second line on (the threshold is only lowered).
R_MIN = 0.35  # lower bound of r_l
R_MAX = 1.3  # upper bound of r_l
G_LINE_MIN = 0.75  # lower bound of the scaled minimum interval (s)

# Shape check (section 5.1.3)
# (b) out-and-back excursion to the right: in the ROUNDTRIP_S before the candidate, the trace
#     rose (window maximum minus window start) by more than ROUNDTRIP_R * d_i.
ROUNDTRIP_S = 1.2
ROUNDTRIP_R = 1.2
# (c) leftward excursion or within-line regression: within RETURN_S after the end of the
#     displacement window, the trace recovered by more than RETURN_R * d_i.
RETURN_S = 0.5
RETURN_R = 0.8


def detect_line_transitions(t, x, line_widths=None, last_line_reached=None, first_at_start=False):
    """Return the times (s) at which the reader enters a new line, in increasing order.

    t : sample times in seconds, with the passage shown at the first sample. Samples that
        are out of time order are sorted.
    x : horizontal gaze position in pixels, same length as t, without missing values. Any
        unit can be used as long as x increases to the right and `line_widths` uses the same.
    line_widths : width of each passage line in pixels, in reading order. If given, the
        line-width factor is applied and detection stops at len(line_widths) transitions (the
        passage line count is the upper bound on the number of lines read).
    last_line_reached : last line reached K, if known (section 6.3.1). Detections beyond K are
        removed by `limit_to_last_line`.
    first_at_start : treat t[0] as the start of line 1 instead of searching for the first-line
        entry, for recordings in which gaze is already on line 1 when the passage appears.

    Raises ValueError if the inputs cannot be a gaze recording in these units.
    """
    t = np.asarray(t, float)
    x = np.asarray(x, float)
    if t.ndim != 1 or t.shape != x.shape:
        raise ValueError("t and x must be one-dimensional and of the same length")
    if len(t) == 0:
        return np.array([], float)
    if not (np.all(np.isfinite(t)) and np.all(np.isfinite(x))):
        raise ValueError("t and x must not contain missing values. Remove those samples first")
    if np.any(np.diff(t) < 0):  # sort by time
        order = np.argsort(t, kind="stable")
        t, x = t[order], x[order]
    if len(t) > 1 and np.median(np.diff(t)) >= 1.0:
        raise ValueError("t must be in seconds (the median sampling interval is 1 s or more)")
    xs, span, win = smooth(t, x)
    transitions = np.array(_detect(t, xs, span, win, line_widths, first_at_start), float)
    if last_line_reached is not None:
        transitions = limit_to_last_line(t, x, transitions, last_line_reached)
    return transitions


def assign_lines(times, transitions, n_lines):
    """Assign each time point (e.g. fixation midpoints, in s) to a line (section 5.3).

    The k-th transition marks the start of line k (1-based), so times before the first
    transition are labeled 0 (before reading onset)."""
    idx = np.searchsorted(np.asarray(transitions, float), np.asarray(times, float), side="right")
    return np.clip(idx, 0, n_lines)


def smooth(t, x):
    """Smoothed coordinate x~, horizontal range S (px) and displacement window w (samples)."""
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 1 / 30
    sm = max(3, int(round(SMOOTH_S / dt)) | 1)  # odd, so that the median window is symmetric
    win = max(2, int(round(T_W / dt)))
    xs = pd.Series(x).rolling(sm, center=True, min_periods=1).median().to_numpy()
    span = float(np.percentile(xs, 95) - np.percentile(xs, 5))
    return xs, span, win


def minimum_interval(t, xs, span):
    """Minimum interval g0 = clip(BETA * S / v, G_MIN, G_MAX) in s.

    v = sum of the positive increments of x~ on a T_W grid / record duration. Returns
    G_FALLBACK when the record is too short or v is not positive."""
    duration = float(t[-1] - t[0])
    if duration <= T_W * 2:
        return G_FALLBACK
    grid = np.arange(t[0], t[-1], T_W)
    progress = np.clip(np.diff(np.interp(grid, t, xs)), 0, None).sum()
    v = progress / duration
    if not np.isfinite(v) or v <= 0:
        return G_FALLBACK
    return float(np.clip(BETA * span / v, G_MIN, G_MAX))


def limit_to_last_line(t, x, transitions, last_line):
    """Keep at most K = `last_line` transitions (known-K condition, section 6.3.1).

    K counts line entries, including the first-line entry. With more than K transitions, the
    first-line entry and the K-1 others with the largest leftward displacement over the
    displacement window are kept (ties keep the earlier one). With K or fewer the transitions
    are returned unchanged, and none is added or moved."""
    if last_line < 1:
        raise ValueError("last_line must be a positive integer")
    transitions = np.asarray(transitions, float)
    if len(transitions) <= last_line:
        return transitions
    xs, _, win = smooth(t, x)
    drops = np.full(len(transitions), np.inf)  # the first-line entry is always kept
    for j, tm in enumerate(transitions[1:], start=1):
        i = int(np.argmin(np.abs(t - tm))) - win // 2
        i = max(0, min(i, len(xs) - win - 1))
        drops[j] = xs[i] - xs[i + win]
    keep = np.sort(np.argsort(-drops, kind="stable")[:last_line])
    return transitions[keep]


def _out_and_back(t, xs, i, drop):
    """Shape check (b): the rise in the ROUNDTRIP_S before sample i exceeds ROUNDTRIP_R x drop."""
    i0 = int(np.searchsorted(t, t[i] - ROUNDTRIP_S))
    seg = xs[i0 : i + 1]
    return len(seg) >= 3 and float(seg.max() - seg[0]) > ROUNDTRIP_R * drop


def _recovers(t, xs, i_end, drop):
    """Shape check (c): within RETURN_S after sample i_end the trace recovers more than
    RETURN_R x drop."""
    i1 = int(np.searchsorted(t, t[i_end] + RETURN_S, side="right"))
    return i1 > i_end + 1 and (xs[i_end:i1].max() - xs[i_end]) > RETURN_R * drop


def _first_line_entry(t, xs, span, disp):
    """Time of the first-line entry (5.1.4), or None.

    Leftward moves within ENTRY_WIN are tried from the largest down. The first one that exceeds
    KAPPA0 * S and passes shape check (b) is accepted and time-stamped at its landing."""
    head = np.flatnonzero(t - t[0] <= ENTRY_WIN)
    head = head[head < len(disp)]
    for j in head[np.argsort(disp[head])]:
        if disp[j] >= -KAPPA0 * span:
            break
        if _out_and_back(t, xs, j, -disp[j]):
            continue
        j1 = int(np.searchsorted(t, t[j] + LANDING_WIN, side="right"))
        return t[j + int(np.argmin(xs[j : max(j1, j + 1)]))]
    return None


def _detect(t, xs, span, win, widths, first_at_start):
    """Transition times from the smoothed trace (steps 1 to 4 of the module docstring)."""
    if span <= 0 or len(xs) <= win:
        return []
    g0 = minimum_interval(t, xs, span)
    disp = xs[win:] - xs[:-win]

    def width_factor(n):
        """r_l of the line being read after n transitions."""
        if 1 <= n <= len(widths):
            return max(R_MIN, min(R_MAX, widths[n - 1] / np.median(widths)))
        return 1.0

    out, last = [], -np.inf
    if first_at_start:
        last = float(t[0])
    else:
        entry = _first_line_entry(t, xs, span, disp)
        if entry is not None:
            last = entry
    if np.isfinite(last):
        out.append(last)
    n = len(out)

    # Candidates are collected with the lowest possible threshold, then each is tested in time
    # order against the threshold and minimum interval of the line being read.
    lowest = KAPPA * (R_MIN if widths is not None else 1.0)
    for i in np.flatnonzero(disp < -lowest * span):
        if widths is not None and n >= len(widths):  # passage line count reached
            break
        tm = t[min(i + win // 2, len(t) - 1)]
        after_entry = n == 1 and not first_at_start
        if widths is not None and n >= 1:
            r = width_factor(n)
            need = max(G_LINE_MIN, g0 * r - (LANDING_WIN if after_entry else 0.0))
            # the factor applies to the threshold from the second transition on and only lowers it
            theta = KAPPA * (min(r, 1.0) if n >= 2 else 1.0) * span
        else:
            need = G_FIRST if after_entry else g0
            theta = KAPPA * span
        if disp[i] >= -theta or tm - last < need:
            continue
        if _out_and_back(t, xs, i, -disp[i]) or _recovers(
            t, xs, min(i + win, len(xs) - 1), -disp[i]
        ):
            continue
        out.append(tm)
        last = tm
        n += 1
    return out
