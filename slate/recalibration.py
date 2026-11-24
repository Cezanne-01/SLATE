"""Horizontal recalibration (article section 5.4, Figure 7).

After line assignment, each line's landing point x_land and reach point x_reach are mapped to
the left edge L and the right edge R of the corresponding text line, which gives a per-line
gain g and offset o:

    g = (x_reach - x_land) / (R* - L),    o = x_land - g * L,    corrected x = (x - o) / g

The reach target is R* = R - ALPHA * (width of the last word). With ALPHA = 0, the default, it
is the right edge of the line. A larger value moves it inward. Where the last fixation of a
line falls depends on the reader, the text and the task, so it is left as a parameter.

The landing point x_land is the minimum of x~ before sustained rightward progress begins after
the line transition. The reach point x_reach is the maximum from the landing until the next
transition (`landing_reach`).
The gain is taken from the line's own two points except for lines whose reach point does not
represent the line end: the last line (no following transition) and short lines (swept width
below SHORT_REL of the trial reference with a shorter time per word). For those lines the gain
is interpolated from neighboring lines. The offset is still computed from the line's own
landing point.

    params = line_parameters(t, x, transitions, lines)   # lines: geometry of each text line
    x_corrected = recalibrate(t, x, transitions, params)
"""

import numpy as np
import pandas as pd

LANDING_WIN = 1.0  # window (s) after a transition in which the initial landing point is taken
ALPHA = 0.0  # reach target = right edge of the line - ALPHA * width of its last word
SMOOTH_N = 5  # median filter width for the anchors (samples), 0.17 s at 30 Hz
SHORT_REL = 0.70  # a line with a gain below this fraction of the trial reference may be short
GAIN_FLOOR = 0.15  # smallest gain used
GAIN_FALLBACK = 0.73  # gain used when no line of the trial yields its own gain

# Landing and reach points. Sustained progress: x~ rises at least one character (CHAR_PX) above
# its running minimum and stays at least half a character above it for the next HOLD_S seconds.
# If too few samples are available or no sustained progress is found, the initial anchors
# (minimum within LANDING_WIN after the transition, maximum before the next transition) are
# kept, and the reason is recorded in `anchor_src`.
CHAR_PX = 33.7  # character width (px)
HOLD_S = 1.0  # time (s) for which progress must be sustained
MIN_LINE_SAMPLES = 6  # minimum number of samples between this transition and the next
MIN_REACH_SAMPLES = 3  # minimum number of samples for the reach search


def landing_reach(t, xs, b, nxt, seg2, p0):
    """(x_land, t_land, x_max, t_max, src) for one line.

    t, xs : time and smoothed x (median filter of SMOOTH_N samples)
    b, nxt: start time of this line, start time of the next line (None for the last line)
    seg2  : reach-interval mask of the initial anchors (finite samples after their landing, up
            to the end of the line)
    p0    : dict(x_land, t_land, x_max, t_max), the initial anchors
    """
    out = dict(p0)
    out["src"] = "sustained progress"
    end = float(nxt) if nxt is not None else float(t[-1])
    m = (t >= b) & (t < end) & np.isfinite(xs)
    idx = np.flatnonzero(m)
    if len(idx) < MIN_LINE_SAMPLES:
        out["src"] = "window minimum (few samples)"
        return out
    tt, xx = t[idx], xs[idx]
    run = np.minimum.accumulate(xx)
    k_start = None
    for k in range(len(xx)):
        if xx[k] < run[k] + CHAR_PX:
            continue
        w = (tt >= tt[k]) & (tt <= tt[k] + HOLD_S)
        if tt[k] + HOLD_S > tt[-1]:
            break
        if np.all(xx[w] >= run[k] + CHAR_PX * 0.5):
            k_start = k
            break
    if k_start is None:
        out["src"] = "window minimum"
        return out
    j = int(np.argmin(xx[: k_start + 1]))
    out["x_land"], out["t_land"] = float(xx[j]), float(tt[j])
    # Reach: from the new landing to the end of the line, within the same mask seg2.
    m2 = seg2 & (t > out["t_land"])
    if m2.sum() >= MIN_REACH_SAMPLES:
        k2 = int(np.argmax(np.where(m2, xs, -np.inf)))
        out["x_max"], out["t_max"] = float(xs[k2]), float(t[k2])
    return out


def line_parameters(t, x, transitions, lines, alpha=ALPHA):
    """Per-line landing point, reach point, gain and offset as a DataFrame, one row per line read.

    t, x        : sample times (s) and horizontal gaze position (px)
    transitions : line-transition times (s) on the same time axis. The k-th is the start of line k
    lines       : one dict per text line in reading order with idx (1-based), left and right
                  edge (px) and n_word (number of words on the line). last_w (width of the
                  last word, px) is needed only when alpha is not 0
    alpha       : how far the reach target lies inside the right edge, in widths of the last word
    """
    xs = pd.Series(x).rolling(SMOOTH_N, center=True, min_periods=1).median().to_numpy()
    starts = sorted(float(b) for b in transitions)
    rows = []
    for i, (start, line) in enumerate(zip(starts, lines)):
        nxt = starts[i + 1] if i + 1 < len(starts) else None

        # initial anchors: lowest x within LANDING_WIN, then highest x up to the next transition
        end = start + LANDING_WIN if nxt is None else min(start + LANDING_WIN, nxt)
        landing = (t >= start) & (t <= end) & np.isfinite(xs)
        if landing.sum() < 3:
            continue
        j = int(np.argmin(xs[landing]))
        x_land, t_land = float(xs[landing][j]), float(t[landing][j])
        # up to the next transition, or to the end of the record for the last line
        reach_end = np.inf if nxt is None else nxt
        reach = (t > t_land) & (t <= reach_end) & np.isfinite(xs)
        if reach.sum() < 3:
            continue
        k = int(np.argmax(xs[reach]))
        anchors = dict(
            x_land=x_land, t_land=t_land, x_max=float(xs[reach][k]), t_max=float(t[reach][k])
        )
        anchors = landing_reach(t, xs, start, nxt, reach, anchors)
        target = line["right"] - (alpha * line["last_w"] if alpha else 0.0)

        rows.append(
            dict(
                line=line["idx"],
                left=line["left"],
                right=line["right"],
                t_transition=start,
                t_land=anchors["t_land"],
                x_land=anchors["x_land"],
                t_reach=anchors["t_max"],
                x_reach=anchors["x_max"],
                anchor_src=anchors["src"],
                last=int(nxt is None),
                gain_raw=(anchors["x_max"] - anchors["x_land"]) / (target - line["left"]),
                time_per_word=(anchors["t_max"] - anchors["t_land"]) / line["n_word"],
            )
        )
    p = pd.DataFrame(rows)
    if p.empty:
        return p

    # Short lines: gain and time per word both below the trial reference (last line excluded).
    body = p[p["last"] == 0]
    ref_gain = float(body.gain_raw.median()) if len(body) else np.nan
    ref_time = float(body.time_per_word.median()) if len(body) else np.nan
    relative = p.gain_raw / ref_gain if np.isfinite(ref_gain) and ref_gain > 0 else np.nan
    p["short"] = ((relative < SHORT_REL) & (p.time_per_word < ref_time)).fillna(False).astype(int)

    # A line keeps its own gain unless it is the last line, a short line or below GAIN_FLOOR.
    # Otherwise the gain is interpolated from the lines that keep theirs.
    own = (p["last"] == 0) & (p.short == 0) & (p.gain_raw > GAIN_FLOOR)
    if own.any():
        p["gain"] = np.interp(p.line, p.line[own], p.gain_raw[own])
        p.loc[own, "gain"] = p.loc[own, "gain_raw"]
        p["gain_src"] = np.where(
            own, "own", np.where(p.short == 1, "neighbors (short line)", "neighbors (last line)")
        )
    else:
        p["gain"] = GAIN_FALLBACK
        p["gain_src"] = "fallback"
    p["offset"] = p.x_land - p.gain * p.left
    return p


def recalibrate(t, x, transitions, params):
    """Recalibrated x per sample: (x - offset) / gain of the line being read. Samples before the
    first transition use the values of the first line. Samples of a line without parameters
    are NaN."""
    out = np.full(len(x), np.nan)
    if params is None or params.empty:
        return out
    starts = np.sort(np.asarray(transitions, float))
    line_of_sample = np.maximum(np.searchsorted(starts, t, side="right"), 1)
    for row in params.itertuples():
        samples = line_of_sample == row.line
        out[samples] = (x[samples] - row.offset) / max(row.gain, GAIN_FLOOR)
    return out
