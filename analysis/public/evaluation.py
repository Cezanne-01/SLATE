"""Helpers for the public oral-reading datasets (article sections 3.2, 6.3.2, 6.4 and 7.3).

Used by analysis.public.run: the trial geometry, the fixation coordinates under a signal
condition (mean of the samples inside each fixation interval, section 6.3.2), and scoring against
the dataset's line labels. The public datasets have no manually marked transition times, so the
reference transitions are derived from the fixation line labels as for the comparison algorithms
(section 6.4).
"""

import zlib

import numpy as np

from analysis import scoring
from analysis.public.degradation import TABLET_PITCH


def trial_seed(s):
    """Seed material that is stable across processes and runs (Python's str hash is randomized
    by PYTHONHASHSEED)."""
    return zlib.crc32(str(s).encode()) % (2**31)


def fix_from_samples(fix, t, x, y):
    """Fixation coordinates from the samples of a signal condition: the mean of the samples
    inside each fixation interval [start, end], or the first sample after the interval if it
    holds none. Returns (N, 4): start, end, x, y."""
    out = fix[:, :4].copy()
    if len(t) == 0:
        return out
    for i in range(len(fix)):
        a, b = np.searchsorted(t, fix[i, 0]), np.searchsorted(t, fix[i, 1], side="right")
        if b > a:
            out[i, 2] = x[a:b].mean()
            out[i, 3] = y[a:b].mean()
        else:
            k = min(a, len(t) - 1)
            out[i, 2] = x[k]
            out[i, 3] = y[k]
    return out


def reference_transitions(fix):
    """Reference line transitions: start of the longest continuous run on each line (discarded fixations excluded)."""
    keep = fix[:, 4] > 0
    f = fix[keep]
    if len(f) == 0:
        return []
    return scoring.transitions_from_lines(f, f[:, 4].astype(int))


def predicted_transitions(pred, fix_in):
    """Predicted labels -> line transitions on the fix_in grid. If fix_in carries reference labels,
    discarded fixations (line_true == 0) are excluded."""
    if pred is None:
        return []
    if fix_in.shape[1] >= 5:
        keep = fix_in[:, 4] > 0
        return (
            scoring.transitions_from_lines(fix_in[keep], np.asarray(pred)[keep])
            if keep.any()
            else []
        )
    return scoring.transitions_from_lines(fix_in, np.asarray(pred))


def score_trial(lines, fixations, transitions, reference):
    """Score one method on one trial -> dict(accuracy, n_fixations, tp, fp, fn).

    accuracy: share of the dataset's fixations assigned to their labeled line. Fixations the
    dataset discarded stay in the denominator and count as errors, as in Carr et al. (2022).
    tp, fp, fn: line-transition counts (analysis.scoring.transition_counts)."""
    truth = fixations[:, 4].astype(int)
    hits = 0 if lines is None else int(np.sum((np.asarray(lines) == truth) & (truth > 0)))
    tp, fp, fn = scoring.transition_counts(list(transitions), list(reference))
    return dict(
        accuracy=hits / len(truth) if len(truth) else np.nan,
        n_fixations=len(truth),
        tp=tp,
        fp=fp,
        fn=fn,
    )


def geometry(trial):
    """(line_Y, line widths, line pitch, text width, text center x) of a public-data trial."""
    line_Y = np.asarray(trial["line_Y"], float)
    line_X = np.asarray(trial["line_X"], float)
    pitch = float(np.median(np.diff(line_Y))) if len(line_Y) > 1 else TABLET_PITCH
    widths = line_X[:, 1] - line_X[:, 0]
    text_w = float(np.nanmax(line_X[:, 1]) - np.nanmin(line_X[:, 0]))
    text_cx = float((np.nanmax(line_X[:, 1]) + np.nanmin(line_X[:, 0])) / 2)
    return line_Y, widths, pitch, text_w, text_cx
