"""Comparison algorithms of Carr et al. (2022) and slice (article section 6.2).

Calls the unmodified algorithms in `baselines.carr_drift` (attach, chain, cluster, compare, merge,
regress, segment, slice, split, stretch, warp) on a fixation sequence and converts the corrected
fixation y to line numbers. All run with the defaults of the implementation. The thresholds of
slice are defined as multiples of the line pitch (Glandorf & Schroeder, 2021), so its pixel
defaults, which refer to a line pitch of 64 px, are rescaled to the line pitch of the text.
"""

import numpy as np

from baselines import carr_drift

ALGOS = ["attach", "chain", "cluster", "compare", "merge", "regress", "segment", "slice",
         "split", "stretch", "warp"]
SLICE_REFERENCE_PITCH = 64.0  # line pitch (px) that the pixel defaults of slice refer to


def line_pitch(line_Y):
    """Line pitch of a text (px): median distance between consecutive lines."""
    return float(np.median(np.diff(line_Y)))


def slice_thresholds(pitch):
    """Thresholds of slice at the line pitch of the text."""
    scale = pitch / SLICE_REFERENCE_PITCH
    return {
        "x_thresh": 192 * scale,
        "y_thresh": 32 * scale,
        "w_thresh": 32 * scale,
        "n_thresh": 90 * scale,
    }


def run_algorithm(name, fix, line_Y, word_XY, pitch, bounded=False):
    """Fixations (start, end, x, y, ...) -> assigned line number (1..len(line_Y)) per fixation,
    or None if the algorithm fails. compare and warp receive the word positions, the others the
    line positions. `pitch` is the line pitch of the text (px), used by slice. With
    bounded=True, compare is limited to min(3, number of candidate lines) nearest lines."""
    if len(fix) < 3:
        return None
    fxy = fix[:, 2:4].copy()
    fn = getattr(carr_drift, name)
    try:
        if name == "compare" and bounded:
            out = fn(fxy, word_XY.copy(), n_nearest_lines=min(3, len(np.unique(word_XY[:, 1]))))
        elif name in ("compare", "warp"):
            out = fn(fxy, word_XY.copy())
        elif name == "slice":
            out = fn(fxy, line_Y.copy(), **slice_thresholds(pitch))
        else:
            out = fn(fxy, line_Y.copy())
    except Exception:
        return None
    if out is None or not np.all(np.isfinite(out[:, 1])):
        return None
    return np.argmin(np.abs(line_Y[None, :] - out[:, 1][:, None]), axis=1) + 1


def majority_vote(predictions, fixations, line_Y):
    """C-WOC: per-fixation majority vote over the algorithms' line numbers. Ties go to the tied
    line nearest the fixation y. `predictions` holds one array per algorithm that did not fail."""
    votes = np.vstack(predictions)
    lines = np.empty(votes.shape[1], dtype=int)
    for j in range(votes.shape[1]):
        counts = np.bincount(votes[:, j])
        tied = np.flatnonzero(counts == counts.max())
        lines[j] = tied[np.argmin(np.abs(line_Y[tied - 1] - fixations[j, 3]))]
    return lines
