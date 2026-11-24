"""Line and corrected position of each fixation in one call (article sections 5.1, 5.3 and 5.4).

    table = correct_fixations(fixations, t, x, lines)

runs the line-transition detection on the gaze samples, assigns every fixation to a line and
applies the horizontal recalibration of that line. The fixations can come from any fixation
detection. SLATE only uses their start and end times and their positions.
"""

import numpy as np
import pandas as pd

from slate.detector import assign_lines, detect_line_transitions
from slate.recalibration import line_parameters


def correct_fixations(
    fixations, t, x, lines, transitions=None, first_at_start=False, last_line_reached=None,
    alpha=0.0,
):
    """Return one row per fixation with its line and corrected position.

    fixations : array with one row per fixation: start (s), end (s), x (px), y (px), on the
        time axis of the samples, or a DataFrame with these four columns
    t, x : gaze sample times in seconds and horizontal gaze position in pixels, as for
        `detect_line_transitions`
    lines : one dict per text line in reading order with `left` and `right` (edges of the line,
        px), `y` (vertical position of the line, px) and `n_word` (number of words on the line).
        `last_w` (width of the last word, px) is needed only when `alpha` is not 0
    transitions : line-transition times, if they were computed before (for example with the
        speech-based supplementation). Detected from the samples if not given
    first_at_start, last_line_reached : passed to `detect_line_transitions`
    alpha : the right end of each line is mapped to the right edge of the text line minus
        `alpha` widths of its last word (section 5.4), 0 by default

    The returned DataFrame keeps the recorded values (start, end, x, y) and adds
        line         line number from 1 (0 for fixations before the first-line entry)
        x_corrected  recalibrated horizontal position, (x - offset) / gain of the line
        y_corrected  vertical position of the assigned text line
    x_corrected and y_corrected are NaN where there is no line or no recalibration.
    """
    if isinstance(fixations, pd.DataFrame) and {"start", "end", "x", "y"} <= set(fixations.columns):
        fixations = fixations[["start", "end", "x", "y"]]
    fixations = np.asarray(fixations, float)
    if fixations.size == 0:
        fixations = fixations.reshape(0, 4)
    t = np.asarray(t, float)
    x = np.asarray(x, float)
    geometry = [dict(line, idx=k) for k, line in enumerate(lines, 1)]
    if transitions is None:
        widths = [line["right"] - line["left"] for line in geometry]
        transitions = detect_line_transitions(
            t, x, line_widths=widths, first_at_start=first_at_start,
            last_line_reached=last_line_reached,
        )
    transitions = np.asarray(transitions, float)

    table = pd.DataFrame(fixations[:, :4], columns=["start", "end", "x", "y"])
    table["line"] = assign_lines((table.start + table.end) / 2, transitions, n_lines=len(geometry))
    table["x_corrected"] = np.nan
    table["y_corrected"] = np.nan
    for k, line in enumerate(geometry, 1):
        table.loc[table.line == k, "y_corrected"] = float(line["y"])
    order = np.argsort(t, kind="stable")
    params = line_parameters(t[order], x[order], transitions, geometry, alpha=alpha)
    for row in params.itertuples():
        own = table.line == row.line
        if np.isfinite(row.gain) and row.gain > 0:
            table.loc[own, "x_corrected"] = (table.x[own] - row.offset) / row.gain
    return table
