"""CONF-LA (Kaltenberger et al., 2026), one of the comparison algorithms (article section 6.2).

`confla_lines` calls `confidence()` from `baselines.confla_algorithms` in post-hoc
mode (one segment = the whole trial) and converts its output to line numbers.
"""

import numpy as np

from baselines import confla_algorithms


def confla_lines(fix, line_Y, text_width, params):
    """Fixations (start, end, cx, cy, ...) -> assigned line number (1..len(line_Y)), or None if
    CONF-LA fails. Fixations that CONF-LA leaves unassigned (low confidence) keep their
    original y and are mapped to the nearest line, as in `baselines.carr.run_algorithm`."""
    if len(fix) < 3:
        return None
    fxy = fix[:, 2:4].copy()
    tms = fix[:, 0:2] * 1000.0  # the original uses ms (eyekit fixation.start/end)
    try:
        out, _, _ = confla_algorithms.confidence(
            fxy, tms, line_Y.astype(float), float(text_width), **params
        )
    except Exception:
        return None
    y = out[:, 1].astype(float)
    if not np.all(np.isfinite(y)):
        return None
    return np.argmin(np.abs(line_Y[None, :] - y[:, None]), axis=1) + 1
