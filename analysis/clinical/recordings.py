"""Gaze samples of a clinical recording (article section 3.1.4).

The gaze record of a trial is the json written by the eye-tracking SDK. Two kinds of samples are
removed: tracking failures (`screenState == "UNKNOWN"`) and samples more than MARGIN outside the
canvas. The coordinates of the remaining samples are not changed. The calibration period of
CALIBRATION_S before the passage is shown is then removed, and time starts at the first sample
after it.
"""

import json

import numpy as np
import pandas as pd

from analysis.paths import RAW_GAZE

CALIBRATION_S = 4.0  # calibration period before the passage is shown (s)
W, H = 1920, 1080  # canvas (px), as in the word boxes of the passages
MARGIN = 200.0  # samples farther than this outside the canvas are removed (px)


def load_gaze(gaze_file):
    """Gaze of one trial after the calibration period: sample times (s) and horizontal
    position (px)."""
    samples = pd.DataFrame(json.load(open(RAW_GAZE / gaze_file))["gazeDataList"])
    x, y = samples.gazeScreenPointX, samples.gazeScreenPointY
    tracked = samples.screenState != "UNKNOWN"
    near_canvas = x.between(-MARGIN, W + MARGIN) & y.between(-MARGIN, H + MARGIN)
    start = samples.timeStamp.iloc[0]
    samples = samples[tracked & near_canvas]
    t = (samples.timeStamp - start).to_numpy(float) / 1000.0
    t = t - t.min()
    x = samples.gazeScreenPointX.to_numpy(float)
    keep = (t >= CALIBRATION_S) & np.isfinite(x)
    return t[keep] - t[keep][0], x[keep]
