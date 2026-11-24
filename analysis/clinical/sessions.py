"""Clinical trials with their ground truth (article sections 3.1.3, 3.1.4 and 6.4).

`load_trials` returns, for one passage, a dict per participant with

    t, x       gaze sample times (s) and horizontal position (px): the input of SLATE
    fixations  fixations of the reading interval [start, end, x, y]: the input of the
               comparison algorithms and the unit of line assignment
    line       ground-truth line of each fixation (-1 = unrelated to reading)
    scored     the fixations that are scored (section 6.4)
    entries    reference line-entry times (s) for the transition scoring
    group      diagnostic group
    site       collection site
    audio      audio recording of the trial

The ground truth is read from three tables in SLATE_CLINICAL_DATA/ground_truth: trials.csv (one
row per trial with its group, site, gaze record and audio recording), fixations.csv (the
fixations with their line) and line_entries.csv (the time at which each line was entered).
"""

import pandas as pd

from analysis.clinical.recordings import load_gaze
from analysis.paths import CLINICAL_DATA

GROUND_TRUTH = CLINICAL_DATA / "ground_truth"
KEYS = {"participant": str, "passage": str}


def trial_table():
    """One row per trial: participant, passage, group, site, gaze_file, audio_file."""
    return pd.read_csv(GROUND_TRUTH / "trials.csv", dtype=KEYS)


def load_trials(passage):
    """Trials of one passage with their ground truth: {participant: dict} (module docstring)."""
    trials = trial_table()
    exact = dict(dtype=KEYS, float_precision="round_trip")
    fixations = pd.read_csv(GROUND_TRUTH / "fixations.csv", **exact)
    entries = pd.read_csv(GROUND_TRUTH / "line_entries.csv", **exact)
    fixations = fixations[fixations.passage == passage].groupby("participant")
    entries = entries[entries.passage == passage].groupby("participant")
    out = {}
    for trial in trials[trials.passage == passage].itertuples():
        t, x = load_gaze(trial.gaze_file)
        f = fixations.get_group(trial.participant)
        out[trial.participant] = dict(
            t=t,
            x=x,
            fixations=f[["start", "end", "x", "y"]].to_numpy(float),
            line=f.line.to_numpy(int),
            scored=f.scored.to_numpy(bool),
            entries=sorted(entries.get_group(trial.participant).time),
            group=trial.group,
            site=trial.site,
            audio=CLINICAL_DATA / trial.audio_file,
        )
    return out
