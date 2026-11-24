"""ETDD70 (article section 3.2.1) -> common format (analysis/public/FORMAT.md).

Czech children (dyslexia 35, control 35), SMI RED 250 Hz, reading aloud. Paths are relative to
SLATE_PUBLIC_DATA/etdd70:

    data/data/Subject_<id>_<task>_raw.csv        samples (time in microseconds, gaze of each eye)
    data/data/Subject_<id>_<task>_fixations.csv  fixations of the providers (start_ms/end_ms on the
                                                 sample clock in ms)
    rois/rois/<Task>_rois.csv                    line and sub-line rectangles (screen px)
    dyslexia_class_label.csv                     subject_id, class_id, label

Conversion:
  - Samples: tracking failures are written as (x, y) = (0, 0). Per eye these become NaN, and x, y
    are the mean of the two eyes (the available eye if only one is present). Time 0 is the first
    sample.
  - Fixations: start, end, orig_fix_x, orig_fix_y, line label. The line label is aoi_line
    "line_00N" -> N, and 0 outside the text.
  - Geometry: line centers and extents from the line rectangles, word positions from the centers
    of the sub-line rectangles, in screen px.

    python -m analysis.public.load_etdd70 [--tasks T4_Meaningful_Text] [--out PKL]

Writes SLATE_PUBLIC_DATA/_common/etdd70.pkl by default.
"""

import argparse
import os
import pickle
import re
import sys
import warnings

import numpy as np
import pandas as pd

from analysis.paths import COMMON, PUBLIC_DATA

ROOT = PUBLIC_DATA / "etdd70"
OUT = COMMON / "etdd70.pkl"
TASKS = {  # task -> ROI file
    "T1_Syllables": "Syllables_rois.csv",
    "T4_Meaningful_Text": "Meaningful_Text_rois.csv",
    "T5_Pseudo_Text": "Pseudo_Text_rois.csv",
}
RATE_HZ = 250.0
NOTES = (
    "ETDD70 (https://doi.org/10.5281/zenodo.13332134), CC BY 4.0. Coordinates in screen px. "
    "Samples: mean of both eyes. Fixations and line labels as provided with the dataset "
    "(aoi_line, 0 = no line)."
)


def read_rois(path):
    """Line and sub-line rectangles of one task -> (line_Y, line_X, word_XY). Sub-line centers
    serve as word positions."""
    r = pd.read_csv(path)
    lines = r[r["kind"] == "line"].sort_values("line")
    assert list(lines["line"].astype(int)) == list(range(1, len(lines) + 1)), path
    line_Y = (lines["y"] + lines["height"] / 2.0).to_numpy(float)
    line_X = np.c_[lines["x"].to_numpy(float), (lines["x"] + lines["width"]).to_numpy(float)]
    subs = r[r["kind"] == "sub-line"].sort_values(["line", "part"])
    wx = (subs["x"] + subs["width"] / 2.0).to_numpy(float)
    wy = line_Y[subs["line"].astype(int).to_numpy() - 1]
    return line_Y, line_X, np.c_[wx, wy]


def eye_xy(raw, side):
    """Gaze of one eye, with the (0, 0) tracking failures set to NaN."""
    x = raw[f"gaze_x_{side}"].to_numpy(float).copy()
    y = raw[f"gaze_y_{side}"].to_numpy(float).copy()
    lost = (x == 0) & (y == 0)
    x[lost] = np.nan
    y[lost] = np.nan
    return x, y


def read_samples(path):
    """Raw sample file -> (t in s from the first sample, x, y, first time stamp in microseconds)."""
    raw = pd.read_csv(path)
    t_us = raw["time"].to_numpy(np.int64)
    t0 = t_us[0]
    xl, yl = eye_xy(raw, "left")
    xr, yr = eye_xy(raw, "right")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # samples lost in both eyes
        x = np.nanmean(np.c_[xl, xr], axis=1)
        y = np.nanmean(np.c_[yl, yr], axis=1)
    return {"t": (t_us - t0) / 1e6, "x": x, "y": y}, t0


def read_fixations(path, t0):
    """Fixation file -> (N, 5) array: start, end (s on the sample time axis), x, y, line label."""
    fix = pd.read_csv(path)
    line = fix["aoi_line"].astype(str).str.extract(r"line_(\d+)")[0]
    return np.c_[
        fix["start_ms"].to_numpy(float) * 1e3 / 1e6 - t0 / 1e6,
        fix["end_ms"].to_numpy(float) * 1e3 / 1e6 - t0 / 1e6,
        fix["orig_fix_x"].to_numpy(float),
        fix["orig_fix_y"].to_numpy(float),
        line.fillna(0).astype(int).to_numpy(float),
    ]


def load_trial(root, sid, task, label, rois):
    """One subject and task -> trial dict in the common format."""
    d = os.path.join(root, "data", "data")
    samples, t0 = read_samples(f"{d}/Subject_{sid}_{task}_raw.csv")
    line_Y, line_X, word_XY = rois
    return {
        "dataset": "etdd70",
        "corpus": task,
        "trial_id": f"{sid}_{task}",
        "participant": str(sid),
        "group": label,
        "condition": "aloud",
        "language": "cs",
        "rate_hz": RATE_HZ,
        "samples": samples,
        "samples_source": "raw",
        "fixations": read_fixations(f"{d}/Subject_{sid}_{task}_fixations.csv", t0),
        "line_Y": line_Y.copy(),
        "line_X": line_X.copy(),
        "word_XY": word_XY.copy(),
        "notes": NOTES,
    }


def subject_ids(data_dir):
    """Subject numbers that have at least one raw sample file."""
    found = (re.match(r"Subject_(\d+)_.*_raw\.csv$", f) for f in os.listdir(data_dir))
    return sorted({int(m.group(1)) for m in found if m})


def build(root, tasks):
    """All subjects of the given tasks -> list of trials (task by task, subjects in order)."""
    lab = pd.read_csv(os.path.join(root, "dyslexia_class_label.csv"))
    labels = dict(zip(lab["subject_id"].astype(int), lab["label"]))
    d = os.path.join(root, "data", "data")
    sids = subject_ids(d)
    trials = []
    for task in tasks:
        rois = read_rois(os.path.join(root, "rois", "rois", TASKS[task]))
        for sid in sids:
            if not os.path.exists(f"{d}/Subject_{sid}_{task}_raw.csv"):
                print(f"skip {sid} {task}: no raw samples", file=sys.stderr)
                continue
            trials.append(load_trial(root, sid, task, labels[sid], rois))
    return trials


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--root", default=str(ROOT), help="ETDD70 folder (default: %(default)s)")
    ap.add_argument("--out", default=str(OUT), help="output pkl (default: %(default)s)")
    ap.add_argument("--tasks", nargs="+", default=list(TASKS), choices=list(TASKS))
    a = ap.parse_args()
    trials = build(a.root, a.tasks)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "wb") as fh:
        pickle.dump(trials, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"saved {len(trials)} trials -> {a.out}")


if __name__ == "__main__":
    main()
