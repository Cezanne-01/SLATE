"""Adedeji et al. (2022), "Return-sweep saccades in oral reading" (article section 3.2.2) ->
common format (analysis/public/FORMAT.md).

Samples, reading mode and stimulus geometry are read from the EyeLink ASC files, with fixations and
line labels from the processed copy of the DIST benchmark, aligned to the ASC time axis. Paths
are relative to SLATE_PUBLIC_DATA:

    adedeji2022/*/ABREV<n>.asc (or adedeji2022/*/*/ABREV<n>.asc)          40 participants
    dist/Abbr/processed_data/ABREV<n>_trial_id_<id>_fixations.csv        0-based assigned_line
    dist/Abbr/processed_data/ABREV<n>_trial_id_<id>_trial.json           ASC line of the TRIALID

ASC layout used here: monocular samples at 1000 Hz, "<time>\t<x>\t<y>\t..." with '.' for a
missing value. A trial runs from "MSG <time> TRIALID <id>" through "REGION CHAR" character boxes,
"DISPLAY ON" (time 0), "DISPLAY OFF" and "TRIAL OK". Aborted trials are skipped, and a TRIALID
completed twice is taken from the ASC line DIST names. The reading mode is the last
"ALOUD/SILENT BLOCK INSTRUCTION PRESENTED" message before the TRIALID. Only experimental trials
(TRIALID E*) are converted. The files contain Latin-1 bytes.

    python -m analysis.public.load_adedeji [--nproc 8]

Writes SLATE_PUBLIC_DATA/_common/adedeji2022.pkl.
"""

import argparse
import collections
import glob
import json
import os
import pickle
import re
from multiprocessing import Pool

import numpy as np
import pandas as pd

from analysis.paths import COMMON, PUBLIC_DATA

RAW_ROOT = str(PUBLIC_DATA / "adedeji2022")
DIST_DIR = str(PUBLIC_DATA / "dist" / "Abbr" / "processed_data")
OUT_PKL = str(COMMON / "adedeji2022.pkl")

DATASET = "adedeji2022"
CORPUS = "Abbr_raw"
LANGUAGE = "en"
RATE_HZ = 1000.0

NOTES = (
    "Adedeji et al. (2022), raw EyeLink ASC files (https://osf.io/ym76z/). Fixations and line "
    "labels from the DIST benchmark (https://osf.io/zt9gn/, assigned_line). Coordinates in "
    "screen px. Samples between DISPLAY ON (t = 0) and DISPLAY OFF. Line and word positions from "
    "the character boxes of the ASC files."
)

_RE_PNUM = re.compile(r"ABREV(\d+)\.asc$")
_RE_REGION = re.compile(r"^MSG\t(\d+) REGION CHAR (\d+) (\d+) (.) (-?\d+) (-?\d+) (-?\d+) (-?\d+)")
_RE_MSG = re.compile(r"^MSG\t(\d+) (.*)$")
BLOCKS = {
    "ALOUD BLOCK INSTRUCTION PRESENTED": "aloud",
    "SILENT BLOCK INSTRUCTION PRESENTED": "silent",
}


def find_asc_files():
    """ASC files of all participants, in participant order."""
    files = glob.glob(os.path.join(RAW_ROOT, "*", "ABREV*.asc")) + glob.glob(
        os.path.join(RAW_ROOT, "*", "*", "ABREV*.asc")
    )
    files = [f for f in files if _RE_PNUM.search(f)]
    return sorted(set(files), key=lambda p: int(_RE_PNUM.search(p).group(1)))


def _float_or_nan(s):
    s = s.strip()
    return np.nan if s in (".", "") else float(s)


def _new_trial(tid, line_idx, condition, eye):
    return dict(
        tid=tid,
        line_idx=line_idx,  # 0-based line of the TRIALID message, as trial_id_idx in DIST
        condition=condition,
        eye=eye,
        regions=[],
        samp=[],
        t_on=None,
        t_off=None,
        ok=False,
        aborted=False,
    )


def _completed(trial):
    """A trial counts when it was shown, closed with TRIAL OK and not aborted."""
    return (
        not trial["aborted"]
        and trial["t_on"] is not None
        and trial["t_off"] is not None
        and trial["ok"]
        and trial["tid"].startswith("E")
    )


def parse_asc(path):
    """One ASC file -> completed experimental trials, each with samples between DISPLAY ON and
    DISPLAY OFF, character boxes, reading mode and recorded eye."""
    with open(path, "r", encoding="latin-1") as fh:
        lines = fh.read().split("\n")
    trials = []
    cur = None
    condition = None
    eye = None
    for i, line in enumerate(lines):
        if not line:
            continue
        c0 = line[0]
        if c0.isdigit():
            if (
                cur is not None
                and cur["t_on"] is not None
                and cur["t_off"] is None
                and cur["tid"].startswith("E")
            ):
                f = line.split("\t")
                cur["samp"].append((int(f[0]), _float_or_nan(f[1]), _float_or_nan(f[2])))
        elif c0 == "M":
            m = _RE_REGION.match(line)
            if m:
                if cur is not None:
                    cur["regions"].append(
                        (int(m.group(2)), m.group(4)) + tuple(int(m.group(k)) for k in (5, 6, 7, 8))
                    )
                continue
            m = _RE_MSG.match(line)
            if not m:
                continue
            body = m.group(2).rstrip("\r")
            if body.startswith("TRIALID "):
                if cur is not None and _completed(cur):
                    trials.append(cur)
                cur = _new_trial(body.split()[1], i, condition, eye)
            elif body in BLOCKS:
                condition = BLOCKS[body]
            elif cur is not None:
                if body == "DISPLAY ON":
                    cur["t_on"] = int(m.group(1))
                elif body == "DISPLAY OFF":
                    cur["t_off"] = int(m.group(1))
                elif body == "TRIAL OK":
                    cur["ok"] = True
                elif body == "TRIAL ABORTED":
                    cur["aborted"] = True
        elif c0 == "S" and line.startswith("START"):
            f = line.split()
            eye = f[2] if len(f) > 2 else eye
            if cur is not None:
                cur["eye"] = eye
    if cur is not None and _completed(cur):
        trials.append(cur)
    return trials


def geometry_from_regions(regions):
    """REGION CHAR boxes -> line_Y, line_X, word_XY (word = run of non-space characters)."""
    regs = sorted(regions, key=lambda r: r[0])  # in character order
    yc = np.array([(r[3] + r[5]) / 2.0 for r in regs])
    line_Y = np.unique(yc)  # top -> bottom = reading order
    line_of = {y: k for k, y in enumerate(line_Y)}
    line_X = np.full((len(line_Y), 2), np.nan)
    words = []
    cur_w = None  # [xmin, xmax, line]
    for r, y in zip(regs, yc):
        k = line_of[y]
        if r[1] != " ":
            line_X[k, 0] = np.nanmin([line_X[k, 0], r[2]])
            line_X[k, 1] = np.nanmax([line_X[k, 1], r[4]])
            if cur_w is None or cur_w[2] != k:
                if cur_w is not None:
                    words.append(cur_w)
                cur_w = [r[2], r[4], k]
            else:
                cur_w[1] = max(cur_w[1], r[4])
        elif cur_w is not None:
            words.append(cur_w)
            cur_w = None
    if cur_w is not None:
        words.append(cur_w)
    word_XY = np.array([[(w[0] + w[1]) / 2.0, line_Y[w[2]]] for w in words], dtype=float)
    return line_Y.astype(float), line_X, word_XY


def build_trial(pnum, tr, dist_fx):
    """ASC trial + DIST fixations -> trial dict in the common format."""
    t_on = tr["t_on"]
    samp = np.array(tr["samp"], dtype=float).reshape(-1, 3)
    line_Y, line_X, word_XY = geometry_from_regions(tr["regions"])
    fixations = np.column_stack(
        [
            (dist_fx["start_time"].values.astype(float) - t_on) / 1000.0,
            (dist_fx["end_time"].values.astype(float) - t_on) / 1000.0,
            dist_fx["x"].values.astype(float),
            dist_fx["y"].values.astype(float),
            dist_fx["assigned_line"].values.astype(float) + 1.0,
        ]
    )
    return dict(
        dataset=DATASET,
        corpus=CORPUS,
        trial_id=f"ABREV{pnum}_{tr['tid']}",
        participant=f"ABREV{pnum}",
        group=None,
        condition=tr["condition"],
        language=LANGUAGE,
        rate_hz=RATE_HZ,
        samples=dict(t=(samp[:, 0] - t_on) / 1000.0, x=samp[:, 1], y=samp[:, 2]),
        samples_source="raw",
        fixations=fixations,
        line_Y=line_Y,
        line_X=line_X,
        word_XY=word_XY,
        notes=NOTES + f" eye={tr['eye']}.",
    )


def process_file(path):
    """One ASC file -> its trials that have DIST fixations, in the order of the ASC file."""
    pnum = int(_RE_PNUM.search(path).group(1))
    parsed = parse_asc(path)
    by_tid = collections.defaultdict(list)
    for tr in parsed:
        by_tid[tr["tid"]].append(tr)
    out = []
    for tid, cands in by_tid.items():
        stem = os.path.join(DIST_DIR, f"ABREV{pnum}_trial_id_{tid}")
        if not (os.path.exists(stem + "_trial.json") and os.path.exists(stem + "_fixations.csv")):
            continue
        tr = cands[-1]
        if len(cands) > 1:  # TRIALID completed more than once: the one DIST used
            with open(stem + "_trial.json") as fh:
                line_idx = json.load(fh)["trial_id_idx"]
            hit = [c for c in cands if c["line_idx"] == line_idx]
            tr = hit[0] if hit else cands[-1]
        out.append(build_trial(pnum, tr, pd.read_csv(stem + "_fixations.csv", index_col=0)))
    order = {tr["tid"]: tr["line_idx"] for tr in parsed}
    out.sort(key=lambda d: order[d["trial_id"].split("_", 1)[1]])
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--nproc", type=int, default=8, help="worker processes")
    a = ap.parse_args()
    files = find_asc_files()
    with Pool(a.nproc) as pool:
        trials = [d for file_trials in pool.map(process_file, files) for d in file_trials]
    os.makedirs(os.path.dirname(OUT_PKL), exist_ok=True)
    with open(OUT_PKL, "wb") as fh:
        pickle.dump(trials, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"saved {len(trials)} trials from {len(files)} ASC files -> {OUT_PKL}")


if __name__ == "__main__":
    main()
