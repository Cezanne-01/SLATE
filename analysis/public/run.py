"""Run SLATE and the comparison algorithms on a public dataset (article sections 6.2, 6.3.2,
7.3.1 and 7.3.2).

Two signal conditions (analysis.public.degradation): `original`, the recording as published
(Table 3 and the left side of Figure 9), and `tablet-like`, the recording with the
five-component degradation applied (right side of Figure 9). SLATE receives the gaze samples.
The comparison algorithms receive, in both conditions, the fixation intervals supplied with the
dataset, with each fixation placed at the mean of the samples of that condition inside its
interval, so both conditions are scored on the same fixations and line labels.

SLATE runs with the same detector constants as for the clinical data and without speech, which
the public datasets do not contain. The comparison algorithms are the ten of Carr et al.
(2022), slice, C-WOC (their majority vote) and CONF-LA with the authors' best_params_N.
numpy's global generator is seeded before each call of a Carr et al. algorithm from (trial,
signal condition, algorithm), so that the k-means of `cluster` and `split`, and with them
C-WOC, are reproducible.

    python -m analysis.public.run --dataset etdd70 --corpus T4_Meaningful_Text --out outputs/public_etdd70
    python -m analysis.public.run --dataset adedeji2022 --out outputs/public_adedeji2022

Reads SLATE_PUBLIC_DATA/_common/<dataset>.pkl (written by the loaders). Writes rows.csv with one
row per trial, signal condition and method (columns in analysis/public/FORMAT.md).
"""

import argparse
import pickle
import zlib
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.paths import COMMON
from analysis.public import degradation, evaluation
from baselines import carr, confla_algorithms
from baselines.confla import confla_lines
from slate import detect_line_transitions

SEED = 20261015  # seed of the tablet-like degradation


def comparison_lines(fixations, line_Y, word_XY, text_width, trial_id, signal):
    """Line of each fixation from the thirteen comparison algorithms (None where one fails)."""
    lines = {}
    for name in carr.ALGOS:
        np.random.seed(zlib.crc32(f"{trial_id}:{signal}:{name}".encode()))
        lines[name] = carr.run_algorithm(
            name, fixations.copy(), line_Y, word_XY, carr.line_pitch(line_Y)
        )
    voters = [p for p in lines.values() if p is not None]
    lines["cwoc"] = carr.majority_vote(voters, fixations, line_Y) if voters else None
    lines["confla"] = confla_lines(
        fixations.copy(), line_Y, text_width, confla_algorithms.best_params_N
    )
    return lines


def run_trial(trial, signals=degradation.SIGNALS):
    """All methods on one trial under each signal condition -> list of row dicts."""
    fixations = np.asarray(trial["fixations"], float)  # start, end, x, y, line label
    line_Y, widths, pitch, text_width, text_cx = evaluation.geometry(trial)
    word_XY = np.asarray(trial["word_XY"], float)
    t0, x0, y0 = (np.asarray(trial["samples"][k], float) for k in ("t", "x", "y"))
    reference = evaluation.reference_transitions(fixations)
    row = dict(
        dataset=trial["dataset"],
        corpus=trial["corpus"],
        participant=trial["participant"],
        trial_id=trial["trial_id"],
        reading_mode=trial["condition"],
    )
    rows = []
    for signal in signals:
        t, x, y = degradation.make_signal(
            t0, x0, y0, signal, pitch, text_width, text_cx, float(np.mean(line_Y)),
            SEED, evaluation.trial_seed(trial["trial_id"]), ev=reference,
        )
        if len(t) < 20:
            continue

        def add(method, lines, transitions):
            score = evaluation.score_trial(lines, fixations, transitions, reference)
            rows.append(dict(row, signal=signal, method=method, **score))

        # the dataset's fixations with coordinates taken from the samples of this condition
        observed = fixations.copy()
        observed[:, :4] = evaluation.fix_from_samples(fixations, t, x, y)
        midpoints = (observed[:, 0] + observed[:, 1]) / 2

        transitions = detect_line_transitions(t, x, line_widths=widths, first_at_start=True)
        add("slate", np.searchsorted(transitions, midpoints, side="right"), transitions)

        predicted = comparison_lines(
            observed, line_Y, word_XY, text_width, trial["trial_id"], signal
        )
        for method, lines in predicted.items():
            add(method, lines, evaluation.predicted_transitions(lines, observed))
    return rows


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--dataset", required=True, choices=["etdd70", "adedeji2022"])
    ap.add_argument("--corpus", help="restrict to a sub-corpus (ETDD70: T4_Meaningful_Text)")
    ap.add_argument("--limit", type=int, help="first N trials only")
    ap.add_argument("--nproc", type=int, default=8, help="worker processes")
    ap.add_argument("--out", required=True, help="output folder for rows.csv")
    args = ap.parse_args()

    with open(COMMON / f"{args.dataset}.pkl", "rb") as f:
        trials = pickle.load(f)
    if args.corpus:
        trials = [t for t in trials if t["corpus"] == args.corpus]
    trials = trials[: args.limit]
    with Pool(args.nproc) as pool:
        rows = [row for trial_rows in pool.imap(run_trial, trials) for row in trial_rows]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "rows.csv", index=False)
    print(f"saved {out / 'rows.csv'}: {len(rows)} rows")


if __name__ == "__main__":
    main()
