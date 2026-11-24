"""Results on the public oral-reading datasets (article section 7.3): Table 3 and the accuracy
in the original and the tablet-like condition.

Reads the per-trial rows written by analysis.public.run.

table3.csv (section 7.3.1): SLATE (gaze only) and the thirteen comparison algorithms of Table 1
on the original signal, with line-transition precision, recall and F1 and the mean and median
accuracy across trials.

degradation.csv (section 7.3.2, where Figure 9 shows SLATE and five of the comparison algorithms):
for every method, the mean accuracy across trials in the original and in the tablet-like condition
and the drop in percentage points. 95% CIs are participant-level
bootstrap intervals (2,000 resamples, with the same resamples for both conditions, so the drop is
paired).

    python -m analysis.public.results \
        "ETDD70 T4=outputs/public_etdd70/rows.csv" \
        "Adedeji et al. (2022)=outputs/public_adedeji2022/rows.csv:aloud" [--out outputs/public]

Each argument is LABEL=ROWS.csv[:READING_MODE].
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from analysis import scoring

N_BOOT, SEED = 2000, 20261105
SIGNALS = ("original", "tablet-like")
METHODS = ["slate", "warp", "segment", "split", "compare", "merge", "cluster", "regress",
           "stretch", "chain", "attach", "slice", "cwoc", "confla"]
LABEL = {"slate": "SLATE (gaze)", "cwoc": "C-WOC", "confla": "CONF-LA"}


def read_rows(spec):
    """Parse LABEL=ROWS.csv[:READING_MODE] and return (label, rows)."""
    label, rest = spec.split("=", 1)
    path, _, reading_mode = rest.partition(":")
    rows = pd.read_csv(path, low_memory=False)
    if reading_mode:
        rows = rows[rows.reading_mode == reading_mode]
    return label, rows


def table3(datasets):
    """Mean and median accuracy across trials and transition P/R/F1 on the original signal."""
    table = []
    for label, rows in datasets:
        rows = rows[rows.signal == "original"]
        for method in METHODS:
            r = rows[rows.method == method]
            tp, fp, fn = (int(r[c].sum()) for c in ("tp", "fp", "fn"))
            table.append(
                dict(
                    dataset=label,
                    method=method,
                    label=LABEL.get(method, method),
                    trials=len(r),
                    participants=int(r.participant.nunique()),
                    mean=float(r.accuracy.mean()),
                    median=float(r.accuracy.median()),
                    **scoring.precision_recall_f1(tp, fp, fn),
                    tp=tp,
                    fp=fp,
                    fn=fn,
                )
            )
    return pd.DataFrame(table)


def interval(values):
    """Percentile 95% interval."""
    return np.percentile(values, 2.5), np.percentile(values, 97.5)


def degradation(datasets):
    """Mean accuracy across trials per dataset, method and signal condition, with participant-level
    bootstrap CIs paired across the two conditions."""
    rng = np.random.default_rng(SEED)
    table = []
    for label, rows in datasets:
        participants = np.array(sorted(rows.participant.unique()))
        draws = rng.integers(0, len(participants), size=(N_BOOT, len(participants)))
        for method in METHODS:
            point, boot = {}, {}
            for signal in SIGNALS:
                accuracy = rows[(rows.method == method) & (rows.signal == signal)].accuracy
                # sum of the trial accuracies and number of trials of each participant
                per = (
                    accuracy.groupby(rows.participant)
                    .agg(["sum", "count"])
                    .reindex(participants)
                    .fillna(0)
                    .to_numpy()
                )
                point[signal] = accuracy.mean()
                boot[signal] = per[draws, 0].sum(1) / per[draws, 1].sum(1)
            original, tablet = SIGNALS
            (o_lo, o_hi), (t_lo, t_hi) = interval(boot[original]), interval(boot[tablet])
            d_lo, d_hi = interval(100 * (boot[original] - boot[tablet]))
            table.append(
                dict(
                    dataset=label, method=method,
                    original=point[original], original_lo=o_lo, original_hi=o_hi,
                    tablet_like=point[tablet], tablet_like_lo=t_lo, tablet_like_hi=t_hi,
                    drop_pp=100 * (point[original] - point[tablet]), drop_lo=d_lo, drop_hi=d_hi,
                )
            )
    return pd.DataFrame(table)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("specs", nargs="+", help="LABEL=ROWS.csv[:READING_MODE]")
    ap.add_argument("--out", default="outputs/public", help="output folder")
    args = ap.parse_args()
    datasets = [read_rows(spec) for spec in args.specs]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    t3 = table3(datasets)
    t3.to_csv(out / "table3.csv", index=False)
    columns = ["dataset", "label", "precision", "recall", "f1", "mean", "median", "trials"]
    print(t3[columns].to_string(index=False, float_format="{:.4f}".format))

    deg = degradation(datasets)
    deg.to_csv(out / "degradation.csv", index=False)
    columns = ["dataset", "method", "original", "tablet_like", "drop_pp"]
    print(deg[columns].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
