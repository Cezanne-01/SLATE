"""Line-assignment accuracy by diagnostic group and collection site (article section 7.2).

For SLATE (gaze, gaze + speech) in the condition without K, the mean accuracy across trials of
each subgroup is placed next to the overall value, together with the gap between SLATE and the
best comparison algorithm of that subgroup. The gap gets a 95% CI from a participant-level
bootstrap within the subgroup (2,000 replicates, both passages of a participant drawn together).

Input: per_trial.csv written by analysis.clinical.table1_table2. Output: subgroups.csv.

    python -m analysis.clinical.subgroups outputs/table1_table2/per_trial.csv [--out outputs/subgroups]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SLATE = ["slate_gaze", "slate_speech"]
N_BOOT, SEED = 2000, 20261029
FACTORS = {"group": ["SCD", "MCI", "Major NCD"], "site": ["SVH", "WKUH", "BSH"]}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("per_trial", help="per_trial.csv written by analysis.clinical.table1_table2")
    ap.add_argument("--out", default="outputs/subgroups", help="output folder")
    args = ap.parse_args()

    per_trial = pd.read_csv(args.per_trial, dtype={"pid": str})
    per_trial = per_trial[per_trial.condition == "displayed"]
    per_trial = per_trial.assign(accuracy=per_trial.hit / per_trial.n)
    # sum of the trial accuracies and number of trials per participant and method
    by_participant = per_trial.groupby(["pid", "group", "site", "method"], as_index=False).agg(
        total=("accuracy", "sum"), trials=("accuracy", "count")
    )
    total = by_participant.pivot(index="pid", columns="method", values="total")
    trials = by_participant.pivot(index="pid", columns="method", values="trials")
    participants = by_participant.drop_duplicates("pid").set_index("pid").loc[total.index]
    assert len(participants) == 116  # participants of the article
    comparison = sorted(set(total.columns) - set(SLATE))
    rng = np.random.default_rng(SEED)

    def mean_accuracy(method, rows):
        return total[method].to_numpy()[rows].sum() / trials[method].to_numpy()[rows].sum()

    slices = []
    for factor, levels in FACTORS.items():
        members = {g: np.flatnonzero(participants[factor].to_numpy() == g) for g in levels}
        resamples = [
            {g: rng.choice(members[g], len(members[g])) for g in levels} for _ in range(N_BOOT)
        ]
        for method in SLATE:
            overall = mean_accuracy(method, slice(None))
            accuracy = {g: mean_accuracy(method, members[g]) for g in levels}
            best = {g: max(comparison, key=lambda b: mean_accuracy(b, members[g])) for g in levels}
            gap = {g: accuracy[g] - mean_accuracy(best[g], members[g]) for g in levels}
            # bootstrap of each subgroup's gap over the stratified resamples
            gap_draws = {
                g: [
                    mean_accuracy(method, draw[g]) - mean_accuracy(best[g], draw[g])
                    for draw in resamples
                ]
                for g in levels
            }
            for g in levels:
                lo, hi = np.percentile(gap_draws[g], [2.5, 97.5])
                slices.append(
                    dict(
                        factor=factor,
                        subgroup=g,
                        method=method,
                        participants=len(members[g]),
                        mean=accuracy[g],
                        diff_from_overall=accuracy[g] - overall,
                        best_comparison=best[g],
                        gap_to_best_comparison=gap[g],
                        gap_lo=lo,
                        gap_hi=hi,
                    )
                )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    slices = pd.DataFrame(slices)
    slices.to_csv(out / "subgroups.csv", index=False)
    print(slices.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
