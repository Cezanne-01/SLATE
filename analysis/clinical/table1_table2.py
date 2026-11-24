"""Tables 1 and 2: line assignment on the clinical recordings (article sections 6.2, 6.3.1, 6.4,
7.1 and 7.2).

Every trial is processed by
  * SLATE, gaze only and gaze + speech (sections 5.1 to 5.3), each without and with the last line
    reached K (section 6.3.1), and
  * the comparison algorithms: the ten of Carr et al. (2022), slice, C-WOC (their majority
    vote) and CONF-LA with the authors' parameter set `best_params_N`, whose return-sweep
    threshold of 172 deg/s is converted to the 40 cm viewing distance (120 cm/s). They receive
    the fixations, in two conditions: `displayed` (all passage lines, Table 1 without K) and
    `knownK` (lines and words 1 to K only, Table 1 with K).
A trial is scored by the number of correctly assigned fixations and by its line transitions
matched to the reference line entries (analysis.scoring).

Outputs: table1.csv (per condition and method: mean accuracy across trials with a
participant-level bootstrap 95% CI, median accuracy, and precision, recall and F1 of the line
transitions),
table2_by_group.csv and table2_by_site.csv (the same by diagnostic group and collection site),
and per_trial.csv (the counts of each trial).

    python -m analysis.clinical.table1_table2 [--out outputs/table1_table2]
"""

import argparse
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from analysis import scoring
from analysis.clinical.passages import TASKS, line_widths, text_geometry
from analysis.clinical.sessions import load_trials
from analysis.clinical.speech_cues import supplement_recording
from baselines import carr, confla_algorithms
from baselines.confla import confla_lines
from slate import detect_line_transitions
from slate.detector import limit_to_last_line

N_BOOT, SEED = 2000, 20261026
GROUPS = ["SCD", "MCI", "Major NCD"]
SITES = {
    "SVH": "St. Vincent's Hospital",
    "WKUH": "Wonkwang University Hospital",
    "BSH": "Bucheon Sejong Hospital",
}
CONDITIONS = {
    "displayed": ["slate_gaze", "slate_speech"],
    "knownK": ["slate_gaze_K", "slate_speech_K"],
}
COMPARISON = ["warp", "segment", "split", "compare", "merge", "cluster", "regress", "stretch",
              "chain", "attach", "slice", "C-WOC", "CONF-LA"]
CONFLA_PARAMS = dict(confla_algorithms.best_params_N, return_sweep_threshold=120.0)  # cm/s


def slate_transitions(trial, task, pid, widths, last_line):
    """Transitions of the four SLATE rows of one trial, keyed by method name."""
    t, x = trial["t"], trial["x"]
    gaze = detect_line_transitions(t, x, line_widths=widths)
    speech = np.unique(supplement_recording(list(gaze), t, x, trial["audio"], pid, task, widths))
    return {
        "slate_gaze": gaze,
        "slate_speech": speech,
        "slate_gaze_K": detect_line_transitions(
            t, x, line_widths=widths, last_line_reached=last_line
        ),
        "slate_speech_K": limit_to_last_line(t, x, speech, last_line),
    }


def comparison_lines(trial, task, pid, line_Y, word_XY, text_width, pitch):
    """Line of each fixation from the thirteen comparison algorithms (None where one fails).
    `pitch` is the line pitch of the passage (px)."""
    fixations = trial["fixations"]
    lines = {}
    for name in carr.ALGOS:
        np.random.seed(
            zlib.crc32(f"{task}:{pid}:{name}:{False}".encode())
        )  # k-means in cluster, split
        lines[name] = carr.run_algorithm(
            name, fixations, line_Y, word_XY, pitch, bounded=name == "compare"
        )
    voters = [p for p in lines.values() if p is not None]
    lines["C-WOC"] = carr.majority_vote(voters, fixations, line_Y) if voters else None
    if len(line_Y) == 1:
        lines["CONF-LA"] = np.ones(len(fixations), dtype=int)
    else:
        lines["CONF-LA"] = confla_lines(fixations.copy(), line_Y, text_width, CONFLA_PARAMS)
    return lines


def per_trial_counts():
    """One row per passage, participant, condition and method with the counts behind the tables."""
    rows = []
    for task in TASKS:
        trials = load_trials(task)
        widths = line_widths(task)
        line_Y, word_XY = text_geometry(task)
        for pid, trial in trials.items():
            fixations, truth, scored = trial["fixations"], trial["line"], trial["scored"]
            midpoints = (fixations[:, 0] + fixations[:, 1]) / 2
            last_line = int(truth[(truth > 0) & (truth <= len(line_Y))].max())

            def add(condition, method, lines, transitions):
                tp, fp, fn = scoring.transition_counts(transitions, trial["entries"])
                rows.append(
                    dict(
                        task=task, pid=pid, group=trial["group"], site=trial["site"],
                        condition=condition, method=method,
                        n=int(scored.sum()),
                        hit=0 if lines is None else int(((lines == truth) & scored).sum()),
                        failed=int(lines is None),
                        tp=tp, fp=fp, fn=fn,
                    )
                )

            slate = slate_transitions(trial, task, pid, widths, last_line)
            for condition, methods in CONDITIONS.items():
                for method in methods:
                    lines = np.searchsorted(slate[method], midpoints, side="right")
                    add(condition, method, lines, slate[method])
                n_lines = len(line_Y) if condition == "displayed" else last_line
                shown_Y = line_Y[:n_lines]
                shown_words = word_XY[np.isin(word_XY[:, 1], shown_Y)]
                predicted = comparison_lines(
                    trial, task, pid, shown_Y, shown_words, max(widths), carr.line_pitch(line_Y)
                )
                for method, lines in predicted.items():
                    transitions = (
                        [] if lines is None else scoring.transitions_from_lines(fixations, lines)
                    )
                    add(condition, method, lines, transitions)
        print("done", task, flush=True)
    return pd.DataFrame(rows)


def bootstrap_ci(counts, rng):
    """Participant-level bootstrap 95% CI of the mean accuracy (both passages drawn together)."""
    accuracy = (counts.hit / counts.n).groupby(counts.pid).agg(["sum", "count"])
    total, trials = accuracy["sum"].to_numpy(float), accuracy["count"].to_numpy(float)
    draws = rng.integers(0, len(total), size=(N_BOOT, len(total)))
    means = total[draws].sum(1) / trials[draws].sum(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarize(counts, rng):
    """One table row: mean accuracy across trials with CI, median accuracy, transition P/R/F1."""
    accuracy = counts.hit / counts.n
    tp, fp, fn = (int(counts[c].sum()) for c in ("tp", "fp", "fn"))
    ci_lo, ci_hi = bootstrap_ci(counts, rng)
    return dict(
        participants=int(counts.pid.nunique()),
        trials=len(counts),
        n=int(counts.n.sum()),
        mean=float(accuracy.mean()),
        ci_lo=ci_lo,
        ci_hi=ci_hi,
        median=float(accuracy.median()),
        failed=int(counts.failed.sum()),
        tp=tp,
        fp=fp,
        fn=fn,
        **scoring.precision_recall_f1(tp, fp, fn),
    )


def summary_tables(per_trial, out):
    """Write Table 1 (table1.csv) and Table 2 (table2_by_group.csv, table2_by_site.csv)."""
    rng = np.random.default_rng(SEED)
    overall, by_group, by_site = [], [], []
    for condition, slate_rows in CONDITIONS.items():
        for method in slate_rows + COMPARISON:
            key = dict(condition=condition, method=method)
            g = per_trial[(per_trial.condition == condition) & (per_trial.method == method)]
            assert (len(g), g.pid.nunique()) == (232, 116), key  # the sample of the article
            overall.append(dict(key, **summarize(g, rng)))
            for group in GROUPS:
                by_group.append(dict(key, group=group, **summarize(g[g.group == group], rng)))
            for site, name in SITES.items():
                by_site.append(
                    dict(key, site=site, site_name=name, **summarize(g[g.site == site], rng))
                )
    pd.DataFrame(overall).to_csv(out / "table1.csv", index=False)
    pd.DataFrame(by_group).to_csv(out / "table2_by_group.csv", index=False)
    pd.DataFrame(by_site).to_csv(out / "table2_by_site.csv", index=False)
    columns = ["condition", "method", "precision", "recall", "f1", "mean", "median", "failed"]
    print(pd.DataFrame(overall)[columns].to_string(index=False, float_format="{:.4f}".format))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", default="outputs/table1_table2", help="output folder")
    out = Path(ap.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    per_trial = per_trial_counts()
    per_trial.to_csv(out / "per_trial.csv", index=False)
    summary_tables(per_trial, out)
    participants = per_trial.drop_duplicates("pid")
    pd.crosstab(participants.group, participants.site, margins=True).to_csv(
        out / "participants_group_by_site.csv"
    )


if __name__ == "__main__":
    main()
