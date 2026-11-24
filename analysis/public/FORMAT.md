# Common format for public datasets

Each loader (`load_etdd70.py`, `load_adedeji.py`) writes `<SLATE_PUBLIC_DATA>/_common/<dataset>.pkl`, a pickled **list of dicts**, one per trial.

| Key | Type | Meaning |
| --- | --- | --- |
| dataset | str | e.g. "adedeji2022", "etdd70" |
| corpus | str | sub-corpus or language (e.g. "T4_Meaningful_Text") |
| trial_id | str | unique within the dataset |
| participant | str | participant id as published |
| group | str or None | group label as published |
| condition | str or None | reading mode: "aloud" / "silent" / None (column `reading_mode` of rows.csv) |
| language | str | ISO code |
| rate_hz | float | sampling rate of the recording |
| samples | dict | {"t": seconds from trial start, "x": px, "y": px}, with missing samples as NaN |
| samples_source | str | "raw": the gaze samples as published |
| fixations | ndarray (N, 5) | [start_s, end_s, x, y, line_true], where line_true is 1-based, 0 = discarded / outside the text |
| line_Y | ndarray (L,) | line center y, in reading order |
| line_X | ndarray (L, 2) | (xmin, xmax) of each line in px |
| word_XY | ndarray (W, 2) or None | word centers (x, y of the line) |
| notes | str | source of the data |

Rules
- Coordinates are screen pixels, y increasing downward. Line pitch is the median of diff(line_Y).
- Time 0 is the start of the trial: the first sample of the recording (ETDD70) or DISPLAY ON (Adedeji et al., 2022). Fixation times are on the same axis.
- The datasets themselves are not stored in this repository.

## Output of analysis.public.run: rows.csv

One row per trial, signal condition and method.

| Column | Meaning |
| --- | --- |
| dataset, corpus, participant, trial_id | as in the trial dict |
| reading_mode | the trial's `condition` ("aloud" / "silent" / empty) |
| signal | "original" or "tablet-like" (analysis.public.degradation) |
| method | "slate", the Carr et al. algorithms and slice ("attach", "chain", "cluster", "compare", "merge", "regress", "segment", "slice", "split", "stretch", "warp"), "cwoc" or "confla" |
| accuracy | share of the dataset's fixations assigned to their labeled line, with discarded fixations (line_true = 0) counted as errors (section 6.4) |
| n_fixations | number of the dataset's fixations (the denominator of accuracy, the same in both signal conditions) |
| tp, fp, fn | line transitions matched one-to-one within 1.5 s, first-line entry excluded (section 6.4) |
