# Data

## Clinical recordings

The clinical recordings contain voice and clinical information and cannot be shared. In their place, this repository provides the degradation that gives a recording of a public eye-tracking dataset the signal properties we measured in the tablet recordings (`analysis/public/degradation.py`). With it, SLATE and the comparison algorithms can be tested on tablet-like data built from public datasets: `examples/demo.ipynb` does this for one recording, and the README describes how to do it for the two datasets used in the paper.

The scripts in `analysis/clinical/` are the ones that produced Tables 1 and 2. They read the gaze record and the audio of each trial and the ground truth (the line of each fixation and the time at which each line was entered) from the folder given by the environment variable `SLATE_CLINICAL_DATA`. We ran them in this order:

```shell
python -m analysis.clinical.speech_cues --task 6.2
python -m analysis.clinical.speech_cues --task 6.3
python -m analysis.clinical.table1_table2 --out outputs/table1_table2
python -m analysis.clinical.subgroups outputs/table1_table2/per_trial.csv --out outputs/subgroups
```

The size of the degradation was set from measurements of the tablet recordings (Appendix A of the paper). `analysis/public/session_affine.csv` holds the gains and offsets measured per recording, which the degradation draws from. It contains no participant identifiers.

## Public datasets

Table 3 and the comparison between the original and the tablet-like signal (Figure 9) use two public oral-reading datasets, which are not redistributed here. Download them from their original sources and place them in the folder given by `SLATE_PUBLIC_DATA` (default: `public_data/` in the repository):

| Dataset | Location | Source |
| --- | --- | --- |
| ETDD70 | `etdd70/` (`data/data/`, `rois/rois/`, `dyslexia_class_label.csv`) | https://doi.org/10.5281/zenodo.13332134 |
| Adedeji et al. (2022), raw EyeLink files | `adedeji2022/*/ABREV<n>.asc` | https://osf.io/ym76z/ |
| Adedeji et al. (2022), fixations and line labels | `dist/Abbr/processed_data/` | folder `Abbr/processed_data.zip` of the DIST benchmark (Mercier et al., 2024), https://osf.io/zt9gn/ |

The loaders write the converted trials to `_common/<dataset>.pkl` in the same folder (`analysis/public/FORMAT.md`).
