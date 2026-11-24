# Third-party material

The MIT license in `LICENSE` covers the original code in this repository. The files below
are third-party work redistributed under their own terms (CC BY 4.0: https://creativecommons.org/licenses/by/4.0/).

| File | Source | License | Changes |
| --- | --- | --- | --- |
| `baselines/carr_drift.py` | Carr, Pescuma, Furlan, Ktori, & Crepaldi (2022). Algorithms for the automated correction of vertical drift in eye-tracking data. *Behavior Research Methods, 54*(1), 287-310. https://doi.org/10.3758/s13428-021-01554-0. Code: https://github.com/jwcarr/drift (`algorithms/Python/drift_algorithms.py`) | CC BY 4.0 | None (header comment added) |
| `baselines/confla_algorithms.py` | Kaltenberger, Chen, Thaqi, & Kasneci (2026). Sure About That Line? Approaching Confidence-Based, Real-Time Line Assignment in Reading Gaze Data. https://doi.org/10.1145/3803540. Code: https://gitlab.lrz.de/hctl/conf-la (commit c25442d) | CC BY 4.0 | Three functions copied verbatim, and the screen-geometry module is replaced by a small shim with the physical screen size of the tablet used in this study |

Reading passages in `stimuli/` are public-domain Korean poems (Han Yong-un, 1926, and Kim Yeong-rang, 1935).

`examples/etdd70_trial.json` is one trial of the ETDD70 dataset (Dostalova, Svaricek,
Sedmidubsky, Culemann, Sasinka, Zezula, & Cenek, 2024, https://doi.org/10.5281/zenodo.13332134),
licensed CC BY 4.0, converted to the format of `analysis/public/FORMAT.md`.

The public datasets used for the transfer and degradation analyses are otherwise not
redistributed here. Obtain them from their original sources and cite them (see `README.md`).
