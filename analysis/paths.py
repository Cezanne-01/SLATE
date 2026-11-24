"""Locations of the repository files and of the data, which are not distributed with it.

Environment variables (see data/README.md):

    SLATE_CLINICAL_DATA   folder with the clinical audio recordings, ground truth and speech
                          recognition cache (default: the repository root)
    SLATE_RAW_GAZE        folder with the gaze records of the clinical recordings
                          (default: SLATE_CLINICAL_DATA/raw)
    SLATE_PUBLIC_DATA     folder with the public datasets (default: the repository's public_data/)
"""

import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
STIMULI = REPO / "stimuli"
CLINICAL_DATA = Path(os.environ.get("SLATE_CLINICAL_DATA", REPO))
RAW_GAZE = Path(os.environ.get("SLATE_RAW_GAZE", CLINICAL_DATA / "raw"))
PUBLIC_DATA = Path(os.environ.get("SLATE_PUBLIC_DATA", REPO / "public_data"))
COMMON = PUBLIC_DATA / "_common"  # converted public datasets (analysis/public/FORMAT.md)
