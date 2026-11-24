"""SLATE: Sequential Line Assignment with trial-adaptive thresholds for Tablet Eye tracking.

    from slate import detect_line_transitions, assign_lines, correct_fixations

`slate.detector`       adaptive line-transition detection and line assignment (sections 5.1, 5.3)
`slate.speech`         speech-based supplementation of the transitions (section 5.2)
`slate.recalibration`  horizontal recalibration (section 5.4)
`slate.correction`     the three steps on a set of fixations in one call
"""

from slate.correction import correct_fixations
from slate.detector import assign_lines, detect_line_transitions

__all__ = ["detect_line_transitions", "assign_lines", "correct_fixations"]
