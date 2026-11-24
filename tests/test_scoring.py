"""Tests of the scoring rules (analysis/scoring.py and analysis/public/evaluation.py, article
section 6.4).

python -m tests.test_scoring
"""

import numpy as np

from analysis.public.evaluation import score_trial
from analysis.scoring import match_transitions, transition_counts


def test_transition_counts():
    assert transition_counts([1, 5, 9], [1, 5, 9]) == (2, 0, 0)
    assert transition_counts([0.2, 1, 5], [1, 5, 9]) == (1, 0, 1)  # gaze before reading dropped
    assert transition_counts([5, 9], [1, 5, 9]) == (2, 0, 0)  # a missed first entry is not scored
    assert transition_counts([1, 3, 5, 9], [1, 5, 9]) == (
        2,
        1,
        0,
    )  # extra detection: false positive
    assert transition_counts([], [1, 5, 9]) == (0, 0, 2)
    assert transition_counts([1.2, 5.1, 9.0, 12.0], [1, 5, 9]) == (2, 1, 0)
    assert transition_counts([0.5], [1, 5]) == (0, 0, 1)  # matched to the first entry
    assert transition_counts([1, 5], []) == (0, 2, 0)


def test_matching_is_one_to_one():
    assert match_transitions([1, 5, 9], [1.2, 5.1]) == {0: 0, 1: 1}
    assert match_transitions([1.0, 1.4], [1.3]) == {1: 0}  # the closest pair wins


def test_discarded_fixations_count_as_errors():
    """A fixation the dataset discarded (label 0) is an error whatever line it is assigned."""
    fixations = np.array([[0.0, 0.2, 100.0, 100.0, 0], [0.3, 0.5, 200.0, 100.0, 1]])
    assert score_trial(np.array([0, 1]), fixations, [], [])["accuracy"] == 0.5
    assert score_trial(np.array([1, 1]), fixations, [], [])["accuracy"] == 0.5


if __name__ == "__main__":
    test_transition_counts()
    test_matching_is_one_to_one()
    test_discarded_fixations_count_as_errors()
    print("scoring tests passed")
