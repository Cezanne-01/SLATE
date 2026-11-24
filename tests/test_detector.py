"""Tests of the line-transition detection (slate/detector.py, article section 5.1).

python -m tests.test_detector

A reader reads the first four lines of a ten-line passage. The same trace is then compressed or
stretched horizontally and displaced, as the recorded range differs between tablet trials.
"""

import numpy as np

from slate import assign_lines, correct_fixations, detect_line_transitions

RATE_HZ = 30.0
LINE_S = 4.0  # reading time of one line (s)
LINE_WIDTHS = [1000.0] * 10  # ten displayed lines of equal width (px)
LINES_READ = 4


def reading_trace():
    """Horizontal gaze position of four lines read from left to right, each followed by a
    return sweep: (t, x)."""
    t = np.arange(0, LINES_READ * LINE_S, 1 / RATE_HZ)
    x = 400.0 + 1000.0 * (t % LINE_S) / LINE_S
    return t, x


def test_same_lines_for_any_recorded_range():
    t, x = reading_trace()
    midpoints = LINE_S * (np.arange(LINES_READ) + 0.5)
    reference = None
    for gain, offset in ((1.0, 0.0), (0.45, 300.0), (1.6, -200.0)):
        transitions = detect_line_transitions(
            t, gain * x + offset, line_widths=LINE_WIDTHS, first_at_start=True
        )
        # no vertical coordinate and no last line reached: four of the ten lines are found
        assert len(transitions) == LINES_READ
        assert np.allclose(transitions[1:], LINE_S * np.arange(1, LINES_READ), atol=0.2)
        lines = assign_lines(midpoints, transitions, n_lines=len(LINE_WIDTHS))
        assert list(lines) == [1, 2, 3, 4]
        if reference is None:
            reference = transitions
        assert np.allclose(transitions, reference)


def test_invalid_input_is_rejected():
    t, x = reading_trace()
    missing = x.copy()
    missing[100:110] = np.nan
    for bad_t, bad_x in (
        (t, missing),  # missing samples
        (t * 1000, x),  # times in ms
        (t[:-1], x),  # different lengths
    ):
        try:
            detect_line_transitions(bad_t, bad_x)
        except ValueError:
            continue
        raise AssertionError("invalid input was accepted")
    assert len(detect_line_transitions([], [])) == 0


def test_samples_out_of_order_are_sorted():
    t, x = reading_trace()
    expected = detect_line_transitions(t, x, line_widths=LINE_WIDTHS, first_at_start=True)
    swapped = np.arange(len(t))
    swapped[[200, 201]] = swapped[[201, 200]]
    found = detect_line_transitions(
        t[swapped], x[swapped], line_widths=LINE_WIDTHS, first_at_start=True
    )
    assert np.allclose(found, expected)


def test_fixation_table():
    """Fixations recorded at half the horizontal range are put on their line and moved back to
    their position in the text."""
    t, x = reading_trace()
    starts = np.arange(0.2, LINES_READ * LINE_S - 0.3, 0.5)
    in_text = 400.0 + 1000.0 * ((starts + 0.125) % LINE_S) / LINE_S
    recorded = 0.5 * in_text + 100.0
    fixations = np.column_stack([starts, starts + 0.25, recorded, np.full_like(starts, 300.0)])
    lines = [dict(left=400.0, right=1400.0, y=200.0 + 80.0 * k, n_word=10) for k in range(10)]
    table = correct_fixations(fixations, t, 0.5 * x + 100.0, lines, first_at_start=True)
    assert list(table.columns) == ["start", "end", "x", "y", "line", "x_corrected", "y_corrected"]
    assert np.allclose(table[["x", "y"]], fixations[:, 2:4])  # recorded values are kept
    assert list(table.line) == list((starts // LINE_S).astype(int) + 1)
    assert np.allclose(table.y_corrected, 200.0 + 80.0 * (table.line - 1))
    assert np.abs(table.x_corrected - in_text).max() < 50.0  # 5% of the line width

    # alpha moves the right target inward by that many widths of the last word
    lines = [dict(line, last_w=100.0) for line in lines]
    inward = correct_fixations(fixations, t, 0.5 * x + 100.0, lines, first_at_start=True, alpha=1.0)
    own = table.line < LINES_READ  # the last line read takes its gain from the other lines
    ratio = (inward.x_corrected[own] - 400.0) / (table.x_corrected[own] - 400.0)
    assert np.allclose(ratio, 0.9)


if __name__ == "__main__":
    test_same_lines_for_any_recorded_range()
    test_fixation_table()
    test_invalid_input_is_rejected()
    test_samples_out_of_order_are_sorted()
    print("detector tests passed")
