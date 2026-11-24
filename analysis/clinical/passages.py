"""Layout of the two clinical passages (article section 3.1.2).

The word boxes of a passage (stimuli/task<passage>_words.json, px on the 1920 x 1080 canvas)
map one to one to its words in reading order. Passages are identified by their task number in
the manifest, "6.2" and "6.3".
"""

import json

import numpy as np

from analysis.paths import STIMULI

TASKS = ("6.2", "6.3")
TEXT = {"6.2": STIMULI / "task6.2_nim-ui-chimmuk.txt", "6.3": STIMULI / "task6.3_youngrang.txt"}
N_LINES = 10  # both passages are displayed on ten lines
ROW_TOL = 25  # boxes whose top edges differ by less than this (px) are on the same line


def word_boxes(task):
    """Word boxes of a passage: list of dicts with x1, y1, x2, y2."""
    return json.loads((STIMULI / f"task{task}_words.json").read_text(encoding="utf-8"))


def word_box_rows(task):
    """Word boxes grouped into text lines, lines and words in reading order."""
    rows = []
    for box in sorted(word_boxes(task), key=lambda b: (b["y1"], b["x1"])):
        for row in rows:
            if abs(row[0]["y1"] - box["y1"]) < ROW_TOL:
                row.append(box)
                break
        else:
            rows.append([box])
    return [sorted(row, key=lambda b: b["x1"]) for row in rows]


def line_widths(task):
    """Width of each passage line (px), in reading order."""
    return [max(b["x2"] for b in row) - min(b["x1"] for b in row) for row in word_box_rows(task)]


def text_geometry(task):
    """Input of the comparison algorithms: line_Y, the center y of each line, and word_XY, one
    row per word with its center x and the y of its line."""
    boxes = word_boxes(task)
    centers = sorted((b["y1"] + b["y2"]) / 2 for b in boxes)
    lines, current = [], [centers[0]]
    for y in centers[1:]:
        if y - current[-1] > 40:  # a new line starts
            lines.append(current)
            current = []
        current.append(y)
    lines.append(current)
    line_Y = np.array([np.mean(line) for line in lines])
    word_XY = np.array(
        sorted(
            (
                (
                    (b["x1"] + b["x2"]) / 2,
                    line_Y[np.argmin(np.abs(line_Y - (b["y1"] + b["y2"]) / 2))],
                )
                for b in boxes
            ),
            key=lambda w: (w[1], w[0]),
        )
    )
    return line_Y, word_XY
