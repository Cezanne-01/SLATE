"""Line-transition scoring (article section 6.4), shared by the clinical and public analyses.

Detected and reference transitions are matched one-to-one within TOL = 1.5 s, closest pairs
first. The first-line entry is not scored, because its definition depends on how a task starts
and is not comparable across methods and datasets: after matching, the reference first entry and
the detection matched to it are dropped, as are unmatched detections that precede it (gaze
before reading).

`transitions_from_lines` converts a line assignment of fixations (comparison algorithms, line
labels of the public datasets) to transition times.
"""

TOL = 1.5  # matching tolerance (s)


def match_transitions(detected, reference, tol=TOL):
    """Greedy one-to-one matching, closest pairs first. Returns {detected index: reference index}."""
    pairs = sorted(
        (abs(a - b), i, j)
        for i, a in enumerate(detected)
        for j, b in enumerate(reference)
        if abs(a - b) <= tol
    )
    matched, used_reference = {}, set()
    for _, i, j in pairs:
        if i not in matched and j not in used_reference:
            matched[i] = j
            used_reference.add(j)
    return matched


def transition_counts(detected, reference, tol=TOL):
    """(TP, FP, FN) for transitions into the second and later lines (first-line entry excluded)."""
    detected, reference = sorted(detected), sorted(reference)
    if not reference:
        return 0, len(detected), 0
    matched = match_transitions(detected, reference, tol)
    tp = fp = 0
    for i, time in enumerate(detected):
        j = matched.get(i)
        if j == 0 or (j is None and time < reference[0]):
            continue  # the first-line entry, or gaze before reading: not scored
        if j is None:
            fp += 1
        else:
            tp += 1
    return tp, fp, len(reference) - 1 - tp


def precision_recall_f1(tp, fp, fn):
    """Precision, recall and F1 = 2TP / (2TP + FP + FN) from pooled counts."""
    return dict(
        precision=tp / (tp + fp) if tp + fp else 0.0,
        recall=tp / (tp + fn) if tp + fn else 0.0,
        f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
    )


def transitions_from_lines(fixations, lines):
    """Transition times from a line assignment: for each line, the start time of its longest
    (in time) contiguous run of assigned fixations. `fixations` has columns start, end, ..."""
    longest = {}
    i, n = 0, len(lines)
    while i < n:
        j = i
        while j + 1 < n and lines[j + 1] == lines[i]:
            j += 1
        duration = fixations[j, 1] - fixations[i, 0]
        if lines[i] not in longest or duration > longest[lines[i]][1]:
            longest[lines[i]] = (fixations[i, 0], duration)
        i = j + 1
    return sorted(start for start, _ in longest.values())
