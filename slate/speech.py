"""Speech-based supplementation of the detected line transitions (article section 5.2).

Optional step applied to the gaze-only transitions of `slate.detector` when the oral-reading
audio was recorded. Speech only decides where to search again. The time of any added or moved
transition is always taken from the horizontal gaze trace.

  line cues: recognized words are aligned in order to the passage text. The first aligned word
     on each line l gives the line cue c_l (`line_cues`).
  first-line entry check: a first detection well before c_1, followed by silence, is treated as
     gaze movement before reading and removed or moved to a leftward move near c_1.
  speech amount: if the voiced time between two neighboring transitions is at least
     LAMBDA = 1.7 times the amount expected for that line's width, a smaller leftward movement
     inside the interval is recovered as the missed transition.
  line-cue shortfall: if c_l shows line l being read but fewer than l transitions precede it, a
     missed transition is searched for before the cue.
"""

import numpy as np

from slate import detector
from slate.speech_activity import speech_cum

LAMBDA = 1.7  # an interval is searched again when its voiced time >= LAMBDA x expected amount
# (speech rate x line width). A missed transition gives a ratio of about
# 1 + next line width / this line width
RECOVER_DROP = 0.4  # minimum leftward movement to recover, as a fraction of the main threshold
CUE_WIN = 1.2  # search within +/- this many seconds of a line cue
ONSET_GUARD = 2.5  # a first detection earlier than (c_1 - ONSET_GUARD) is checked
ONSET_SILENCE = 0.45  # ... and treated as pre-reading only if the voiced fraction of the following
# 4 s is below this value


def leftward_candidates(xs, t, span, win, lo, hi, min_drop):
    """Leftward-movement candidates inside (lo, hi) of at least min_drop, largest first. A
    candidate whose preceding ROUNDTRIP_S window spans (max - min) more than ROUNDTRIP_R x its
    leftward move is skipped. Candidates closer than 0.8 s to a kept one are dropped."""
    disp = xs[win:] - xs[:-win]
    idx = np.flatnonzero((t[: len(disp)] > lo) & (t[: len(disp)] < hi) & (disp < -min_drop))
    out = []
    for i in idx[np.argsort(disp[idx])]:
        tm = t[min(i + win // 2, len(t) - 1)]
        i0 = int(np.searchsorted(t, t[i] - detector.ROUNDTRIP_S))
        seg = xs[i0 : i + 1]
        if len(seg) >= 3 and (seg.max() - seg.min()) > detector.ROUNDTRIP_R * (-disp[i]):
            continue
        if all(abs(tm - o) > 0.8 for o in out):
            out.append(tm)
    return out


def voiced_fraction(segs, a, b):
    """Voiced fraction of [a, b]."""
    return speech_cum(segs, a, b) / max(b - a, 1e-9)


def supplement(det, t, x, segs, widths=None, cues=None):
    """Supplement the gaze-only detections `det` with speech.

    segs   = voice-activity segments [(start, end) s]
    widths = horizontal width of each passage line (px), giving the expected speech amount per line
    cues  = {line number: first speech time}, the line cues c_l (may cover only some lines)
    Returns the supplemented, sorted list of transition times (s)."""
    out = sorted(det)
    xs, span, win = detector.smooth(t, x)
    thr = detector.KAPPA * span
    cues = cues or {}

    if 1 in cues and out:
        # First-line entry check. While the first detection is well before c_1, the second one
        # lies no later than CUE_WIN after c_1 and the 4 s after the first are mostly silent,
        # the first detection is removed (gaze reaches line 1 when reading aloud begins).
        t1 = cues[1]
        while (
            len(out) >= 2
            and out[0] < t1 - ONSET_GUARD
            and out[1] <= t1 + CUE_WIN
            and segs is not None
            and len(segs)
            and voiced_fraction(segs, out[0], out[0] + 4.0) < ONSET_SILENCE
        ):
            out = out[1:]
        if out[0] < t1 - ONSET_GUARD:
            # otherwise move the entry to the landing point of a leftward movement near the cue
            cand = leftward_candidates(xs, t, span, win, t1 - 3.0, t1 + 0.5, detector.KAPPA0 * span)
            if cand:
                k0 = int(np.searchsorted(t, cand[0]))
                k1 = int(np.searchsorted(t, cand[0] + detector.LANDING_WIN, side="right"))
                out = sorted([float(t[k0 + int(np.argmin(xs[k0 : max(k1, k0 + 1)]))])] + out[1:])

    if segs is not None and len(segs) and widths is not None and len(widths) and len(out) >= 4:
        # Interval k between detections corresponds to line k+1. The speech rate (s/px), pooled
        # over the trial, gives the expected speech amount per line. Intervals reaching LAMBDA
        # times that amount are searched again (at most one addition per interval).
        pairs = [(a, b, widths[k]) for k, (a, b) in enumerate(zip(out, out[1:])) if k < len(widths)]
        sp = [speech_cum(segs, a, b) for a, b, _ in pairs]
        wsum = sum(w for _, _, w in pairs)
        if wsum > 0 and sum(sp) > 0:
            rate = sum(sp) / wsum
            ins = []
            for (a, b, w), s in zip(pairs, sp):
                if s >= LAMBDA * rate * w:
                    c = leftward_candidates(xs, t, span, win, a + 0.8, b - 0.8, RECOVER_DROP * thr)
                    if c:
                        ins.append(c[0])
                    elif cues:
                        # a line cue inside this interval: search its neighborhood again with a
                        # lower threshold
                        for la in cues.values():
                            if a + 0.8 < la < b - 0.8:
                                c = leftward_candidates(
                                    xs, t, span, win, la - CUE_WIN, la + CUE_WIN, 0.25 * thr
                                )
                                if c:
                                    ins.append(c[0])
                                    break
            out = sorted(out + ins)

    if cues and len(out) >= 3:
        # Line-cue shortfall: the cue shows line L being read at time tau but fewer than L
        # transitions precede tau, so a transition was missed before it (the eye leads the voice).
        ins = []
        for li, tm in sorted(cues.items()):
            c = sum(1 for v in out if v <= tm)
            if c and c <= li - 1:
                a = out[c - 1]
                b = out[c] if c < len(out) else tm
                if b - a > 1.6:
                    cc = leftward_candidates(xs, t, span, win, a + 0.8, b - 0.8, RECOVER_DROP * thr)
                    if cc:
                        ins.append(cc[0])
                break  # one insertion per pass
        out = sorted(out + ins)
    return out


def line_cues(aligned, line_of):
    """Line cues c_l from passage words aligned to the speech.

    aligned = [(passage word, speech time or None)] in reading order, line_of = line number of
    each passage word. Returns {line number: time of the first aligned word on that line}. Only
    cues in increasing time order are kept."""
    first = {}
    for k, (_, tm) in enumerate(aligned):
        if tm is not None:
            li = line_of[k]
            first[li] = min(first.get(li, 1e9), tm)
    ok, last = {}, -1e9
    for li in sorted(first):
        if first[li] > last:
            ok[li] = first[li]
            last = first[li]
    return ok
