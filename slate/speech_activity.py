"""
Voice activity for the speech-based supplementation (article section 5.2).

`vad_segments` is an energy-based voice activity detection: 25 ms RMS frames every 10 ms, speech
where the level exceeds the 10th percentile by 35% of the 10th-95th percentile range, gaps
shorter than 100 ms filled, segments shorter than 120 ms removed. `speech_cum` is the speech
amount (total voiced time) within an interval. Time axis: 0 = start of passage presentation,
shared by gaze (calibration removed) and speech.
"""

import numpy as np


def vad_segments(a, sr, hop_ms=10):
    """Return speech segments [(start, end) s] of the audio signal `a` at sampling rate `sr`."""
    import librosa

    fl, hl = int(sr * 0.025), int(sr * hop_ms / 1000)
    db = librosa.amplitude_to_db(
        librosa.feature.rms(y=a, frame_length=fl, hop_length=hl)[0] + 1e-10
    )
    floor, peak = np.percentile(db, 10), np.percentile(db, 95)
    speech = db > floor + 0.35 * (peak - floor)
    minf = int(100 / hop_ms)
    i = 0
    while i < len(speech):
        if not speech[i]:
            j = i
            while j < len(speech) and not speech[j]:
                j += 1
            if (j - i) < minf and 0 < i and j < len(speech):
                speech[i:j] = True
            i = j
        else:
            i += 1
    t = np.arange(len(speech)) * hop_ms / 1000
    segs, i = [], 0
    while i < len(speech):
        if speech[i]:
            j = i
            while j < len(speech) and speech[j]:
                j += 1
            if (j - i) >= int(120 / hop_ms):
                segs.append((float(t[i]), float(t[min(j, len(t) - 1)])))
            i = j
        else:
            i += 1
    return segs


def speech_cum(segs, t0, t1):
    """Total speech time (s) within [t0, t1]."""
    return sum(max(0.0, min(e, t1) - max(s, t0)) for s, e in segs)
