"""Audio decoding for the speech-based supplementation (article section 5.2)."""

import av
import librosa
import numpy as np


def load_audio(path, target_sr=16000):
    """Decode an audio recording (for example m4a) to float32 mono at target_sr."""
    container = av.open(str(path))
    frames, sr = [], None
    for frame in container.decode(audio=0):
        sr = frame.sample_rate
        frames.append(frame.to_ndarray())
    a = np.concatenate(frames, axis=1)
    a = a.mean(axis=0) if a.shape[0] > 1 else a[0].astype(np.float32)
    if sr != target_sr:
        a = librosa.resample(a, orig_sr=sr, target_sr=target_sr)
    return a
