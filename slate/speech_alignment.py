"""
Word timing of oral reading by speech recognition and in-order alignment to the passage text
(article sections 5.2 and 6.1).

  (1) faster-whisper (small model, Korean) gives word-level timestamps of what was spoken.
  (2) The recognized words are aligned in order to the passage words (longest matching blocks,
      difflib), giving a speech time for each matched passage word. Words that are not in the
      passage are not matched.
"""

import difflib
import re
from pathlib import Path

WHISPER_MODEL = "small"
ASR_LANG = "ko"


def get_model():
    """faster-whisper model (GPU if available, else CPU)."""
    from faster_whisper import WhisperModel

    try:
        return WhisperModel(WHISPER_MODEL, device="cuda", compute_type="float16")
    except Exception:
        return WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")


def transcribe_words(model, audio):
    """Word-level timestamps. Returns [(word, start, end), ...]."""
    segs, _ = model.transcribe(
        audio, language=ASR_LANG, beam_size=1, word_timestamps=True, vad_filter=True
    )
    words = []
    for s in segs:
        for w in s.words or []:
            tok = w.word.strip()
            if tok:
                words.append((tok, w.start, w.end))
    return words


def norm(tok):
    """Normalization for alignment: keep only Unicode word characters, lowercase."""
    return re.sub(r"[^\w]", "", tok, flags=re.UNICODE).lower()


def load_ref(path):
    """Passage words in reading order (blank lines and lines starting with # are skipped)."""
    lines = []
    for ln in Path(path).read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if ln and not ln.startswith("#"):
            lines.append(ln)
    return " ".join(lines).split()


def align_to_ref(words, ref_words):
    """
    Align the ASR word list in order to the passage word list (difflib matching blocks).
    Returns: list of (passage word, ASR start time or None), and coverage (0-1).
    """
    asr_norm = [norm(w) for w, _, _ in words]
    ref_norm = [norm(w) for w in ref_words]
    sm = difflib.SequenceMatcher(a=ref_norm, b=asr_norm, autojunk=False)
    ref_time = [None] * len(ref_words)
    for i, j, n in sm.get_matching_blocks():
        for k in range(n):
            ref_time[i + k] = words[j + k][1]  # ASR start
    matched = sum(t is not None for t in ref_time)
    return list(zip(ref_words, ref_time)), matched / len(ref_words) if ref_words else 0
