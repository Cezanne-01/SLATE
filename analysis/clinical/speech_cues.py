"""Speech input of the clinical recordings for the speech-based supplementation (article
sections 5.2 and 6.1).

For one trial, the audio is decoded and segmented by voice activity, and the line cues c_l are
taken from the cached word-level speech recognition aligned to the passage text. The cache holds
one json per participant and passage and is built once with

    python -m analysis.clinical.speech_cues --task 6.2
    python -m analysis.clinical.speech_cues --task 6.3

(faster-whisper small model, Korean, GPU if available). It is written to
speech_words/<passage>/ inside SLATE_CLINICAL_DATA.
"""

import argparse
import json

from analysis.clinical.passages import TEXT, word_box_rows
from analysis.clinical.sessions import trial_table
from analysis.paths import CLINICAL_DATA
from slate import speech
from slate import speech_alignment as asr
from slate.audio import load_audio
from slate.speech_activity import vad_segments


def cache_dir(task):
    """Word-level speech recognition cache of a passage (one json per participant)."""
    return CLINICAL_DATA / "speech_words" / task


def recording_line_cues(pid, task):
    """Line cues c_l of one trial: {line number: time of the first recognized word on that line}.
    Empty if the trial has no cache entry."""
    cache = cache_dir(task) / f"{pid}.json"
    if not cache.exists():
        return {}
    words = [tuple(w) for w in json.loads(cache.read_text())]  # (word, start, end)
    aligned, _ = asr.align_to_ref(words, asr.load_ref(TEXT[task]))
    line_of = [line for line, row in enumerate(word_box_rows(task), 1) for _ in row]
    return speech.line_cues(aligned, line_of)


def supplement_recording(transitions, t, x, audio, pid, task, widths):
    """`slate.speech.supplement` for one clinical trial, reading its audio and line cues."""
    segments = vad_segments(load_audio(audio), 16000)
    return speech.supplement(
        transitions, t, x, segments, widths=widths, cues=recording_line_cues(pid, task)
    )


def transcribe(task):
    """Write the word-level recognition cache of every participant with a voice recording."""
    cd = cache_dir(task)
    cd.mkdir(parents=True, exist_ok=True)
    trials = trial_table()
    model = asr.get_model()
    for trial in trials[trials.passage == task].itertuples():
        pid = trial.participant
        f = cd / f"{pid}.json"
        if f.exists():
            continue
        words = asr.transcribe_words(model, load_audio(CLINICAL_DATA / trial.audio_file))
        f.write_text(json.dumps(words, ensure_ascii=False))
        print(f"{pid}: {len(words)} words")


def main():
    ap = argparse.ArgumentParser(
        description="Build the word-level speech recognition cache used for the line cues."
    )
    ap.add_argument("--task", default="6.2", choices=["6.2", "6.3"], help="passage")
    transcribe(ap.parse_args().task)


if __name__ == "__main__":
    main()
