"""Speaker Diarization utilities leveraging pyannote-audio pre-trained pipelines.

This module provides a thin wrapper around `pyannote.audio`’s state-of-the-art diarization
pipeline so users can obtain "who-spoke-when" annotations with a single call.
The heavy ML model is downloaded on-demand the first time it is used and cached
locally (via HuggingFace Hub).

Requires optional dependency: `pyannote.audio` (install with `pip install audiofeat[models]`).

Example
-------
>>> import torch
>>> from audiofeat.diarization import diarize
>>> diarization = diarize("example.wav")
>>> for speech_turn in diarization:
...     print(f"Speaker {speech_turn[2]}: {speech_turn[0]:.2f}s – {speech_turn[1]:.2f}s")
"""
from __future__ import annotations

from typing import List, Tuple, Union
from pathlib import Path

# Lazy import so that base installation remains lightweight


def _lazy_load_pipeline():
    try:
        from pyannote.audio import Pipeline  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "pyannote.audio is required for speaker diarization. "
            "Install with `pip install audiofeat[models]` or `pip install pyannote.audio`."
        ) from exc

    # Using the publicly available pre-trained pipeline (may change over time)
    return Pipeline.from_pretrained("pyannote/speaker-diarization@2.1", use_auth_token=None)


def diarize(
    audio: Union[str, Path],
    *,
    num_speakers: int | None = None,
    min_speaker_turn_duration: float = 0.0,
) -> List[Tuple[float, float, str]]:
    """Perform speaker diarization on an audio file.

    Parameters
    ----------
    audio : str or Path
        Path to mono/stereo WAV/FLAC/MP3 file.
    num_speakers : int, optional
        If known, hint the number of speakers to the pipeline.
    min_speaker_turn_duration : float, optional
        Post-processing: merge segments shorter than this (in seconds).

    Returns
    -------
    list of (start, end, speaker_label)
        Time-stamped speaker segments in seconds.
    """
    pipeline = _lazy_load_pipeline()

    # pyannote handles file paths directly
    diarization = pipeline(str(audio), num_speakers=num_speakers)

    # Convert pyannote Annotation → list[tuple]
    segments: List[Tuple[float, float, str]] = []
    for segment, _, speaker in diarization.itertracks(yield_label=True):
        start = float(segment.start)
        end = float(segment.end)
        if end - start < min_speaker_turn_duration:
            continue
        segments.append((start, end, str(speaker)))

    # Sort just in case
    segments.sort(key=lambda x: x[0])
    return segments
