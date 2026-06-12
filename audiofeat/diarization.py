"""Speaker Diarization utilities leveraging pyannote-audio pretrained pipelines.

This module provides a thin wrapper around `pyannote.audio`'s state-of-the-art
diarization pipeline so users can obtain "who-spoke-when" annotations with a
single call. The heavy ML model is downloaded on-demand the first time it is
used and cached locally (via HuggingFace Hub).

The default checkpoint (``pyannote/speaker-diarization-3.1``) is **gated** on the
Hub: you must accept its conditions and supply a HuggingFace access token, via
the ``auth_token`` argument or the ``HF_TOKEN`` environment variable.

Requires optional dependency ``pyannote.audio``
(``pip install audiofeat[diarization]``).

Example
-------
>>> import torch
>>> from audiofeat.diarization import diarize
>>> diarization = diarize("example.wav", auth_token="hf_...")
>>> for start, end, speaker in diarization:
...     print(f"Speaker {speaker}: {start:.2f}s - {end:.2f}s")
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple, Union

# pyannote 3.x default pipeline checkpoint.
_DEFAULT_CHECKPOINT = "pyannote/speaker-diarization-3.1"


def _lazy_load_pipeline(checkpoint: str, auth_token: str | None):
    """Import pyannote (deferred) and build the diarization pipeline.

    ``ImportError`` is caught broadly because ``pyannote.audio`` can fail to
    import (not merely be absent) against incompatible torch/torchaudio builds.
    """
    try:
        from pyannote.audio import Pipeline  # type: ignore
    except ImportError as exc:
        raise ModuleNotFoundError(
            "`pyannote.audio` is required for speaker diarization. "
            "Install with `pip install audiofeat[diarization]`."
        ) from exc

    pipeline = Pipeline.from_pretrained(checkpoint, use_auth_token=auth_token)
    if pipeline is None:
        # pyannote returns None when a gated model is requested without a valid
        # token (or its user conditions have not been accepted).
        raise RuntimeError(
            f"Failed to load gated diarization model '{checkpoint}'. Accept its "
            f"conditions on the HuggingFace Hub and pass a token via the "
            f"`auth_token` argument or the HF_TOKEN environment variable."
        )
    return pipeline


def diarize(
    audio: Union[str, Path],
    *,
    num_speakers: int | None = None,
    min_speaker_turn_duration: float = 0.0,
    auth_token: str | None = None,
    checkpoint: str = _DEFAULT_CHECKPOINT,
) -> List[Tuple[float, float, str]]:
    """Perform speaker diarization on an audio file.

    Parameters
    ----------
    audio : str or Path
        Path to mono/stereo WAV/FLAC/MP3 file.
    num_speakers : int, optional
        If known, hint the number of speakers to the pipeline.
    min_speaker_turn_duration : float, optional
        Post-processing: drop segments shorter than this (in seconds).
    auth_token : str, keyword-only, optional
        HuggingFace access token for the gated diarization model. If omitted,
        the ``HF_TOKEN`` environment variable is used.
    checkpoint : str, keyword-only, optional
        Pipeline checkpoint to load (defaults to ``pyannote/speaker-diarization-3.1``).

    Returns
    -------
    list of (start, end, speaker_label)
        Time-stamped speaker segments in seconds, sorted by start time.
    """
    token = auth_token or os.environ.get("HF_TOKEN")
    pipeline = _lazy_load_pipeline(checkpoint, token)

    # pyannote handles file paths directly.
    diarization = pipeline(str(audio), num_speakers=num_speakers)

    # Convert pyannote Annotation -> list[tuple].
    segments: List[Tuple[float, float, str]] = []
    for segment, _, speaker in diarization.itertracks(yield_label=True):
        start = float(segment.start)
        end = float(segment.end)
        if end - start < min_speaker_turn_duration:
            continue
        segments.append((start, end, str(speaker)))

    segments.sort(key=lambda x: x[0])
    return segments
