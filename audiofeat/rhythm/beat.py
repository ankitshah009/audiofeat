from __future__ import annotations

import numpy as np
import torch

from ..temporal.beat import beat_track as _beat_track_frames


def beat_detection(
    signal: torch.Tensor,
    sample_rate: int,
    window_size: float = 0.05,
    hop_size: float = 0.025,
):
    """
    Estimate BPM and confidence, aligned with librosa beat tracking where available.
    """
    if sample_rate <= 0:
        raise ValueError("sample_rate must be > 0.")
    if window_size <= 0 or hop_size <= 0:
        raise ValueError("window_size and hop_size must be > 0.")

    n_fft = max(64, int(round(window_size * sample_rate)))
    hop_length = max(1, int(round(hop_size * sample_rate)))
    tempo, beat_frames = _beat_track_frames(
        signal,
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
    )
    bpm = float(tempo.item())
    if beat_frames.numel() < 2:
        return bpm, 0.0

    ibis = torch.diff(beat_frames.float())
    ibis = ibis[ibis > 0]
    if ibis.numel() == 0:
        return bpm, 0.0
    cv = float((ibis.std(unbiased=False) / (ibis.mean() + 1e-8)).item())
    confidence = float(np.clip(1.0 - cv, 0.0, 1.0))
    return bpm, confidence
