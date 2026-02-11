from __future__ import annotations

import torch

from .beat import beat_track as _beat_track_frames


def tempo(
    audio_data: torch.Tensor,
    sample_rate: int,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> float:
    """
    Estimate tempo in BPM using the shared beat tracker backend.
    """
    bpm, _ = _beat_track_frames(
        audio_data,
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
    )
    return float(bpm.item())


def beat_track(
    audio_data: torch.Tensor,
    sample_rate: int,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> torch.Tensor:
    """
    Return beat times in seconds.
    """
    _, beat_frames = _beat_track_frames(
        audio_data,
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
    )
    if beat_frames.numel() == 0:
        return torch.zeros(0, dtype=torch.float32, device=audio_data.device)
    beat_times = beat_frames.float() * float(hop_length) / float(sample_rate)
    return beat_times.to(device=audio_data.device)
