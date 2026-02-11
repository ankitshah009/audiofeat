from __future__ import annotations

import torch

from .spectrogram import cqt_spectrogram


def cqt(
    waveform: torch.Tensor,
    sample_rate: int,
    hop_length: int = 512,
    f_min: float = 32.70,
    n_bins: int = 84,
    bins_per_octave: int = 12,
) -> torch.Tensor:
    """
    Compute CQT magnitude spectrogram, aligned with librosa when available.
    """
    return cqt_spectrogram(
        waveform,
        sample_rate=sample_rate,
        hop_length=hop_length,
        fmin=f_min,
        n_bins=n_bins,
        bins_per_octave=bins_per_octave,
    )
