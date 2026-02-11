from __future__ import annotations

import torch

from ._erb import erb_cepstral_coefficients


def gtcc(
    audio_data: torch.Tensor,
    sample_rate: int,
    n_gtcc: int = 20,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_bands: int = 128,
    fmin: float = 50.0,
    fmax: float | None = None,
) -> torch.Tensor:
    """
    Compute GTCCs using an ERB/gammatone-style filterbank pipeline.
    """
    return erb_cepstral_coefficients(
        waveform=audio_data,
        sample_rate=sample_rate,
        n_coeffs=n_gtcc,
        n_fft=n_fft,
        hop_length=hop_length,
        n_bands=n_bands,
        fmin=fmin,
        fmax=fmax,
    )
