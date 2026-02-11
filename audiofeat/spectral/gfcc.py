from __future__ import annotations

import torch

from ..cepstral._erb import erb_cepstral_coefficients


def gfcc(
    waveform: torch.Tensor,
    sample_rate: int,
    n_gfcc: int = 40,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_bands: int = 128,
    fmin: float = 50.0,
    fmax: float | None = None,
) -> torch.Tensor:
    """
    Compute GFCCs using an ERB/gammatone-style filterbank pipeline.
    """
    return erb_cepstral_coefficients(
        waveform=waveform,
        sample_rate=sample_rate,
        n_coeffs=n_gfcc,
        n_fft=n_fft,
        hop_length=hop_length,
        n_bands=n_bands,
        fmin=fmin,
        fmax=fmax,
    )
