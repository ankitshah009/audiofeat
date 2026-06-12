"""Source separation wrapper using Asteroid ConvTasNet.

Despite the legacy module name, this provides **source separation** (splitting a
mixture into its constituent sources), not spatial localisation/ITD/ILD. It
wraps Asteroid's pretrained ConvTasNet to separate a 2-speaker mixture.

Requires the optional dependency group ``separation``
(``pip install audiofeat[separation]``).
"""
from __future__ import annotations

from typing import List

import torch

from ._optional import require


@torch.inference_mode()
def separate_sources(waveform: torch.Tensor, sample_rate: int) -> List[torch.Tensor]:
    """Separate *waveform* into its estimated sources.

    Parameters
    ----------
    waveform : torch.Tensor
        Mixture audio ``(samples,)`` or ``(channels, samples)``.
    sample_rate : int
        Sampling rate of *waveform*; resampled to 16 kHz for the model.

    Returns
    -------
    list of torch.Tensor
        One waveform tensor per estimated source.
    """
    asteroid_models = require("asteroid.models", extra="separation", pip_name="asteroid")
    ConvTasNet = asteroid_models.ConvTasNet

    model = ConvTasNet.from_pretrained("JorisCos/ConvTasNet_Libri2Mix_sepclean_16k")
    if sample_rate != 16000:
        import torchaudio.transforms as T

        waveform = T.Resample(sample_rate, 16000)(waveform)

    # Use the module forward, NOT model.separate() — the latter is a file-I/O
    # convenience that writes WAVs to disk. forward returns (batch, n_src, samples).
    est_sources = model(waveform)
    est_sources = est_sources.squeeze(0)  # drop the batch dim -> (n_src, samples)
    return [est_sources[i].cpu() for i in range(est_sources.size(0))]
