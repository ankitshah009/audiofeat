"""Source separation / localisation wrapper using Asteroid ConvTasNet."""
from __future__ import annotations
from typing import List

import torch


@torch.inference_mode()
def separate_sources(waveform: torch.Tensor, sample_rate: int) -> List[torch.Tensor]:
    try:
        from asteroid.models import ConvTasNet  # type: ignore
        from asteroid.utils import torch_utils
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Install asteroid via `pip install audiofeat[models]`.") from exc

    model = ConvTasNet.from_pretrained("JorisCos/ConvTasNet_Libri2Mix_sepclean_16k")
    if sample_rate != 16000:
        import torchaudio.transforms as T

        waveform = T.Resample(sample_rate, 16000)(waveform)
    est_sources = model.separate(waveform)
    return [s.cpu() for s in est_sources]
