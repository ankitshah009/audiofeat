"""RNNoise RNN-based noise suppression wrapper."""
from __future__ import annotations
import torch


@torch.inference_mode()
def denoise_rnn(waveform: torch.Tensor, sample_rate: int = 48000):
    """Suppress noise using rnnoise-torch if installed."""
    try:
        from rnnoise_torch import RNNoise  # type: ignore
    except ImportError as exc:
        raise ModuleNotFoundError(
            "`rnnoise-torch` is required for RNN denoising. "
            "Install with `pip install audiofeat[denoise]`."
        ) from exc

    model = RNNoise()
    return model(waveform, sample_rate)
