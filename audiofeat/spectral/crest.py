
import torch
from ..temporal.rms import hann_window

__all__ = ["spectral_crest_factor"]


def spectral_crest_factor(x: torch.Tensor, n_fft: int):
    """
    Computes the spectral crest factor of an audio signal.

    The spectral crest factor is the ratio of the peak of the power spectrum to
    its mean: ``max(P) / mean(P)``. It is >= 1 by construction; a pure tone
    yields a large value while white noise yields a value close to 1.

    Args:
        x (torch.Tensor): The audio signal.
        n_fft (int): The number of FFT points.

    Returns:
        torch.Tensor: The spectral crest factor.
    """
    if x.numel() < 2:
        return torch.tensor(0.0, device=x.device, dtype=x.dtype)

    X = torch.fft.rfft(x * hann_window(x.numel()).to(x.device), n=n_fft)
    P = X.abs() ** 2

    max_magnitude = torch.max(P)
    mean_magnitude = torch.mean(P)

    if mean_magnitude == 0:
        return torch.tensor(0.0, device=x.device, dtype=x.dtype)

    return max_magnitude / mean_magnitude
