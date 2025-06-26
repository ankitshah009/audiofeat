
import torch
from ..temporal.rms import hann_window

def spectral_crest_factor(x: torch.Tensor, n_fft: int):
    """
    Computes the spectral crest factor of an audio signal.

    Args:
        x (torch.Tensor): The audio signal.
        n_fft (int): The number of FFT points.

    Returns:
        torch.Tensor: The spectral crest factor.
    """
    X = torch.fft.rfft(x * hann_window(x.numel()).to(x.device), n=n_fft)
    P = X.abs() ** 2
    
    max_magnitude = torch.max(P)
    sum_magnitudes = torch.sum(P)

    if sum_magnitudes == 0:
        return torch.tensor(0.0, device=x.device)

    return max_magnitude / sum_magnitudes
