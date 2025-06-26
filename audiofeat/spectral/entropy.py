import torch
from ..temporal.rms import hann_window

def spectral_entropy(x: torch.Tensor, n_fft: int):
    """Spectral entropy of a frame."""
    X = torch.fft.rfft(x * hann_window(x.numel()).to(x.device), n=n_fft)
    P = (X.abs() ** 2)
    P = P / P.sum()
    return -(P * torch.log2(P + 1e-12)).sum()