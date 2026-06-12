import torch
from ..temporal.rms import hann_window

__all__ = ["spectral_entropy"]


def spectral_entropy(x: torch.Tensor, n_fft: int):
    """Spectral entropy of a frame.

    Returns a finite ``0.0`` for silent input (where the power spectrum sums to
    zero and normalisation would otherwise produce NaN) and for length-1 input.
    """
    if x.numel() < 2:
        return torch.tensor(0.0, device=x.device, dtype=x.dtype)

    X = torch.fft.rfft(x * hann_window(x.numel()).to(x.device), n=n_fft)
    P = (X.abs() ** 2)
    denom = P.sum()
    if denom <= 0:
        return torch.tensor(0.0, device=x.device, dtype=x.dtype)
    P = P / denom
    return -(P * torch.log2(P + 1e-12)).sum()
