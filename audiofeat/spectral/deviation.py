
import torch
from ..temporal.rms import hann_window

__all__ = ["spectral_deviation"]


def spectral_deviation(x: torch.Tensor, n_fft: int):
    """
    Quantifies the "jaggedness" of the local spectrum.

    Returns a finite ``0.0`` for silent input (where the power spectrum sums to
    zero) and for length-1 input.

    Args:
        x (torch.Tensor): The audio signal.
        n_fft (int): The number of FFT points.

    Returns:
        torch.Tensor: The spectral deviation.
    """
    if x.numel() < 2:
        return torch.tensor(0.0, device=x.device, dtype=x.dtype)

    X = torch.fft.rfft(x * hann_window(x.numel()).to(x.device), n=n_fft)
    P = X.abs() ** 2

    # Normalize spectrum (guard against all-zero / silent input)
    denom = torch.sum(P)
    if denom <= 0:
        return torch.tensor(0.0, device=x.device, dtype=x.dtype)
    P_norm = P / denom

    # Calculate spectral deviation
    deviation = torch.sum(torch.abs(P_norm[1:] - P_norm[:-1]))

    return deviation
