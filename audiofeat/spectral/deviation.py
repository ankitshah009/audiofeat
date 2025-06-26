
import torch
from ..temporal.rms import hann_window

def spectral_deviation(x: torch.Tensor, n_fft: int):
    """
    Quantifies the "jaggedness" of the local spectrum.

    Args:
        x (torch.Tensor): The audio signal.
        n_fft (int): The number of FFT points.

    Returns:
        torch.Tensor: The spectral deviation.
    """
    X = torch.fft.rfft(x * hann_window(x.numel()).to(x.device), n=n_fft)
    P = X.abs() ** 2
    
    # Normalize spectrum
    P_norm = P / torch.sum(P)

    # Calculate spectral deviation
    deviation = torch.sum(torch.abs(P_norm[1:] - P_norm[:-1]))
    
    return deviation
