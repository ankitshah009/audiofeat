
import torch
from ..temporal.rms import hann_window

def spectral_slope(x: torch.Tensor, n_fft: int):
    """
    Computes the spectral slope of an audio signal.

    Args:
        x (torch.Tensor): The audio signal.
        n_fft (int): The number of FFT points.

    Returns:
        torch.Tensor: The spectral slope.
    """
    X = torch.fft.rfft(x * hann_window(x.numel()).to(x.device), n=n_fft)
    log_mag = torch.log(X.abs() + 1e-8)
    freqs = torch.linspace(0, n_fft // 2, log_mag.numel(), device=x.device)

    # Linear regression of log-magnitude against frequency
    # slope = (N * sum(xy) - sum(x) * sum(y)) / (N * sum(x^2) - (sum(x))^2)
    N = log_mag.numel()
    sum_xy = torch.sum(freqs * log_mag)
    sum_x = torch.sum(freqs)
    sum_y = torch.sum(log_mag)
    sum_x2 = torch.sum(freqs ** 2)

    numerator = N * sum_xy - sum_x * sum_y
    denominator = N * sum_x2 - sum_x ** 2

    if denominator == 0:
        return torch.tensor(0.0, device=x.device)
    
    return numerator / denominator
