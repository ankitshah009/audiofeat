
import torch
from ..temporal.rms import hann_window

__all__ = ["spectral_slope"]


def spectral_slope(x: torch.Tensor, n_fft: int, sample_rate: int = 22050):
    """
    Computes the spectral slope of an audio signal.

    The spectral slope is the slope of a linear regression of the
    log-magnitude spectrum against frequency (in Hz). A signal whose magnitude
    decays as ``exp(-alpha * f)`` yields a slope of ``-alpha``.

    Args:
        x (torch.Tensor): The audio signal.
        n_fft (int): The number of FFT points.
        sample_rate (int): Sampling rate of the audio (Hz). The frequency axis
            is built with ``rfftfreq`` so the slope is expressed per Hz rather
            than per FFT bin.

    Returns:
        torch.Tensor: The spectral slope (per Hz).
    """
    if x.numel() < 2:
        return torch.tensor(0.0, device=x.device, dtype=x.dtype)

    X = torch.fft.rfft(x * hann_window(x.numel()).to(x.device), n=n_fft)
    log_mag = torch.log(X.abs() + 1e-8)
    freqs = torch.fft.rfftfreq(n_fft, d=1.0 / sample_rate).to(x.device)

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
        return torch.tensor(0.0, device=x.device, dtype=x.dtype)

    return numerator / denominator
