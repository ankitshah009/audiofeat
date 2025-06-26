
import torch
from ..temporal.rms import hann_window

def alpha_ratio(x: torch.Tensor, fs: int, n_fft: int = 2048):
    """
    Computes the Alpha Ratio: ratio of energy in 50-1000 Hz to 1000-5000 Hz.

    Args:
        x (torch.Tensor): The audio signal.
        fs (int): The sample rate of the audio.
        n_fft (int): The number of FFT points.

    Returns:
        torch.Tensor: The Alpha Ratio.
    """
    X = torch.fft.rfft(x * hann_window(x.numel()).to(x.device), n=n_fft)
    P = X.abs() ** 2
    freqs = torch.linspace(0, fs / 2, P.numel(), device=x.device)

    low_band_mask = (freqs >= 50) & (freqs < 1000)
    high_band_mask = (freqs >= 1000) & (freqs < 5000)

    low_band_energy = P[low_band_mask].sum()
    high_band_energy = P[high_band_mask].sum()

    return 10 * torch.log10(low_band_energy / (high_band_energy + 1e-8))
