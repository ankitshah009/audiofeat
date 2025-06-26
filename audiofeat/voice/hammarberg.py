
import torch
from ..temporal.rms import hann_window

def hammarberg_index(x: torch.Tensor, fs: int, n_fft: int = 2048):
    """
    Computes the Hammarberg Index: ratio of highest energy peak in 0-2 kHz to 2-5 kHz.

    Args:
        x (torch.Tensor): The audio signal.
        fs (int): The sample rate of the audio.
        n_fft (int): The number of FFT points.

    Returns:
        torch.Tensor: The Hammarberg Index.
    """
    X = torch.fft.rfft(x * hann_window(x.numel()).to(x.device), n=n_fft)
    P = X.abs() ** 2
    freqs = torch.linspace(0, fs / 2, P.numel(), device=x.device)

    low_band_mask = (freqs >= 0) & (freqs < 2000)
    high_band_mask = (freqs >= 2000) & (freqs < 5000)

    low_band_peak = torch.max(P[low_band_mask]) if low_band_mask.sum() > 0 else torch.tensor(1e-8)
    high_band_peak = torch.max(P[high_band_mask]) if high_band_mask.sum() > 0 else torch.tensor(1e-8)

    return 10 * torch.log10(low_band_peak / (high_band_peak + 1e-8))
