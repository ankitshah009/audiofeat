
import torch
from ..temporal.rms import hann_window

def sibilant_spectral_peak_frequency(x: torch.Tensor, fs: int, n_fft: int = 1024):
    """Peak frequency of sibilant energy between 3 and 12 kHz."""
    X = torch.fft.rfft(x * hann_window(x.numel()).to(x.device), n=n_fft)
    P = X.abs() ** 2
    freqs = torch.linspace(0, fs / 2, P.numel(), device=x.device)
    mask = (freqs >= 3000) & (freqs <= 12000)
    if mask.sum() == 0:
        return torch.tensor(0.0, device=x.device)
    peak_idx = P[mask].argmax()
    return freqs[mask][peak_idx]
