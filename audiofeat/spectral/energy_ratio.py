
import torch
from ..temporal.rms import hann_window

def low_high_energy_ratio(x: torch.Tensor, fs: int, n_fft: int = 1024):
    """Ratio of energy below 1 kHz to that above 3 kHz."""
    X = torch.fft.rfft(x * hann_window(x.numel()).to(x.device), n=n_fft)
    P = X.abs() ** 2
    freqs = torch.linspace(0, fs / 2, P.numel(), device=x.device)
    low = P[freqs < 1000].sum()
    high = P[freqs > 3000].sum()
    return 10 * torch.log10(low / (high + 1e-8))
