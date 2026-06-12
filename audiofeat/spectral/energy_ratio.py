
import torch
from ..temporal.rms import hann_window

__all__ = ["low_high_energy_ratio"]


def low_high_energy_ratio(x: torch.Tensor, fs: int, n_fft: int = 1024):
    """Ratio of energy below 1 kHz to that above 3 kHz, in dB.

    The epsilon is applied symmetrically to numerator and denominator so the
    result stays finite for silence and for signals with no low-band energy
    (which would otherwise produce ``log10(0) = -inf``).
    """
    if x.numel() < 2:
        return torch.tensor(0.0, device=x.device, dtype=x.dtype)

    X = torch.fft.rfft(x * hann_window(x.numel()).to(x.device), n=n_fft)
    P = X.abs() ** 2
    freqs = torch.linspace(0, fs / 2, P.numel(), device=x.device)
    low = P[freqs < 1000].sum()
    high = P[freqs > 3000].sum()
    return 10 * torch.log10((low + 1e-8) / (high + 1e-8))
