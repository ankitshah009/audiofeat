
import torch
from ..temporal.rms import hann_window

def nasality_index(nasal: torch.Tensor, oral: torch.Tensor, fs: int, n_fft: int = 1024):
    """Compute nasality index from nasal and oral microphone signals."""
    N = torch.fft.rfft(nasal * hann_window(nasal.numel()).to(nasal.device), n=n_fft)
    O = torch.fft.rfft(oral * hann_window(oral.numel()).to(oral.device), n=n_fft)
    freqs = torch.linspace(0, fs / 2, N.numel(), device=nasal.device)
    mask = (freqs >= 300) & (freqs <= 800)
    n_power = (N.abs() ** 2)[mask].sum()
    o_power = (O.abs() ** 2)[mask].sum()
    return 10 * torch.log10((n_power + 1e-8) / (o_power + 1e-8))
