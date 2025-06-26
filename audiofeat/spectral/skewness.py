
import torch
from ..temporal.rms import hann_window

def spectral_skewness(x: torch.Tensor, n_fft: int):
    X = torch.fft.rfft(x * hann_window(x.numel()).to(x.device), n=n_fft)
    P = X.abs() ** 2
    freqs = torch.linspace(0, n_fft // 2, P.numel(), device=x.device)
    mean = torch.sum(freqs * P) / torch.sum(P)
    var = torch.sum((freqs - mean) ** 2 * P) / torch.sum(P)
    skew = torch.sum((freqs - mean) ** 3 * P) / (torch.sum(P) * var.sqrt() ** 3)
    kurt = torch.sum((freqs - mean) ** 4 * P) / (torch.sum(P) * var ** 2) - 3
    return skew, kurt
