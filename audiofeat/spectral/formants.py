
import torch
import numpy as np
from scipy.linalg import toeplitz

def formant_frequencies(x: torch.Tensor, fs: int, order: int):
    """Estimate formant frequencies using LPC."""
    x = x - 0.97 * torch.nn.functional.pad(x[:-1], (1,0))
    autocorr = torch.fft.irfft(torch.fft.rfft(x, n=2*order), n=2*order)
    R = torch.from_numpy(toeplitz(autocorr[:order].numpy()))
    r = autocorr[1:order+1]
    coeffs = torch.linalg.solve(R, r)
    a = torch.cat([torch.ones(1, device=x.device), -coeffs])
    roots = np.roots(a.numpy())
    roots = roots[(roots.imag >= 0)]
    angles = np.angle(roots)
    formants = angles * fs / (2 * np.pi)
    formants = np.sort(formants)
    return torch.from_numpy(formants)

def formant_bandwidths(a: torch.Tensor, fs: int):
    """Formant bandwidths from LPC polynomial roots."""
    roots = np.roots(a.numpy())
    roots = roots[(roots.imag >= 0)]
    freqs = np.angle(roots) * fs / (2 * np.pi)
    bandwidths = -2 * fs * np.log(np.abs(roots)) / (2 * np.pi)
    order = freqs.argsort()
    return torch.from_numpy(bandwidths[order])

def formant_dispersion(formants: torch.Tensor):
    """Average spacing between first five formants."""
    if formants.numel() < 5:
        return torch.tensor(0.0, device=formants.device)
    d = 0
    for i in range(4):
        d += formants[i+1] - formants[i]
    return d / 4
