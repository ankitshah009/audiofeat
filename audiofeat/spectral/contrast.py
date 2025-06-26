
import torch
from ..temporal.rms import hann_window

def spectral_contrast(x: torch.Tensor, fs: int, n_fft: int = 2048, n_bands: int = 6):
    """
    Computes the spectral contrast of an audio signal.

    Args:
        x (torch.Tensor): The audio signal.
        fs (int): The sample rate of the audio.
        n_fft (int): The number of FFT points.
        n_bands (int): The number of frequency bands.

    Returns:
        torch.Tensor: The spectral contrast for each band.
    """
    X = torch.fft.rfft(x * hann_window(x.numel()).to(x.device), n=n_fft)
    P = X.abs() ** 2
    
    # Divide spectrum into bands
    band_edges = torch.linspace(0, n_fft // 2, n_bands + 1, dtype=torch.long)
    
    contrast = []
    for i in range(n_bands):
        band_start = band_edges[i]
        band_end = band_edges[i+1]
        
        if band_start == band_end:
            contrast.append(torch.tensor(0.0, device=x.device))
            continue

        band_spectrum = P[band_start:band_end]
        
        if band_spectrum.numel() == 0:
            contrast.append(torch.tensor(0.0, device=x.device))
            continue

        # Find peaks and valleys
        peaks = torch.max(band_spectrum)
        valleys = torch.min(band_spectrum)
        
        if valleys == 0:
            contrast.append(torch.tensor(0.0, device=x.device))
            continue

        contrast.append(torch.log(peaks / valleys))
        
    return torch.tensor(contrast)
