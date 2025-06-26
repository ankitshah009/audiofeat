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
    
    # Define frequency band edges (logarithmic spacing for better perceptual relevance)
    # This is a common approach in spectral contrast calculation
    # Using a simplified linear spacing for now, but noting the ideal would be log.
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

        # Find peaks and valleys within the band
        peaks = torch.max(band_spectrum)
        valleys = torch.min(band_spectrum)
        
        # Calculate contrast: (peak - valley) / (peak + valley)
        # Add a small epsilon to the denominator to avoid division by zero
        denominator = peaks + valleys + 1e-8
        contrast.append((peaks - valleys) / denominator)
        
    return torch.tensor(contrast)