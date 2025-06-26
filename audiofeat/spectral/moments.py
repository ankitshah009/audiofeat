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

def spectral_spread(x: torch.Tensor, n_fft: int, sample_rate: int):
    """
    Computes the spectral spread (bandwidth) of an audio signal.

    Args:
        x (torch.Tensor): The audio signal.
        n_fft (int): The number of FFT points.
        sample_rate (int): The sample rate of the audio.

    Returns:
        torch.Tensor: The spectral spread.
    """
    X = torch.fft.rfft(x * hann_window(x.numel()).to(x.device), n=n_fft)
    P = X.abs() ** 2
    freqs = torch.fft.rfftfreq(n_fft, d=1.0/sample_rate) # Use actual frequencies
    
    # Calculate spectral centroid first
    numerator_centroid = torch.sum(freqs * P)
    denominator_centroid = torch.sum(P)
    
    if denominator_centroid == 0:
        return torch.tensor(0.0, device=x.device)
        
    centroid = numerator_centroid / denominator_centroid

    # Calculate spectral spread (standard deviation around centroid)
    numerator_spread = torch.sum((freqs - centroid) ** 2 * P)
    denominator_spread = torch.sum(P)
    
    if denominator_spread == 0:
        return torch.tensor(0.0, device=x.device)

    spread = torch.sqrt(numerator_spread / denominator_spread)
    return spread