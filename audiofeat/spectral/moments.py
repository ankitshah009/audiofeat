import torch
from ..temporal.rms import hann_window

__all__ = ["spectral_skewness", "spectral_spread"]


def spectral_skewness(x: torch.Tensor, n_fft: int):
    """Spectral skewness and (excess) kurtosis of a frame.

    Returns finite values for degenerate inputs: silence (total power 0) and a
    pure tone (zero spectral variance) both yield ``(0, 0)`` rather than NaN.
    """
    if x.numel() < 2:
        zero = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        return zero, zero.clone()

    X = torch.fft.rfft(x * hann_window(x.numel()).to(x.device), n=n_fft)
    P = X.abs() ** 2
    freqs = torch.linspace(0, n_fft // 2, P.numel(), device=x.device)

    total = torch.sum(P)
    if total <= 0:
        zero = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        return zero, zero.clone()

    mean = torch.sum(freqs * P) / total
    var = torch.clamp(torch.sum((freqs - mean) ** 2 * P) / total, min=0.0)

    if var <= 0:
        # Pure tone / single active bin: higher moments are undefined; report 0.
        zero = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        return zero, zero.clone()

    std = var.sqrt()
    skew = torch.sum((freqs - mean) ** 3 * P) / (total * std ** 3)
    kurt = torch.sum((freqs - mean) ** 4 * P) / (total * var ** 2) - 3
    return skew, kurt


def spectral_spread(x: torch.Tensor, n_fft: int, sample_rate: int, power: float = 2.0):
    """
    Computes the spectral spread (bandwidth) of an audio signal.

    Args:
        x (torch.Tensor): The audio signal.
        n_fft (int): The number of FFT points.
        sample_rate (int): The sample rate of the audio.
        power (float): Exponent applied to the magnitude spectrum used to weight
            the spread. ``2.0`` (default) weights by the power spectrum (the
            original behaviour); ``1.0`` weights by the magnitude spectrum,
            matching librosa's magnitude-based ``spectral_bandwidth``.

    Returns:
        torch.Tensor: The spectral spread.
    """
    if x.numel() < 2:
        return torch.tensor(0.0, device=x.device, dtype=x.dtype)

    X = torch.fft.rfft(x * hann_window(x.numel()).to(x.device), n=n_fft)
    P = X.abs() ** power
    freqs = torch.fft.rfftfreq(n_fft, d=1.0 / sample_rate).to(x.device)  # Use actual frequencies

    denominator = torch.sum(P)
    if denominator <= 0:
        return torch.tensor(0.0, device=x.device, dtype=x.dtype)

    centroid = torch.sum(freqs * P) / denominator

    numerator_spread = torch.sum((freqs - centroid) ** 2 * P)
    spread = torch.sqrt(torch.clamp(numerator_spread / denominator, min=0.0))
    return spread
