
import torch
from ..temporal.rms import frame_signal

def _geometric_mean(x: torch.Tensor, dim: int = -1, keepdim: bool = False):
    """
    Computes the geometric mean of a tensor along a given dimension.
    """
    # Avoid log(0) by adding a small epsilon
    return torch.exp(torch.mean(torch.log(x + 1e-8), dim=dim, keepdim=keepdim))

def spectral_flatness(audio_data: torch.Tensor, frame_length=2048, hop_length=512):
    """
    Computes the spectral flatness of an audio signal.

    Args:
        audio_data (torch.Tensor): The audio signal.
        frame_length (int): The length of each frame in samples.
        hop_length (int): The number of samples to slide the window.

    Returns:
        torch.Tensor: The spectral flatness for each frame.
    """
    frames = frame_signal(audio_data, frame_length, hop_length)
    magnitude_spectrum = torch.abs(torch.fft.rfft(frames))
    
    geometric_mean = _geometric_mean(magnitude_spectrum, dim=1)
    arithmetic_mean = torch.mean(magnitude_spectrum, dim=1)

    # Avoid division by zero
    flatness = torch.where(arithmetic_mean != 0, geometric_mean / arithmetic_mean, torch.zeros_like(geometric_mean))
    return flatness
