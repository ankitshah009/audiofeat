import torch
from ..temporal.rms import frame_signal

def spectral_flux(audio_data: torch.Tensor, frame_length=2048, hop_length=512):
    """
    Computes the spectral flux of an audio signal.

    Args:
        audio_data (torch.Tensor): The audio signal.
        frame_length (int): The length of each frame in samples.
        hop_length (int): The number of samples to slide the window.

    Returns:
        torch.Tensor: The spectral flux for each frame.
    """
    frames = frame_signal(audio_data, frame_length, hop_length)
    magnitude_spectrum = torch.abs(torch.fft.rfft(frames))
    return torch.sqrt(torch.sum(torch.diff(magnitude_spectrum, dim=0)**2, dim=1))