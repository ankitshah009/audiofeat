import torch
from ..temporal.rms import frame_signal
from scipy.stats import gmean

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
    return torch.tensor([gmean(frame.numpy()) for frame in magnitude_spectrum]) / torch.mean(magnitude_spectrum, dim=1)