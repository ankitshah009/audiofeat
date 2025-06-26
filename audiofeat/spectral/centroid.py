import torch
from ..temporal.rms import frame_signal

def spectral_centroid(audio_data: torch.Tensor, frame_length=2048, hop_length=512):
    """
    Computes the spectral centroid of an audio signal.

    Args:
        audio_data (torch.Tensor): The audio signal.
        frame_length (int): The length of each frame in samples.
        hop_length (int): The number of samples to slide the window.

    Returns:
        torch.Tensor: The spectral centroid for each frame.
    """
    frames = frame_signal(audio_data, frame_length, hop_length)
    magnitude_spectrum = torch.abs(torch.fft.rfft(frames))
    frequency_bins = torch.fft.rfftfreq(frame_length, d=1.0/22050) # d is sample spacing, 1/fs
    
    # Ensure frequency_bins has the same number of dimensions as magnitude_spectrum for broadcasting
    # magnitude_spectrum is (num_frames, num_bins)
    # frequency_bins is (num_bins,)
    # We need to expand frequency_bins to (1, num_bins) for element-wise multiplication
    
    numerator = torch.sum(magnitude_spectrum * frequency_bins.unsqueeze(0), dim=1)
    denominator = torch.sum(magnitude_spectrum, dim=1)

    # Avoid division by zero
    centroid = torch.where(denominator != 0, numerator / denominator, torch.zeros_like(numerator))
    return centroid