import torch
from .rms import frame_signal

def zero_crossing_count(audio_data: torch.Tensor, frame_length=2048, hop_length=512):
    """
    Computes the number of zero-crossings in each frame of an audio signal.

    Args:
        audio_data (torch.Tensor): The audio signal.
        frame_length (int): The length of each frame in samples.
        hop_length (int): The number of samples to slide the window.

    Returns:
        torch.Tensor: The zero-crossing count for each frame.
    """
    frames = frame_signal(audio_data, frame_length, hop_length)
    return torch.sum(torch.abs(torch.diff(torch.sign(frames))), dim=1) / 2

def zero_crossing_rate(audio_data: torch.Tensor, frame_length=2048, hop_length=512):
    """
    Computes the normalized zero-crossing rate of an audio signal.

    Args:
        audio_data (torch.Tensor): The audio signal.
        frame_length (int): The length of each frame in samples.
        hop_length (int): The number of samples to slide the window.

    Returns:
        torch.Tensor: The normalized zero-crossing rate for each frame.
    """
    return zero_crossing_count(audio_data, frame_length, hop_length) / frame_length