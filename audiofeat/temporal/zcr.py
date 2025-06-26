import torch
from .rms import frame_signal

def zero_crossing_rate(audio_data: torch.Tensor, frame_length=2048, hop_length=512):
    """
    Computes the zero-crossing rate of an audio signal.

    Args:
        audio_data (torch.Tensor): The audio signal.
        frame_length (int): The length of each frame in samples.
        hop_length (int): The number of samples to slide the window.

    Returns:
        torch.Tensor: The zero-crossing rate for each frame.
    """
    frames = frame_signal(audio_data, frame_length, hop_length)
    return torch.sum(torch.abs(torch.diff(torch.sign(frames))), dim=1) / (2 * frame_length)