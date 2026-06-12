import torch
from .rms import frame_signal

__all__ = ["zero_crossing_count", "zero_crossing_rate"]


def zero_crossing_count(audio_data: torch.Tensor, frame_length=2048, hop_length=512):
    """
    Computes the number of zero-crossings in each frame of an audio signal.

    A zero crossing occurs between two consecutive samples whose *sign bit*
    differs.  Following ``librosa.feature.zero_crossing_rate`` (which uses
    ``numpy.signbit``), samples equal to ``0.0`` are treated as
    non-negative (sign bit clear) rather than getting a dedicated zero
    sign as ``torch.sign`` would.  This makes the count agree with librosa
    on signals that touch exactly zero.

    Args:
        audio_data (torch.Tensor): The audio signal.
        frame_length (int): The length of each frame in samples.
        hop_length (int): The number of samples to slide the window.

    Returns:
        torch.Tensor: The zero-crossing count for each frame.
    """
    frames = frame_signal(audio_data, frame_length, hop_length)
    # signbit-equivalent: True where value is >= 0 (matches numpy.signbit
    # convention used by librosa, where +0.0 is non-negative).
    non_neg = frames >= 0
    crossings = non_neg[:, 1:] ^ non_neg[:, :-1]
    return crossings.sum(dim=1).to(frames.dtype)


def zero_crossing_rate(audio_data: torch.Tensor, frame_length=2048, hop_length=512):
    """
    Computes the normalized zero-crossing rate of an audio signal.

    The count of sign-bit changes per frame is divided by ``frame_length``,
    matching ``librosa.feature.zero_crossing_rate``.

    Args:
        audio_data (torch.Tensor): The audio signal.
        frame_length (int): The length of each frame in samples.
        hop_length (int): The number of samples to slide the window.

    Returns:
        torch.Tensor: The normalized zero-crossing rate for each frame.
    """
    return zero_crossing_count(audio_data, frame_length, hop_length) / frame_length