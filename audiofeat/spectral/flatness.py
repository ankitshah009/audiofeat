
import torch
from ..temporal.rms import frame_signal

__all__ = ["spectral_flatness"]


def spectral_flatness(
    audio_data: torch.Tensor,
    frame_length: int = 2048,
    hop_length: int = 512,
    power: float = 2.0,
    amin: float = 1e-10,
):
    """
    Computes the spectral flatness of an audio signal.

    Matches ``librosa.feature.spectral_flatness``: the spectrum is raised to
    ``power`` (2.0 by default, i.e. the power spectrum), floored at ``amin``,
    and the geometric mean is divided by the arithmetic mean.

    Args:
        audio_data (torch.Tensor): The audio signal.
        frame_length (int): The length of each frame in samples.
        hop_length (int): The number of samples to slide the window.
        power (float): Exponent applied to the magnitude spectrum. ``2.0``
            (default) operates on the power spectrum like librosa; ``1.0``
            operates on the magnitude spectrum.
        amin (float): Floor applied to the spectrum before taking logs,
            matching librosa's ``amin`` (default ``1e-10``).

    Returns:
        torch.Tensor: The spectral flatness for each frame (values in [0, 1]).
    """
    frames = frame_signal(audio_data, frame_length, hop_length)
    magnitude_spectrum = torch.abs(torch.fft.rfft(frames))

    # librosa: S_thresh = max(amin, S ** power); flatness = gmean / amean
    spec = torch.clamp(magnitude_spectrum ** power, min=amin)
    geometric_mean = torch.exp(torch.mean(torch.log(spec), dim=1))
    arithmetic_mean = torch.mean(spec, dim=1)

    flatness = geometric_mean / arithmetic_mean
    return flatness
