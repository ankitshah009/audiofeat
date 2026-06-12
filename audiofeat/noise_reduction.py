"""Simple spectral-subtraction noise reduction."""
from __future__ import annotations

import torch
import torchaudio


@torch.inference_mode()
def noise_reduction(
    waveform: torch.Tensor,
    sample_rate: int,
    threshold: float = 0.01,
) -> torch.Tensor:
    """Suppress stationary noise via magnitude spectral subtraction.

    Parameters
    ----------
    waveform : torch.Tensor
        Audio samples ``(samples,)`` or ``(channels, samples)``.
    sample_rate : int
        Sampling rate (unused by the heuristic; kept for API parity).
    threshold : float, default 0.01
        Fraction of the peak magnitude used as the subtracted noise floor.

    Returns
    -------
    torch.Tensor
        The denoised waveform, same shape family as the input.
    """
    del sample_rate  # not needed for the magnitude-threshold heuristic
    n_fft, hop_length = 1024, 512

    # Complex spectrogram (power=None) so magnitude AND phase are meaningful.
    # A power spectrogram returns real values, making torch.angle all-zero and
    # the reconstruction garbage.
    spec = torchaudio.transforms.Spectrogram(
        n_fft=n_fft, hop_length=hop_length, power=None
    )(waveform)
    magnitude = spec.abs()
    phase = torch.angle(spec)

    # Estimate noise floor as a fraction of the peak magnitude, then subtract.
    noise_floor = threshold * torch.max(magnitude)
    reduced_magnitude = torch.clamp(magnitude - noise_floor, min=0.0)

    # Reconstruct via the inverse complex STFT.
    reduced_spec = torch.polar(reduced_magnitude, phase)
    reduced_waveform = torchaudio.transforms.InverseSpectrogram(
        n_fft=n_fft, hop_length=hop_length
    )(reduced_spec)
    return reduced_waveform
