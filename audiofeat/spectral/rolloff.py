import torch
from ..temporal.rms import frame_signal

__all__ = ["spectral_rolloff"]


def spectral_rolloff(
    audio_data: torch.Tensor,
    frame_length: int = 2048,
    hop_length: int = 512,
    rolloff_percent: float = 0.85,
    sample_rate: int = 22050,
    power: float = 1.0,
):
    """
    Computes the spectral rolloff of an audio signal.

    Args:
        audio_data (torch.Tensor): The audio signal.
        frame_length (int): The length of each frame in samples.
        hop_length (int): The number of samples to slide the window.
        rolloff_percent (float): The percentage of the total energy to capture (e.g., 0.85, 0.90, 0.95).
        sample_rate (int): Sampling rate of the audio (Hz).
        power (float): Exponent applied to the magnitude spectrum before
            accumulation. ``1.0`` (default) cumulates the magnitude spectrum
            and matches ``librosa.feature.spectral_rolloff``; ``2.0`` cumulates
            the power spectrum.

    Returns:
        torch.Tensor: The spectral rolloff for each frame.
    """
    if sample_rate <= 0:
        raise ValueError("sample_rate must be > 0.")
    if not 0.0 < rolloff_percent <= 1.0:
        raise ValueError("rolloff_percent must satisfy 0 < rolloff_percent <= 1.")

    frames = frame_signal(audio_data, frame_length, hop_length)
    magnitude_spectrum = torch.abs(torch.fft.rfft(frames))
    # librosa.feature.spectral_rolloff cumulates the MAGNITUDE spectrum
    # (power=1), not the power spectrum; ``power`` is exposed for callers that
    # explicitly want power-based accumulation.
    accum_spectrum = magnitude_spectrum if power == 1.0 else magnitude_spectrum ** power
    total_energy = torch.sum(accum_spectrum, dim=1)
    cumulative_energy = torch.cumsum(accum_spectrum, dim=1)

    # Find the frequency bin where the cumulative energy exceeds the rolloff_percent
    # We need to handle cases where total_energy is zero to avoid NaN in threshold
    threshold = rolloff_percent * total_energy.unsqueeze(1)
    
    # Use searchsorted to find the index where cumulative_energy crosses the threshold
    # Add a small epsilon to cumulative_energy to handle exact matches at the boundary
    rolloff_index = torch.searchsorted(cumulative_energy + 1e-8, threshold)
    
    # Clamp the index to be within valid bounds (0 to num_bins - 1)
    num_bins = magnitude_spectrum.shape[1]
    rolloff_index = torch.clamp(rolloff_index, 0, num_bins - 1)

    # Convert bin index to frequency
    frequencies = torch.fft.rfftfreq(frame_length, d=1.0 / sample_rate)
    
    return frequencies.to(rolloff_index.device)[rolloff_index].squeeze(1)
