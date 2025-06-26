import torch
from ..temporal.rms import frame_signal

def spectral_rolloff(audio_data: torch.Tensor, frame_length=2048, hop_length=512, rolloff_percent=0.85):
    """
    Computes the spectral rolloff of an audio signal.

    Args:
        audio_data (torch.Tensor): The audio signal.
        frame_length (int): The length of each frame in samples.
        hop_length (int): The number of samples to slide the window.
        rolloff_percent (float): The percentage of the total energy to capture.

    Returns:
        torch.Tensor: The spectral rolloff for each frame.
    """
    frames = frame_signal(audio_data, frame_length, hop_length)
    magnitude_spectrum = torch.abs(torch.fft.rfft(frames))
    total_energy = torch.sum(magnitude_spectrum, dim=1)
    cumulative_energy = torch.cumsum(magnitude_spectrum, dim=1)
    rolloff_index = torch.searchsorted(cumulative_energy, rolloff_percent * total_energy.unsqueeze(1))
    return torch.fft.rfftfreq(frame_length)[rolloff_index].squeeze(1)