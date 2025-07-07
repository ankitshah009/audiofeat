
import torch
from ..temporal.energy import energy

def silence_removal(signal: torch.Tensor, sample_rate: int, window_size: float = 0.05, hop_size: float = 0.025, threshold: float = 0.1):
    """
    Removes silent segments from an audio signal based on energy.

    Args:
        signal (torch.Tensor): The input audio signal.
        sample_rate (int): The sample rate of the audio signal.
        window_size (float, optional): The size of the analysis window in seconds. Defaults to 0.05.
        hop_size (float, optional): The hop size between consecutive windows in seconds. Defaults to 0.025.
        threshold (float, optional): The energy threshold for detecting non-silent segments. Defaults to 0.1.

    Returns:
        torch.Tensor: The audio signal with silent segments removed.
    """
    # 1. Calculate short-term energy
    energies = energy(signal, sample_rate, window_size, hop_size).squeeze()

    # 2. Find segments above the threshold
    non_silent_frames = torch.where(energies > threshold)[0]

    if len(non_silent_frames) == 0:
        return torch.tensor([])

    # 3. Group consecutive non-silent frames into segments
    segment_starts = [non_silent_frames[0]]
    segment_ends = []

    for i in range(1, len(non_silent_frames)):
        if non_silent_frames[i] > non_silent_frames[i-1] + 1:
            segment_ends.append(non_silent_frames[i-1])
            segment_starts.append(non_silent_frames[i])
    segment_ends.append(non_silent_frames[-1])

    # 4. Convert frame indices to sample indices and concatenate segments
    result_signal = []
    for start, end in zip(segment_starts, segment_ends):
        start_sample = int(start * hop_size * sample_rate)
        end_sample = int(end * hop_size * sample_rate + window_size * sample_rate)
        result_signal.append(signal[start_sample:end_sample])

    return torch.cat(result_signal)
