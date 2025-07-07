
import torch

def energy(signal: torch.Tensor, sample_rate: int, window_size: float = 0.05, hop_size: float = 0.025):
    """
    Calculates the short-term energy of an audio signal.

    Args:
        signal (torch.Tensor): The input audio signal.
        sample_rate (int): The sample rate of the audio signal.
        window_size (float, optional): The size of the analysis window in seconds. Defaults to 0.05.
        hop_size (float, optional): The hop size between consecutive windows in seconds. Defaults to 0.025.

    Returns:
        torch.Tensor: A tensor containing the energy for each frame.
    """
    win_length = int(window_size * sample_rate)
    hop_length = int(hop_size * sample_rate)
    window = torch.hann_window(win_length)

    # Unfold the signal into frames
    frames = signal.unfold(0, win_length, hop_length)

    # Apply the window and calculate the energy
    frames = frames * window
    energies = torch.sum(frames**2, dim=1)

    return energies.unsqueeze(0)
