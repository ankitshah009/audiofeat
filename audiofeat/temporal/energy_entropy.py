
import torch
from ..temporal.rms import frame_signal

def entropy_of_energy(audio_data: torch.Tensor, frame_length: int, hop_length: int, n_sub_frames: int = 10):
    """
    Computes the entropy of energy of an audio signal.

    Args:
        audio_data (torch.Tensor): The audio signal.
        frame_length (int): The length of each frame in samples.
        hop_length (int): The number of samples to slide the window.
        n_sub_frames (int): The number of sub-frames to divide each frame into.

    Returns:
        torch.Tensor: The entropy of energy for each frame.
    """
    frames = frame_signal(audio_data, frame_length, hop_length)
    sub_frame_length = frame_length // n_sub_frames
    
    entropy = []
    for frame in frames:
        sub_frames = frame_signal(frame, sub_frame_length, sub_frame_length)
        energy = torch.sum(sub_frames**2, dim=1)
        total_energy = torch.sum(energy)
        if total_energy == 0:
            entropy.append(0.0)
            continue
        prob = energy / total_energy
        entropy.append(-(prob * torch.log2(prob + 1e-8)).sum().item())
    return torch.tensor(entropy)
