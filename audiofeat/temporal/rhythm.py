import torch
from ..temporal.rms import frame_signal

def breath_group_duration(env: torch.Tensor, fs: int):
    """Estimate breath group durations from envelope."""
    threshold = env.mean() * 0.25
    below = (env < threshold).float()
    indices = torch.nonzero(below).squeeze()
    if indices.numel() == 0:
        return torch.tensor([])
    diffs = indices[1:] - indices[:-1]
    starts = indices[:-1][diffs > int(0.25 * fs)]
    if starts.numel() < 2:
        return torch.tensor([])
    durations = (starts[1:] - starts[:-1]).float() / fs
    return durations

def speech_rate(x: torch.Tensor, fs: int, threshold_ratio: float = 0.3, min_gap: float = 0.1):
    """Estimate speech rate in syllables per second."""
    env = torch.abs(x)
    win_len = max(1, int(0.02 * fs))
    kernel = torch.ones(win_len, device=x.device) / win_len
    env = torch.nn.functional.conv1d(env.view(1,1,-1), kernel.view(1,1,-1), padding=win_len//2).squeeze()
    threshold = env.mean() * threshold_ratio
    peaks = (env[1:-1] > env[:-2]) & (env[1:-1] > env[2:]) & (env[1:-1] > threshold)
    indices = torch.nonzero(peaks).squeeze() + 1
    if indices.numel() == 0:
        return 0.0
    keep = torch.cat([torch.tensor([True], device=x.device), (indices[1:] - indices[:-1]) > int(min_gap * fs)])
    syllables = indices[keep]
    return float(syllables.numel()) / (x.numel() / fs)

def temporal_centroid(audio_data: torch.Tensor, frame_length: int, hop_length: int):
    """
    Computes the temporal centroid of an audio signal.

    Args:
        audio_data (torch.Tensor): The audio signal.
        frame_length (int): The length of each frame in samples.
        hop_length (int): The number of samples to slide the window.

    Returns:
        torch.Tensor: The temporal centroid for each frame.
    """
    frames = frame_signal(audio_data, frame_length, hop_length)
    sample_energy = frames**2

    time_indices = torch.arange(0, frame_length, device=audio_data.device, dtype=torch.float32)

    numerator = torch.sum(sample_energy * time_indices, dim=1)
    denominator = torch.sum(sample_energy, dim=1)

    temporal_centroids = torch.where(denominator != 0, numerator / denominator, torch.zeros_like(numerator))

    return temporal_centroids