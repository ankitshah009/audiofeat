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