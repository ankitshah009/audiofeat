
import torch
from ..temporal.rms import frame_signal

def amplitude_modulation_depth(env: torch.Tensor, window: int):
    """Amplitude modulation depth over a sliding window."""
    if env.numel() < window:
        return torch.tensor(0.0, device=env.device)
    frames = frame_signal(env, window, window)
    max_e = frames.max(dim=1).values
    min_e = frames.min(dim=1).values
    return ((max_e - min_e) / (max_e + min_e + 1e-8)).mean()
