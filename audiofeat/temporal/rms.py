import torch

def frame_signal(x: torch.Tensor, frame_length: int, hop_length: int):
    """Frame a 1D signal into overlapping frames."""
    num_frames = 1 + (x.numel() - frame_length) // hop_length
    strides = (x.stride(0) * hop_length, x.stride(0))
    shape = (num_frames, frame_length)
    return x.as_strided(shape, strides)

def hann_window(L: int):
    """Return an L-point Hann window."""
    n = torch.arange(L, dtype=torch.float32)
    return 0.5 * (1 - torch.cos(2 * torch.pi * n / (L - 1)))

def rms(x: torch.Tensor, frame_length: int, hop_length: int):
    """Root-mean-square amplitude per frame."""
    frames = frame_signal(x, frame_length, hop_length)
    w = hann_window(frame_length).to(x.device)
    win_frames = frames * w
    return torch.sqrt(torch.mean(win_frames ** 2, dim=1))

def short_time_energy(x: torch.Tensor, frame_length: int, hop_length: int):
    """
    Computes the short-time energy of an audio signal.

    Args:
        x (torch.Tensor): The audio signal.
        frame_length (int): The length of each frame in samples.
        hop_length (int): The number of samples to slide the window.

    Returns:
        torch.Tensor: The short-time energy for each frame.
    """
    frames = frame_signal(x, frame_length, hop_length)
    return torch.sum(frames ** 2, dim=1)