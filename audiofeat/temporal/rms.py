import torch

__all__ = ["rms", "short_time_energy"]


def frame_signal(x: torch.Tensor, frame_length: int, hop_length: int):
    """Frame a 1D signal into overlapping frames."""
    if frame_length <= 0:
        raise ValueError("frame_length must be > 0.")
    if hop_length <= 0:
        raise ValueError("hop_length must be > 0.")
    if x.numel() == 0:
        raise ValueError("Input signal must be non-empty.")

    x = x.flatten()
    if x.numel() < frame_length:
        x = torch.cat([x, x.new_zeros(frame_length - x.numel())], dim=0)

    num_frames = 1 + (x.numel() - frame_length) // hop_length
    strides = (x.stride(0) * hop_length, x.stride(0))
    shape = (num_frames, frame_length)
    return x.as_strided(shape, strides)

def hann_window(L: int):
    """Return an L-point Hann window."""
    n = torch.arange(L, dtype=torch.float32)
    return 0.5 * (1 - torch.cos(2 * torch.pi * n / (L - 1)))

def rms(x: torch.Tensor, frame_length: int, hop_length: int, window: str = "hann"):
    """Root-mean-square amplitude per frame.

    Args:
        x (torch.Tensor): The audio signal.
        frame_length (int): The length of each frame in samples.
        hop_length (int): The number of samples to slide the window.
        window (str): Analysis window. ``"hann"`` (default) applies a Hann
            window with window-energy normalization (the historical
            behavior of this function). ``"rect"`` / ``"boxcar"`` / ``"none"``
            use an unwindowed (rectangular) frame, i.e. plain
            ``sqrt(mean(frame**2))``, which matches
            ``librosa.feature.rms`` (with ``center=False``) for parity.

    Returns:
        torch.Tensor: RMS amplitude for each frame.

    Notes:
        librosa's ``rms`` uses a rectangular window and is the canonical
        reference; pass ``window="rect"`` to reproduce it. The default
        ``"hann"`` is retained for backward compatibility with existing
        callers and serialized pipelines.
    """
    frames = frame_signal(x, frame_length, hop_length)
    win = window.lower()
    if win in ("rect", "rectangular", "boxcar", "none"):
        # Rectangular window: sqrt(mean(frame**2)) — matches librosa.
        return torch.sqrt(torch.mean(frames ** 2, dim=1))
    if win != "hann":
        raise ValueError(
            f"Unsupported window {window!r}; expected 'hann' or 'rect'/'boxcar'."
        )
    w = hann_window(frame_length).to(x.device)
    win_frames = frames * w
    win_energy = torch.sum(w ** 2)
    return torch.sqrt(torch.sum(win_frames ** 2, dim=1) / win_energy)

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
