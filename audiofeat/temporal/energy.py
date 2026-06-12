
import torch

from .rms import frame_signal, hann_window

__all__ = ["energy"]


def energy(signal: torch.Tensor, sample_rate: int, window_size: float = 0.05, hop_size: float = 0.025):
    """
    Calculates the short-term (windowed) energy of an audio signal.

    Args:
        signal (torch.Tensor): The input audio signal.
        sample_rate (int): The sample rate of the audio signal.
        window_size (float, optional): The size of the analysis window in seconds. Defaults to 0.05.
        hop_size (float, optional): The hop size between consecutive windows in seconds. Defaults to 0.025.

    Returns:
        torch.Tensor: A tensor of shape ``(1, n_frames)`` containing the
        energy for each frame.

    Notes:
        Framing is done via :func:`audiofeat.temporal.rms.frame_signal`,
        which zero-pads inputs shorter than one window. This avoids the
        ``RuntimeError`` that ``Tensor.unfold`` raises when the signal is
        shorter than the window length. The Hann window is materialized on
        the input signal's device so the function is device-safe (CPU/GPU).
    """
    win_length = max(1, int(window_size * sample_rate))
    hop_length = max(1, int(hop_size * sample_rate))

    # frame_signal zero-pads short signals, so this never crashes on
    # inputs shorter than win_length.
    frames = frame_signal(signal, win_length, hop_length)

    # Window on the same device as the input (device-safe).
    window = hann_window(win_length).to(device=frames.device, dtype=frames.dtype)

    frames = frames * window
    energies = torch.sum(frames ** 2, dim=1)

    return energies.unsqueeze(0)
