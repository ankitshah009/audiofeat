
import torch

from ..temporal.rms import frame_signal

__all__ = ["entropy_of_energy"]


def entropy_of_energy(audio_data: torch.Tensor, frame_length: int, hop_length: int, n_sub_frames: int = 10):
    """
    Computes the (Shannon) entropy of energy of an audio signal.

    Each frame is split into ``n_sub_frames`` contiguous sub-frames; the
    normalized sub-frame energy distribution ``p`` yields the entropy
    ``-sum(p * log2(p))`` (in bits). A perfectly flat distribution gives
    ``log2(n_sub_frames)``; an impulse concentrated in one sub-frame gives
    ``~0``.

    Args:
        audio_data (torch.Tensor): The audio signal.
        frame_length (int): The length of each frame in samples.
        hop_length (int): The number of samples to slide the window.
        n_sub_frames (int): The number of sub-frames per frame.

    Returns:
        torch.Tensor: The entropy of energy for each frame, on the same
        device as ``audio_data``.

    Notes:
        Zero-probability bins are masked out via ``p * log2(p.clamp_min(eps))``
        so they contribute exactly 0 (``0 * log2(0) -> 0``), rather than the
        ``log2(p + 1e-8)`` anti-pattern which biases the result on sparse
        frames. Frames with zero total energy return an entropy of 0.
    """
    frames = frame_signal(audio_data, frame_length, hop_length)  # (n_frames, frame_length)
    device = frames.device
    sub_frame_length = max(1, frame_length // n_sub_frames)
    usable = sub_frame_length * n_sub_frames

    # Reshape each frame into n_sub_frames contiguous sub-frames (vectorized,
    # no Python loop). Trailing samples that don't fill a sub-frame are dropped,
    # matching the previous frame_signal(..., sub_frame_length, sub_frame_length)
    # behavior (which discarded the partial tail frame).
    sub = frames[:, :usable].reshape(frames.shape[0], n_sub_frames, sub_frame_length)
    sub_energy = torch.sum(sub ** 2, dim=2)  # (n_frames, n_sub_frames)

    total_energy = sub_energy.sum(dim=1, keepdim=True)  # (n_frames, 1)
    prob = sub_energy / total_energy.clamp_min(1e-12)

    # Mask zero-probability bins: p * log2(clamp_min(p)) -> 0 when p == 0.
    log_p = torch.log2(prob.clamp_min(1e-12))
    entropy = -(prob * log_p).sum(dim=1)

    # Frames with no energy -> entropy 0.
    entropy = torch.where(total_energy.squeeze(1) > 0, entropy, torch.zeros_like(entropy))
    return entropy.to(device)
