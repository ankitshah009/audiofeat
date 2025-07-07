import torch

__all__ = ["decay_time"]


def decay_time(
    x: torch.Tensor,
    sample_rate: int,
    threshold_db: float = -20.0,
) -> torch.Tensor:
    """Compute *decay time* of an audio signal.

    The decay time is defined here as the interval between the peak
    amplitude and the first time the amplitude envelope drops below a
    given threshold (in dB relative to that peak).  Typical values are
    -20 dB (fast decay) or -60 dB (reverberation time T60).

    Args:
        x (torch.Tensor): Mono signal.
        sample_rate (int): Sampling rate in Hz.
        threshold_db (float): Negative dB value relative to the peak.

    Returns:
        torch.Tensor: Scalar time in seconds (0 if envelope never drops
            below threshold).
    """
    if x.dim() != 1:
        raise ValueError("`decay_time` expects a mono 1-D signal.")

    # Compute amplitude envelope using absolute value and moving maximum
    env = x.abs()
    peak_idx = torch.argmax(env)
    peak_amp = env[peak_idx]
    if peak_amp == 0:
        return torch.tensor(0.0, device=x.device)

    # Threshold linear value
    thr = peak_amp * (10.0 ** (threshold_db / 20.0))

    # Search forward from peak for first sample below threshold
    after_peak = env[peak_idx:]
    below = torch.where(after_peak < thr)[0]
    if below.numel() == 0:
        return torch.tensor(0.0, device=x.device)

    decay_samples = below[0].item()
    return torch.tensor(decay_samples / sample_rate, dtype=torch.float32, device=x.device) 