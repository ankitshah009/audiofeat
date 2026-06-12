import torch

__all__ = ["decay_time"]


def _smooth_envelope(
    waveform: torch.Tensor,
    sample_rate: int,
    window_ms: float = 5.0,
) -> torch.Tensor:
    """Compute a smoothed amplitude envelope via moving-average of ``|x|``.

    Mirrors :func:`audiofeat.temporal.attack._smooth_envelope` so attack and
    decay descriptors operate on the same envelope definition.
    """
    x = waveform.float().abs()
    win_len = max(1, int(sample_rate * window_ms / 1000.0))
    if win_len % 2 == 0:
        win_len += 1
    if win_len <= 1 or x.numel() < 2:
        return x
    kernel = torch.ones(1, 1, win_len, device=x.device, dtype=x.dtype) / win_len
    pad = min(win_len // 2, x.numel() - 1)
    padded = torch.nn.functional.pad(x.view(1, 1, -1), (pad, pad), mode="reflect")
    env = torch.nn.functional.conv1d(padded, kernel).squeeze()
    # conv1d with reflect padding preserves length when pad == win_len // 2.
    if env.numel() != x.numel():
        # Fallback for tiny signals where pad was clamped: trim/pad to match.
        env = env.flatten()[: x.numel()]
        if env.numel() < x.numel():
            env = torch.cat([env, x[env.numel():]])
    return env


def decay_time(
    x: torch.Tensor,
    sample_rate: int,
    threshold_db: float = -20.0,
    window_ms: float = 5.0,
) -> torch.Tensor:
    """Compute *decay time* of an audio signal.

    The decay time is the interval between the peak of the **smoothed**
    amplitude envelope and the first time that envelope drops below a
    threshold (in dB relative to the peak). Typical values are -20 dB
    (fast decay) or -60 dB (reverberation time T60).

    A smoothed envelope is essential: operating on the raw ``|x|`` of a
    tonal signal is meaningless because every period crosses zero, so the
    first inter-sample trough immediately falls below the threshold and
    yields a spurious near-zero decay time. Smoothing with a short
    moving-average window collapses the per-period ripple, so the search
    tracks the true amplitude decay.

    Args:
        x (torch.Tensor): Mono 1-D signal.
        sample_rate (int): Sampling rate in Hz.
        threshold_db (float): Negative dB value relative to the peak.
        window_ms (float): Envelope smoothing window in milliseconds.

    Returns:
        torch.Tensor: Scalar time in seconds (0 if the envelope never drops
            below the threshold after its peak).
    """
    if x.dim() != 1:
        raise ValueError("`decay_time` expects a mono 1-D signal.")
    if x.numel() == 0:
        return torch.tensor(0.0, device=x.device)

    # Smoothed amplitude envelope (suppresses per-period zero crossings).
    env = _smooth_envelope(x, sample_rate, window_ms=window_ms)

    peak_idx = torch.argmax(env)
    peak_amp = env[peak_idx]
    if peak_amp <= 0:
        return torch.tensor(0.0, device=x.device)

    # Linear threshold value.
    thr = peak_amp * (10.0 ** (threshold_db / 20.0))

    # Search forward from the peak for the first sample below threshold.
    after_peak = env[peak_idx:]
    below = torch.where(after_peak < thr)[0]
    if below.numel() == 0:
        return torch.tensor(0.0, device=x.device)

    decay_samples = int(below[0].item())
    return torch.tensor(decay_samples / sample_rate, dtype=torch.float32, device=x.device)
