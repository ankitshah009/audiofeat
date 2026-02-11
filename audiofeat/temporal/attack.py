"""Log Attack Time — MPEG-7 inspired temporal descriptor.

Computes the attack time of an audio signal using a smoothed amplitude
envelope.  The attack is defined as the interval between the points
where the envelope crosses a lower threshold (default 10 % of peak)
and an upper threshold (default 90 % of peak).

LAT = log10(t_upper - t_lower)

If no valid attack is found (e.g. silence), returns a very small
sentinel value (log10(1e-8) ≈ -8).
"""

from __future__ import annotations

import torch


def _smooth_envelope(
    waveform: torch.Tensor,
    sample_rate: int,
    window_ms: float = 5.0,
) -> torch.Tensor:
    """Compute a smoothed amplitude envelope via RMS windowing."""
    x = waveform.float().abs()
    win_len = max(1, int(sample_rate * window_ms / 1000.0))
    if win_len % 2 == 0:
        win_len += 1
    kernel = torch.ones(1, 1, win_len, device=x.device, dtype=x.dtype) / win_len
    padded = torch.nn.functional.pad(x.view(1, 1, -1), (win_len // 2, win_len // 2), mode="reflect")
    env = torch.nn.functional.conv1d(padded, kernel).squeeze()
    return env


def log_attack_time(
    audio_data: torch.Tensor,
    sample_rate: int,
    low_threshold: float = 0.10,
    high_threshold: float = 0.90,
    window_ms: float = 5.0,
) -> float:
    """Compute log-attack-time of an audio signal.

    Parameters
    ----------
    audio_data : torch.Tensor
        Mono audio waveform.
    sample_rate : int
        Sampling rate in Hz.
    low_threshold : float
        Lower envelope fraction (default 0.10 = 10 % of peak).
    high_threshold : float
        Upper envelope fraction (default 0.90 = 90 % of peak).
    window_ms : float
        Smoothing window in milliseconds for the envelope.

    Returns
    -------
    float
        log10 of the attack time in seconds.
    """
    if audio_data.ndim > 1:
        audio_data = audio_data.squeeze(0) if audio_data.shape[0] == 1 else audio_data[0]
    audio_data = audio_data.float()

    env = _smooth_envelope(audio_data, sample_rate, window_ms=window_ms)
    peak = env.max().item()
    if peak < 1e-12:
        return torch.log10(torch.tensor(1e-8)).item()

    low_level = low_threshold * peak
    high_level = high_threshold * peak

    # Find first crossing of low threshold
    above_low = (env >= low_level).nonzero(as_tuple=True)[0]
    if above_low.numel() == 0:
        return torch.log10(torch.tensor(1e-8)).item()
    start_idx = int(above_low[0].item())

    # Find first crossing of high threshold after start
    above_high = (env[start_idx:] >= high_level).nonzero(as_tuple=True)[0]
    if above_high.numel() == 0:
        return torch.log10(torch.tensor(1e-8)).item()
    end_idx = start_idx + int(above_high[0].item())

    if end_idx <= start_idx:
        return torch.log10(torch.tensor(1e-8)).item()

    attack_time = float(end_idx - start_idx) / sample_rate
    return torch.log10(torch.tensor(max(attack_time, 1e-8))).item()
