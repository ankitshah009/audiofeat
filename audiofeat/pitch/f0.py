
import torch

from ..temporal.rms import frame_signal, hann_window


def fundamental_frequency_autocorr(
    x: torch.Tensor,
    fs: int,
    frame_length: int,
    hop_length: int,
    fmin: int = 50,
    fmax: int = 600,
):
    """Estimate F0 via autocorrelation per frame."""
    if fmin <= 0 or fmax <= fmin:
        raise ValueError("Expected 0 < fmin < fmax.")

    x = x.flatten().float()
    frames = frame_signal(x, frame_length, hop_length)
    w = hann_window(frame_length).to(x.device)
    win = frames * w
    spec = torch.fft.rfft(win, n=2 * frame_length)
    autocorr = torch.fft.irfft(spec * torch.conj(spec), n=2 * frame_length)
    autocorr = autocorr[:, :frame_length]
    min_lag = int(fs / fmax)
    max_lag = int(fs / fmin)
    if max_lag <= min_lag:
        raise ValueError("Invalid lag range. Check fmin/fmax and sample rate.")

    ac_segment = autocorr[:, min_lag:max_lag]
    lag = ac_segment.argmax(dim=1) + min_lag
    return fs / lag.float()

def fundamental_frequency_yin(
    x: torch.Tensor,
    fs: int,
    frame_length: int,
    hop_length: int,
    fmin: int = 50,
    fmax: int = 600,
    threshold: float = 0.1,
):
    """Estimate F0 per frame using the YIN algorithm."""
    if fmin <= 0 or fmax <= fmin:
        raise ValueError("Expected 0 < fmin < fmax.")

    x = x.flatten().float()
    frames = frame_signal(x, frame_length, hop_length)
    w = hann_window(frame_length).to(x.device)
    win = frames * w
    n = frame_length

    spec = torch.fft.rfft(win, n=2 * n)
    power = spec.abs() ** 2
    autocorr = torch.fft.irfft(power, n=2 * n)
    autocorr = autocorr[:, :n]
    sq = (win ** 2).sum(dim=1, keepdim=True)
    diff = sq + sq - 2 * autocorr
    diff[:, 0] = 0

    cumsum = torch.cumsum(diff[:, 1:], dim=1)
    denom = torch.arange(1, n, device=x.device).float()
    cmnd = torch.zeros_like(diff)
    cmnd[:, 0] = 1
    cmnd[:, 1:] = diff[:, 1:] * denom / (cumsum + 1e-8)

    min_lag = int(fs / fmax)
    max_lag = int(fs / fmin)
    if max_lag <= min_lag:
        raise ValueError("Invalid lag range. Check fmin/fmax and sample rate.")

    segment = cmnd[:, min_lag:max_lag]
    minima, min_idx = segment.min(dim=1)
    below = segment < threshold
    first_idx = torch.zeros_like(min_idx)
    if below.any():
        first_idx[below.any(dim=1)] = below[below.any(dim=1)].float().argmax(dim=1)
    lag = torch.where(below.any(dim=1), first_idx, min_idx) + min_lag
    lag = lag.clamp(min=1).float()
    return fs / lag
