
import torch

from ..temporal.rms import frame_signal, hann_window


def fundamental_frequency_autocorr(
    x: torch.Tensor,
    fs: int,
    frame_length: int,
    hop_length: int,
    fmin: float = 50.0,
    fmax: float = 600.0,
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
    fmin: float = 50.0,
    fmax: float = 600.0,
    threshold: float = 0.1,
):
    """Estimate F0 per frame using the YIN algorithm."""
    if fmin <= 0 or fmax <= fmin:
        raise ValueError("Expected 0 < fmin < fmax.")

    x = x.flatten().float()
    frames = frame_signal(x, frame_length, hop_length)
    n = frame_length

    # YIN difference function (de Cheveigne & Kawahara 2002).
    # The frame is NOT Hann-windowed: YIN operates on the raw frame.
    # d(tau) = sum_j (x_j - x_{j+tau})^2
    #        = r0(0) + r_tau(tau) - 2 * acf(tau)
    # where acf(tau) is the autocorrelation and r_tau(tau) is the energy of the
    # lagged window sum_{j} x_{j+tau}^2 (a running energy term, NOT a constant).
    spec = torch.fft.rfft(frames, n=2 * n, dim=1)
    power = spec.abs() ** 2
    autocorr = torch.fft.irfft(power, n=2 * n, dim=1)
    autocorr = autocorr[:, :n]

    # Cumulative energy from both ends to form the two running-energy terms.
    sq = frames ** 2
    total_energy = sq.sum(dim=1, keepdim=True)                      # r(0)
    cum_energy = torch.cumsum(sq, dim=1)                            # sum_{j<=tau} x_j^2
    # Energy of the first (n - tau) samples: sum_{j=0}^{n-tau-1} x_j^2
    lagged_energy = torch.zeros_like(autocorr)
    lagged_energy[:, 0] = total_energy.squeeze(1)
    lagged_energy[:, 1:] = total_energy - cum_energy[:, :-1]

    diff = total_energy + lagged_energy - 2 * autocorr
    diff = diff.clamp(min=0.0)
    diff[:, 0] = 0

    # Cumulative mean normalized difference (CMND).
    cumsum = torch.cumsum(diff[:, 1:], dim=1)
    denom = torch.arange(1, n, device=x.device).float()
    cmnd = torch.zeros_like(diff)
    cmnd[:, 0] = 1
    cmnd[:, 1:] = diff[:, 1:] * denom / (cumsum + 1e-8)

    min_lag = int(fs / fmax)
    max_lag = int(fs / fmin)
    if max_lag <= min_lag:
        raise ValueError("Invalid lag range. Check fmin/fmax and sample rate.")
    max_lag = min(max_lag, n - 1)

    segment = cmnd[:, min_lag:max_lag]
    minima, min_idx = segment.min(dim=1)

    # YIN absolute-threshold step (de Cheveigne & Kawahara 2002, section 4):
    # choose the FIRST lag that is below ``threshold`` AND is a local minimum.
    # A point merely on the descending slope of the valley can already be below
    # threshold; selecting it (instead of the valley bottom) biases F0 high. We
    # therefore restrict candidates to local minima and fall back to the global
    # minimum when no below-threshold dip exists.
    L = segment.shape[1]
    is_local_min = torch.ones_like(segment, dtype=torch.bool)
    if L >= 2:
        is_local_min[:, :-1] &= segment[:, :-1] <= segment[:, 1:]
        is_local_min[:, 1:] &= segment[:, 1:] < segment[:, :-1]
    candidate = (segment < threshold) & is_local_min
    has_candidate = candidate.any(dim=1)
    cand_idx = candidate.float().argmax(dim=1)
    first_idx = torch.where(has_candidate, cand_idx, min_idx)
    lag = (first_idx + min_lag).clamp(min=1, max=n - 1)

    # Parabolic interpolation around the chosen lag for sub-sample period
    # (de Cheveigne & Kawahara 2002, section 5). Refines tau using the three
    # CMND values straddling the minimum.
    lag_lo = (lag - 1).clamp(min=0, max=n - 1)
    lag_hi = (lag + 1).clamp(min=0, max=n - 1)
    rows = torch.arange(cmnd.shape[0], device=x.device)
    a = cmnd[rows, lag_lo]
    b = cmnd[rows, lag]
    cc = cmnd[rows, lag_hi]
    denom_interp = (a - 2 * b + cc)
    shift = torch.where(
        denom_interp.abs() > 1e-12,
        0.5 * (a - cc) / denom_interp,
        torch.zeros_like(denom_interp),
    )
    # Only apply when the central point is a genuine interior minimum.
    interior = (lag > 0) & (lag < n - 1)
    shift = torch.where(interior, shift.clamp(-1.0, 1.0), torch.zeros_like(shift))
    refined_lag = lag.float() + shift
    refined_lag = refined_lag.clamp(min=1.0)
    return fs / refined_lag
