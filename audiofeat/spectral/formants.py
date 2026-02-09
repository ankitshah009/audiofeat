from __future__ import annotations

import math

import numpy as np
import torch
import torchaudio.functional as AF


def _burg_lpc(x: np.ndarray, order: int) -> np.ndarray:
    """
    Compute LPC coefficients using Burg recursion.

    This mirrors the family of Burg estimators used in speech formant pipelines.
    """
    n = len(x)
    if n <= order:
        return np.concatenate([[1.0], np.zeros(order, dtype=np.float64)])

    ef = x.astype(np.float64, copy=True)
    eb = x.astype(np.float64, copy=True)
    a = np.zeros(order + 1, dtype=np.float64)
    a[0] = 1.0

    for i in range(order):
        ef_slice = ef[i + 1 :]
        eb_slice = eb[i:-1]
        num = -2.0 * np.dot(ef_slice, eb_slice)
        denom = np.dot(ef_slice, ef_slice) + np.dot(eb_slice, eb_slice)
        if denom < 1e-12:
            break

        k = float(np.clip(num / denom, -0.9999, 0.9999))
        a_prev = a.copy()
        for j in range(1, i + 2):
            a[j] = a_prev[j] + k * a_prev[i + 1 - j]

        ef_new = ef[i + 1 :] + k * eb[i:-1]
        eb_new = eb[i:-1] + k * ef[i + 1 :]
        ef[i + 1 :] = ef_new
        eb[i:-1] = eb_new

    return a


def _roots_to_formants(
    a: np.ndarray,
    fs: int,
    *,
    min_freq: float = 50.0,
    max_freq: float | None = None,
    max_bandwidth: float = 700.0,
) -> tuple[np.ndarray, np.ndarray]:
    if max_freq is None:
        max_freq = fs / 2 - 50.0

    roots = np.roots(a)
    roots = roots[np.abs(roots) < 1.0]
    roots = roots[roots.imag >= 0]
    if len(roots) == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    angles = np.angle(roots)
    freqs = angles * fs / (2.0 * np.pi)
    with np.errstate(divide="ignore", invalid="ignore"):
        bandwidths = -fs * np.log(np.abs(roots)) / np.pi

    mask = (
        (freqs >= min_freq)
        & (freqs <= max_freq)
        & np.isfinite(freqs)
        & np.isfinite(bandwidths)
        & (bandwidths > 0.0)
        & (bandwidths <= max_bandwidth)
    )
    freqs = freqs[mask]
    bandwidths = bandwidths[mask]
    order = np.argsort(freqs)
    return freqs[order], bandwidths[order]


def _gaussian_window(length: int) -> np.ndarray:
    # Praat uses a Gaussian-like analysis window for Burg formants.
    x = np.linspace(-1.0, 1.0, length, dtype=np.float64)
    sigma = 0.4
    w = np.exp(-0.5 * (x / sigma) ** 2)
    return w / (np.max(w) + 1e-12)


def _frame_signal_numpy(x: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    if x.size < frame_length:
        x = np.pad(x, (0, frame_length - x.size), mode="constant")
    n_frames = 1 + (x.size - frame_length) // hop_length
    shape = (n_frames, frame_length)
    strides = (x.strides[0] * hop_length, x.strides[0])
    return np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides).copy()


def _estimate_pre_emphasis_coeff(pre_emphasis: float, fs: int) -> float:
    if 0.0 < pre_emphasis < 1.0:
        return float(pre_emphasis)
    if pre_emphasis <= 0:
        return 0.0
    # If user passes frequency in Hz by mistake, convert to coefficient.
    return float(math.exp(-2.0 * math.pi * float(pre_emphasis) / float(fs)))


def _apply_pre_emphasis(x: np.ndarray, coeff: float) -> np.ndarray:
    if coeff <= 0:
        return x
    out = np.empty_like(x)
    out[0] = x[0]
    out[1:] = x[1:] - coeff * x[:-1]
    return out


def _resample_for_formants(x: torch.Tensor, fs: int, max_formant: float) -> tuple[torch.Tensor, int]:
    target_fs = int(max(8000, round(2.0 * max_formant)))
    if fs == target_fs:
        return x, fs
    if fs <= target_fs:
        return x, fs
    return AF.resample(x, fs, target_fs), target_fs


def _build_formant_ranges(num_formants: int, max_formant: float) -> list[tuple[float, float]]:
    defaults = [
        (150.0, 1200.0),
        (500.0, 3500.0),
        (1200.0, 5000.0),
        (2000.0, 6500.0),
        (2500.0, 8000.0),
    ]
    ranges: list[tuple[float, float]] = []
    low = 100.0
    high_step = max_formant / max(num_formants + 1, 1)
    for i in range(num_formants):
        if i < len(defaults):
            lo, hi = defaults[i]
        else:
            lo = low + i * 0.5 * high_step
            hi = low + (i + 2) * high_step
        ranges.append((max(50.0, lo), min(max_formant, hi)))
    return ranges


def _track_formants(
    frame_candidates: list[tuple[np.ndarray, np.ndarray]],
    num_formants: int,
    max_formant: float,
) -> tuple[np.ndarray, np.ndarray]:
    n_frames = len(frame_candidates)
    freq_arr = np.full((n_frames, num_formants), np.nan, dtype=np.float64)
    bw_arr = np.full((n_frames, num_formants), np.nan, dtype=np.float64)
    ranges = _build_formant_ranges(num_formants, max_formant)

    prev = np.full(num_formants, np.nan, dtype=np.float64)
    for i, (freqs, bws) in enumerate(frame_candidates):
        if freqs.size == 0:
            continue
        used: set[int] = set()
        for j in range(num_formants):
            low, high = ranges[j]
            prev_same_frame = freq_arr[i, j - 1] if j > 0 else np.nan
            min_allowed = max(low, prev_same_frame + 20.0) if np.isfinite(prev_same_frame) else low
            candidate_idx = [
                k
                for k in range(freqs.size)
                if k not in used and min_allowed <= freqs[k] <= high
            ]
            if not candidate_idx:
                candidate_idx = [
                    k
                    for k in range(freqs.size)
                    if k not in used and freqs[k] >= min_allowed and freqs[k] <= max_formant
                ]
            if not candidate_idx:
                continue

            if np.isfinite(prev[j]):
                best = min(candidate_idx, key=lambda k: abs(freqs[k] - prev[j]))
            else:
                best = min(candidate_idx, key=lambda k: freqs[k])

            freq_arr[i, j] = float(freqs[best])
            bw_arr[i, j] = float(bws[best])
            used.add(best)

        prev = freq_arr[i].copy()

    return freq_arr, bw_arr


def _robust_track_median(track: np.ndarray) -> float:
    values = track[np.isfinite(track)]
    if values.size == 0:
        return float("nan")
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    if mad <= 0:
        return median
    robust_sigma = 1.4826 * mad
    inliers = values[np.abs(values - median) <= 3.5 * robust_sigma]
    if inliers.size == 0:
        return median
    return float(np.median(inliers))


def _praat_contours_from_tensor(
    x: np.ndarray,
    fs: int,
    *,
    num_formants: int,
    max_formant: float,
    frame_length_ms: float,
    hop_length_ms: float,
    pre_emphasis: float,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        import parselmouth  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "praat-parselmouth is required for `method='praat'`. "
            "Install with `pip install \"audiofeat[validation]\"`."
        ) from exc

    sound = parselmouth.Sound(x.astype(np.float64), sampling_frequency=float(fs))
    if pre_emphasis < 1.0:
        pre_emphasis_from_hz = -fs * math.log(max(pre_emphasis, 1e-8)) / (2.0 * math.pi)
    else:
        pre_emphasis_from_hz = float(pre_emphasis)
    formant = sound.to_formant_burg(
        time_step=float(hop_length_ms) / 1000.0,
        max_number_of_formants=float(num_formants),
        maximum_formant=float(max_formant),
        window_length=float(frame_length_ms) / 1000.0,
        pre_emphasis_from=float(pre_emphasis_from_hz),
    )

    duration = len(x) / float(fs)
    times = np.arange(0.0, duration, float(hop_length_ms) / 1000.0, dtype=np.float64)
    freq_arr = np.full((len(times), num_formants), np.nan, dtype=np.float64)
    bw_arr = np.full((len(times), num_formants), np.nan, dtype=np.float64)
    for i, t in enumerate(times):
        for j in range(num_formants):
            value = formant.get_value_at_time(j + 1, float(t))
            if value is not None and np.isfinite(float(value)) and float(value) > 0:
                freq_arr[i, j] = float(value)
            try:
                bw = formant.get_bandwidth_at_time(j + 1, float(t))
            except Exception:
                bw = None
            if bw is not None and np.isfinite(float(bw)) and float(bw) > 0:
                bw_arr[i, j] = float(bw)
    return freq_arr, bw_arr


def formant_contours(
    x: torch.Tensor,
    fs: int,
    order: int | None = 10,
    num_formants: int = 5,
    max_formant: float = 5500.0,
    frame_length_ms: float = 25.0,
    hop_length_ms: float | None = None,
    pre_emphasis: float = 0.97,
    max_bandwidth: float = 700.0,
    method: str = "burg",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Extract time-varying formant contours.

    Args:
        x: Audio waveform tensor.
        fs: Sample rate in Hz.
        order: LPC order for Burg method. If None, uses `2 * num_formants + 2`.
        num_formants: Number of formants to track.
        max_formant: Upper frequency bound (Hz). Typical values:
            5000 (male), 5500 (female), 8000 (children).
        frame_length_ms: Effective analysis window in milliseconds.
        hop_length_ms: Hop step in milliseconds. Defaults to `frame_length_ms / 4`.
        pre_emphasis: Pre-emphasis coefficient in (0, 1). Also accepts Hz-like value.
        max_bandwidth: Maximum valid formant bandwidth in Hz.
        method: `burg` (torch/numpy implementation) or `praat` (parselmouth backend).
    """
    if num_formants <= 0:
        raise ValueError("num_formants must be > 0.")
    if max_formant <= 0:
        raise ValueError("max_formant must be > 0.")
    if hop_length_ms is None:
        hop_length_ms = frame_length_ms / 4.0
    if hop_length_ms <= 0:
        raise ValueError("hop_length_ms must be > 0.")

    x = x.flatten().float()
    if x.numel() == 0:
        raise ValueError("Input signal must be non-empty.")

    x, fs = _resample_for_formants(x, fs, max_formant)
    x_np = x.detach().cpu().numpy().astype(np.float64, copy=False)

    if method == "praat":
        freqs, bws = _praat_contours_from_tensor(
            x_np,
            fs,
            num_formants=num_formants,
            max_formant=max_formant,
            frame_length_ms=frame_length_ms,
            hop_length_ms=float(hop_length_ms),
            pre_emphasis=pre_emphasis,
        )
        return torch.from_numpy(freqs), torch.from_numpy(bws)

    if method != "burg":
        raise ValueError("method must be one of {'burg', 'praat'}.")

    pre_emph_coeff = _estimate_pre_emphasis_coeff(pre_emphasis, fs)
    x_np = _apply_pre_emphasis(x_np, pre_emph_coeff)

    if order is None:
        order = max(10, 2 * num_formants + 2)
    order = int(order)
    if order <= 0:
        raise ValueError("order must be > 0.")

    # Praat's "window length" is effective Gaussian width; actual window is roughly 2x.
    actual_window_ms = 2.0 * float(frame_length_ms)
    frame_length = int(max(order * 4, round(actual_window_ms * fs / 1000.0)))
    hop_length = int(max(1, round(float(hop_length_ms) * fs / 1000.0)))

    frames = _frame_signal_numpy(x_np, frame_length=frame_length, hop_length=hop_length)
    window = _gaussian_window(frame_length)

    frame_candidates: list[tuple[np.ndarray, np.ndarray]] = []
    for frame in frames:
        if frame.size < order * 2:
            frame_candidates.append((np.array([]), np.array([])))
            continue
        a = _burg_lpc(frame * window, order)
        freqs, bws = _roots_to_formants(
            a,
            fs,
            min_freq=50.0,
            max_freq=max_formant,
            max_bandwidth=max_bandwidth,
        )
        frame_candidates.append((freqs, bws))

    freq_arr, bw_arr = _track_formants(frame_candidates, num_formants, max_formant)
    return torch.from_numpy(freq_arr), torch.from_numpy(bw_arr)


def formant_frequencies(
    x: torch.Tensor,
    fs: int,
    order: int | None = 10,
    num_formants: int = 5,
    max_formant: float = 5500.0,
    frame_length_ms: float = 25.0,
    hop_length_ms: float | None = None,
    pre_emphasis: float = 0.97,
    max_bandwidth: float = 700.0,
    method: str = "burg",
):
    """
    Estimate median formant frequencies [F1, F2, ..., Fn].
    """
    contours, _bws = formant_contours(
        x,
        fs=fs,
        order=order,
        num_formants=num_formants,
        max_formant=max_formant,
        frame_length_ms=frame_length_ms,
        hop_length_ms=hop_length_ms,
        pre_emphasis=pre_emphasis,
        max_bandwidth=max_bandwidth,
        method=method,
    )
    contours_np = contours.detach().cpu().numpy()
    medians = [_robust_track_median(contours_np[:, i]) for i in range(contours_np.shape[1])]
    return torch.tensor(medians, dtype=torch.float64, device=x.device)


def formant_bandwidths(a: torch.Tensor, fs: int):
    """Formant bandwidths from LPC polynomial roots."""
    roots = np.roots(a.detach().cpu().numpy())
    roots = roots[roots.imag >= 0]
    freqs = np.angle(roots) * fs / (2 * np.pi)
    with np.errstate(divide="ignore", invalid="ignore"):
        bandwidths = -fs * np.log(np.abs(roots)) / np.pi
    order = np.argsort(freqs)
    bandwidths = bandwidths[order]
    bandwidths = bandwidths[np.isfinite(bandwidths)]
    return torch.from_numpy(bandwidths)


def formant_dispersion(formants: torch.Tensor):
    """Average spacing between consecutive finite formants."""
    formants = formants[torch.isfinite(formants)]
    if formants.numel() < 2:
        return torch.tensor(0.0, device=formants.device)
    diffs = formants[1:] - formants[:-1]
    return diffs.mean()
