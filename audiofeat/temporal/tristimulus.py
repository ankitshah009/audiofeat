"""Tristimulus — spectral balance descriptor (Pollard & Jansson, 1982).

Computes the energy distribution across harmonic partials:
  T1 = a1 / sum(a_k)          — fundamental
  T2 = (a2 + a3 + a4) / sum   — 2nd–4th harmonics
  T3 = sum(a5..N) / sum        — remaining harmonics

Harmonics are identified by finding peaks closest to k * f0 in the
magnitude STFT, where f0 is estimated globally via autocorrelation.
The result is averaged over voiced frames.
"""

from __future__ import annotations

import torch
import torchaudio.transforms as T


def _estimate_f0_autocorr(
    waveform: torch.Tensor,
    sample_rate: int,
    fmin: float = 60.0,
    fmax: float = 1000.0,
) -> float:
    """Simple autocorrelation-based f0 estimate on the full waveform."""
    x = waveform.float()
    x = x - x.mean()
    if x.numel() < 4:
        return 0.0

    n = x.numel()
    # Use FFT-based autocorrelation for speed
    fft_size = 1
    while fft_size < 2 * n:
        fft_size *= 2
    X = torch.fft.rfft(x, n=fft_size)
    ac = torch.fft.irfft(X * X.conj(), n=fft_size)[:n]

    min_lag = max(1, int(sample_rate / fmax))
    max_lag = min(n - 1, int(sample_rate / fmin))
    if max_lag <= min_lag:
        return 0.0

    ac_region = ac[min_lag : max_lag + 1]
    if ac_region.numel() == 0:
        return 0.0
    best_lag = int(ac_region.argmax().item()) + min_lag
    if best_lag == 0:
        return 0.0
    return float(sample_rate) / best_lag


def _harmonic_amplitudes(
    spec_frame: torch.Tensor,
    f0_hz: float,
    sample_rate: int,
    n_fft: int,
    n_harmonics: int = 10,
    tolerance_hz: float = 30.0,
) -> torch.Tensor:
    """Extract amplitudes of the first `n_harmonics` partials from one STFT frame."""
    n_bins = spec_frame.shape[0]
    freq_per_bin = float(sample_rate) / n_fft
    amps = torch.zeros(n_harmonics, device=spec_frame.device, dtype=spec_frame.dtype)

    for k in range(n_harmonics):
        target_hz = f0_hz * (k + 1)
        center_bin = int(round(target_hz / freq_per_bin))
        tol_bins = max(1, int(round(tolerance_hz / freq_per_bin)))
        lo = max(0, center_bin - tol_bins)
        hi = min(n_bins, center_bin + tol_bins + 1)
        if lo >= n_bins or hi <= 0:
            continue
        amps[k] = spec_frame[lo:hi].max()
    return amps


def tristimulus(
    waveform: torch.Tensor,
    sample_rate: int,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_harmonics: int = 10,
) -> torch.Tensor:
    """Compute tristimulus values (T1, T2, T3) from harmonic amplitudes.

    Parameters
    ----------
    waveform : torch.Tensor
        Mono audio, shape ``(samples,)`` or ``(1, samples)``.
    sample_rate : int
        Sampling rate in Hz.
    n_fft : int
        FFT size.
    hop_length : int
        Hop length for STFT.
    n_harmonics : int
        Number of harmonics to track (>= 5 recommended).

    Returns
    -------
    torch.Tensor
        Shape ``(3,)`` — [T1, T2, T3], each in [0, 1], summing to ~1.
    """
    if waveform.ndim > 1:
        waveform = waveform.squeeze(0) if waveform.shape[0] == 1 else waveform[0]
    if waveform.ndim == 0:
        raise ValueError("Input waveform cannot be a scalar.")

    waveform = waveform.float()
    n_harmonics = max(n_harmonics, 5)  # need at least 5 for T3

    # Estimate f0 from the whole waveform
    f0 = _estimate_f0_autocorr(waveform, sample_rate)
    if f0 < 20.0:
        # Cannot determine f0 — return uniform distribution
        return torch.tensor([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], device=waveform.device)

    # Compute magnitude STFT
    spec_transform = T.Spectrogram(n_fft=n_fft, hop_length=hop_length, power=1.0)
    mag = spec_transform(waveform)  # (n_bins, n_frames)

    n_frames = mag.shape[1]
    t1_acc = 0.0
    t2_acc = 0.0
    t3_acc = 0.0
    count = 0

    for i in range(n_frames):
        amps = _harmonic_amplitudes(
            mag[:, i], f0, sample_rate, n_fft,
            n_harmonics=n_harmonics,
        )
        total = amps.sum().item()
        if total < 1e-12:
            continue
        t1_acc += amps[0].item() / total
        t2_acc += (amps[1] + amps[2] + amps[3]).item() / total
        t3_acc += amps[4:].sum().item() / total
        count += 1

    if count == 0:
        return torch.tensor([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], device=waveform.device)

    t1 = t1_acc / count
    t2 = t2_acc / count
    t3 = t3_acc / count
    return torch.tensor([t1, t2, t3], device=waveform.device, dtype=torch.float32)
