"""ERB / gammatone filterbank and cepstral coefficient pipeline.

Implements:
  1. Glasberg & Moore (1990) ERB-rate scale conversions.
  2. 4th-order gammatone magnitude-response filterbank.
  3. Log-energy + DCT pipeline for GFCC / GTCC computation.

The gammatone impulse response is:
    g(t) = t^(n-1) exp(-2π b t) cos(2π f_c t),   n = 4
with bandwidth  b = 1.019 · ERB(f_c).

The magnitude-squared frequency response of order n=4 is:
    |H(f)|^2 ∝ 1 / (1 + ((f - f_c) / b)^2)^n
This gives proper 4th-order selectivity matching the analytical
gammatone transfer function.
"""

from __future__ import annotations

import torch
import torchaudio.functional as AF


def to_mono_waveform(waveform: torch.Tensor) -> torch.Tensor:
    if waveform.dim() == 1:
        return waveform.float()
    if waveform.dim() == 2:
        if waveform.shape[0] == 1:
            return waveform.squeeze(0).float()
        return waveform.mean(dim=0).float()
    raise ValueError("waveform must be 1-D or 2-D (channels, samples).")


def hz_to_erb(freq_hz: torch.Tensor) -> torch.Tensor:
    """Convert Hz to ERB-rate (Glasberg & Moore 1990)."""
    return 21.4 * torch.log10(4.37e-3 * freq_hz + 1.0)


def erb_to_hz(freq_erb: torch.Tensor) -> torch.Tensor:
    """Convert ERB-rate back to Hz."""
    return (torch.pow(10.0, freq_erb / 21.4) - 1.0) / 4.37e-3


def gammatone_filterbank(
    sample_rate: int,
    n_fft: int,
    n_bands: int,
    fmin: float = 50.0,
    fmax: float | None = None,
    order: int = 4,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build a gammatone-style magnitude filterbank.

    Parameters
    ----------
    sample_rate : int
        Audio sample rate.
    n_fft : int
        FFT size.
    n_bands : int
        Number of filters.
    fmin, fmax : float
        Frequency range (Hz). ``fmax`` defaults to Nyquist.
    order : int
        Gammatone filter order (default 4).
    device, dtype : torch
        Tensor placement.

    Returns
    -------
    torch.Tensor
        Shape ``(n_bands, n_fft // 2 + 1)`` — energy weights.
    """
    if sample_rate <= 0 or n_fft <= 0 or n_bands <= 0:
        raise ValueError("sample_rate, n_fft, and n_bands must be > 0.")
    if fmin <= 0:
        raise ValueError("fmin must be > 0.")
    if fmax is None:
        fmax = sample_rate / 2.0
    if fmax <= fmin:
        raise ValueError("fmax must be > fmin.")

    n_bins = n_fft // 2 + 1
    freqs = torch.linspace(0.0, float(sample_rate) / 2.0, n_bins, device=device, dtype=dtype)

    # ERB-spaced center frequencies
    erb_min = hz_to_erb(torch.tensor(float(fmin), device=device, dtype=dtype))
    erb_max = hz_to_erb(torch.tensor(float(fmax), device=device, dtype=dtype))
    erb_centers = torch.linspace(erb_min, erb_max, n_bands, device=device, dtype=dtype)
    cf = erb_to_hz(erb_centers)

    # Bandwidth: b = 1.019 * ERB(f_c)
    erb_bw = 24.7 * (4.37 * (cf / 1000.0) + 1.0)
    bw = 1.019 * erb_bw

    # Vectorised magnitude response: |H(f)|^2 ∝ (1 + ((f - cf)/bw)^2)^(-order)
    f = freqs.unsqueeze(0)     # (1, n_bins)
    c = cf.unsqueeze(1)        # (n_bands, 1)
    b = bw.unsqueeze(1)        # (n_bands, 1)

    fb = torch.pow(1.0 + torch.pow((f - c) / (b + 1e-12), 2.0), -float(order))

    # Area-normalize each filter so energy is comparable across bands
    fb = fb / (fb.sum(dim=1, keepdim=True) + 1e-12)
    return fb


def erb_cepstral_coefficients(
    waveform: torch.Tensor,
    sample_rate: int,
    n_coeffs: int,
    n_fft: int,
    hop_length: int,
    n_bands: int,
    fmin: float = 50.0,
    fmax: float | None = None,
    eps: float = 1e-10,
) -> torch.Tensor:
    """Compute gammatone cepstral coefficients (GFCC / GTCC).

    Pipeline: STFT → power spectrum → gammatone filterbank →
    log energies → DCT (type-II, ortho).

    Parameters
    ----------
    waveform : torch.Tensor
        Mono audio.
    sample_rate : int
        Sample rate.
    n_coeffs : int
        Number of cepstral coefficients to return.
    n_fft : int
        FFT window size.
    hop_length : int
        Hop size.
    n_bands : int
        Number of gammatone filters.
    fmin, fmax : float
        Frequency range.
    eps : float
        Floor for log computation.

    Returns
    -------
    torch.Tensor
        Shape ``(n_coeffs, frames)``.
    """
    x = to_mono_waveform(waveform)
    if x.numel() == 0:
        raise ValueError("waveform must be non-empty.")
    if n_coeffs <= 0 or n_bands <= 0:
        raise ValueError("n_coeffs and n_bands must be > 0.")

    n_coeffs = min(int(n_coeffs), int(n_bands))

    window = torch.hann_window(int(n_fft), device=x.device, dtype=x.dtype)
    spec = torch.stft(
        x,
        n_fft=int(n_fft),
        hop_length=int(hop_length),
        window=window,
        return_complex=True,
    )
    power_spec = spec.abs().pow(2.0)

    fb = gammatone_filterbank(
        sample_rate=int(sample_rate),
        n_fft=int(n_fft),
        n_bands=int(n_bands),
        fmin=float(fmin),
        fmax=None if fmax is None else float(fmax),
        order=4,
        device=x.device,
        dtype=x.dtype,
    )
    band_energies = fb @ power_spec
    log_energies = torch.log(band_energies + float(eps))

    # DCT-II with ortho normalization
    dct = AF.create_dct(int(n_coeffs), int(n_bands), norm="ortho")
    dct = dct.to(device=x.device, dtype=x.dtype)
    coeffs = torch.matmul(log_energies.transpose(0, 1), dct).transpose(0, 1)
    return coeffs
