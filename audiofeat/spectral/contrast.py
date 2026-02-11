"""Spectral contrast — Jiang et al. (2002).

Each frame of a magnitude spectrogram is divided into octave-spaced
sub-bands.  For each sub-band the *peak* (mean of top-quantile bins)
and *valley* (mean of bottom-quantile bins) are computed and the
contrast is returned as the dB difference: ``dB(peak) - dB(valley)``.

Primary path delegates to ``librosa.feature.spectral_contrast``.
The fallback faithfully replicates librosa's algorithm:

1. Band edges: ``octa[0] = 0``, ``octa[k>=1] = fmin * 2^(k-1)``
   so the first band is ``[0, fmin]`` and subsequent bands are
   ``[fmin, 2*fmin]``, ``[2*fmin, 4*fmin]``, …
2. Within each band: sort magnitude bins, take mean of bottom
   ``quantile`` fraction as valley, top ``quantile`` as peak.
3. Contrast = power_to_db(peak) − power_to_db(valley)  (librosa v0.11+)
"""

from __future__ import annotations

import numpy as np
import torch


def _to_mono_tensor(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 1:
        return x.float()
    if x.dim() == 2:
        if x.shape[0] == 1:
            return x.squeeze(0).float()
        return x.mean(dim=0).float()
    raise ValueError("Input must be 1-D or 2-D (channels, samples).")


def _power_to_db(x: torch.Tensor, ref: float = 1.0, amin: float = 1e-10, top_db: float = 80.0) -> torch.Tensor:
    """Convert to dB, matching ``librosa.power_to_db`` (v0.11+, ref=1.0).

    Formula: 10 * log10(max(S, amin)) - 10 * log10(max(ref, amin))
    then clamp to [global_max - top_db, global_max].
    """
    log_spec = 10.0 * torch.log10(torch.clamp(x, min=amin))
    log_ref = 10.0 * torch.log10(torch.tensor(max(ref, amin)))
    log_spec = log_spec - log_ref
    if top_db is not None:
        log_spec = torch.clamp(log_spec, min=log_spec.max() - top_db)
    return log_spec


def _fallback_spectral_contrast(
    x: torch.Tensor,
    fs: int,
    n_fft: int,
    hop_length: int,
    n_bands: int,
    quantile: float,
    fmin: float,
) -> torch.Tensor:
    """Reproduce librosa's spectral_contrast in pure PyTorch."""
    # librosa uses center=True with pad_mode='constant' (zero-padding).
    # torch.stft defaults to pad_mode='reflect', so we pad manually.
    pad_length = n_fft // 2
    x_padded = torch.nn.functional.pad(x, (pad_length, pad_length), mode='constant', value=0.0)
    window = torch.hann_window(n_fft, device=x.device)
    spec = torch.stft(
        x_padded, n_fft=n_fft, hop_length=hop_length, window=window,
        center=False, return_complex=True,
    ).abs()  # magnitude, shape (n_bins, n_frames)

    freqs = torch.fft.rfftfreq(n_fft, d=1.0 / fs).to(x.device)
    n_bins, n_frames = spec.shape

    # Octave band edges matching librosa exactly
    # octa[0] = 0, octa[k>=1] = fmin * 2^(k-1)
    octa = torch.zeros(n_bands + 2, device=x.device, dtype=torch.float32)
    for k in range(1, n_bands + 2):
        octa[k] = fmin * (2.0 ** (k - 1))

    valley = torch.zeros(n_bands + 1, n_frames, device=x.device, dtype=spec.dtype)
    peak = torch.zeros_like(valley)

    for k in range(n_bands + 1):
        f_low = octa[k]
        f_high = octa[k + 1]
        current_band = (freqs >= f_low) & (freqs <= f_high)
        idx = current_band.nonzero(as_tuple=True)[0]

        if idx.numel() == 0:
            continue

        # librosa: if k > 0, include one bin below
        if k > 0 and idx[0] > 0:
            current_band[idx[0] - 1] = True

        # librosa: if k == n_bands (last band), include all bins above
        if k == n_bands:
            current_band[idx[-1] + 1 :] = True

        # librosa: compute quantile count BEFORE dropping the last bin
        n_q = max(1, round(quantile * int(current_band.sum())))

        sub_band = spec[current_band, :]  # (band_bins, n_frames)

        # librosa: for non-last bands, drop the last bin
        if k < n_bands and sub_band.shape[0] > 1:
            sub_band = sub_band[:-1, :]

        # Sort along frequency axis
        sorted_band, _ = torch.sort(sub_band, dim=0)

        valley[k, :] = sorted_band[:n_q, :].mean(dim=0)
        peak[k, :] = sorted_band[-n_q:, :].mean(dim=0)

    # power_to_db difference — librosa applies power_to_db to magnitudes directly
    contrast = _power_to_db(peak) - _power_to_db(valley)
    return contrast


def spectral_contrast(
    x: torch.Tensor,
    fs: int,
    n_fft: int = 2048,
    n_bands: int = 6,
    hop_length: int = 512,
    quantile: float = 0.02,
    fmin: float = 200.0,
) -> torch.Tensor:
    """Compute spectral contrast, matching librosa when available.

    Parameters
    ----------
    x : torch.Tensor
        Audio waveform.
    fs : int
        Sample rate in Hz.
    n_fft : int
        FFT window size.
    n_bands : int
        Number of octave-spaced frequency bands.
    hop_length : int
        Hop length for STFT.
    quantile : float
        Quantile for peak/valley selection, in (0, 1).
    fmin : float
        Lowest frequency band edge (Hz).

    Returns
    -------
    torch.Tensor
        Shape ``(n_bands + 1, frames)`` — spectral contrast in dB.
    """
    if fs <= 0:
        raise ValueError("fs must be > 0.")
    if n_fft <= 0 or hop_length <= 0 or n_bands <= 0:
        raise ValueError("n_fft, hop_length, and n_bands must be > 0.")
    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must satisfy 0 < quantile < 1.")
    if fmin <= 0:
        raise ValueError("fmin must be > 0.")

    waveform = _to_mono_tensor(x)
    device = waveform.device

    try:
        import librosa  # type: ignore
    except ModuleNotFoundError:
        return _fallback_spectral_contrast(
            waveform, fs, n_fft, hop_length, n_bands, quantile, fmin,
        )

    contrast_np = librosa.feature.spectral_contrast(
        y=waveform.detach().cpu().numpy().astype(np.float32, copy=False),
        sr=int(fs),
        n_fft=int(n_fft),
        hop_length=int(hop_length),
        n_bands=int(n_bands),
        quantile=float(quantile),
        fmin=float(fmin),
    )
    return torch.from_numpy(contrast_np.astype(np.float32, copy=False)).to(device=device)
