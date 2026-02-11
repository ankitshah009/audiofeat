"""Chromagram (STFT-based) — Ellis (2007).

Computes a chroma filterbank by projecting FFT bins onto pitch classes
using Gaussian bumps in log-frequency space, with an octave-dominance
weighting.

Primary path delegates to ``librosa.feature.chroma_stft``.
The fallback builds the filterbank in PyTorch following the same
algorithm as ``librosa.filters.chroma`` so that the two paths produce
comparable output.
"""

from __future__ import annotations

import numpy as np
import torch
import torchaudio.transforms as T


def _to_mono_numpy(audio_data: torch.Tensor) -> np.ndarray:
    x = audio_data.detach().float().cpu()
    if x.dim() == 0:
        raise ValueError("Input audio_data cannot be a scalar.")
    if x.dim() == 1:
        return x.numpy()
    if x.dim() == 2:
        if x.shape[0] == 1:
            return x.squeeze(0).numpy()
        return x.mean(dim=0).numpy()
    raise ValueError("audio_data must be 1-D or 2-D (channels, samples).")


def _chroma_filterbank(
    sample_rate: int,
    n_fft: int,
    n_chroma: int = 12,
    tuning: float = 0.0,
    ctroct: float = 5.0,
    octwidth: float = 2.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build a chroma filterbank matching librosa.filters.chroma.

    Follows the same algorithm: Gaussian bumps in log-frequency,
    column L2-normalized, with octave-dominance Gaussian weighting,
    rolled to start at C.
    """
    # Frequencies for bins 1..n_fft-1 (skip DC)
    # librosa: np.linspace(0, sr, n_fft, endpoint=False)[1:]
    freqs = (torch.arange(1, n_fft, device=device, dtype=dtype) * (float(sample_rate) / n_fft))

    # Convert Hz to fractional chroma bins using librosa.hz_to_octs:
    #   A440_adj = 440 * 2^(tuning/n_chroma)
    #   octs = log2(freq / (A440_adj / 16))
    # Then frqbins = n_chroma * octs
    A440_adj = 440.0 * (2.0 ** (tuning / n_chroma))
    frqbins = n_chroma * torch.log2(freqs / (A440_adj / 16.0))

    # DC bin: 1.5 octaves below bin 1
    dc_val = frqbins[0] - 1.5 * n_chroma
    frqbins = torch.cat([dc_val.unsqueeze(0), frqbins])  # length n_fft

    # Bin widths
    binwidths = torch.cat([
        torch.clamp(frqbins[1:] - frqbins[:-1], min=1.0),
        torch.ones(1, device=device, dtype=dtype),
    ])

    # Distance matrix: D[c, f] = frqbins[f] - c
    chroma_idx = torch.arange(n_chroma, device=device, dtype=dtype)
    D = frqbins.unsqueeze(0) - chroma_idx.unsqueeze(1)  # (n_chroma, n_fft)

    # Wrap into [-n_chroma/2, n_chroma/2]
    n_chroma2 = round(n_chroma / 2.0)
    D = torch.remainder(D + n_chroma2 + 10 * n_chroma, n_chroma) - n_chroma2

    # Gaussian bumps (narrowed by factor of 2)
    wts = torch.exp(-0.5 * (2.0 * D / binwidths.unsqueeze(0)) ** 2)

    # Column-normalize (L2)
    col_norms = torch.norm(wts, p=2, dim=0, keepdim=True)
    wts = wts / (col_norms + 1e-12)

    # Octave dominance weighting
    if octwidth > 0:
        octs_for_weight = frqbins / n_chroma
        oct_weight = torch.exp(-0.5 * ((octs_for_weight - ctroct) / octwidth) ** 2)
        wts = wts * oct_weight.unsqueeze(0)

    # Roll to start at C (base_c=True)
    wts = torch.roll(wts, shifts=-3 * (n_chroma // 12), dims=0)

    # Keep only rfft bins: [0, n_fft//2]
    return wts[:, : n_fft // 2 + 1]


def _fallback_chroma(
    audio_data: torch.Tensor,
    sample_rate: int,
    n_fft: int,
    hop_length: int,
    n_chroma: int,
) -> torch.Tensor:
    """Compute chroma using the PyTorch filterbank + STFT."""
    x = audio_data.float()
    if x.dim() == 2:
        x = x.mean(dim=0) if x.shape[0] > 1 else x.squeeze(0)
    x = x.flatten()
    device = x.device

    # Power spectrogram — librosa uses center=True with zero-padding
    stft = T.Spectrogram(n_fft=n_fft, hop_length=hop_length, power=2.0, pad_mode='constant')(x)

    # Build filterbank
    fb = _chroma_filterbank(
        sample_rate, n_fft, n_chroma, tuning=0.0,
        device=device, dtype=stft.dtype,
    )

    raw_chroma = fb @ stft

    # L-inf normalize each frame (column)
    max_vals = raw_chroma.abs().max(dim=0, keepdim=True).values
    chroma_norm = raw_chroma / (max_vals + 1e-12)
    return chroma_norm


def chroma(
    audio_data: torch.Tensor,
    sample_rate: int,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_chroma: int = 12,
) -> torch.Tensor:
    """Compute chroma features, matching ``librosa.feature.chroma_stft``.

    Parameters
    ----------
    audio_data : torch.Tensor
        Audio waveform.
    sample_rate : int
        Sample rate in Hz.
    n_fft : int
        FFT window size.
    hop_length : int
        Hop length for STFT.
    n_chroma : int
        Number of chroma bins (default 12).

    Returns
    -------
    torch.Tensor
        Shape ``(n_chroma, frames)``.
    """
    if sample_rate <= 0:
        raise ValueError("sample_rate must be > 0.")
    if n_fft <= 0 or hop_length <= 0 or n_chroma <= 0:
        raise ValueError("n_fft, hop_length, and n_chroma must be > 0.")

    device = audio_data.device

    try:
        import librosa  # type: ignore
    except ModuleNotFoundError:
        return _fallback_chroma(audio_data, sample_rate, n_fft, hop_length, n_chroma).to(device)

    waveform = _to_mono_numpy(audio_data)
    if waveform.size == 0:
        raise ValueError("audio_data must be non-empty.")

    chroma_np = librosa.feature.chroma_stft(
        y=waveform,
        sr=int(sample_rate),
        n_fft=int(n_fft),
        hop_length=int(hop_length),
        n_chroma=int(n_chroma),
        tuning=0.0,
    )
    return torch.from_numpy(chroma_np.astype(np.float32, copy=False)).to(device=device)
