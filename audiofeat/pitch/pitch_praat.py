"""Praat-based pitch extraction using parselmouth.

This module provides exact Praat parity for F0 extraction, useful for
research workflows that require reproducible results matching Praat's output.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import torch


def fundamental_frequency_praat(
    x: torch.Tensor,
    fs: int,
    *,
    pitch_floor: float = 75.0,
    pitch_ceiling: float = 600.0,
    time_step: float | None = None,
    very_accurate: bool = False,
    fill_unvoiced: float = 0.0,
) -> torch.Tensor:
    """
    Extract F0 using Praat's autocorrelation method via parselmouth.

    This provides exact Praat parity for pitch extraction, using Praat's
    robust autocorrelation algorithm with pathfinding that produces smoother
    and more accurate pitch contours than YIN-based methods.

    Args:
        x: Audio waveform tensor.
        fs: Sample rate in Hz.
        pitch_floor: Minimum F0 in Hz (affects analysis window length).
        pitch_ceiling: Maximum F0 in Hz.
        time_step: Time step in seconds. If None, uses 0.75 / pitch_floor
            (Praat's default for accurate mode).
        very_accurate: If True, uses longer analysis windows for more
            accurate pitch tracking at the expense of time resolution.
        fill_unvoiced: Value to use for unvoiced frames. Set to float('nan')
            to keep NaN for unvoiced frames.

    Returns:
        1D tensor of F0 values in Hz, with unvoiced frames set to fill_unvoiced.

    Raises:
        ModuleNotFoundError: If parselmouth is not installed.
        ValueError: If pitch_floor >= pitch_ceiling or inputs are invalid.

    Example:
        >>> import torch
        >>> from audiofeat.pitch.pitch_praat import fundamental_frequency_praat
        >>> waveform = torch.randn(22050 * 2)  # 2 seconds of audio
        >>> f0 = fundamental_frequency_praat(waveform, fs=22050)
        >>> print(f0.shape)
    """
    try:
        import parselmouth  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "praat-parselmouth is required for Praat pitch extraction. "
            "Install with `pip install \"audiofeat[validation]\"`."
        ) from exc

    if pitch_floor <= 0:
        raise ValueError("pitch_floor must be > 0.")
    if pitch_ceiling <= pitch_floor:
        raise ValueError("pitch_ceiling must be > pitch_floor.")

    x = x.flatten().float()
    if x.numel() == 0:
        raise ValueError("Input waveform must be non-empty.")

    waveform_np = x.detach().cpu().numpy().astype(np.float64, copy=False)

    sound = parselmouth.Sound(waveform_np, sampling_frequency=float(fs))

    # Use Praat's default time step if not specified
    if time_step is None:
        time_step = 0.0  # Praat will use its default

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pitch = sound.to_pitch_ac(
            time_step=float(time_step) if time_step > 0 else None,
            pitch_floor=float(pitch_floor),
            pitch_ceiling=float(pitch_ceiling),
            very_accurate=bool(very_accurate),
        )

    pitch_values = np.asarray(pitch.selected_array["frequency"], dtype=np.float64)

    # Handle unvoiced frames
    if np.isnan(fill_unvoiced):
        # Keep NaN for unvoiced frames
        pass
    else:
        unvoiced_mask = pitch_values == 0
        pitch_values[unvoiced_mask] = float(fill_unvoiced)

    return torch.from_numpy(pitch_values.astype(np.float32)).to(device=x.device)


def fundamental_frequency_praat_cc(
    x: torch.Tensor,
    fs: int,
    *,
    pitch_floor: float = 75.0,
    pitch_ceiling: float = 600.0,
    time_step: float | None = None,
    very_accurate: bool = False,
    fill_unvoiced: float = 0.0,
) -> torch.Tensor:
    """
    Extract F0 using Praat's cross-correlation method via parselmouth.

    The cross-correlation method is typically better for noisy signals
    or for voice types with strong higher harmonics.

    Args:
        x: Audio waveform tensor.
        fs: Sample rate in Hz.
        pitch_floor: Minimum F0 in Hz.
        pitch_ceiling: Maximum F0 in Hz.
        time_step: Time step in seconds. If None, uses Praat's default.
        very_accurate: If True, uses longer analysis windows.
        fill_unvoiced: Value to use for unvoiced frames.

    Returns:
        1D tensor of F0 values in Hz.
    """
    try:
        import parselmouth  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "praat-parselmouth is required for Praat pitch extraction. "
            "Install with `pip install \"audiofeat[validation]\"`."
        ) from exc

    if pitch_floor <= 0:
        raise ValueError("pitch_floor must be > 0.")
    if pitch_ceiling <= pitch_floor:
        raise ValueError("pitch_ceiling must be > pitch_floor.")

    x = x.flatten().float()
    if x.numel() == 0:
        raise ValueError("Input waveform must be non-empty.")

    waveform_np = x.detach().cpu().numpy().astype(np.float64, copy=False)
    sound = parselmouth.Sound(waveform_np, sampling_frequency=float(fs))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pitch = sound.to_pitch_cc(
            time_step=float(time_step) if time_step and time_step > 0 else None,
            pitch_floor=float(pitch_floor),
            pitch_ceiling=float(pitch_ceiling),
            very_accurate=bool(very_accurate),
        )

    pitch_values = np.asarray(pitch.selected_array["frequency"], dtype=np.float64)

    if not np.isnan(fill_unvoiced):
        unvoiced_mask = pitch_values == 0
        pitch_values[unvoiced_mask] = float(fill_unvoiced)

    return torch.from_numpy(pitch_values.astype(np.float32)).to(device=x.device)


def pitch_strength_praat(
    x: torch.Tensor,
    fs: int,
    *,
    pitch_floor: float = 75.0,
    pitch_ceiling: float = 600.0,
    time_step: float | None = None,
) -> torch.Tensor:
    """
    Extract pitch strength (autocorrelation at F0 lag) using Praat.

    Pitch strength indicates the confidence of the pitch detection.
    Values close to 1 indicate strong periodicity, values close to 0
    indicate weak or aperiodic signals.

    Args:
        x: Audio waveform tensor.
        fs: Sample rate in Hz.
        pitch_floor: Minimum F0 in Hz.
        pitch_ceiling: Maximum F0 in Hz.
        time_step: Time step in seconds.

    Returns:
        1D tensor of pitch strength values (0-1 range).
    """
    try:
        import parselmouth  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "praat-parselmouth is required for Praat pitch extraction. "
            "Install with `pip install \"audiofeat[validation]\"`."
        ) from exc

    x = x.flatten().float()
    if x.numel() == 0:
        raise ValueError("Input waveform must be non-empty.")

    waveform_np = x.detach().cpu().numpy().astype(np.float64, copy=False)
    sound = parselmouth.Sound(waveform_np, sampling_frequency=float(fs))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pitch = sound.to_pitch_ac(
            time_step=float(time_step) if time_step and time_step > 0 else None,
            pitch_floor=float(pitch_floor),
            pitch_ceiling=float(pitch_ceiling),
        )

    strength_values = np.asarray(pitch.selected_array["strength"], dtype=np.float64)
    strength_values = np.nan_to_num(strength_values, nan=0.0)

    return torch.from_numpy(strength_values.astype(np.float32)).to(device=x.device)
