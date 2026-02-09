"""Praat-based voice quality metrics using parselmouth.

This module provides exact Praat parity for voice quality metrics including
jitter, shimmer, and harmonic-to-noise ratio (HNR) using Praat's PointProcess
analysis via parselmouth.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch


def jitter_shimmer_praat(
    x: torch.Tensor | str | Path,
    fs: int | None = None,
    *,
    pitch_floor: float = 75.0,
    pitch_ceiling: float = 600.0,
    period_floor: float = 0.0001,
    period_ceiling: float = 0.02,
    max_period_factor: float = 1.3,
    max_amplitude_factor: float = 1.6,
) -> dict[str, float]:
    """
    Extract jitter and shimmer metrics using Praat's PointProcess analysis.

    This provides exact Praat parity for voice quality metrics, using Praat's
    robust period detection and perturbation measurement algorithms.

    Args:
        x: Either a waveform tensor (requires fs) or a file path to an audio file.
        fs: Sample rate in Hz. Required if x is a tensor.
        pitch_floor: Minimum F0 in Hz for pitch detection.
        pitch_ceiling: Maximum F0 in Hz for pitch detection.
        period_floor: Minimum period in seconds (typically 0.0001).
        period_ceiling: Maximum period in seconds (typically 0.02).
        max_period_factor: Maximum ratio between consecutive periods (1.3 = 30%).
        max_amplitude_factor: Maximum ratio between consecutive amplitudes.

    Returns:
        Dictionary with Praat voice quality metrics:
        - jitter_local_percent: Local jitter (%)
        - jitter_local_abs_sec: Local jitter (absolute, seconds)
        - jitter_rap_percent: Relative average perturbation (%)
        - jitter_ppq5_percent: Five-point period perturbation quotient (%)
        - jitter_ddp_percent: Difference of differences of periods (%)
        - shimmer_local_percent: Local shimmer (%)
        - shimmer_local_db: Local shimmer (dB)
        - shimmer_apq3_percent: Three-point amplitude perturbation quotient (%)
        - shimmer_apq5_percent: Five-point amplitude perturbation quotient (%)
        - shimmer_apq11_percent: 11-point amplitude perturbation quotient (%)
        - hnr_db: Harmonics-to-noise ratio (dB)
        - num_periods: Number of detected periods

    Raises:
        ModuleNotFoundError: If parselmouth is not installed.
        ValueError: If inputs are invalid.

    Example:
        >>> import torch
        >>> from audiofeat.voice.praat_voice import jitter_shimmer_praat
        >>> waveform = torch.randn(22050 * 2)  # 2 seconds of audio
        >>> metrics = jitter_shimmer_praat(waveform, fs=22050)
        >>> print(metrics["jitter_local_percent"])
    """
    try:
        import parselmouth  # type: ignore
        from parselmouth.praat import call  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "praat-parselmouth is required for Praat voice quality extraction. "
            "Install with `pip install \"audiofeat[validation]\"`."
        ) from exc

    # Handle file path vs tensor input
    if isinstance(x, (str, Path)):
        sound = parselmouth.Sound(str(x))
    else:
        if fs is None:
            raise ValueError("Sample rate (fs) is required when input is a tensor.")
        x = x.flatten().float()
        if x.numel() == 0:
            raise ValueError("Input waveform must be non-empty.")
        waveform_np = x.detach().cpu().numpy().astype(np.float64, copy=False)
        sound = parselmouth.Sound(waveform_np, sampling_frequency=float(fs))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # Create pitch object and point process
        pitch = call(
            sound,
            "To Pitch",
            0.0,  # time step (0 = auto)
            float(pitch_floor),
            float(pitch_ceiling),
        )
        point_process = call(
            [sound, pitch],
            "To PointProcess (cc)",
        )

        # Get number of periods
        num_points = call(point_process, "Get number of points")
        num_periods = max(0, num_points - 1)

        # Helper to safely extract values
        def _safe_call(func_name: str, *args) -> float:
            try:
                val = call(point_process, func_name, *args)
                return float(val) if val is not None and np.isfinite(val) else float("nan")
            except Exception:
                return float("nan")

        def _safe_sound_call(func_name: str, *args) -> float:
            try:
                val = call([sound, point_process], func_name, *args)
                return float(val) if val is not None and np.isfinite(val) else float("nan")
            except Exception:
                return float("nan")

        # Extract jitter metrics
        jitter_local = _safe_call(
            "Get jitter (local)",
            0.0,  # start time
            0.0,  # end time (0 = all)
            float(period_floor),
            float(period_ceiling),
            float(max_period_factor),
        )
        jitter_local_abs = _safe_call(
            "Get jitter (local, absolute)",
            0.0,
            0.0,
            float(period_floor),
            float(period_ceiling),
            float(max_period_factor),
        )
        jitter_rap = _safe_call(
            "Get jitter (rap)",
            0.0,
            0.0,
            float(period_floor),
            float(period_ceiling),
            float(max_period_factor),
        )
        jitter_ppq5 = _safe_call(
            "Get jitter (ppq5)",
            0.0,
            0.0,
            float(period_floor),
            float(period_ceiling),
            float(max_period_factor),
        )
        jitter_ddp = _safe_call(
            "Get jitter (ddp)",
            0.0,
            0.0,
            float(period_floor),
            float(period_ceiling),
            float(max_period_factor),
        )

        # Extract shimmer metrics
        shimmer_local = _safe_sound_call(
            "Get shimmer (local)",
            0.0,
            0.0,
            float(period_floor),
            float(period_ceiling),
            float(max_period_factor),
            float(max_amplitude_factor),
        )
        shimmer_local_db = _safe_sound_call(
            "Get shimmer (local_dB)",
            0.0,
            0.0,
            float(period_floor),
            float(period_ceiling),
            float(max_period_factor),
            float(max_amplitude_factor),
        )
        shimmer_apq3 = _safe_sound_call(
            "Get shimmer (apq3)",
            0.0,
            0.0,
            float(period_floor),
            float(period_ceiling),
            float(max_period_factor),
            float(max_amplitude_factor),
        )
        shimmer_apq5 = _safe_sound_call(
            "Get shimmer (apq5)",
            0.0,
            0.0,
            float(period_floor),
            float(period_ceiling),
            float(max_period_factor),
            float(max_amplitude_factor),
        )
        shimmer_apq11 = _safe_sound_call(
            "Get shimmer (apq11)",
            0.0,
            0.0,
            float(period_floor),
            float(period_ceiling),
            float(max_period_factor),
            float(max_amplitude_factor),
        )

        # Extract HNR
        try:
            harmonicity = call(
                sound,
                "To Harmonicity (cc)",
                0.01,  # time step
                float(pitch_floor),
                0.1,  # silence threshold
                1.0,  # periods per window
            )
            hnr = call(harmonicity, "Get mean", 0.0, 0.0)
            hnr = float(hnr) if hnr is not None and np.isfinite(hnr) else float("nan")
        except Exception:
            hnr = float("nan")

    return {
        "jitter_local_percent": jitter_local * 100.0 if np.isfinite(jitter_local) else float("nan"),
        "jitter_local_abs_sec": jitter_local_abs,
        "jitter_rap_percent": jitter_rap * 100.0 if np.isfinite(jitter_rap) else float("nan"),
        "jitter_ppq5_percent": jitter_ppq5 * 100.0 if np.isfinite(jitter_ppq5) else float("nan"),
        "jitter_ddp_percent": jitter_ddp * 100.0 if np.isfinite(jitter_ddp) else float("nan"),
        "shimmer_local_percent": shimmer_local * 100.0 if np.isfinite(shimmer_local) else float("nan"),
        "shimmer_local_db": shimmer_local_db,
        "shimmer_apq3_percent": shimmer_apq3 * 100.0 if np.isfinite(shimmer_apq3) else float("nan"),
        "shimmer_apq5_percent": shimmer_apq5 * 100.0 if np.isfinite(shimmer_apq5) else float("nan"),
        "shimmer_apq11_percent": shimmer_apq11 * 100.0 if np.isfinite(shimmer_apq11) else float("nan"),
        "hnr_db": hnr,
        "num_periods": num_periods,
    }


def hnr_praat(
    x: torch.Tensor | str | Path,
    fs: int | None = None,
    *,
    pitch_floor: float = 75.0,
    time_step: float = 0.01,
    silence_threshold: float = 0.1,
    periods_per_window: float = 1.0,
) -> float:
    """
    Extract harmonics-to-noise ratio (HNR) using Praat's autocorrelation method.

    Args:
        x: Either a waveform tensor (requires fs) or a file path.
        fs: Sample rate in Hz. Required if x is a tensor.
        pitch_floor: Minimum F0 in Hz.
        time_step: Time step in seconds.
        silence_threshold: Frames with amplitude < this * max are set to undefined.
        periods_per_window: Number of periods per analysis window.

    Returns:
        Mean HNR in dB.
    """
    try:
        import parselmouth  # type: ignore
        from parselmouth.praat import call  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "praat-parselmouth is required for Praat HNR extraction. "
            "Install with `pip install \"audiofeat[validation]\"`."
        ) from exc

    if isinstance(x, (str, Path)):
        sound = parselmouth.Sound(str(x))
    else:
        if fs is None:
            raise ValueError("Sample rate (fs) is required when input is a tensor.")
        x = x.flatten().float()
        if x.numel() == 0:
            raise ValueError("Input waveform must be non-empty.")
        waveform_np = x.detach().cpu().numpy().astype(np.float64, copy=False)
        sound = parselmouth.Sound(waveform_np, sampling_frequency=float(fs))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        harmonicity = call(
            sound,
            "To Harmonicity (cc)",
            float(time_step),
            float(pitch_floor),
            float(silence_threshold),
            float(periods_per_window),
        )
        hnr = call(harmonicity, "Get mean", 0.0, 0.0)

    return float(hnr) if hnr is not None and np.isfinite(hnr) else float("nan")
