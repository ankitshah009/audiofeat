from __future__ import annotations

import platform
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchaudio

from ..pitch.f0 import fundamental_frequency_autocorr, fundamental_frequency_yin
from ..pitch.pyin import fundamental_frequency_pyin
from ..spectral.centroid import spectral_centroid
from ..spectral.formants import formant_frequencies
from ..spectral.rolloff import spectral_rolloff
from ..temporal.rms import hann_window, rms
from ..temporal.zcr import zero_crossing_rate
from .praat import compare_audio_to_praat_reference, evaluate_praat_report, extract_praat_reference


def _relative_error(value: float, reference: float) -> float:
    denom = max(abs(reference), 1e-8)
    return float(abs(value - reference) / denom)


def _sine_wave(
    *,
    sample_rate: int,
    frequency_hz: float,
    duration_sec: float,
    amplitude: float = 0.8,
) -> torch.Tensor:
    t = torch.arange(int(sample_rate * duration_sec), dtype=torch.float32) / sample_rate
    return amplitude * torch.sin(2 * torch.pi * float(frequency_hz) * t)


def _speech_like_signal(sample_rate: int, duration_sec: float = 1.8) -> torch.Tensor:
    t = torch.arange(int(sample_rate * duration_sec), dtype=torch.float32) / sample_rate
    source = torch.zeros_like(t)
    f0 = 120.0
    for k in range(1, 15):
        source += torch.sin(2 * torch.pi * (k * f0) * t) / k

    x = source.detach().cpu().numpy().astype(np.float64, copy=False)
    for center_hz, q in ((500.0, 3.0), (1500.0, 4.0), (2500.0, 5.5)):
        b, a = _iirpeak(center_hz / (sample_rate / 2.0), q)
        x = _lfilter(b, a, x)
    return torch.from_numpy(x.astype(np.float32))


def _iirpeak(w0: float, q: float) -> tuple[np.ndarray, np.ndarray]:
    from scipy.signal import iirpeak

    return iirpeak(float(w0), float(q))


def _lfilter(b: np.ndarray, a: np.ndarray, x: np.ndarray) -> np.ndarray:
    from scipy.signal import lfilter

    return lfilter(b, a, x)


def _is_module_available(module_name: str) -> bool:
    try:
        __import__(module_name)
        return True
    except Exception:
        return False


def _check_result(
    *,
    name: str,
    max_points: float,
    passed: bool,
    details: dict[str, Any],
    skipped: bool = False,
    reason: str | None = None,
) -> dict[str, Any]:
    result = {
        "name": name,
        "passed": bool(passed),
        "skipped": bool(skipped),
        "max_points": float(max_points),
        "earned_points": float(max_points if passed and not skipped else 0.0),
        "details": details,
    }
    if reason:
        result["reason"] = reason
    return result


def run_gold_standard_scorecard(
    *,
    sample_rate: int = 22050,
    frame_length: int = 1024,
    hop_length: int = 256,
    praat_audio_path: str | Path | None = None,
    include_optional: bool = True,
) -> dict[str, Any]:
    """
    Run reproducible quality checks and produce a score normalized to 100.

    The score is based on deterministic signal-theory checks, pitch/formant checks,
    and optional parity checks with external references (librosa, Praat/parselmouth).
    """
    torch.manual_seed(0)
    np.random.seed(0)

    if sample_rate <= 0:
        raise ValueError("sample_rate must be > 0.")
    if frame_length <= 0 or hop_length <= 0:
        raise ValueError("frame_length and hop_length must be > 0.")

    checks: list[dict[str, Any]] = []

    # 1) RMS theoretical sanity on a constant signal.
    ones = torch.ones(sample_rate, dtype=torch.float32)
    measured_rms = float(rms(ones, frame_length=frame_length, hop_length=hop_length).mean().item())
    expected_rms = 1.0  # constant signal of ones: sqrt(sum(w^2 * 1^2) / sum(w^2)) = 1
    rms_rel_err = _relative_error(measured_rms, expected_rms)
    checks.append(
        _check_result(
            name="rms_theoretical_consistency",
            max_points=10.0,
            passed=rms_rel_err <= 0.01,
            details={
                "measured_rms": measured_rms,
                "expected_rms": expected_rms,
                "relative_error": rms_rel_err,
                "threshold": 0.01,
            },
        )
    )

    # 2) ZCR sanity on a pure tone.
    tone_freq = float(sample_rate * 10 / frame_length)
    sine_220 = _sine_wave(sample_rate=sample_rate, frequency_hz=tone_freq, duration_sec=2.0)
    measured_zcr = float(
        zero_crossing_rate(
            sine_220,
            frame_length=frame_length,
            hop_length=hop_length,
        ).median().item()
    )
    expected_zcr = float((2.0 * tone_freq) / float(sample_rate))
    zcr_abs_err = abs(measured_zcr - expected_zcr)
    checks.append(
        _check_result(
            name="zcr_theoretical_consistency",
            max_points=10.0,
            passed=zcr_abs_err <= 0.005,
            details={
                "measured_zcr": measured_zcr,
                "expected_zcr": expected_zcr,
                "absolute_error": zcr_abs_err,
                "threshold": 0.005,
            },
        )
    )

    # 3) Spectral centroid sanity.
    centroid = spectral_centroid(
        sine_220,
        frame_length=frame_length,
        hop_length=hop_length,
        sample_rate=sample_rate,
    )
    measured_centroid = float(centroid.median().item())
    centroid_abs_err = abs(measured_centroid - tone_freq)
    checks.append(
        _check_result(
            name="spectral_centroid_tone_consistency",
            max_points=8.0,
            passed=centroid_abs_err <= 30.0,
            details={
                "measured_hz": measured_centroid,
                "expected_hz": tone_freq,
                "absolute_error_hz": centroid_abs_err,
                "threshold_hz": 60.0,
            },
        )
    )

    # 4) Spectral rolloff sanity.
    rolloff = spectral_rolloff(
        sine_220,
        frame_length=frame_length,
        hop_length=hop_length,
        rolloff_percent=0.85,
        sample_rate=sample_rate,
    )
    measured_rolloff = float(rolloff.median().item())
    rolloff_abs_err = abs(measured_rolloff - tone_freq)
    checks.append(
        _check_result(
            name="spectral_rolloff_tone_consistency",
            max_points=7.0,
            passed=rolloff_abs_err <= 90.0,
            details={
                "measured_hz": measured_rolloff,
                "expected_hz": tone_freq,
                "absolute_error_hz": rolloff_abs_err,
                "threshold_hz": 90.0,
            },
        )
    )

    # 5) Pitch detection checks.
    for method_name, max_points, estimator in (
        ("pitch_autocorr_tone_accuracy", 10.0, "autocorr"),
        ("pitch_yin_tone_accuracy", 10.0, "yin"),
    ):
        if estimator == "autocorr":
            f0 = fundamental_frequency_autocorr(
                sine_220,
                fs=sample_rate,
                frame_length=frame_length,
                hop_length=hop_length,
                fmin=max(50, int(tone_freq * 0.5)),
                fmax=max(200, int(tone_freq * 2.0)),
            )
        else:
            f0 = fundamental_frequency_yin(
                sine_220,
                fs=sample_rate,
                frame_length=frame_length,
                hop_length=hop_length,
                fmin=max(50, int(tone_freq * 0.5)),
                fmax=max(200, int(tone_freq * 2.0)),
            )
        measured = float(f0.median().item())
        rel_err = _relative_error(measured, tone_freq)
        checks.append(
            _check_result(
                name=method_name,
                max_points=max_points,
                passed=rel_err <= 0.05,
                details={
                    "measured_hz": measured,
                    "expected_hz": tone_freq,
                    "relative_error": rel_err,
                    "threshold": 0.05,
                },
            )
        )

    # 6) Formant sanity on a deterministic speech-like signal.
    speech_like = _speech_like_signal(sample_rate=sample_rate)
    formants = formant_frequencies(
        speech_like,
        fs=sample_rate,
        order=12,
        num_formants=3,
        max_formant=5000.0,
        method="burg",
    )
    f1, f2, f3 = [float(formants[i].item()) for i in range(3)]
    finite = all(np.isfinite(v) for v in (f1, f2, f3))
    ordered = f1 < f2 < f3
    plausible = 150.0 <= f1 <= 1200.0 and 500.0 <= f2 <= 3200.0
    checks.append(
        _check_result(
            name="formant_monotonicity_plausibility",
            max_points=10.0,
            passed=finite and ordered and plausible,
            details={
                "f1_hz": f1,
                "f2_hz": f2,
                "f3_hz": f3,
                "finite": finite,
                "ordered": ordered,
                "plausible": plausible,
            },
        )
    )

    librosa_available = _is_module_available("librosa")
    parselmouth_available = _is_module_available("parselmouth")

    # 7) Optional pYIN check.
    if include_optional and librosa_available:
        f0_pyin = fundamental_frequency_pyin(
            sine_220,
            fs=sample_rate,
            frame_length=frame_length,
            hop_length=hop_length,
            fmin=max(50.0, tone_freq * 0.5),
            fmax=max(200.0, tone_freq * 2.0),
            fill_unvoiced=float("nan"),
        )
        finite_values = f0_pyin[torch.isfinite(f0_pyin)]
        measured = float(finite_values.median().item()) if finite_values.numel() else float("nan")
        rel_err = _relative_error(measured, tone_freq) if np.isfinite(measured) else float("inf")
        checks.append(
            _check_result(
                name="pitch_pyin_tone_accuracy",
                max_points=10.0,
                passed=np.isfinite(measured) and rel_err <= 0.05,
                details={
                    "measured_hz": measured,
                    "expected_hz": tone_freq,
                    "relative_error": rel_err,
                    "threshold": 0.05,
                },
            )
        )
    else:
        checks.append(
            _check_result(
                name="pitch_pyin_tone_accuracy",
                max_points=10.0,
                passed=False,
                skipped=True,
                reason="librosa is not installed or optional checks disabled.",
                details={},
            )
        )

    # 8/9) Optional Praat alignment checks.
    if include_optional and parselmouth_available:
        pitch_method = "pyin" if librosa_available else "yin"
        cleanup_path = False
        if praat_audio_path is None:
            fd, temp_path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            audio_path = Path(temp_path)
            cleanup_path = True
            torchaudio.save(
                str(audio_path),
                speech_like.unsqueeze(0),
                sample_rate,
            )
        else:
            audio_path = Path(praat_audio_path)

        try:
            praat_reference = extract_praat_reference(
                audio_path,
                speaker_profile="neutral",
                num_formants=5,
                max_formant=5500.0,
                time_step=0.01,
                formant_window_length=0.025,
                pre_emphasis_from_hz=50.0,
            )
            native_report = compare_audio_to_praat_reference(
                audio_path,
                praat_reference,
                sample_rate=sample_rate,
                speaker_profile="neutral",
                pitch_method=pitch_method,
                formant_method="burg",
                time_step_sec=0.01,
                formant_window_length_sec=0.025,
                pre_emphasis_from_hz=50.0,
            )
            native_tolerances = {
                "pitch_mean": 0.15,
                "pitch_median": 0.15,
                # Native Burg formants are expected to be coarser than Praat.
                "formant_f1": 0.80,
                "formant_f2": 0.50,
            }
            native_eval = evaluate_praat_report(
                native_report,
                tolerances=native_tolerances,
            )
            checks.append(
                _check_result(
                    name="praat_native_alignment",
                    max_points=12.0,
                    passed=bool(native_eval["passed"]),
                    details={
                        "pitch_method": pitch_method,
                        "tolerances": native_tolerances,
                        "relative_error": native_report.get("relative_error", {}),
                        "tolerance_check": native_eval,
                    },
                )
            )

            praat_backend_report = compare_audio_to_praat_reference(
                audio_path,
                praat_reference,
                sample_rate=sample_rate,
                speaker_profile="neutral",
                pitch_method=pitch_method,
                formant_method="praat",
                time_step_sec=0.01,
                formant_window_length_sec=0.025,
                pre_emphasis_from_hz=50.0,
            )
            rel = dict(praat_backend_report.get("relative_error", {}))
            max_rel = max(
                float(rel.get("pitch_mean", 0.0)),
                float(rel.get("pitch_median", 0.0)),
                float(rel.get("formant_f1", 0.0)),
                float(rel.get("formant_f2", 0.0)),
            )
            checks.append(
                _check_result(
                    name="praat_backend_parity",
                    max_points=13.0,
                    passed=max_rel <= 0.03,
                    details={
                        "pitch_method": pitch_method,
                        "relative_error": rel,
                        "max_relative_error": max_rel,
                        "threshold": 0.03,
                    },
                )
            )
        finally:
            if cleanup_path:
                try:
                    audio_path.unlink()
                except Exception:
                    pass
    else:
        reason = "parselmouth is not installed or optional checks disabled."
        checks.append(
            _check_result(
                name="praat_native_alignment",
                max_points=12.0,
                passed=False,
                skipped=True,
                reason=reason,
                details={},
            )
        )
        checks.append(
            _check_result(
                name="praat_backend_parity",
                max_points=13.0,
                passed=False,
                skipped=True,
                reason=reason,
                details={},
            )
        )

    available_points = sum(c["max_points"] for c in checks if not c["skipped"])
    earned_points = sum(c["earned_points"] for c in checks if not c["skipped"])
    score = float(100.0 * earned_points / available_points) if available_points > 0 else 0.0

    passed_required = all(c["passed"] for c in checks if not c["skipped"])
    failed_checks = [c["name"] for c in checks if (not c["skipped"]) and (not c["passed"])]
    skipped_checks = [c["name"] for c in checks if c["skipped"]]

    return {
        "score": score,
        "passed": passed_required,
        "summary": {
            "available_points": available_points,
            "earned_points": earned_points,
            "total_checks": len(checks),
            "failed_checks": failed_checks,
            "skipped_checks": skipped_checks,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": str(torch.__version__),
            "torchaudio": str(torchaudio.__version__),
            "librosa_available": librosa_available,
            "parselmouth_available": parselmouth_available,
        },
        "checks": checks,
    }
