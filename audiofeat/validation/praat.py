from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import numpy as np
import torch

from ..io.features import load_audio
from ..pitch.f0 import fundamental_frequency_autocorr, fundamental_frequency_yin
from ..pitch.pyin import fundamental_frequency_pyin
from ..spectral.formants import formant_frequencies


DEFAULT_TIME_STEP_SEC = 0.01

SPEAKER_PROFILES = {
    "male": {"pitch_floor": 75.0, "pitch_ceiling": 300.0, "max_formant": 5000.0},
    "female": {"pitch_floor": 100.0, "pitch_ceiling": 500.0, "max_formant": 5500.0},
    "child": {"pitch_floor": 150.0, "pitch_ceiling": 800.0, "max_formant": 8000.0},
    "neutral": {"pitch_floor": 75.0, "pitch_ceiling": 300.0, "max_formant": 5500.0},
}

DEFAULT_TOLERANCE_PRESETS = {
    "male": {"pitch_mean": 0.15, "pitch_median": 0.15, "formant_f1": 0.20, "formant_f2": 0.35},
    "female": {"pitch_mean": 0.15, "pitch_median": 0.15, "formant_f1": 0.20, "formant_f2": 0.40},
    "child": {"pitch_mean": 0.20, "pitch_median": 0.20, "formant_f1": 0.25, "formant_f2": 0.45},
    "neutral": {"pitch_mean": 0.15, "pitch_median": 0.15, "formant_f1": 0.20, "formant_f2": 0.40},
}


def _to_float(value: float | int | None) -> float:
    if value is None:
        return float("nan")
    return float(value)


def _resolve_profile(profile: str | None) -> str:
    if profile is None:
        return "neutral"
    normalized = profile.lower().strip()
    if normalized not in SPEAKER_PROFILES:
        raise ValueError(
            f"Unsupported speaker profile: {profile}. "
            f"Choose one of {sorted(SPEAKER_PROFILES)}."
        )
    return normalized


def apply_speaker_profile(
    *,
    speaker_profile: str = "neutral",
    pitch_floor: float | None = None,
    pitch_ceiling: float | None = None,
    max_formant: float | None = None,
) -> dict[str, float]:
    profile_key = _resolve_profile(speaker_profile)
    defaults = SPEAKER_PROFILES[profile_key]
    return {
        "pitch_floor": float(defaults["pitch_floor"] if pitch_floor is None else pitch_floor),
        "pitch_ceiling": float(defaults["pitch_ceiling"] if pitch_ceiling is None else pitch_ceiling),
        "max_formant": float(defaults["max_formant"] if max_formant is None else max_formant),
    }


def relative_error(value: float, reference: float) -> float:
    if not np.isfinite(value) or not np.isfinite(reference):
        return float("nan")
    denom = max(abs(reference), 1e-8)
    return float(abs(value - reference) / denom)


def load_praat_reference(reference_json: str | Path) -> dict:
    with Path(reference_json).open("r") as f:
        data = json.load(f)
    if "praat" in data:
        return data["praat"]
    return data


def save_json(data: Mapping, output_path: str | Path) -> Path:
    def _sanitize(obj):
        if isinstance(obj, dict):
            return {k: _sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_sanitize(v) for v in obj]
        if isinstance(obj, tuple):
            return tuple(_sanitize(v) for v in obj)
        if isinstance(obj, (float, np.floating)):
            value = float(obj)
            if not np.isfinite(value):
                return None
            return value
        return obj

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(_sanitize(dict(data)), f, indent=2, allow_nan=False)
    return path


def _audiofeat_pitch_stats(
    waveform: torch.Tensor,
    sample_rate: int,
    *,
    frame_length: int,
    hop_length: int,
    pitch_floor: float,
    pitch_ceiling: float,
    method: str = "autocorr",
    yin_threshold: float = 0.1,
) -> dict[str, float]:
    if method == "autocorr":
        f0 = fundamental_frequency_autocorr(
            waveform,
            fs=sample_rate,
            frame_length=frame_length,
            hop_length=hop_length,
            fmin=pitch_floor,
            fmax=pitch_ceiling,
        )
    elif method == "yin":
        f0 = fundamental_frequency_yin(
            waveform,
            fs=sample_rate,
            frame_length=frame_length,
            hop_length=hop_length,
            fmin=pitch_floor,
            fmax=pitch_ceiling,
            threshold=yin_threshold,
        )
    elif method == "pyin":
        f0 = fundamental_frequency_pyin(
            waveform,
            fs=sample_rate,
            frame_length=frame_length,
            hop_length=hop_length,
            fmin=pitch_floor,
            fmax=pitch_ceiling,
            fill_unvoiced=0.0,
        )
    elif method == "praat":
        try:
            from ..pitch.pitch_praat import fundamental_frequency_praat
        except ImportError as exc:
            raise ImportError(
                "praat-parselmouth is required for pitch_method='praat'. "
                "Install with `pip install \"audiofeat[validation]\"`."
            ) from exc
        f0 = fundamental_frequency_praat(
            waveform,
            fs=sample_rate,
            pitch_floor=pitch_floor,
            pitch_ceiling=pitch_ceiling,
            fill_unvoiced=0.0,
        )
    else:
        raise ValueError("pitch method must be one of {'autocorr', 'yin', 'pyin', 'praat'}.")

    voiced = f0[f0 > 0]
    if voiced.numel() == 0:
        return {
            "mean_hz": float("nan"),
            "median_hz": float("nan"),
            "voiced_ratio": 0.0,
        }
    return {
        "mean_hz": float(voiced.mean().item()),
        "median_hz": float(voiced.median().item()),
        "voiced_ratio": float((f0 > 0).float().mean().item()),
    }


def _audiofeat_formant_stats(
    waveform: torch.Tensor,
    sample_rate: int,
    *,
    formant_order: int | None,
    num_formants: int,
    max_formant: float,
    formant_method: str,
    formant_window_length_sec: float,
    formant_time_step_sec: float,
    pre_emphasis_from_hz: float,
) -> dict[str, float]:
    formants = formant_frequencies(
        waveform,
        fs=sample_rate,
        order=formant_order,
        num_formants=num_formants,
        max_formant=max_formant,
        frame_length_ms=1000.0 * formant_window_length_sec,
        hop_length_ms=1000.0 * formant_time_step_sec,
        pre_emphasis=pre_emphasis_from_hz,
        method=formant_method,
    )
    result: dict[str, float] = {}
    for idx in range(min(formants.numel(), num_formants)):
        result[f"f{idx + 1}_median_hz"] = float(formants[idx].item())
    result.setdefault("f1_median_hz", float("nan"))
    result.setdefault("f2_median_hz", float("nan"))
    return result


def compute_audiofeat_reference(
    waveform: torch.Tensor,
    sample_rate: int,
    *,
    frame_length: int,
    hop_length: int,
    speaker_profile: str = "neutral",
    pitch_floor: float | None = None,
    pitch_ceiling: float | None = None,
    pitch_method: str = "autocorr",
    yin_threshold: float = 0.1,
    formant_order: int | None = None,
    num_formants: int = 5,
    max_formant: float | None = None,
    formant_method: str = "burg",
    formant_window_length_sec: float = 0.025,
    formant_time_step_sec: float = DEFAULT_TIME_STEP_SEC,
    pre_emphasis_from_hz: float = 50.0,
) -> dict[str, dict[str, float]]:
    settings = apply_speaker_profile(
        speaker_profile=speaker_profile,
        pitch_floor=pitch_floor,
        pitch_ceiling=pitch_ceiling,
        max_formant=max_formant,
    )
    if formant_order is None:
        formant_order = max(10, 2 * num_formants + 2)

    return {
        "pitch": _audiofeat_pitch_stats(
            waveform,
            sample_rate,
            frame_length=frame_length,
            hop_length=hop_length,
            pitch_floor=settings["pitch_floor"],
            pitch_ceiling=settings["pitch_ceiling"],
            method=pitch_method,
            yin_threshold=yin_threshold,
        ),
        "formants": _audiofeat_formant_stats(
            waveform,
            sample_rate,
            formant_order=formant_order,
            num_formants=num_formants,
            max_formant=settings["max_formant"],
            formant_method=formant_method,
            formant_window_length_sec=formant_window_length_sec,
            formant_time_step_sec=formant_time_step_sec,
            pre_emphasis_from_hz=pre_emphasis_from_hz,
        ),
    }


def extract_praat_reference(
    audio_path: str | Path,
    *,
    speaker_profile: str = "neutral",
    pitch_floor: float | None = None,
    pitch_ceiling: float | None = None,
    num_formants: int = 5,
    max_formant: float | None = None,
    time_step: float = DEFAULT_TIME_STEP_SEC,
    formant_window_length: float = 0.025,
    pre_emphasis_from_hz: float = 50.0,
) -> dict[str, dict[str, float]]:
    """
    Extract Praat reference values directly using parselmouth.
    """
    try:
        import parselmouth  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        raise ModuleNotFoundError(
            "praat-parselmouth is required for direct Praat extraction. "
            "Install with `pip install \"audiofeat[validation]\"`."
        ) from exc

    profile = apply_speaker_profile(
        speaker_profile=speaker_profile,
        pitch_floor=pitch_floor,
        pitch_ceiling=pitch_ceiling,
        max_formant=max_formant,
    )

    sound = parselmouth.Sound(str(audio_path))
    pitch = sound.to_pitch(
        time_step=float(time_step),
        pitch_floor=float(profile["pitch_floor"]),
        pitch_ceiling=float(profile["pitch_ceiling"]),
    )
    pitch_values = np.asarray(pitch.selected_array["frequency"], dtype=np.float64)
    voiced = pitch_values[pitch_values > 0]
    pitch_stats = {
        "mean_hz": float(np.mean(voiced)) if voiced.size > 0 else float("nan"),
        "median_hz": float(np.median(voiced)) if voiced.size > 0 else float("nan"),
    }

    formant = sound.to_formant_burg(
        time_step=float(time_step),
        max_number_of_formants=float(num_formants),
        maximum_formant=float(profile["max_formant"]),
        window_length=float(formant_window_length),
        pre_emphasis_from=float(pre_emphasis_from_hz),
    )
    times = np.arange(0.0, float(sound.duration), float(time_step), dtype=np.float64)
    formant_stats: dict[str, float] = {}
    for idx in range(1, num_formants + 1):
        values = []
        for t in times:
            val = formant.get_value_at_time(idx, float(t))
            if val is None:
                continue
            val = float(val)
            if np.isfinite(val) and val > 0:
                values.append(val)
        formant_stats[f"f{idx}_median_hz"] = float(np.median(values)) if values else float("nan")

    return {"pitch": pitch_stats, "formants": formant_stats}


def build_praat_comparison_report(
    waveform: torch.Tensor,
    sample_rate: int,
    praat_reference: Mapping[str, Mapping[str, float]],
    *,
    frame_length: int,
    hop_length: int,
    speaker_profile: str = "neutral",
    pitch_floor: float | None = None,
    pitch_ceiling: float | None = None,
    pitch_method: str = "autocorr",
    yin_threshold: float = 0.1,
    formant_order: int | None = None,
    num_formants: int = 5,
    max_formant: float | None = None,
    formant_method: str = "burg",
    formant_window_length_sec: float = 0.025,
    formant_time_step_sec: float = DEFAULT_TIME_STEP_SEC,
    pre_emphasis_from_hz: float = 50.0,
) -> dict:
    profile = apply_speaker_profile(
        speaker_profile=speaker_profile,
        pitch_floor=pitch_floor,
        pitch_ceiling=pitch_ceiling,
        max_formant=max_formant,
    )
    if formant_order is None:
        formant_order = max(10, 2 * num_formants + 2)

    praat_pitch = dict(praat_reference.get("pitch", {}))
    praat_formants = dict(praat_reference.get("formants", {}))
    audiofeat_reference = compute_audiofeat_reference(
        waveform,
        sample_rate,
        frame_length=frame_length,
        hop_length=hop_length,
        speaker_profile=speaker_profile,
        pitch_floor=profile["pitch_floor"],
        pitch_ceiling=profile["pitch_ceiling"],
        pitch_method=pitch_method,
        yin_threshold=yin_threshold,
        formant_order=formant_order,
        num_formants=num_formants,
        max_formant=profile["max_formant"],
        formant_method=formant_method,
        formant_window_length_sec=formant_window_length_sec,
        formant_time_step_sec=formant_time_step_sec,
        pre_emphasis_from_hz=pre_emphasis_from_hz,
    )
    audiofeat_pitch = audiofeat_reference["pitch"]
    audiofeat_formants = audiofeat_reference["formants"]

    return {
        "settings": {
            "sample_rate": sample_rate,
            "frame_length": frame_length,
            "hop_length": hop_length,
            "speaker_profile": _resolve_profile(speaker_profile),
            "pitch_floor": profile["pitch_floor"],
            "pitch_ceiling": profile["pitch_ceiling"],
            "pitch_method": pitch_method,
            "yin_threshold": yin_threshold,
            "formant_order": formant_order,
            "num_formants": num_formants,
            "max_formant": profile["max_formant"],
            "formant_method": formant_method,
            "formant_window_length_sec": formant_window_length_sec,
            "formant_time_step_sec": formant_time_step_sec,
            "pre_emphasis_from_hz": pre_emphasis_from_hz,
        },
        "praat": {"pitch": praat_pitch, "formants": praat_formants},
        "audiofeat": {"pitch": audiofeat_pitch, "formants": audiofeat_formants},
        "relative_error": {
            "pitch_mean": relative_error(
                _to_float(audiofeat_pitch.get("mean_hz")),
                _to_float(praat_pitch.get("mean_hz")),
            ),
            "pitch_median": relative_error(
                _to_float(audiofeat_pitch.get("median_hz")),
                _to_float(praat_pitch.get("median_hz")),
            ),
            "formant_f1": relative_error(
                _to_float(audiofeat_formants.get("f1_median_hz")),
                _to_float(praat_formants.get("f1_median_hz")),
            ),
            "formant_f2": relative_error(
                _to_float(audiofeat_formants.get("f2_median_hz")),
                _to_float(praat_formants.get("f2_median_hz")),
            ),
        },
    }


def compare_audio_to_praat_reference(
    audio_path: str | Path,
    praat_reference: Mapping[str, Mapping[str, float]],
    *,
    sample_rate: int = 22050,
    frame_length: int | None = None,
    hop_length: int | None = None,
    speaker_profile: str = "neutral",
    pitch_floor: float | None = None,
    pitch_ceiling: float | None = None,
    pitch_method: str = "autocorr",
    yin_threshold: float = 0.1,
    formant_order: int | None = None,
    num_formants: int = 5,
    max_formant: float | None = None,
    formant_method: str = "burg",
    time_step_sec: float = DEFAULT_TIME_STEP_SEC,
    formant_window_length_sec: float = 0.025,
    pre_emphasis_from_hz: float = 50.0,
) -> dict:
    profile = apply_speaker_profile(
        speaker_profile=speaker_profile,
        pitch_floor=pitch_floor,
        pitch_ceiling=pitch_ceiling,
        max_formant=max_formant,
    )
    waveform, sample_rate = load_audio(audio_path, target_sample_rate=sample_rate, mono=True)

    if frame_length is None:
        frame_length = max(32, int(3.0 * sample_rate / profile["pitch_floor"]))
    if hop_length is None:
        hop_length = max(16, int(time_step_sec * sample_rate))

    return build_praat_comparison_report(
        waveform,
        sample_rate,
        praat_reference,
        frame_length=frame_length,
        hop_length=hop_length,
        speaker_profile=speaker_profile,
        pitch_floor=profile["pitch_floor"],
        pitch_ceiling=profile["pitch_ceiling"],
        pitch_method=pitch_method,
        yin_threshold=yin_threshold,
        formant_order=formant_order,
        num_formants=num_formants,
        max_formant=profile["max_formant"],
        formant_method=formant_method,
        formant_window_length_sec=formant_window_length_sec,
        formant_time_step_sec=time_step_sec,
        pre_emphasis_from_hz=pre_emphasis_from_hz,
    )


def evaluate_praat_report(
    report: Mapping,
    tolerances: Mapping[str, float] | None = None,
) -> dict[str, object]:
    settings = dict(report.get("settings", {}))
    profile = _resolve_profile(str(settings.get("speaker_profile", "neutral")))
    thresholds = dict(DEFAULT_TOLERANCE_PRESETS[profile])
    if tolerances:
        thresholds.update({k: float(v) for k, v in tolerances.items()})

    rel = dict(report.get("relative_error", {}))
    praat_pitch = dict(dict(report.get("praat", {})).get("pitch", {}))
    praat_formants = dict(dict(report.get("praat", {})).get("formants", {}))
    audio_pitch = dict(dict(report.get("audiofeat", {})).get("pitch", {}))
    audio_formants = dict(dict(report.get("audiofeat", {})).get("formants", {}))

    availability_pairs = {
        "pitch_mean": (audio_pitch.get("mean_hz"), praat_pitch.get("mean_hz")),
        "pitch_median": (audio_pitch.get("median_hz"), praat_pitch.get("median_hz")),
        "formant_f1": (audio_formants.get("f1_median_hz"), praat_formants.get("f1_median_hz")),
        "formant_f2": (audio_formants.get("f2_median_hz"), praat_formants.get("f2_median_hz")),
    }
    checks = {}
    passed = True
    for key, tol in thresholds.items():
        value = _to_float(rel.get(key))
        if not np.isfinite(value):
            a_val, p_val = availability_pairs.get(key, (float("nan"), float("nan")))
            if not np.isfinite(_to_float(a_val)) and not np.isfinite(_to_float(p_val)):
                checks[key] = {
                    "value": value,
                    "tolerance": tol,
                    "passed": True,
                    "skipped": True,
                    "reason": "metric unavailable in both Praat and audiofeat outputs",
                }
            else:
                checks[key] = {"value": value, "tolerance": tol, "passed": False}
                passed = False
            continue
        ok = value <= tol
        checks[key] = {"value": value, "tolerance": tol, "passed": ok}
        passed = passed and ok

    return {"passed": passed, "checks": checks}
