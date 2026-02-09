from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

import torch
import torchaudio

from ..pitch.f0 import fundamental_frequency_yin
from ..pitch.strength import pitch_strength
from ..spectral.centroid import spectral_centroid
from ..spectral.mfcc import mfcc
from ..spectral.rolloff import spectral_rolloff
from ..stats.functionals import compute_functionals
from ..temporal.rms import rms
from ..temporal.zcr import zero_crossing_rate


DEFAULT_SAMPLE_RATE = 22050
DEFAULT_FRAME_LENGTH = 2048
DEFAULT_HOP_LENGTH = 512
SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}


def _looks_like_text_placeholder(path: Path) -> bool:
    try:
        raw = path.read_bytes()[:256]
    except OSError:
        return False
    prefixes = (
        b"404:",
        b"File not found:",
        b"version https://git-lfs.github.com/spec/v1",
    )
    return any(raw.startswith(prefix) for prefix in prefixes)


def to_mono(waveform: torch.Tensor) -> torch.Tensor:
    """Convert a loaded waveform to mono and return shape `(num_samples,)`."""
    if waveform.dim() == 1:
        return waveform
    if waveform.dim() != 2:
        raise ValueError("Expected waveform with shape (channels, samples) or (samples,).")
    if waveform.size(0) == 1:
        return waveform.squeeze(0)
    return waveform.mean(dim=0)


def resample_if_needed(
    waveform: torch.Tensor,
    sample_rate: int,
    target_sample_rate: int,
) -> torch.Tensor:
    if sample_rate == target_sample_rate:
        return waveform
    return torchaudio.functional.resample(waveform, sample_rate, target_sample_rate)


def load_audio(
    audio_path: str | Path,
    *,
    target_sample_rate: int | None = DEFAULT_SAMPLE_RATE,
    mono: bool = True,
) -> tuple[torch.Tensor, int]:
    """Load audio from disk with optional mono conversion and resampling."""
    path = Path(audio_path)
    try:
        waveform, sample_rate = torchaudio.load(str(path))
    except Exception as exc:  # pragma: no cover - backend-specific errors
        extra = ""
        if path.exists() and _looks_like_text_placeholder(path):
            extra = (
                " The file appears to be a text placeholder (for example a 404 page "
                "or a missing Git LFS object), not valid audio."
            )
        raise RuntimeError(f"Failed to decode audio file '{path}'.{extra}") from exc

    if mono:
        waveform = to_mono(waveform)
    if target_sample_rate is not None and sample_rate != target_sample_rate:
        waveform = resample_if_needed(waveform, sample_rate, target_sample_rate)
        sample_rate = target_sample_rate
    return waveform, sample_rate


def summarize_series(prefix: str, series: torch.Tensor) -> dict[str, float]:
    series = series.reshape(-1, 1)
    stats = compute_functionals(series).detach().cpu()
    keys = ["mean", "std", "min", "max", "skewness", "kurtosis"]
    return {f"{prefix}_{k}": float(stats[i].item()) for i, k in enumerate(keys)}


def summarize_matrix(prefix: str, matrix: torch.Tensor) -> dict[str, float]:
    if matrix.dim() == 1:
        matrix = matrix.unsqueeze(0)
    mean = matrix.mean(dim=1).detach().cpu()
    std = matrix.std(dim=1, unbiased=False).detach().cpu()
    out: dict[str, float] = {}
    for idx in range(mean.numel()):
        out[f"{prefix}_mean_{idx}"] = float(mean[idx].item())
        out[f"{prefix}_std_{idx}"] = float(std[idx].item())
    return out


def extract_core_features(
    waveform: torch.Tensor,
    sample_rate: int,
    *,
    frame_length: int = DEFAULT_FRAME_LENGTH,
    hop_length: int = DEFAULT_HOP_LENGTH,
) -> dict[str, float]:
    """Extract a compact, production-friendly feature summary."""
    waveform = waveform.flatten().float()

    rms_series = rms(waveform, frame_length=frame_length, hop_length=hop_length)
    zcr_series = zero_crossing_rate(waveform, frame_length=frame_length, hop_length=hop_length)
    centroid_series = spectral_centroid(
        waveform,
        frame_length=frame_length,
        hop_length=hop_length,
        sample_rate=sample_rate,
    )
    rolloff_series = spectral_rolloff(
        waveform,
        frame_length=frame_length,
        hop_length=hop_length,
        sample_rate=sample_rate,
    )
    f0_yin = fundamental_frequency_yin(
        waveform,
        fs=sample_rate,
        frame_length=frame_length,
        hop_length=hop_length,
    )
    strength_series = pitch_strength(
        waveform,
        fs=sample_rate,
        frame_length=frame_length,
        hop_length=hop_length,
    )
    mfcc_matrix = mfcc(
        waveform,
        sample_rate,
        n_fft=frame_length,
        hop_length=hop_length,
    )

    features: dict[str, float] = {
        "sample_rate": int(sample_rate),
        "num_samples": int(waveform.numel()),
        "duration_sec": float(waveform.numel() / max(sample_rate, 1)),
    }

    features.update(summarize_series("rms", rms_series))
    features.update(summarize_series("zcr", zcr_series))
    features.update(summarize_series("spectral_centroid", centroid_series))
    features.update(summarize_series("spectral_rolloff", rolloff_series))
    features.update(summarize_series("pitch_strength", strength_series))

    voiced = f0_yin[f0_yin > 0]
    features["f0_voiced_ratio"] = float((f0_yin > 0).float().mean().item())
    if voiced.numel() > 0:
        features["f0_mean_hz"] = float(voiced.mean().item())
        features["f0_median_hz"] = float(voiced.median().item())
    else:
        features["f0_mean_hz"] = float("nan")
        features["f0_median_hz"] = float("nan")

    features.update(summarize_matrix("mfcc", mfcc_matrix))
    return features


def extract_features_from_file(
    audio_path: str | Path,
    *,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    frame_length: int = DEFAULT_FRAME_LENGTH,
    hop_length: int = DEFAULT_HOP_LENGTH,
) -> dict[str, float | str]:
    waveform, sr = load_audio(audio_path, target_sample_rate=sample_rate)
    features = extract_core_features(
        waveform,
        sr,
        frame_length=frame_length,
        hop_length=hop_length,
    )
    features["path"] = str(audio_path)
    return features


def iter_audio_files(
    input_dir: str | Path,
    *,
    extensions: Iterable[str] = SUPPORTED_AUDIO_EXTENSIONS,
) -> list[Path]:
    ext_set = {ext.lower() for ext in extensions}
    root = Path(input_dir)
    return sorted(
        path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in ext_set
    )


def extract_features_for_directory(
    input_dir: str | Path,
    *,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    frame_length: int = DEFAULT_FRAME_LENGTH,
    hop_length: int = DEFAULT_HOP_LENGTH,
    skip_errors: bool = True,
    errors: list[str] | None = None,
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for path in iter_audio_files(input_dir):
        try:
            rows.append(
                extract_features_from_file(
                    path,
                    sample_rate=sample_rate,
                    frame_length=frame_length,
                    hop_length=hop_length,
                )
            )
        except Exception as exc:
            if errors is not None:
                errors.append(f"{path}: {exc}")
            if not skip_errors:
                raise
    return rows


def write_feature_rows_to_csv(
    rows: list[dict[str, float | str]],
    output_csv: str | Path,
) -> Path:
    if not rows:
        raise ValueError("No rows to write.")
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output_path
