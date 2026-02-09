from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch


OpenSmileFeatureSet = Literal[
    "ComParE_2016",
    "GeMAPSv01a",
    "GeMAPSv01b",
    "eGeMAPSv01a",
    "eGeMAPSv01b",
    "eGeMAPSv02",
]
OpenSmileFeatureLevel = Literal["Functionals", "LowLevelDescriptors", "LowLevelDescriptors_Deltas"]


def _import_opensmile():
    try:
        import opensmile  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "opensmile is required for standardized feature sets. "
            "Install with `pip install \"audiofeat[standards]\"`."
        ) from exc
    return opensmile


def available_opensmile_feature_sets() -> list[str]:
    """Return available standard openSMILE feature sets."""
    opensmile = _import_opensmile()
    return sorted(
        name
        for name in dir(opensmile.FeatureSet)
        if not name.startswith("_")
    )


def available_opensmile_feature_levels() -> list[str]:
    """Return available standard openSMILE feature levels."""
    opensmile = _import_opensmile()
    return sorted(
        name
        for name in dir(opensmile.FeatureLevel)
        if not name.startswith("_")
    )


def extract_opensmile_features(
    audio: str | Path | torch.Tensor,
    *,
    sample_rate: int | None = None,
    feature_set: OpenSmileFeatureSet = "eGeMAPSv02",
    feature_level: OpenSmileFeatureLevel = "Functionals",
    flatten: bool = True,
):
    """
    Extract standardized openSMILE descriptors (e.g., eGeMAPSv02 or ComParE_2016).

    Args:
        audio:
            Path to an audio file or an in-memory waveform tensor.
        sample_rate:
            Required when `audio` is a tensor.
        feature_set:
            openSMILE feature-set enum name.
        feature_level:
            openSMILE feature-level enum name.
        flatten:
            When True, return first-row dict for easier JSON serialization.
            When False, return the native DataFrame-like output from openSMILE.
    """
    opensmile = _import_opensmile()
    if not hasattr(opensmile.FeatureSet, feature_set):
        raise ValueError(f"Unsupported openSMILE feature_set: {feature_set}")
    if not hasattr(opensmile.FeatureLevel, feature_level):
        raise ValueError(f"Unsupported openSMILE feature_level: {feature_level}")

    smile = opensmile.Smile(
        feature_set=getattr(opensmile.FeatureSet, feature_set),
        feature_level=getattr(opensmile.FeatureLevel, feature_level),
    )

    if isinstance(audio, (str, Path)):
        df = smile.process_file(str(audio))
    else:
        if sample_rate is None:
            raise ValueError("sample_rate is required when extracting from tensor input.")
        waveform = audio.flatten().detach().cpu().numpy()
        df = smile.process_signal(waveform, sample_rate)

    if not flatten:
        return df

    if getattr(df, "shape", (0, 0))[0] == 0:
        return {}

    # Pandas row -> dict[str, float], but avoid hard dependency in type hints.
    row = df.iloc[0]
    return {str(k): float(v) for k, v in row.to_dict().items()}
