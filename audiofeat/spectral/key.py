from __future__ import annotations

import torch

from .chroma import chroma


_KEYS = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
_MAJOR_TEMPLATE = torch.tensor(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
    dtype=torch.float32,
)
_MINOR_TEMPLATE = torch.tensor(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17],
    dtype=torch.float32,
)


def _zscore(x: torch.Tensor) -> torch.Tensor:
    x = x - x.mean()
    std = x.std(unbiased=False)
    return x / (std + 1e-12)


def key_detect(
    waveform: torch.Tensor,
    sample_rate: int,
    n_fft: int = 4096,
    hop_length: int = 2048,
    n_chroma: int = 12,
) -> str:
    """
    Detect key using Krumhansl-Schmuckler template matching on chroma features.
    """
    if sample_rate <= 0:
        raise ValueError("sample_rate must be > 0.")
    if n_chroma != 12:
        raise ValueError("n_chroma must be 12 for key detection.")

    chroma_feat = chroma(
        waveform,
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_chroma=n_chroma,
    )
    if chroma_feat.numel() == 0:
        return "Unknown"

    chroma_vec = _zscore(chroma_feat.mean(dim=1))
    major_tpl = _zscore(_MAJOR_TEMPLATE.to(chroma_vec.device))
    minor_tpl = _zscore(_MINOR_TEMPLATE.to(chroma_vec.device))

    best_score = float("-inf")
    best_key = "Unknown"
    for idx, key_name in enumerate(_KEYS):
        rotated = torch.roll(chroma_vec, shifts=-idx, dims=0)
        major_score = torch.dot(rotated, major_tpl) / len(rotated)
        if major_score.item() > best_score:
            best_score = major_score.item()
            best_key = f"{key_name} major"

        minor_score = torch.dot(rotated, minor_tpl) / len(rotated)
        if minor_score.item() > best_score:
            best_score = minor_score.item()
            best_key = f"{key_name} minor"

    return best_key
