
import math

import torch

def harmonic_richness_factor(magnitudes: torch.Tensor):
    """Harmonic richness factor given harmonic magnitudes starting at F0.

    Computes ``10 * log10(sum(|H_k|^2 for k>=2) / |H_1|^2)`` — the ratio of the
    energy in the upper harmonics to the energy of the fundamental.

    Returns:
        torch.Tensor: Scalar value in **decibels (dB)**.
    """
    if magnitudes.numel() < 2:
        return torch.tensor(0.0, device=magnitudes.device)
    numerator = magnitudes[1:].pow(2).sum()
    denominator = magnitudes[0].pow(2)
    return 10 * torch.log10(numerator / (denominator + 1e-8))

def inharmonicity_index(peaks: torch.Tensor, f0: float):
    """Inharmonicity from peak frequencies and fundamental.

    Measures the mean relative deviation of detected partials from the ideal
    harmonic series ``k * f0``: ``mean(|peaks_k / (k * f0) - 1|)``.

    Args:
        peaks (torch.Tensor): Partial (peak) frequencies, ordered so element
            ``k-1`` corresponds to the k-th harmonic.
        f0 (float): Fundamental frequency in Hz. Must be strictly positive.

    Returns:
        torch.Tensor: Scalar inharmonicity index (0 for a perfectly harmonic
        spectrum). Returns NaN if ``f0 <= 0`` or is non-finite, since the
        harmonic grid ``k * f0`` is then undefined (avoids division by zero).
    """
    if f0 is None or not math.isfinite(float(f0)) or float(f0) <= 0.0:
        return torch.tensor(float("nan"), device=peaks.device)
    if peaks.numel() == 0:
        return torch.tensor(float("nan"), device=peaks.device)
    k = torch.arange(1, peaks.numel() + 1, device=peaks.device, dtype=peaks.dtype)
    return torch.mean(torch.abs(peaks / (k * f0) - 1))
