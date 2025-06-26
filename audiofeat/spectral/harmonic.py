
import torch

def harmonic_richness_factor(magnitudes: torch.Tensor):
    """Harmonic richness factor given harmonic magnitudes starting at F0."""
    if magnitudes.numel() < 2:
        return torch.tensor(0.0, device=magnitudes.device)
    numerator = magnitudes[1:].pow(2).sum()
    denominator = magnitudes[0].pow(2)
    return 10 * torch.log10(numerator / (denominator + 1e-8))

def inharmonicity_index(peaks: torch.Tensor, f0: float):
    """Inharmonicity from peak frequencies and fundamental."""
    k = torch.arange(1, peaks.numel() + 1, device=peaks.device)
    return torch.mean(torch.abs(peaks / (k * f0) - 1))
