
import torch

__all__ = ["phase_coherence"]

def phase_coherence(phases: torch.Tensor):
    """Compute phase coherence from instantaneous phase."""
    return torch.abs(torch.mean(torch.exp(1j * phases)))
