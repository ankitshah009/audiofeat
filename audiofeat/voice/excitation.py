
import torch

def glottal_to_noise_excitation(spec: torch.Tensor):
    """Approximate GNE using band cross-correlations."""
    bands = spec.view(6, -1)
    corr = torch.nn.functional.conv1d(bands.unsqueeze(1), bands.unsqueeze(1), padding=0).max(dim=-1).values
    g = corr.max()
    return 10 * torch.log10(g / (1 - g + 1e-8))
