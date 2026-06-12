
import torch

def semitone_sd(f0: torch.Tensor):
    """Standard deviation of F0 in semitones."""
    voiced = f0[f0 > 0]
    # Unbiased std of a single element is NaN; <2 voiced frames => no spread.
    if voiced.numel() < 2:
        return torch.tensor(0.0, device=f0.device)
    mean_f0 = voiced.mean()
    if torch.isnan(mean_f0):
        return torch.tensor(0.0, device=f0.device)
    semitones = 12 * torch.log2(voiced / mean_f0)
    return semitones.std()
