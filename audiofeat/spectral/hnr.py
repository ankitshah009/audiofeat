
import torch

def harmonic_to_noise_ratio(harmonic_energy: torch.Tensor, noise_energy: torch.Tensor):
    """
    Computes the Harmonic-to-Noise Ratio (HNR).

    Args:
        harmonic_energy (torch.Tensor): The energy of the harmonic components.
        noise_energy (torch.Tensor): The energy of the noise components.

    Returns:
        torch.Tensor: The HNR in dB.
    """
    return 10 * torch.log10(harmonic_energy / (noise_energy + 1e-8))
