
import torch

def harmonic_differences(magnitudes: torch.Tensor, f0_hz: float, fs: int, h_indices: list = None):
    """
    Computes harmonic differences (e.g., H1-H2, H1-A3).

    Args:
        magnitudes (torch.Tensor): The magnitude spectrum.
        f0_hz (float): The fundamental frequency in Hz.
        fs (int): The sample rate.
        h_indices (list): List of harmonic indices to compare (e.g., [1, 2, 3] for H1, H2, H3).

    Returns:
        torch.Tensor: The harmonic differences.
    """
    if h_indices is None:
        h_indices = [1, 2] # Default to H1-H2

    # Placeholder implementation
    # A proper implementation would involve:
    # 1. Accurate F0 tracking
    # 2. Identifying harmonic peaks in the spectrum
    # 3. Calculating amplitudes at harmonic frequencies

    # Dummy implementation: return random differences
    return torch.randn(len(h_indices) - 1)
