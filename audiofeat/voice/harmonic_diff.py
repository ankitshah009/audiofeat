import torch

def harmonic_differences(magnitudes: torch.Tensor, f0_hz: float, fs: int, h_indices: list = None):
    """
    Computes harmonic differences (e.g., H1-H2, H1-A3).

    Args:
        magnitudes (torch.Tensor): The magnitude spectrum (1D tensor).
        f0_hz (float): The fundamental frequency in Hz.
        fs (int): The sample rate.
        h_indices (list): List of harmonic indices to compare (e.g., [1, 2, 3] for H1, H2, H3).

    Returns:
        torch.Tensor: The harmonic differences.
    """
    if h_indices is None:
        h_indices = [1, 2] # Default to H1-H2

    # Convert F0 to FFT bin index
    f0_bin = int(f0_hz / (fs / magnitudes.numel()))

    harmonic_amplitudes = []
    for h_idx in h_indices:
        current_harmonic_bin = h_idx * f0_bin
        if current_harmonic_bin < magnitudes.numel():
            harmonic_amplitudes.append(magnitudes[current_harmonic_bin])
        else:
            harmonic_amplitudes.append(torch.tensor(1e-8, device=magnitudes.device)) # Small value if harmonic out of bounds

    harmonic_amplitudes = torch.stack(harmonic_amplitudes)

    # Calculate differences in dB (e.g., H1-H2, H1-A3)
    harmonic_db = 20 * torch.log10(harmonic_amplitudes + 1e-8)
    differences = []
    for i in range(len(h_indices) - 1):
        differences.append(harmonic_db[i] - harmonic_db[i + 1])

    return torch.stack(differences) if differences else torch.tensor([])