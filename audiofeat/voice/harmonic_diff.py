import torch

def harmonic_differences(magnitudes: torch.Tensor, f0_hz: float, fs: int, h_indices: list = None, search_bins: int = 3):
    """
    Computes harmonic differences (e.g., H1-H2, H1-A3) in dB.

    Args:
        magnitudes (torch.Tensor): The magnitude spectrum (1D tensor), i.e. the
            output of ``torch.fft.rfft(...).abs()`` of length ``n_fft // 2 + 1``.
        f0_hz (float): The fundamental frequency in Hz.
        fs (int): The sample rate.
        h_indices (list): List of harmonic indices to compare (e.g., [1, 2, 3]
            for H1, H2, H3).
        search_bins (int): Half-width (in bins) of the peak-picking window placed
            around each expected harmonic location.

    Returns:
        torch.Tensor: The harmonic differences (consecutive, in dB).
    """
    if h_indices is None:
        h_indices = [1, 2]  # Default to H1-H2

    n_bins = magnitudes.numel()
    # rfft length is n_fft/2 + 1, so n_fft = 2 * (n_bins - 1) and the spacing
    # between adjacent bins is fs / n_fft = fs / (2 * (n_bins - 1)).
    if n_bins < 2:
        return torch.tensor([], device=magnitudes.device)
    bin_width = fs / (2.0 * (n_bins - 1))

    harmonic_amplitudes = []
    for h_idx in h_indices:
        center = int(round(h_idx * f0_hz / bin_width))
        if center >= n_bins:
            harmonic_amplitudes.append(torch.tensor(1e-8, device=magnitudes.device))
            continue
        # Peak-pick within +/- search_bins around the expected location so that
        # spectral leakage / quantization does not place the harmonic in the
        # wrong (e.g. half-frequency) bin.
        lo = max(center - search_bins, 0)
        hi = min(center + search_bins + 1, n_bins)
        harmonic_amplitudes.append(magnitudes[lo:hi].max())

    harmonic_amplitudes = torch.stack(harmonic_amplitudes)

    # Calculate differences in dB (e.g., H1-H2, H1-A3)
    harmonic_db = 20 * torch.log10(harmonic_amplitudes + 1e-8)
    differences = []
    for i in range(len(h_indices) - 1):
        differences.append(harmonic_db[i] - harmonic_db[i + 1])

    return torch.stack(differences) if differences else torch.tensor([], device=magnitudes.device)
