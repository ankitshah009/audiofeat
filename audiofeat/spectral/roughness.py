import torch
import math
from ..temporal.rms import hann_window


def spectral_roughness(
    x: torch.Tensor,
    sample_rate: int = 22050,
    n_fft: int = 2048,
    top_db: float = 60.0,
) -> torch.Tensor:
    """Compute the *spectral roughness* of an audio signal.

    The implementation follows the perceptual model described by
    Sethares (1998) and Vassilakis (2001, 2005).  Given two partials
    with frequencies *f1*, *f2* and amplitudes *A1*, *A2*, the pairwise
    roughness is approximated by ::

        R(A1, A2, f1, f2) = (A1 * A2) ** 0.1 \
                            * ( 2 * min(A1, A2) / (A1 + A2) ) ** 3.11 \
                            * [exp(-b1 * s * d) - exp(-b2 * s * d)]

    where ``d = |f2 - f1|`` , ``s = 0.24 / (0.021 * f_min + 19)``,
    ``b1 = 3.5`` and ``b2 = 5.75``.  The total roughness of the signal
    is obtained by summing the contributions of all partial pairs.

    To keep the computation tractable, only frequency bins whose
    magnitude is within *top_db* of the maximum magnitude are
    considered.

    Args:
        x (torch.Tensor): 1-D audio signal (mono).
        sample_rate (int): Sampling rate of *x*.
        n_fft (int): FFT length to use for the analysis.
        top_db (float): Minimum amplitude (in dB) relative to the peak
            to retain a frequency bin for the roughness calculation.

    Returns:
        torch.Tensor: Scalar tensor containing the estimated roughness.
    """
    if x.dim() != 1:
        raise ValueError("`spectral_roughness` expects a 1-D (mono) signal.")

    # Window the signal to reduce spectral leakage
    windowed = x * hann_window(x.numel()).to(x.device)

    # Compute single-sided magnitude spectrum
    spectrum = torch.fft.rfft(windowed, n=n_fft)
    magnitudes = spectrum.abs()

    # Convert threshold from dB to linear scale
    ref = magnitudes.max()
    if ref == 0:
        return torch.tensor(0.0, device=x.device)
    mag_db = 20 * torch.log10(magnitudes / ref + 1e-12)
    keep_mask = mag_db >= -top_db

    if keep_mask.sum() <= 1:
        # Not enough components for a pairwise metric
        return torch.tensor(0.0, device=x.device)

    mags = magnitudes[keep_mask]
    # Normalise amplitudes between 0 and 1 for robustness
    mags = mags / mags.max()

    freqs = torch.linspace(0, sample_rate / 2, magnitudes.numel(), device=x.device)
    freqs = freqs[keep_mask]

    # Build pairwise matrices (broadcasted) ---------------------------------
    A_i = mags.unsqueeze(1)  # (N, 1)
    A_j = mags.unsqueeze(0)  # (1, N)

    f_i = freqs.unsqueeze(1)
    f_j = freqs.unsqueeze(0)

    # Prepare terms of the model
    A_min = torch.minimum(A_i, A_j)
    A_max = torch.maximum(A_i, A_j)

    X = (A_min * A_max).pow(0.1)
    Y = (2.0 * A_min / (A_min + A_max + 1e-12)).pow(3.11)

    f_min = torch.minimum(f_i, f_j)
    s = 0.24 / (0.021 * f_min + 19.0)

    d = (f_j - f_i).abs()
    Z = torch.exp(-3.5 * s * d) - torch.exp(-5.75 * s * d)

    rough_matrix = X * Y * Z

    # We only need upper triangular part (i < j) to avoid double counting
    roughness = rough_matrix.triu(diagonal=1).sum()
    return roughness.detach()  # Return a leaf tensor for convenience 