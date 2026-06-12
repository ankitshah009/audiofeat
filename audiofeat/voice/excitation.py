
import torch

def glottal_to_noise_excitation(spec: torch.Tensor):
    """Approximate Glottal-to-Noise Excitation (GNE) from band envelopes.

    The full GNE (Michaelis et al., 1997) cross-correlates the Hilbert
    envelopes of overlapping frequency bands of the inverse-filtered signal and
    takes the maximum correlation. Here we compute a defensible approximation:
    the maximum *normalized* cross-correlation coefficient between the band
    rows of ``spec`` (each row treated as a band envelope). Normalization bounds
    the coefficient to ``[-1, 1]``; we clamp to ``[0, 1)`` before the logit so
    the result is always finite.

    Args:
        spec (torch.Tensor): A ``(n_bands, n_frames)`` non-negative band
            envelope / spectrogram. A 1D tensor is treated as a single band.

    Returns:
        torch.Tensor: A finite scalar GNE value (dB-like ``10*log10(g/(1-g))``).
    """
    if spec.ndim == 1:
        spec = spec.unsqueeze(0)
    bands = spec.float()
    n_bands = bands.shape[0]

    # Zero-mean, unit-norm each band so that the dot product between any pair is
    # a Pearson correlation coefficient in [-1, 1].
    centered = bands - bands.mean(dim=1, keepdim=True)
    norms = centered.norm(dim=1, keepdim=True)
    normalized = centered / (norms + 1e-8)

    # Cross-correlation matrix between bands; ignore the diagonal (self-corr = 1).
    corr = normalized @ normalized.t()
    if n_bands > 1:
        eye = torch.eye(n_bands, device=corr.device, dtype=corr.dtype)
        corr = corr - 2.0 * eye  # push diagonal below any real off-diagonal value
        g = corr.max()
    else:
        g = corr.max()

    # Bound strictly inside [0, 1) so log10(g/(1-g)) is finite.
    g = g.clamp(min=0.0, max=1.0 - 1e-6)
    return 10 * torch.log10((g + 1e-8) / (1 - g + 1e-8))
