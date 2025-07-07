import torch
from ..temporal.rms import hann_window


def spectral_irregularity(x: torch.Tensor, n_fft: int = 2048) -> torch.Tensor:
    """Compute Jensen's *spectral irregularity* of a signal.

    The irregularity index reflects how smoothly the magnitudes of
    successive partials vary across the spectrum.  Low values indicate
    a smooth spectral envelope (e.g., harmonic tones) whereas high
    values correspond to spectra with large fluctuations between
    neighbour bins/partials (e.g., noisy sounds).

    The formulation used here follows Jensen (1999)::

        Irregularity = \sum_{k=1}^{N-1} |m[k+1] - m[k]| 
                        / \sum_{k=1}^{N-1} (m[k+1] + m[k])

    Args:
        x (torch.Tensor): 1-D audio signal.
        n_fft (int): FFT size.

    Returns:
        torch.Tensor: Scalar irregularity value in the range [0, 1].
    """
    if x.dim() != 1:
        raise ValueError("`spectral_irregularity` expects a mono signal (1-D tensor).")

    windowed = x * hann_window(x.numel()).to(x.device)
    spectrum = torch.fft.rfft(windowed, n=n_fft)
    mags = spectrum.abs()

    if mags.numel() < 2:
        return torch.tensor(0.0, device=x.device)

    diff = torch.abs(mags[1:] - mags[:-1])
    denom = mags[1:] + mags[:-1] + 1e-12  # avoid divide-by-zero

    irregularity = (diff.sum() / denom.sum()).clamp(0.0, 1.0)
    return irregularity.detach() 