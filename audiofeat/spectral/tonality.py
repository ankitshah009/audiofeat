import torch
from ..temporal.rms import hann_window

__all__ = ["spectral_tonality"]


def spectral_tonality(
    x: torch.Tensor,
    n_fft: int = 2048,
    top_db: float = 60.0,
) -> torch.Tensor:
    r"""Compute a simple *tonality coefficient* (0..1).

    The ECMA-418-2 standard defines a sophisticated critical-band based
    metric; here we approximate it with a *spectral crest factor* (also
    called *tonality index*) which correlates well with perceived
    tonality and is often used in perceptual audio coding.

        SCF = max(|X(f)|) / (mean(|X(f)|) + eps)
        T   = (SCF - 1) / (N - 1)  \in [0, 1]

    where *N* is the number of bins considered.  A perfectly tonal
    signal (single spectral line) yields T ≈ 1 whereas white noise
    yields T ≈ 0.
    """
    if x.dim() != 1:
        raise ValueError("`spectral_tonality` expects a mono 1-D signal.")

    windowed = x * hann_window(x.numel()).to(x.device)
    spectrum = torch.fft.rfft(windowed, n=n_fft)
    mags = spectrum.abs()

    # Apply threshold similar to MPEG psychoacoustic model
    ref = mags.max()
    if ref == 0:
        return torch.tensor(0.0, device=x.device)
    mag_db = 20 * torch.log10(mags / ref + 1e-12)
    mask = mag_db >= -top_db
    if mask.sum() == 0:
        return torch.tensor(0.0, device=x.device)

    mags = mags[mask]
    N = mags.numel()
    scf = mags.max() / (mags.mean() + 1e-12)

    tonality = (scf - 1.0) / (N - 1.0 + 1e-12)
    tonality = tonality.clamp(0.0, 1.0)
    return tonality.detach() 
