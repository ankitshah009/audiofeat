import torch
import math
from ..temporal.rms import hann_window

__all__ = ["spectral_sharpness"]

# Pre-computed Bark scale frequencies (Hz) for centre of 24 bands (Zwicker)
# Source: DIN 45692 Appendix
_BARK_CF = torch.tensor([
    50,   150,  250,  350,  450,  570,  700,  840,
    1000, 1170, 1370, 1600, 1850, 2150, 2500, 2900,
    3400, 4000, 4800, 5800, 7000, 8500, 10500, 13500
], dtype=torch.float32)

# Corresponding critical band edges (Hz) for 24 Bark bands
_BARK_EDGES = torch.tensor([
    0,   100,  200,  300,  400,  510,  630,  770,
    920, 1080, 1270, 1480, 1720, 2000, 2320, 2700,
    3150, 3700, 4400, 5300, 6400, 7700, 9500, 12000, 15500
], dtype=torch.float32)  # length 25

# Weighting function g(z) according to DIN 45692

def _sharpness_weight(z: torch.Tensor) -> torch.Tensor:
    """Weighting function g(z) for sharpness where *z* is Bark value."""
    g = torch.ones_like(z)
    mask = z > 16.0
    g[mask] = 0.066 * torch.exp(0.171 * (z[mask] - 16.0))
    return g


def _hz_to_bark(f: torch.Tensor) -> torch.Tensor:
    """Convert frequency (Hz) to Bark scale using Zwicker formula."""
    return 13 * torch.atan(0.00076 * f) + 3.5 * torch.atan((f / 7500.0) ** 2)


def spectral_sharpness(
    x: torch.Tensor,
    sample_rate: int = 22050,
    n_fft: int = 2048,
    power: float = 2.0,
) -> torch.Tensor:
    r"""Compute Zwicker *sharpness* (in acum) of an audio signal.

    This implementation approximates the DIN 45692 / ECMA-418-2 standard:

    S = \frac{\sum_{z} N'(z) \; g(z) \; z}{\sum_{z} N'(z)} \; [\text{acum}]

    where
      N'(z) = specific loudness in the Bark band *z* (approximated with band power),
      g(z)   = weighting function (unity up to 16 Bark, exponential beyond).

    A true standard-compliant loudness model would use ISO 532-1; here we
    approximate N'(z) by energy in the critical-band filters which is
    sufficient for comparative use cases (e.g. brightness ranking).
    """
    if x.dim() != 1:
        raise ValueError("`spectral_sharpness` expects a 1-D mono signal.")

    # Window and FFT
    windowed = x * hann_window(x.numel()).to(x.device)
    spectrum = torch.fft.rfft(windowed, n=n_fft)
    power_spectrum = (spectrum.abs() ** power)

    freqs = torch.linspace(0, sample_rate / 2, power_spectrum.numel(), device=x.device)
    bark_vals = _hz_to_bark(freqs)

    # Integrate power into 24 Bark bands
    band_power = torch.zeros(24, device=x.device)
    for b in range(24):
        f_low = _BARK_EDGES[b]
        f_high = _BARK_EDGES[b + 1]
        mask = (freqs >= f_low) & (freqs < f_high)
        if mask.any():
            band_power[b] = power_spectrum[mask].mean()  # mean power per band

    if band_power.sum() == 0:
        return torch.tensor(0.0, device=x.device)

    z = torch.arange(1, 25, device=x.device, dtype=torch.float32)  # 1..24 Bark indices
    g = _sharpness_weight(z)

    sharpness = (band_power * g * z).sum() / band_power.sum()
    return sharpness.detach() 
