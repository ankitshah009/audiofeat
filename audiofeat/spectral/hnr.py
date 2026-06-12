
import torch


def harmonic_to_noise_ratio(harmonic_energy: torch.Tensor, noise_energy: torch.Tensor):
    """
    Energy-ratio helper: ``10 * log10(harmonic_energy / noise_energy)`` in dB.

    .. note::
        This is **not** an HNR *estimator* — it merely converts two
        pre-computed energy scalars into a dB ratio. It does not analyse a
        signal. To estimate HNR directly from a waveform use
        :func:`harmonic_to_noise_ratio_acf`.

    Args:
        harmonic_energy (torch.Tensor): Energy of the harmonic component.
        noise_energy (torch.Tensor): Energy of the noise component.

    Returns:
        torch.Tensor: The energy ratio expressed in dB.
    """
    return 10 * torch.log10(harmonic_energy / (noise_energy + 1e-8))


def harmonic_to_noise_ratio_acf(
    signal: torch.Tensor,
    sample_rate: int,
    fmin: float = 75.0,
    fmax: float = 500.0,
) -> torch.Tensor:
    """
    Estimate the Harmonic-to-Noise Ratio (HNR) of a frame via autocorrelation.

    Implements Boersma's method (Praat): the autocorrelation of the windowed
    signal is divided by the autocorrelation of the window itself to remove the
    windowing bias, and the largest peak ``r_max`` in the lag range
    ``[sr/fmax, sr/fmin]`` is interpreted as the harmonic fraction. The HNR is::

        HNR = 10 * log10(r_max / (1 - r_max))    dB

    Args:
        signal (torch.Tensor): A single mono frame/segment (1D tensor).
        sample_rate (int): Sampling rate in Hz.
        fmin (float): Lowest expected fundamental frequency in Hz.
        fmax (float): Highest expected fundamental frequency in Hz.

    Returns:
        torch.Tensor: Scalar HNR in dB. Returns ``-inf`` for a degenerate
        (silent / fully aperiodic) frame.

    References:
        Boersma, P. (1993). "Accurate short-term analysis of the fundamental
        frequency and the harmonics-to-noise ratio of a sampled sound."
    """
    if sample_rate <= 0:
        raise ValueError("sample_rate must be > 0.")
    if not (0 < fmin < fmax):
        raise ValueError("Require 0 < fmin < fmax.")

    x = signal.flatten().to(torch.float64)
    n = x.numel()
    if n < 4:
        return torch.tensor(float("-inf"), dtype=torch.float64, device=signal.device)

    # Remove DC then apply a Hann window.
    x = x - x.mean()
    window = torch.hann_window(n, periodic=False, dtype=torch.float64, device=x.device)
    xw = x * window

    # Normalised autocorrelation of the signal and of the window, via FFT.
    nfft = 1
    while nfft < 2 * n:
        nfft *= 2

    def _acf(v: torch.Tensor) -> torch.Tensor:
        V = torch.fft.rfft(v, n=nfft)
        ac = torch.fft.irfft(V * torch.conj(V), n=nfft)[:n]
        return ac

    r_signal = _acf(xw)
    r_window = _acf(window)

    r0 = r_signal[0]
    if r0 <= 0:
        return torch.tensor(float("-inf"), dtype=torch.float64, device=signal.device)

    # Boersma: divide signal ACF by window ACF to undo the window's taper bias.
    eps = 1e-12
    r = (r_signal / r0) / (r_window / r_window[0] + eps)

    lag_min = max(1, int(round(sample_rate / fmax)))
    lag_max = min(n - 1, int(round(sample_rate / fmin)))
    if lag_max <= lag_min:
        return torch.tensor(float("-inf"), dtype=torch.float64, device=signal.device)

    r_max = torch.max(r[lag_min : lag_max + 1])
    # Clamp strictly inside (0, 1) so the dB conversion is finite.
    r_max = torch.clamp(r_max, min=1e-6, max=1.0 - 1e-6)

    hnr = 10.0 * torch.log10(r_max / (1.0 - r_max))
    return hnr.to(torch.float64)
