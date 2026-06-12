import torch
import torchaudio
import torchaudio.transforms as T


def _median_filter_1d_axis(mag: torch.Tensor, kernel_size: int, axis: int) -> torch.Tensor:
    """
    Sliding-window median filter along ``axis`` of a 2D tensor, using a
    reflect-padded ``unfold`` (vectorised; no Python double loop).

    The kernel is forced odd and clamped to the axis length, and the reflect
    pad is clamped so it never exceeds ``size - 1`` (torch's reflect-pad limit)
    for short signals.
    """
    size = mag.shape[axis]
    k = int(kernel_size)
    if k < 1:
        k = 1
    if k % 2 == 0:
        k += 1
    if k > size:
        k = size if size % 2 == 1 else max(1, size - 1)
    half = k // 2
    # reflect padding requires pad < size; clamp for very short axes.
    pad = min(half, max(0, size - 1))

    work = mag if axis == 1 else mag.transpose(0, 1)  # filter along last dim
    if pad > 0:
        work = torch.nn.functional.pad(work, (pad, pad), mode="reflect")
    # If clamping made the effective window shorter than k, shrink k to fit.
    eff_k = min(k, work.shape[-1])
    windows = work.unfold(dimension=-1, size=eff_k, step=1)
    med = windows.median(dim=-1).values
    # Crop back to original length (handles the clamped-pad case).
    med = med[..., :size]
    if med.shape[-1] < size:  # pathological tiny axis; pad-edge replicate
        med = torch.nn.functional.pad(med, (0, size - med.shape[-1]), mode="replicate")
    return med if axis == 1 else med.transpose(0, 1)


def hps(waveform: torch.Tensor, sample_rate: int, n_fft: int = 2048, hop_length: int = 512,
        margin_h: float = 3.0, margin_p: float = 3.0,
        kernel_size_h: int | None = None, kernel_size_p: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Performs Harmonic-Percussive Separation (HPS) on an audio waveform.

    Separates an audio signal into its harmonic and percussive components via
    median filtering of the magnitude spectrogram (Fitzgerald, 2010). A median
    filter along the **frequency** axis suppresses sharp harmonic peaks (giving
    the percussive estimate's complement) while a median filter along the
    **time** axis suppresses transients (giving the harmonic estimate); soft
    Wiener-style masks are then derived from the two filtered spectra.

    Parameters
    ----------
    waveform : torch.Tensor
        Mono audio waveform tensor. Expected shape: (num_samples,) or (1, num_samples).
    sample_rate : int
        Sampling rate of the waveform.
    n_fft : int
        Size of the FFT window.
    hop_length : int
        Number of samples between successive frames.
    margin_h : float
        Legacy half-width control for the harmonic (time-axis) median filter.
        Used only when ``kernel_size_h`` is None and ``margin_h`` differs from
        its default; the kernel length is ``2*margin_h + 1``.
    margin_p : float
        Legacy half-width control for the percussive (frequency-axis) median
        filter. Used only when ``kernel_size_p`` is None and ``margin_p``
        differs from its default; the kernel length is ``2*margin_p + 1``.
    kernel_size_h : int, optional
        Explicit (odd) kernel length for the harmonic median filter along the
        time axis. Defaults to 31 (a sensible Fitzgerald/librosa value).
    kernel_size_p : int, optional
        Explicit (odd) kernel length for the percussive median filter along the
        frequency axis. Defaults to 17.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        (harmonic_waveform, percussive_waveform).

    Notes
    -----
    Median filtering is vectorised with ``unfold`` + ``median`` (no Python
    double loop). Reflect padding is clamped for short signals.
    Requires 'torch' and 'torchaudio'.
    """
    if waveform.ndim > 1 and waveform.shape[0] > 1:
        waveform = waveform[0]
    elif waveform.ndim == 0:
        raise ValueError("Input waveform cannot be a scalar.")
    waveform = waveform.flatten()

    # Resolve kernel sizes. Honour explicit kernel_size_* first; otherwise fall
    # back to a sensible default unless the caller overrode the legacy margin.
    _DEFAULT_MARGIN = 3.0
    if kernel_size_h is None:
        kernel_size_h = 31 if margin_h == _DEFAULT_MARGIN else int(margin_h * 2 + 1)
    if kernel_size_p is None:
        kernel_size_p = 17 if margin_p == _DEFAULT_MARGIN else int(margin_p * 2 + 1)

    # Compute STFT (power=None -> complex spectrogram)
    stft_transform = T.Spectrogram(
        n_fft=n_fft,
        hop_length=hop_length,
        power=None,
    )
    stft_matrix = stft_transform(waveform)
    magnitude_spectrogram = torch.abs(stft_matrix)
    phase_spectrogram = torch.angle(stft_matrix)

    # Harmonic estimate: median filter along the TIME axis (axis=1).
    harmonic_median = _median_filter_1d_axis(
        magnitude_spectrogram, kernel_size_h, axis=1
    )
    # Percussive estimate: median filter along the FREQUENCY axis (axis=0).
    percussive_median = _median_filter_1d_axis(
        magnitude_spectrogram, kernel_size_p, axis=0
    )

    # Soft masking (Wiener / power law)
    denom = harmonic_median + percussive_median + 1e-8
    harmonic_mask = (harmonic_median / denom) ** 2
    percussive_mask = (percussive_median / denom) ** 2

    # Apply masks to the original STFT magnitude
    harmonic_spectrogram = magnitude_spectrogram * harmonic_mask
    percussive_spectrogram = magnitude_spectrogram * percussive_mask

    # Reconstruct complex spectrograms
    harmonic_stft = harmonic_spectrogram * torch.exp(1j * phase_spectrogram)
    percussive_stft = percussive_spectrogram * torch.exp(1j * phase_spectrogram)

    # Inverse STFT to get time-domain waveforms
    istft_transform = T.InverseSpectrogram(
        n_fft=n_fft,
        hop_length=hop_length,
    )
    harmonic_waveform = istft_transform(harmonic_stft)
    percussive_waveform = istft_transform(percussive_stft)

    return harmonic_waveform, percussive_waveform
