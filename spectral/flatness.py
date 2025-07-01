import torch
import torchaudio
import torchaudio.transforms as T

def spectral_flatness(waveform: torch.Tensor, sample_rate: int, n_fft: int = 2048, hop_length: int = 512) -> torch.Tensor:
    """
    Computes the spectral flatness (or tonality coefficient) of an audio waveform.

    Spectral flatness is a measure used in digital signal processing to
    characterize an audio spectrum. A high spectral flatness indicates that
    the spectrum is similar to white noise; a low spectral flatness indicates
    that the spectrum is similar to a pure tone.

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

    Returns
    -------
    torch.Tensor
        Spectral flatness per frame.

    Notes
    -----
    This implementation uses torchaudio for spectrogram computation.
    Requires 'torch' and 'torchaudio' to be installed.
    Spectral flatness is calculated as the ratio of the geometric mean
    to the arithmetic mean of the power spectrum.
    """
    if waveform.ndim > 1 and waveform.shape[0] > 1:
        waveform = waveform[0]
    elif waveform.ndim == 0:
        raise ValueError("Input waveform cannot be a scalar.")

    # Compute the STFT power spectrogram
    spectrogram_transform = T.Spectrogram(
        n_fft=n_fft,
        hop_length=hop_length,
        power=2.0, # Power spectrogram
    )
    power_spectrogram = spectrogram_transform(waveform)

    # Add a small epsilon to avoid log(0) and division by zero
    epsilon = 1e-10
    power_spectrogram = power_spectrogram + epsilon

    # Calculate geometric mean: exp(mean(log(x)))
    geometric_mean = torch.exp(torch.mean(torch.log(power_spectrogram), dim=0))

    # Calculate arithmetic mean: mean(x)
    arithmetic_mean = torch.mean(power_spectrogram, dim=0)

    # Calculate spectral flatness
    flatness = geometric_mean / arithmetic_mean

    return flatness
