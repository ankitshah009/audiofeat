import torch
import torchaudio
import torchaudio.transforms as T

def log_mel_spectrogram(waveform: torch.Tensor, sample_rate: int, n_fft: int = 2048, hop_length: int = 512, n_mels: int = 128, f_min: float = 0.0, f_max: float = None) -> torch.Tensor:
    """
    Computes the log-Mel spectrogram of an audio waveform using torchaudio.

    Parameters
    ----------
    waveform : torch.Tensor
        Mono audio waveform tensor. Expected shape: (num_samples,) or (1, num_samples).
    sample_rate : int
        Sampling rate of the waveform.
    n_fft : int
        Length of the FFT window.
    hop_length : int
        Number of samples between successive frames.
    n_mels : int
        Number of Mel bands to generate.
    f_min : float
        Minimum frequency (Hz) for the Mel filterbank.
    f_max : float or None
        Maximum frequency (Hz) for the Mel filterbank. If None, defaults to sample_rate / 2.

    Returns
    -------
    torch.Tensor
        Log-Mel spectrogram.
    """
    if waveform.ndim > 1 and waveform.shape[0] > 1:
        # Assuming mono or taking the first channel if multi-channel
        waveform = waveform[0]
    elif waveform.ndim == 0:
        raise ValueError("Input waveform cannot be a scalar.")

    if f_max is None:
        f_max = float(sample_rate / 2)

    # Create a MelSpectrogram transform
    mel_spectrogram_transform = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        f_min=f_min,
        f_max=f_max,
        power=2.0,  # Power spectrogram
    )

    # Compute the Mel spectrogram
    mel_spectrogram = mel_spectrogram_transform(waveform)

    # Convert to log scale
    log_mel_s = torch.log(mel_spectrogram + 1e-6) # Add a small epsilon to avoid log(0)

    return log_mel_s