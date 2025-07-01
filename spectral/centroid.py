import torch
import torchaudio
import torchaudio.transforms as T

def spectral_centroid(waveform: torch.Tensor, sample_rate: int, n_fft: int = 2048, hop_length: int = 512) -> torch.Tensor:
    """
    Computes the spectral centroid of an audio waveform.

    The spectral centroid indicates the "center of mass" of the spectrum
    and is a good indicator of the "brightness" of a sound. Higher centroid
    values correspond to "brighter" sounds.

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
        Spectral centroid per frame.

    Notes
    -----
    This implementation uses torchaudio for spectrogram computation.
    Requires 'torch' and 'torchaudio' to be installed.
    """
    if waveform.ndim > 1 and waveform.shape[0] > 1:
        waveform = waveform[0]
    elif waveform.ndim == 0:
        raise ValueError("Input waveform cannot be a scalar.")

    # Compute the STFT magnitude spectrogram
    spectrogram_transform = T.Spectrogram(
        n_fft=n_fft,
        hop_length=hop_length,
        power=2.0, # Power spectrogram
    )
    spec = spectrogram_transform(waveform)
    magnitudes = torch.sqrt(spec) # Amplitude spectrogram

    # Create a frequency axis for the spectrogram
    freqs = torch.linspace(0, sample_rate / 2, magnitudes.shape[0], device=waveform.device)
    freqs = freqs.unsqueeze(1) # Make it (n_freq_bins, 1) for broadcasting

    # Calculate spectral centroid
    # Centroid = sum(freq * magnitude) / sum(magnitude)
    sum_freq_mag = torch.sum(freqs * magnitudes, dim=0)
    sum_mag = torch.sum(magnitudes, dim=0)

    # Avoid division by zero for frames with no energy
    spectral_c = torch.where(sum_mag > 0, sum_freq_mag / sum_mag, torch.tensor(0.0, device=waveform.device))

    return spectral_c
