import torch
import torchaudio
import torchaudio.transforms as T

def chroma_stft(waveform: torch.Tensor, sample_rate: int, n_fft: int = 4096, hop_length: int = 2048,
                n_chroma: int = 12, f_min: float = 27.5, f_max: float = 880.0) -> torch.Tensor:
    """
    Computes the Chroma Short-Time Fourier Transform (STFT) features.

    Chroma features are a powerful representation for music audio, where the
    entire spectrum is projected onto 12 bins representing the 12 distinct
    semitones (or chroma) of the musical octave.

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
    n_chroma : int
        Number of chroma bins to produce.
    f_min : float
        Minimum frequency (Hz) for the chroma calculation (typically A0).
    f_max : float
        Maximum frequency (Hz) for the chroma calculation (typically A5 or C6).

    Returns
    -------
    torch.Tensor
        Chroma STFT features, shape (n_chroma, num_frames).

    Notes
    -----
    This implementation uses torchaudio's `Spectrogram` and `MelScale` (conceptually)
    or `Resample` and then `Spectrogram` to align frequencies to chroma bins.
    A direct `torchaudio.transforms.ChromaSTFT` is not available, so this is a
    manual construction using lower-level transforms.
    For a more direct and optimized chroma feature extraction, `librosa` is often used.
    This implementation approximates the process by mapping frequencies to chroma bins.
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
    freqs = torch.linspace(0, sample_rate / 2, magnitudes.shape[0])

    # Map frequencies to chroma bins
    chroma_bins = torch.zeros((n_chroma, magnitudes.shape[1]), device=waveform.device)

    # Calculate the number of semitones from f_min to f_max
    # This is a simplified mapping. A more accurate mapping would involve
    # a dedicated chroma filter bank or resampling to a log-frequency scale.
    for i in range(magnitudes.shape[0]): # Iterate over frequency bins
        freq = freqs[i]
        if freq > 0:
            # Convert frequency to semitone number relative to A0 (27.5 Hz)
            semitone = 12 * torch.log2(freq / f_min)
            chroma_idx = int(semitone.round()) % n_chroma
            chroma_bins[chroma_idx, :] += magnitudes[i, :]

    # Normalize chroma features (optional, but common)
    chroma_bins = torch.nn.functional.normalize(chroma_bins, p=2, dim=0)

    return chroma_bins
