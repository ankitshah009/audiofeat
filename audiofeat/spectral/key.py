import torch
import torchaudio
import torchaudio.transforms as T

def key_detect(waveform: torch.Tensor, sample_rate: int, n_fft: int = 4096, hop_length: int = 2048,
               n_chroma: int = 12) -> str:
    """
    Detects the musical key of an audio waveform.

    This involves computing chroma features and comparing them to pre-defined
    key templates (Krumhansl-Schmuckler or similar).

    Parameters
    ----------
    waveform : torch.Tensor
        Mono audio waveform tensor. Expected shape: (num_samples,) or (1, num_samples).
    sample_rate : int
        Sampling rate of the waveform.
    n_fft : int
        Size of the FFT window for chroma computation.
    hop_length : int
        Number of samples between successive frames for chroma computation.
    n_chroma : int
        Number of chroma bins (should be 12 for standard Western music).

    Returns
    -------
    str
        The detected musical key (e.g., "C major", "A minor").

    Notes
    -----
    This is a simplified key detection algorithm. More advanced methods
    might involve probabilistic models, dynamic programming, or more
    sophisticated chroma feature extraction and template matching.
    Requires 'torch' and 'torchaudio' to be installed.
    """
    if waveform.ndim > 1 and waveform.shape[0] > 1:
        waveform = waveform[0]
    elif waveform.ndim == 0:
        raise ValueError("Input waveform cannot be a scalar.")

    # Compute Chroma STFT features
    # Re-using the chroma_stft function from audiofeat.spectral.chroma
    # For this to work, audiofeat.spectral.chroma needs to be importable.
    # For now, I'll include a simplified chroma computation here.
    # In a real library, you'd import it.

    # Simplified Chroma STFT (similar to the one implemented previously)
    spectrogram_transform = T.Spectrogram(
        n_fft=n_fft,
        hop_length=hop_length,
        power=2.0, # Power spectrogram
    )
    spec = spectrogram_transform(waveform)
    magnitudes = torch.sqrt(spec) # Amplitude spectrogram

    freqs = torch.linspace(0, sample_rate / 2, magnitudes.shape[0])
    chroma_bins = torch.zeros((n_chroma, magnitudes.shape[1]), device=waveform.device)

    f_min = 27.5 # A0
    for i in range(magnitudes.shape[0]):
        freq = freqs[i]
        if freq > 0:
            semitone = 12 * torch.log2(freq / f_min)
            chroma_idx = int(semitone.round()) % n_chroma
            chroma_bins[chroma_idx, :] += magnitudes[i, :]
    chroma_bins = torch.nn.functional.normalize(chroma_bins, p=2, dim=0)

    # Average chroma over time to get a single chroma vector
    chroma_vector = torch.mean(chroma_bins, dim=1)
    chroma_vector = torch.nn.functional.normalize(chroma_vector, p=1, dim=0) # Normalize to sum to 1

    # Krumhansl-Schmuckler key templates (simplified)
    # These are 12-element vectors representing the typical chroma distribution for each key.
    # Major key template (C major)
    major_template = torch.tensor([
        6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88
    ], device=waveform.device)
    major_template = torch.nn.functional.normalize(major_template, p=1, dim=0)

    # Minor key template (A minor)
    minor_template = torch.tensor([
        6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 3.98, 2.69, 3.34, 3.17, 2.51
    ], device=waveform.device)
    minor_template = torch.nn.functional.normalize(minor_template, p=1, dim=0)

    keys = [
        "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"
    ]

    best_key = "Unknown"
    max_correlation = -1.0

    for i in range(n_chroma):
        # Rotate the chroma vector to align with the current key's root
        rotated_chroma = torch.roll(chroma_vector, shifts=-i, dims=0)

        # Correlate with major template
        corr_major = torch.dot(rotated_chroma, major_template)
        if corr_major > max_correlation:
            max_correlation = corr_major
            best_key = keys[i] + " major"

        # Correlate with minor template
        corr_minor = torch.dot(rotated_chroma, minor_template)
        if corr_minor > max_correlation:
            max_correlation = corr_minor
            best_key = keys[i] + " minor"

    return best_key
