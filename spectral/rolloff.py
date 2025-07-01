import torch
import torchaudio
import torchaudio.transforms as T

def spectral_rolloff(waveform: torch.Tensor, sample_rate: int, n_fft: int = 2048, hop_length: int = 512, roll_percent: float = 0.85) -> torch.Tensor:
    """
    Computes the spectral rolloff of an audio waveform.

    Spectral rolloff is the frequency below which a specified percentage
    (e.g., 85%) of the total spectral energy lies. It's often used to
    distinguish between voiced and unvoiced speech, or to characterize
    the spectral shape of sounds.

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
    roll_percent : float
        The percentage of total spectral energy to calculate the rolloff point.
        Must be between 0 and 1.

    Returns
    -------
    torch.Tensor
        Spectral rolloff per frame.

    Notes
    -----
    This implementation uses torchaudio for spectrogram computation.
    Requires 'torch' and 'torchaudio' to be installed.
    """
    if waveform.ndim > 1 and waveform.shape[0] > 1:
        waveform = waveform[0]
    elif waveform.ndim == 0:
        raise ValueError("Input waveform cannot be a scalar.")

    if not (0.0 < roll_percent < 1.0):
        raise ValueError("roll_percent must be between 0 and 1 (exclusive).")

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

    # Calculate the cumulative sum of magnitudes for each frame
    cumulative_magnitudes = torch.cumsum(magnitudes, dim=0)

    # Calculate the total energy for each frame
    total_energy = torch.sum(magnitudes, dim=0)

    # Find the frequency bin where the cumulative sum exceeds roll_percent of total energy
    rolloff_freqs = torch.zeros(magnitudes.shape[1], device=waveform.device)

    for i in range(magnitudes.shape[1]): # Iterate over frames
        target_energy = total_energy[i] * roll_percent
        # Find the first frequency bin where cumulative energy is >= target_energy
        # Use torch.searchsorted for efficiency
        idx = torch.searchsorted(cumulative_magnitudes[:, i], target_energy)
        # Clamp index to valid range
        idx = torch.clamp(idx, 0, len(freqs) - 1)
        rolloff_freqs[i] = freqs[idx]

    return rolloff_freqs
