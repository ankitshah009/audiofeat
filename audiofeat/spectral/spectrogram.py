
import torch
import torchaudio.transforms as T
from ..temporal.rms import hann_window

def linear_spectrogram(audio_data: torch.Tensor, n_fft: int = 2048, hop_length: int = 512):
    """
    Computes the linear spectrogram (STFT) of an audio signal.

    Args:
        audio_data (torch.Tensor): The audio signal.
        n_fft (int): The number of FFT points.
        hop_length (int): The number of samples to slide the window.

    Returns:
        torch.Tensor: The magnitude spectrogram.
    """
    window = hann_window(n_fft).to(audio_data.device)
    stft = torch.stft(audio_data, n_fft, hop_length, window=window, return_complex=True)
    return torch.abs(stft)

def mel_spectrogram(audio_data: torch.Tensor, sample_rate: int, n_fft: int = 2048, hop_length: int = 512, n_mels: int = 128):
    """
    Computes the Mel spectrogram of an audio signal.

    Args:
        audio_data (torch.Tensor): The audio signal.
        sample_rate (int): The sample rate of the audio.
        n_fft (int): The number of FFT points.
        hop_length (int): The number of samples to slide the window.
        n_mels (int): The number of Mel bands.

    Returns:
        torch.Tensor: The Mel spectrogram.
    """
    mel_spectrogram_transform = T.MelSpectrogram(sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    return mel_spectrogram_transform(audio_data)

def cqt_spectrogram(audio_data: torch.Tensor, sample_rate: int, hop_length: int = 512, fmin: float = 32.7, n_bins: int = 84, bins_per_octave: int = 12):
    """
    Computes the Constant-Q Transform (CQT) spectrogram of an audio signal.

    Args:
        audio_data (torch.Tensor): The audio signal.
        sample_rate (int): The sample rate of the audio.
        hop_length (int): The number of samples to slide the window.
        fmin (float): The minimum frequency.
        n_bins (int): The total number of bins.
        bins_per_octave (int): The number of bins per octave.

    Returns:
        torch.Tensor: The CQT spectrogram.
    """
    # Simplified CQT implementation using STFT and a custom filterbank
    # This is a basic approximation and not a full CQT implementation
    # A proper CQT requires variable window lengths and more complex filter design

    q_factor = 1 / (2**(1/bins_per_octave) - 1)
    
    # Calculate frequencies for each bin
    frequencies = fmin * (2**(torch.arange(n_bins).float() / bins_per_octave))
    
    # Determine window lengths for each frequency
    window_lengths = (sample_rate / frequencies * q_factor).int()
    window_lengths[window_lengths < 1] = 1 # Ensure minimum window length of 1

    # Use a fixed n_fft for STFT for simplicity, but a true CQT has variable FFT sizes
    n_fft_stft = 2048 # A common FFT size

    # Compute STFT
    stft_transform = T.Spectrogram(n_fft=n_fft_stft, hop_length=hop_length, window_fn=torch.hann_window, return_complex=True)
    stft_data = stft_transform(audio_data)
    
    # Create a simple CQT-like filterbank (rectangular bins for simplicity)
    cqt_filter_bank = torch.zeros(n_bins, n_fft_stft // 2 + 1, device=audio_data.device)
    for i, freq in enumerate(frequencies):
        # Map CQT frequency to STFT bin
        stft_bin = int(freq / (sample_rate / n_fft_stft))
        if stft_bin < n_fft_stft // 2 + 1:
            cqt_filter_bank[i, stft_bin] = 1.0 # Simple rectangular filter

    # Apply filterbank to STFT magnitude
    cqt_magnitude = torch.matmul(cqt_filter_bank, torch.abs(stft_data))

    return cqt_magnitude
