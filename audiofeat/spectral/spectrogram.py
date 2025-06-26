
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
