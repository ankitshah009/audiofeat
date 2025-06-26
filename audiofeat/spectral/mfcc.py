
import torch
import torchaudio.transforms as T

def mfcc(audio_data: torch.Tensor, sample_rate: int, n_mfcc: int = 40, n_fft: int = 2048, hop_length: int = 512, n_mels: int = 128):
    """
    Computes the Mel-Frequency Cepstral Coefficients (MFCCs) of an audio signal.

    Args:
        audio_data (torch.Tensor): The audio signal.
        sample_rate (int): The sample rate of the audio.
        n_mfcc (int): The number of MFCCs to compute.
        n_fft (int): The number of FFT points for Mel spectrogram.
        hop_length (int): The number of samples to slide the window for Mel spectrogram.
        n_mels (int): The number of Mel bands for Mel spectrogram.

    Returns:
        torch.Tensor: The MFCCs.
    """
    mel_spectrogram_transform = T.MelSpectrogram(sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    mfcc_transform = T.MFCC(sample_rate=sample_rate, n_mfcc=n_mfcc)
    return mfcc_transform(mel_spectrogram_transform(audio_data))
