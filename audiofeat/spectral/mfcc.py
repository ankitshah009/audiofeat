
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
    audio_data = audio_data.flatten().float()
    if audio_data.numel() == 0:
        raise ValueError("audio_data must be non-empty.")

    # Keep MFCC extraction stable for short clips.
    effective_n_fft = int(min(n_fft, max(16, audio_data.numel())))
    effective_hop = int(min(hop_length, max(1, effective_n_fft // 2)))
    effective_n_mels = int(min(n_mels, max(8, effective_n_fft // 2)))
    effective_n_mfcc = int(min(n_mfcc, effective_n_mels))

    mfcc_transform = T.MFCC(
        sample_rate=sample_rate,
        n_mfcc=effective_n_mfcc,
        melkwargs={
            "n_fft": effective_n_fft,
            "hop_length": effective_hop,
            "n_mels": effective_n_mels,
            "center": False,
        },
    )
    return mfcc_transform(audio_data)
