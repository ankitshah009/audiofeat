from __future__ import annotations

import numpy as np
import torch
import torchaudio.transforms as T

from ..temporal.rms import hann_window


def _to_mono_tensor(audio_data: torch.Tensor) -> torch.Tensor:
    x = audio_data.float()
    if x.dim() == 0:
        raise ValueError("audio_data cannot be a scalar.")
    if x.dim() == 1:
        return x
    if x.dim() == 2:
        if x.shape[0] == 1:
            return x.squeeze(0)
        return x.mean(dim=0)
    raise ValueError("audio_data must be 1-D or 2-D (channels, samples).")


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
    waveform = _to_mono_tensor(audio_data)
    window = hann_window(n_fft).to(waveform.device)
    stft = torch.stft(waveform, n_fft, hop_length, window=window, return_complex=True)
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
    waveform = _to_mono_tensor(audio_data)
    mel_spectrogram_transform = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
    )
    return mel_spectrogram_transform(waveform)

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
    waveform = _to_mono_tensor(audio_data)
    device = waveform.device

    try:
        import librosa  # type: ignore
    except ModuleNotFoundError:
        if not hasattr(T, "CQT"):
            raise ModuleNotFoundError(
                "librosa is required for CQT on this torchaudio build. "
                "Install with `pip install \"audiofeat[examples]\"`."
            )
        transform = T.CQT(
            sample_rate=int(sample_rate),
            hop_length=int(hop_length),
            f_min=float(fmin),
            n_bins=int(n_bins),
            bins_per_octave=int(bins_per_octave),
        )
        return torch.abs(transform(waveform))

    cqt_np = librosa.cqt(
        y=waveform.detach().cpu().numpy().astype(np.float32, copy=False),
        sr=int(sample_rate),
        hop_length=int(hop_length),
        fmin=float(fmin),
        n_bins=int(n_bins),
        bins_per_octave=int(bins_per_octave),
    )
    return torch.from_numpy(np.abs(cqt_np).astype(np.float32, copy=False)).to(device=device)
