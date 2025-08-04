import torch
from torch import nn
import torchaudio

def noise_reduction(waveform: torch.Tensor, sample_rate: int, threshold: float = 0.01) -> torch.Tensor:
    # Simple spectral subtraction for noise reduction
    # First, compute the STFT
    stft = torchaudio.transforms.Spectrogram(n_fft=1024, hop_length=512)(waveform)
    magnitude = torch.abs(stft)
    phase = torch.angle(stft)
    
    # Estimate noise floor (simple threshold-based)
    noise_floor = threshold * torch.max(magnitude)
    reduced_magnitude = torch.max(magnitude - noise_floor, torch.zeros_like(magnitude))
    
    # Reconstruct the waveform
    reduced_stft = reduced_magnitude * torch.exp(1j * phase)
    reduced_waveform = torchaudio.transforms.InverseSpectrogram(n_fft=1024, hop_length=512)(reduced_stft)
    return reduced_waveform