
import torch
import pytest
from audiofeat.pitch.f0 import fundamental_frequency_autocorr, fundamental_frequency_yin
from audiofeat.pitch.pyin import fundamental_frequency_pyin
from audiofeat.pitch.semitone import semitone_sd

def test_fundamental_frequency_autocorr():
    audio_data = torch.randn(22050 * 5)
    result = fundamental_frequency_autocorr(audio_data, fs=22050, frame_length=2048, hop_length=512)
    assert isinstance(result, torch.Tensor)
    assert result.shape[0] > 0

def test_fundamental_frequency_yin():
    audio_data = torch.randn(22050 * 5)
    result = fundamental_frequency_yin(audio_data, fs=22050, frame_length=2048, hop_length=512)
    assert isinstance(result, torch.Tensor)
    assert result.shape[0] > 0

def test_semitone_sd():
    f0_data = torch.tensor([100.0, 105.0, 102.0, 110.0, 0.0, 0.0])
    result = semitone_sd(f0_data)
    assert isinstance(result, torch.Tensor)
    assert result.item() >= 0


def test_fundamental_frequency_pyin():
    pytest.importorskip("librosa")
    sr = 16000
    t = torch.arange(sr * 2, dtype=torch.float32) / sr
    audio_data = torch.sin(2 * torch.pi * 220 * t)
    result = fundamental_frequency_pyin(
        audio_data,
        fs=sr,
        frame_length=1024,
        hop_length=256,
        fmin=80,
        fmax=400,
    )
    assert isinstance(result, torch.Tensor)
    assert result.numel() > 0
