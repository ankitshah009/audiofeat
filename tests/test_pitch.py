
import torch
import pytest
from audiofeat.pitch.f0 import fundamental_frequency_autocorr, fundamental_frequency_yin
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
