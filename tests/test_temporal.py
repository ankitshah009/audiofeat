import torch
import pytest
from audiofeat.temporal.rms import rms, short_time_energy
from audiofeat.temporal.zcr import zero_crossing_rate
from audiofeat.temporal.attack import log_attack_time
from audiofeat.temporal.rhythm import temporal_centroid
from audiofeat.temporal.energy_entropy import entropy_of_energy

def test_rms():
    audio_data = torch.randn(22050 * 5) # 5 seconds of audio
    frame_length = 2048
    hop_length = 512
    result = rms(audio_data, frame_length, hop_length)
    assert isinstance(result, torch.Tensor)
    assert result.shape[0] > 0

def test_short_time_energy():
    audio_data = torch.randn(22050 * 5) # 5 seconds of audio
    frame_length = 2048
    hop_length = 512
    result = short_time_energy(audio_data, frame_length, hop_length)
    assert isinstance(result, torch.Tensor)
    assert result.shape[0] > 0

def test_zero_crossing_rate():
    audio_data = torch.randn(22050 * 5) # 5 seconds of audio
    frame_length = 2048
    hop_length = 512
    result = zero_crossing_rate(audio_data, frame_length, hop_length)
    assert isinstance(result, torch.Tensor)
    assert result.shape[0] > 0

def test_log_attack_time():
    audio_data = torch.randn(22050 * 1)
    result = log_attack_time(audio_data, sample_rate=22050)
    assert isinstance(result, float)

def test_temporal_centroid():
    audio_data = torch.randn(22050 * 5)
    frame_length = 2048
    hop_length = 512
    result = temporal_centroid(audio_data, frame_length, hop_length)
    assert isinstance(result, torch.Tensor)
    assert result.shape[0] > 0

def test_entropy_of_energy():
    audio_data = torch.randn(22050 * 5)
    frame_length = 2048
    hop_length = 512
    result = entropy_of_energy(audio_data, frame_length, hop_length)
    assert isinstance(result, torch.Tensor)
    assert result.shape[0] > 0