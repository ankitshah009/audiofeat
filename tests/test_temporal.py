
import torch
import pytest
from audiofeat.temporal.rms import rms, short_time_energy
from audiofeat.temporal.zcr import zero_crossing_rate
from audiofeat.temporal.attack import log_attack_time
from audiofeat.temporal.rhythm import temporal_centroid
from audiofeat.temporal.energy_entropy import entropy_of_energy
from audiofeat.temporal.rhythm_features import tempo, beat_track
from audiofeat.temporal.rms import frame_signal

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

def test_tempo():
    audio_data = torch.randn(22050 * 10) # Longer audio for tempo
    result = tempo(audio_data, sample_rate=22050)
    assert isinstance(result, float)
    assert result >= 0 # Tempo should be non-negative

def test_beat_track():
    audio_data = torch.randn(22050 * 10) # Longer audio for beat tracking
    result = beat_track(audio_data, sample_rate=22050)
    assert isinstance(result, torch.Tensor)
    assert result.dim() == 1 # Should be a 1D tensor of beat times


def test_frame_signal_short_clip_is_zero_padded():
    audio_data = torch.arange(0, 100, dtype=torch.float32)
    frames = frame_signal(audio_data, frame_length=256, hop_length=128)
    assert frames.shape == (1, 256)
    assert torch.allclose(frames[0, :100], audio_data)
    assert torch.allclose(frames[0, 100:], torch.zeros(156))
