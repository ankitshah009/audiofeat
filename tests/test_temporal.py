
import torch
import pytest
from audiofeat.temporal.rms import rms
from audiofeat.temporal.zcr import zero_crossing_rate

def test_rms():
    audio_data = torch.randn(22050 * 5) # 5 seconds of audio
    frame_length = 2048
    hop_length = 512
    result = rms(audio_data, frame_length, hop_length)
    assert isinstance(result, torch.Tensor)
    assert result.shape[0] > 0

def test_zero_crossing_rate():
    audio_data = torch.randn(22050 * 5) # 5 seconds of audio
    frame_length = 2048
    hop_length = 512
    result = zero_crossing_rate(audio_data, frame_length, hop_length)
    assert isinstance(result, torch.Tensor)
    assert result.shape[0] > 0
