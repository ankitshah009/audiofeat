import torch
import pytest
from audiofeat.temporal.rhythm import breath_group_duration, speech_rate

def test_breath_group_duration():
    env = torch.randn(22050 * 5).abs()
    result = breath_group_duration(env, fs=22050)
    assert isinstance(result, torch.Tensor)

def test_speech_rate():
    audio_data = torch.randn(22050 * 5)
    result = speech_rate(audio_data, fs=22050)
    assert isinstance(result, float)