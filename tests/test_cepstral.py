
import torch
import pytest
from audiofeat.cepstral.lpcc import lpcc
from audiofeat.cepstral.gtcc import gtcc
from audiofeat.cepstral.deltas import delta, delta_delta

def test_lpcc():
    audio_data = torch.randn(22050 * 5)
    result = lpcc(audio_data, sample_rate=22050)
    assert isinstance(result, torch.Tensor)
    assert result.shape[0] > 0
    assert result.shape[1] > 0

def test_gtcc():
    audio_data = torch.randn(22050 * 5)
    result = gtcc(audio_data, sample_rate=22050)
    assert isinstance(result, torch.Tensor)
    assert result.shape[0] > 0
    assert result.shape[1] > 0

def test_delta():
    features = torch.randn(10, 5) # time_steps, features
    result = delta(features)
    assert isinstance(result, torch.Tensor)
    assert result.shape == features.shape

def test_delta_delta():
    features = torch.randn(10, 5) # time_steps, features
    result = delta_delta(features)
    assert isinstance(result, torch.Tensor)
    assert result.shape == features.shape
