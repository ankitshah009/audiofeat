
import torch
import pytest
from audiofeat.temporal.amplitude import amplitude_modulation_depth

def test_amplitude_modulation_depth():
    env = torch.randn(22050 * 5).abs()
    result = amplitude_modulation_depth(env, window=512)
    assert isinstance(result, torch.Tensor)
