import torch
from audiofeat.spectral.sharpness import spectral_sharpness
from audiofeat.spectral.tonality import spectral_tonality
from audiofeat.temporal.decay import decay_time


def test_spectral_sharpness():
    x = torch.randn(22050 * 2)  # 2 s noise
    val = spectral_sharpness(x, sample_rate=22050)
    assert isinstance(val, torch.Tensor)
    assert val >= 0


def test_spectral_tonality():
    # Mix of sinusoid and noise
    sr = 22050
    t = torch.linspace(0, 1, sr)
    sine = torch.sin(2 * torch.pi * 440 * t)
    noise = 0.1 * torch.randn_like(sine)
    x = sine + noise
    val = spectral_tonality(x, n_fft=2048)
    assert isinstance(val, torch.Tensor)
    assert 0 <= val <= 1


def test_decay_time():
    sr = 22050
    # signal: 0.5s sine then zeros
    t = torch.linspace(0, 0.5, int(sr * 0.5))
    sig = torch.sin(2 * torch.pi * 440 * t)
    tail = torch.zeros(int(sr * 0.5))
    x = torch.cat([sig, tail])
    val = decay_time(x, sample_rate=sr, threshold_db=-20)
    assert isinstance(val, torch.Tensor)
    assert val >= 0 