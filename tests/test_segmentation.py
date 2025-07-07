
import torch
import pytest
from audiofeat.segmentation.silence import silence_removal
from audiofeat.segmentation.diarization import speaker_diarization
from audiofeat.segmentation.thumbnailing import music_thumbnailing

def test_silence_removal():
    sample_rate = 22050
    # A silent signal with a non-silent part in the middle
    signal = torch.zeros(sample_rate * 5)
    signal[sample_rate*2:sample_rate*3] = torch.randn(sample_rate)
    result = silence_removal(signal, sample_rate)
    assert isinstance(result, torch.Tensor)
    assert result.shape[0] > 0
    assert result.shape[0] < signal.shape[0]

def test_speaker_diarization():
    sample_rate = 22050
    # A signal with two speakers (simulated by different frequencies)
    t = torch.linspace(0, 5, sample_rate * 5) # 5 seconds signal
    speaker1 = torch.sin(2 * torch.pi * 220 * t)
    speaker2 = torch.sin(2 * torch.pi * 440 * t)
    signal = torch.cat([speaker1, speaker2])
    result = speaker_diarization(signal, sample_rate, n_speakers=2)
    assert isinstance(result, torch.Tensor)
    assert result.shape[0] > 0
    assert len(torch.unique(result)) == 2

def test_music_thumbnailing():
    sample_rate = 22050
    # A signal with a repeating pattern
    t = torch.linspace(0, 1, sample_rate)
    pattern = torch.sin(2 * torch.pi * 440 * t)
    signal = torch.cat([torch.randn(sample_rate*5), pattern, torch.randn(sample_rate*5), pattern, torch.randn(sample_rate*5)])
    start_time, end_time = music_thumbnailing(signal, sample_rate)
    assert isinstance(start_time, torch.Tensor)
    assert isinstance(end_time, torch.Tensor)
    assert start_time.item() < end_time.item()
