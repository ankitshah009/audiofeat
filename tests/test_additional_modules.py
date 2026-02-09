import pytest
import torch
import torchaudio.transforms as T

from audiofeat.emotion import detect_emotion
from audiofeat.noise_reduction import noise_reduction
from audiofeat.rhythm.beat import beat_detection
from audiofeat.spectral.bandwidth import spectral_bandwidth
from audiofeat.spectral.cqt import cqt
from audiofeat.spectral.gfcc import gfcc
from audiofeat.spectral.hps import hps
from audiofeat.spectral.irregularity import spectral_irregularity
from audiofeat.spectral.key import key_detect
from audiofeat.spectral.log_mel_spectrogram import log_mel_spectrogram
from audiofeat.spectral.roughness import spectral_roughness
from audiofeat.temporal.beat import beat_track as temporal_beat_track
from audiofeat.temporal.centroid import temporal_centroid
from audiofeat.temporal.loudness import loudness
from audiofeat.temporal.onset import onset_detect
from audiofeat.temporal.teager import teager_energy_operator
from audiofeat.temporal.tristimulus import tristimulus
from audiofeat.voice.jitter import jitter_ddp, jitter_local, jitter_ppq5
from audiofeat.voice.shimmer import shimmer_apq3, shimmer_dda, shimmer_local, shimmer_local_db


def _tone(sr: int = 22050, freq: float = 220.0, dur: float = 1.0) -> torch.Tensor:
    t = torch.arange(int(sr * dur), dtype=torch.float32) / sr
    return torch.sin(2 * torch.pi * freq * t)


def test_spectral_bandwidth_gfcc_hps_key_logmel():
    x = _tone()
    bw = spectral_bandwidth(x, sample_rate=22050, n_fft=1024, hop_length=256)
    assert bw.ndim == 1 and bw.numel() > 0

    g = gfcc(x, sample_rate=22050, n_gfcc=13, n_fft=1024, hop_length=256, n_bands=32)
    assert g.shape[0] == 13

    harm, perc = hps(x, sample_rate=22050, n_fft=512, hop_length=128, margin_h=1.0, margin_p=1.0)
    assert harm.ndim == 1 and perc.ndim == 1
    assert harm.numel() > 0 and perc.numel() > 0

    key = key_detect(x, sample_rate=22050, n_fft=1024, hop_length=256)
    assert isinstance(key, str)
    assert "major" in key or "minor" in key

    lm = log_mel_spectrogram(x, sample_rate=22050, n_fft=1024, hop_length=256, n_mels=32)
    assert lm.shape[0] == 32


def test_spectral_cqt_if_available():
    if not hasattr(T, "CQT"):
        pytest.skip("torchaudio.transforms.CQT is unavailable in this torchaudio build.")
    x = _tone()
    out = cqt(x, sample_rate=22050, hop_length=256, n_bins=24, bins_per_octave=12)
    assert out.shape[0] == 24


def test_roughness_irregularity_teager():
    x = _tone(freq=330.0)
    rough = spectral_roughness(x, sample_rate=22050)
    irr = spectral_irregularity(x, n_fft=1024)
    teo = teager_energy_operator(x)
    assert torch.isfinite(rough)
    assert torch.isfinite(irr)
    assert torch.isfinite(teo)


def test_temporal_advanced_extractors():
    x = _tone(dur=2.0)
    cent = temporal_centroid(x, sample_rate=22050)
    onsets = onset_detect(x, sample_rate=22050, n_fft=512, hop_length=128, backtrack=True)
    tempo, beats = temporal_beat_track(
        x,
        sample_rate=22050,
        n_fft=512,
        hop_length=128,
        tempo_min=60.0,
        tempo_max=240.0,
    )
    loud = loudness(x, sample_rate=22050)
    tri = tristimulus(x, sample_rate=22050, n_fft=512, hop_length=128)
    bpm, confidence = beat_detection(x, sample_rate=22050)

    assert torch.isfinite(cent)
    assert onsets.ndim == 1
    assert torch.isfinite(tempo)
    assert beats.ndim == 1
    assert torch.isfinite(loud).all()
    assert tri.shape == (3,)
    assert 0.0 <= float(tri.sum().item()) <= 1.0001
    assert bpm >= 0.0
    assert confidence <= 1.0


def test_jitter_shimmer_variants():
    periods = torch.tensor([0.010, 0.011, 0.0105, 0.0098, 0.0102, 0.0101], dtype=torch.float32)
    amps = torch.tensor([1.0, 0.98, 1.02, 1.01, 0.99, 1.03], dtype=torch.float32)

    assert jitter_local(periods).item() >= 0.0
    assert jitter_ppq5(periods).item() >= 0.0
    assert jitter_ddp(periods).item() >= 0.0

    assert shimmer_local(amps).item() >= 0.0
    assert shimmer_local_db(amps).item() >= 0.0
    assert shimmer_apq3(amps).item() >= 0.0
    assert shimmer_dda(amps).item() >= 0.0

    assert jitter_local(torch.tensor([0.01])).item() == 0.0
    assert shimmer_local(torch.tensor([1.0])).item() == 0.0


def test_emotion_and_noise_reduction_execute():
    x = _tone(dur=1.5)
    label = detect_emotion(x, sample_rate=22050)
    denoised = noise_reduction(x, sample_rate=22050, threshold=0.02)
    assert isinstance(label, str)
    assert denoised.ndim == 1
    assert denoised.numel() > 0
