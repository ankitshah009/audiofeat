import torch
import pytest
from audiofeat.spectral.centroid import spectral_centroid
from audiofeat.spectral.rolloff import spectral_rolloff
from audiofeat.spectral.flux import spectral_flux
from audiofeat.spectral.flatness import spectral_flatness
from audiofeat.spectral.entropy import spectral_entropy
from audiofeat.spectral.skewness import spectral_skewness
from audiofeat.spectral.energy_ratio import low_high_energy_ratio
from audiofeat.spectral.harmonic import harmonic_richness_factor, inharmonicity_index
from audiofeat.spectral.phase import phase_coherence
from audiofeat.spectral.formants import formant_frequencies, formant_bandwidths, formant_dispersion
from audiofeat.spectral.sibilance import sibilant_spectral_peak_frequency
from audiofeat.spectral.spectrogram import linear_spectrogram, mel_spectrogram
from audiofeat.spectral.mfcc import mfcc

def test_spectral_centroid():
    audio_data = torch.randn(22050 * 5) # 5 seconds of audio
    result = spectral_centroid(audio_data)
    assert isinstance(result, torch.Tensor)
    assert result.shape[0] > 0

def test_spectral_rolloff():
    audio_data = torch.randn(22050 * 5) # 5 seconds of audio
    result = spectral_rolloff(audio_data)
    assert isinstance(result, torch.Tensor)
    assert result.shape[0] > 0

def test_spectral_flux():
    audio_data = torch.randn(22050 * 5) # 5 seconds of audio
    result = spectral_flux(audio_data)
    assert isinstance(result, torch.Tensor)
    assert result.shape[0] > 0

def test_spectral_flatness():
    audio_data = torch.randn(22050 * 5) # 5 seconds of audio
    result = spectral_flatness(audio_data)
    assert isinstance(result, torch.Tensor)
    assert result.shape[0] > 0

def test_spectral_entropy():
    audio_data = torch.randn(2048)
    result = spectral_entropy(audio_data, n_fft=2048)
    assert isinstance(result, torch.Tensor)

def test_spectral_skewness():
    audio_data = torch.randn(2048)
    skew, kurt = spectral_skewness(audio_data, n_fft=2048)
    assert isinstance(skew, torch.Tensor)
    assert isinstance(kurt, torch.Tensor)

def test_low_high_energy_ratio():
    audio_data = torch.randn(22050)
    result = low_high_energy_ratio(audio_data, fs=22050)
    assert isinstance(result, torch.Tensor)

def test_harmonic_richness_factor():
    magnitudes = torch.randn(10)
    result = harmonic_richness_factor(magnitudes)
    assert isinstance(result, torch.Tensor)

def test_inharmonicity_index():
    peaks = torch.randn(10)
    result = inharmonicity_index(peaks, f0=100)
    assert isinstance(result, torch.Tensor)

def test_phase_coherence():
    phases = torch.randn(10)
    result = phase_coherence(phases)
    assert isinstance(result, torch.Tensor)

def test_formant_frequencies():
    audio_data = torch.randn(22050 * 5)
    result = formant_frequencies(audio_data, fs=22050, order=10)
    assert isinstance(result, torch.Tensor)

def test_formant_bandwidths():
    a = torch.randn(10)
    result = formant_bandwidths(a, fs=22050)
    assert isinstance(result, torch.Tensor)

def test_formant_dispersion():
    formants = torch.randn(10)
    result = formant_dispersion(formants)
    assert isinstance(result, torch.Tensor)

def test_sibilant_spectral_peak_frequency():
    audio_data = torch.randn(22050)
    result = sibilant_spectral_peak_frequency(audio_data, fs=22050)
    assert isinstance(result, torch.Tensor)

def test_linear_spectrogram():
    audio_data = torch.randn(22050 * 5)
    result = linear_spectrogram(audio_data)
    assert isinstance(result, torch.Tensor)
    assert result.shape[0] > 0
    assert result.shape[1] > 0

def test_mel_spectrogram():
    audio_data = torch.randn(22050 * 5)
    result = mel_spectrogram(audio_data, sample_rate=22050)
    assert isinstance(result, torch.Tensor)
    assert result.shape[0] > 0
    assert result.shape[1] > 0

def test_mfcc():
    audio_data = torch.randn(22050 * 5)
    result = mfcc(audio_data, sample_rate=22050)
    assert isinstance(result, torch.Tensor)
    assert result.shape[0] > 0
    assert result.shape[1] > 0