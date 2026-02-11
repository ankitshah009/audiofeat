import torch
import pytest
from scipy.signal import iirpeak, lfilter
from audiofeat.spectral.centroid import spectral_centroid
from audiofeat.spectral.rolloff import spectral_rolloff
from audiofeat.spectral.flux import spectral_flux
from audiofeat.spectral.flatness import spectral_flatness
from audiofeat.spectral.entropy import spectral_entropy
from audiofeat.spectral.moments import spectral_skewness, spectral_spread
from audiofeat.spectral.energy_ratio import low_high_energy_ratio
from audiofeat.spectral.harmonic import harmonic_richness_factor, inharmonicity_index
from audiofeat.spectral.phase import phase_coherence
from audiofeat.spectral.formants import formant_frequencies, formant_bandwidths, formant_dispersion
from audiofeat.spectral.sibilance import sibilant_spectral_peak_frequency
from audiofeat.spectral.spectrogram import linear_spectrogram, mel_spectrogram, cqt_spectrogram
from audiofeat.spectral.mfcc import mfcc
from audiofeat.spectral.slope import spectral_slope
from audiofeat.spectral.crest import spectral_crest_factor
from audiofeat.spectral.contrast import spectral_contrast
from audiofeat.spectral.hnr import harmonic_to_noise_ratio
from audiofeat.spectral.deviation import spectral_deviation
from audiofeat.spectral.chroma import chroma
from audiofeat.spectral.tonnetz import tonnetz

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

    # Test with a different rolloff_percent
    result_90 = spectral_rolloff(audio_data, rolloff_percent=0.90)
    assert isinstance(result_90, torch.Tensor)
    assert result_90.shape[0] > 0

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

def test_spectral_spread():
    audio_data = torch.randn(2048)
    result = spectral_spread(audio_data, n_fft=2048, sample_rate=22050)
    assert isinstance(result, torch.Tensor)

def test_spectral_slope():
    audio_data = torch.randn(2048)
    result = spectral_slope(audio_data, n_fft=2048)
    assert isinstance(result, torch.Tensor)

def test_spectral_crest_factor():
    audio_data = torch.randn(2048)
    result = spectral_crest_factor(audio_data, n_fft=2048)
    assert isinstance(result, torch.Tensor)

def test_spectral_contrast():
    audio_data = torch.randn(2048)
    result = spectral_contrast(audio_data, fs=22050)
    assert isinstance(result, torch.Tensor)
    assert result.shape[0] > 0
    # librosa-aligned output is in dB; values are finite and non-negative
    assert torch.all(torch.isfinite(result))

def test_harmonic_to_noise_ratio():
    harmonic_energy = torch.tensor(10.0)
    noise_energy = torch.tensor(1.0)
    result = harmonic_to_noise_ratio(harmonic_energy, noise_energy)
    assert isinstance(result, torch.Tensor)

def test_spectral_deviation():
    audio_data = torch.randn(2048)
    result = spectral_deviation(audio_data, n_fft=2048)
    assert isinstance(result, torch.Tensor)

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

def test_cqt_spectrogram():
    audio_data = torch.randn(22050 * 5)
    result = cqt_spectrogram(audio_data, sample_rate=22050)
    assert isinstance(result, torch.Tensor)
    assert result.shape[0] > 0
    assert result.shape[1] > 0

def test_mfcc():
    audio_data = torch.randn(22050 * 5)
    result = mfcc(audio_data, sample_rate=22050)
    assert isinstance(result, torch.Tensor)
    assert result.shape[0] > 0
    assert result.shape[1] > 0

def test_chroma():
    audio_data = torch.randn(22050 * 5)
    result = chroma(audio_data, sample_rate=22050)
    assert isinstance(result, torch.Tensor)
    assert result.shape[0] == 12
    assert result.shape[1] > 0

def test_tonnetz():
    chroma_features = torch.randn(12, 100) # Dummy chroma features
    result = tonnetz(chroma_features)
    assert isinstance(result, torch.Tensor)
    assert result.shape[0] == 6
    assert result.shape[1] > 0


def test_formant_frequencies_speech_like_signal():
    sr = 16000
    duration = 1.0
    t = torch.arange(int(sr * duration), dtype=torch.float32) / sr
    # Harmonic-rich glottal-like source
    source = torch.zeros_like(t)
    f0 = 120.0
    for k in range(1, 15):
        source += torch.sin(2 * torch.pi * (k * f0) * t) / k
    source_np = source.numpy()

    # Shape with two broad resonances near F1/F2.
    for center_hz, q in [(500.0, 3.0), (1500.0, 4.0)]:
        b, a = iirpeak(center_hz / (sr / 2), q)
        source_np = lfilter(b, a, source_np)

    x = torch.from_numpy(source_np.astype("float32"))
    estimated = formant_frequencies(
        x,
        fs=sr,
        order=12,
        num_formants=3,
        max_formant=5000.0,
    )
    assert estimated.shape[0] >= 2
    assert torch.isfinite(estimated[0])
    assert torch.isfinite(estimated[1])
    assert 150.0 <= float(estimated[0].item()) <= 2000.0
    assert float(estimated[1].item()) > float(estimated[0].item())


def test_spectral_centroid_respects_sample_rate():
    sr = 16000
    freq_hz = 1000.0
    t = torch.arange(sr * 2, dtype=torch.float32) / sr
    x = torch.sin(2 * torch.pi * freq_hz * t)

    centroid_correct_sr = spectral_centroid(
        x,
        frame_length=1024,
        hop_length=256,
        sample_rate=sr,
    ).median()
    centroid_wrong_sr = spectral_centroid(
        x,
        frame_length=1024,
        hop_length=256,
        sample_rate=22050,
    ).median()

    assert abs(float(centroid_correct_sr.item()) - freq_hz) < 60.0
    assert abs(float(centroid_wrong_sr.item()) - float(centroid_correct_sr.item())) > 120.0


def test_spectral_rolloff_respects_sample_rate_and_validates_percent():
    sr = 16000
    freq_hz = 1200.0
    t = torch.arange(sr * 2, dtype=torch.float32) / sr
    x = torch.sin(2 * torch.pi * freq_hz * t)

    rolloff_correct_sr = spectral_rolloff(
        x,
        frame_length=1024,
        hop_length=256,
        sample_rate=sr,
    ).median()
    rolloff_wrong_sr = spectral_rolloff(
        x,
        frame_length=1024,
        hop_length=256,
        sample_rate=22050,
    ).median()

    correct_val = float(rolloff_correct_sr.item())
    wrong_val = float(rolloff_wrong_sr.item())
    assert correct_val > 0.0
    assert wrong_val > correct_val
    scale_ratio = wrong_val / correct_val
    expected_ratio = 22050.0 / sr
    assert abs(scale_ratio - expected_ratio) < 0.05

    with pytest.raises(ValueError):
        spectral_rolloff(x, rolloff_percent=0.0, sample_rate=sr)
