
import torch
import pytest
from audiofeat.voice.quality import jitter, shimmer, subharmonic_to_harmonic_ratio, normalized_amplitude_quotient, closed_quotient, glottal_closure_time, soft_phonation_index, speed_quotient, vocal_fry_index
from audiofeat.voice.cpp import delta_cpp
from audiofeat.voice.onset import voice_onset_time
from audiofeat.voice.excitation import glottal_to_noise_excitation
from audiofeat.voice.flow import maximum_flow_declination_rate
from audiofeat.voice.nasality import nasality_index
from audiofeat.voice.vocal_tract import vocal_tract_length
from audiofeat.voice.alpha_ratio import alpha_ratio
from audiofeat.voice.hammarberg import hammarberg_index
from audiofeat.voice.harmonic_diff import harmonic_differences

def test_jitter():
    periods = torch.randn(10).abs() + 0.01
    result = jitter(periods)
    assert isinstance(result, torch.Tensor)

def test_shimmer():
    amplitudes = torch.randn(10).abs() + 0.01
    result = shimmer(amplitudes)
    assert isinstance(result, torch.Tensor)

def test_subharmonic_to_harmonic_ratio():
    mag = torch.randn(20).abs() + 0.01
    result = subharmonic_to_harmonic_ratio(mag, f0_bin=5, num_harmonics=3)
    assert isinstance(result, torch.Tensor)

def test_normalized_amplitude_quotient():
    peak_flow = torch.tensor(0.5)
    mfdr = torch.tensor(0.1)
    period = torch.tensor(0.01)
    result = normalized_amplitude_quotient(peak_flow, mfdr, period)
    assert isinstance(result, torch.Tensor)

def test_closed_quotient():
    open_time = torch.tensor(0.001)
    close_time = torch.tensor(0.005)
    period = torch.tensor(0.01)
    result = closed_quotient(open_time, close_time, period)
    assert isinstance(result, torch.Tensor)

def test_glottal_closure_time():
    open_times = torch.randn(10).abs()
    close_times = open_times + torch.randn(10).abs()
    periods = torch.randn(10).abs()
    result = glottal_closure_time(open_times, close_times, periods)
    assert isinstance(result, torch.Tensor)

def test_soft_phonation_index():
    low_band_energy = torch.tensor(10.0)
    high_band_energy = torch.tensor(1.0)
    result = soft_phonation_index(low_band_energy, high_band_energy)
    assert isinstance(result, torch.Tensor)

def test_speed_quotient():
    open_times = torch.randn(10).abs()
    close_times = torch.randn(10).abs()
    result = speed_quotient(open_times, close_times)
    assert isinstance(result, torch.Tensor)

def test_vocal_fry_index():
    f0 = torch.tensor([50.0, 55.0, 60.0, 0.0, 0.0, 40.0, 45.0])
    result = vocal_fry_index(f0)
    assert isinstance(result, torch.Tensor)

def test_delta_cpp():
    cpp = torch.randn(10)
    result = delta_cpp(cpp)
    assert isinstance(result, torch.Tensor)

def test_voice_onset_time():
    audio_data = torch.randn(22050 * 2)
    result = voice_onset_time(audio_data, fs=22050, frame_length=2048, hop_length=512)
    assert isinstance(result, float)

def test_glottal_to_noise_excitation():
    spec = torch.randn(6, 10).abs()
    result = glottal_to_noise_excitation(spec)
    assert isinstance(result, torch.Tensor)

def test_maximum_flow_declination_rate():
    flow = torch.randn(100)
    result = maximum_flow_declination_rate(flow, fs=22050)
    assert isinstance(result, torch.Tensor)

def test_nasality_index():
    nasal = torch.randn(22050)
    oral = torch.randn(22050)
    result = nasality_index(nasal, oral, fs=22050)
    assert isinstance(result, torch.Tensor)

def test_vocal_tract_length():
    F1 = 500.0
    F2 = 1500.0
    result = vocal_tract_length(F1, F2)
    assert isinstance(result, float)

def test_alpha_ratio():
    audio_data = torch.randn(22050)
    result = alpha_ratio(audio_data, fs=22050)
    assert isinstance(result, torch.Tensor)

def test_hammarberg_index():
    audio_data = torch.randn(22050)
    result = hammarberg_index(audio_data, fs=22050)
    assert isinstance(result, torch.Tensor)

def test_harmonic_differences():
    magnitudes = torch.randn(1025)
    result = harmonic_differences(magnitudes, f0_hz=100.0, fs=22050)
    assert isinstance(result, torch.Tensor)
