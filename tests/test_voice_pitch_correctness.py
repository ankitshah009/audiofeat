"""Mathematical-correctness tests for the VOICE and PITCH modules.

Each test uses a synthetic signal with known ground truth (sines, harmonic
series, periodic period/amplitude sequences) so that the *value* a function
returns can be checked, not merely its type. Edge cases (silence, length-1 /
short inputs) assert finiteness and absence of crashes.

These tests pin the fixes for the bugs found in the static audit:
  1  voice/onset.py            true autocorrelation (was identity) + NaN sentinel
  2  voice/harmonic_diff.py    correct rfft bin width + peak-picking
  3  voice/quality.py          SHR bounds-safety; SPI = low/high; jitter/shimmer delegation
  4  voice/excitation.py       GNE bounded in [0,1) -> finite output
  5  voice/cpp.py              power cepstrum / 10*log10; peak on regression residual
  6  voice/{alpha_ratio,hammarberg,nasality}.py  framing (no truncation); nasalance %
  7  voice/vocal_tract.py      L = c/(4*F1) ~ 17.2 cm for F1=500
  8  voice/{jitter,shimmer}.py jitter_rap, shimmer_apq5/apq11
  9  pitch/semitone.py         single voiced frame -> 0 (was NaN)
 10  pitch/f0.py               YIN: proper difference fn + parabolic interpolation
"""

import math

import torch
import pytest

from audiofeat.voice.onset import voice_onset_time
from audiofeat.voice.harmonic_diff import harmonic_differences
from audiofeat.voice.quality import (
    subharmonic_to_harmonic_ratio,
    soft_phonation_index,
    jitter,
    shimmer,
)
from audiofeat.voice.excitation import glottal_to_noise_excitation
from audiofeat.voice.cpp import cepstral_peak_prominence
from audiofeat.voice.alpha_ratio import alpha_ratio
from audiofeat.voice.hammarberg import hammarberg_index
from audiofeat.voice.nasality import nasality_index
from audiofeat.voice.vocal_tract import vocal_tract_length
from audiofeat.voice.jitter import jitter_rap, jitter_local
from audiofeat.voice.shimmer import shimmer_apq5, shimmer_apq11, shimmer_local
from audiofeat.pitch.f0 import fundamental_frequency_yin
from audiofeat.pitch.semitone import semitone_sd

SR = 22050


def _sine(freq, dur=1.0, sr=SR, amp=1.0):
    t = torch.arange(int(dur * sr), dtype=torch.float32) / sr
    return amp * torch.sin(2 * math.pi * freq * t)


def _harmonic_series(f0, n_harm=20, dur=0.5, sr=SR):
    t = torch.arange(int(dur * sr), dtype=torch.float32) / sr
    out = torch.zeros_like(t)
    for k in range(1, n_harm + 1):
        if k * f0 < sr / 2:
            out = out + torch.sin(2 * math.pi * k * f0 * t)
    return out


# --------------------------------------------------------------------------- #
# 1. voice_onset_time: true autocorrelation + known VOT gap                    #
# --------------------------------------------------------------------------- #
def test_voice_onset_time_known_gap():
    torch.manual_seed(0)
    sr = SR
    hop, fl = 256, 1024

    silence = torch.zeros(int(0.30 * sr))
    burst = torch.zeros(int(0.05 * sr))
    burst[:50] = torch.randn(50) * 5.0           # transient click (unvoiced)
    gap = torch.zeros(int(0.20 * sr))            # silent VOT gap
    voiced = _sine(150.0, dur=0.5, sr=sr)        # voiced onset

    x = torch.cat([silence, burst, gap, voiced])
    vot = voice_onset_time(x, fs=sr, frame_length=fl, hop_length=hop)

    assert isinstance(vot, torch.Tensor)
    assert torch.isfinite(vot)
    expected = (burst.numel() + gap.numel()) / sr  # ~0.25 s burst->voicing
    # Frame quantization (~hop/sr per frame) -> allow ~3 frames tolerance.
    assert abs(float(vot) - expected) < 0.05


def test_voice_onset_time_nan_when_silent():
    # Pure silence -> no burst -> NaN sentinel (not 0.0, not a crash).
    x = torch.zeros(SR)
    vot = voice_onset_time(x, fs=SR, frame_length=1024, hop_length=256)
    assert isinstance(vot, torch.Tensor)
    assert torch.isnan(vot)


# --------------------------------------------------------------------------- #
# 2. harmonic_differences: correct bin width + peak-picking                    #
# --------------------------------------------------------------------------- #
def test_harmonic_differences_known_ratio():
    sr = SR
    f0 = 200.0
    n = 8192
    a1, a2 = 1.0, 0.3
    t = torch.arange(n, dtype=torch.float32) / sr
    x = a1 * torch.sin(2 * math.pi * f0 * t) + a2 * torch.sin(2 * math.pi * 2 * f0 * t)
    mag = torch.fft.rfft(x * torch.hann_window(n)).abs()  # length n//2 + 1

    hd = harmonic_differences(mag, f0_hz=f0, fs=sr, h_indices=[1, 2])
    expected = 20 * math.log10(a1 / a2)  # ~10.46 dB
    assert hd.numel() == 1
    # If the bin width were wrong (fs/numel), the harmonics would land in the
    # wrong bins and this would be far off; require < 1.5 dB error.
    assert abs(float(hd[0]) - expected) < 1.5


def test_harmonic_differences_short_spectrum_safe():
    # n_bins < 2 -> empty result, no crash.
    out = harmonic_differences(torch.tensor([1.0]), f0_hz=100.0, fs=SR)
    assert out.numel() == 0


# --------------------------------------------------------------------------- #
# 3. quality: SHR bounds-safety, SPI orientation, jitter/shimmer delegation    #
# --------------------------------------------------------------------------- #
def test_shr_out_of_bounds_is_safe():
    torch.manual_seed(0)
    mag = torch.randn(20).abs() + 0.01
    # 10 harmonics * f0_bin 5 = index 50 >> 20: must not raise IndexError.
    r = subharmonic_to_harmonic_ratio(mag, f0_bin=5, num_harmonics=10)
    assert isinstance(r, torch.Tensor)
    assert torch.isfinite(r)


def test_soft_phonation_index_orientation():
    # SPI = 10*log10(low/high). low > high -> POSITIVE.
    spi = soft_phonation_index(torch.tensor(10.0), torch.tensor(1.0))
    assert float(spi) == pytest.approx(10.0, abs=1e-3)
    spi2 = soft_phonation_index(torch.tensor(1.0), torch.tensor(10.0))
    assert float(spi2) == pytest.approx(-10.0, abs=1e-3)


def test_jitter_shimmer_delegate_to_percent():
    # The convenience jitter/shimmer must match the canonical percent versions
    # (previously they returned a raw ratio -> silent 100x discrepancy).
    periods = torch.tensor([0.010, 0.011, 0.0105, 0.0098, 0.0102], dtype=torch.float32)
    amps = torch.tensor([1.0, 0.98, 1.02, 1.01, 0.99], dtype=torch.float32)
    assert float(jitter(periods)) == pytest.approx(float(jitter_local(periods)), abs=1e-6)
    assert float(shimmer(amps)) == pytest.approx(float(shimmer_local(amps)), abs=1e-6)
    # N < 2 guard.
    assert float(jitter(torch.tensor([0.01]))) == 0.0
    assert float(shimmer(torch.tensor([1.0]))) == 0.0


# --------------------------------------------------------------------------- #
# 4. glottal_to_noise_excitation: always finite                               #
# --------------------------------------------------------------------------- #
def test_gne_is_finite():
    torch.manual_seed(0)
    assert torch.isfinite(glottal_to_noise_excitation(torch.randn(6, 100).abs()))
    # Perfectly correlated bands push the coefficient toward 1: must stay finite.
    assert torch.isfinite(glottal_to_noise_excitation(torch.ones(6, 50)))
    # 1-D input (single band) handled.
    assert torch.isfinite(glottal_to_noise_excitation(torch.randn(40).abs()))
    # Anti-correlated / mixed signs: no log of a negative.
    assert torch.isfinite(glottal_to_noise_excitation(torch.randn(8, 64)))


# --------------------------------------------------------------------------- #
# 5. cepstral_peak_prominence: harmonic >> noise, magnitude in plausible band  #
# --------------------------------------------------------------------------- #
def test_cpp_harmonic_vs_noise():
    torch.manual_seed(0)
    sr = SR
    harm = _harmonic_series(150.0, n_harm=30, dur=0.6, sr=sr)
    noise = torch.randn(int(0.6 * sr))

    cpp_h = float(cepstral_peak_prominence(harm, sr).mean())
    cpp_n = float(cepstral_peak_prominence(noise, sr).mean())

    assert cpp_h > cpp_n + 2.0  # clear, periodic harmonic series dominates noise
    # Power-cepstrum / 10*log10 keeps this in a clinically plausible band; the
    # old magnitude-cepstrum / 8.686 scaling roughly doubled it (40-60 territory
    # for very strong signals). Assert it stays well below that.
    assert 0.5 < cpp_h < 30.0


def test_cpp_silence_finite():
    out = cepstral_peak_prominence(torch.zeros(SR), SR)
    assert torch.isfinite(out).all()


# --------------------------------------------------------------------------- #
# 6. alpha_ratio / hammarberg / nasality: framing (no truncation)             #
# --------------------------------------------------------------------------- #
def _two_tone(dur, sr=SR):
    t = torch.arange(int(dur * sr), dtype=torch.float32) / sr
    return torch.sin(2 * math.pi * 300 * t) + 0.5 * torch.sin(2 * math.pi * 3000 * t)


def test_alpha_ratio_truncation_invariance():
    # A long signal with identical spectral content must give ~the same answer
    # as a short one. The old code truncated to the first n_fft samples and
    # returned a wrong (taper-attenuated) value for long signals.
    short = _two_tone(0.2)
    long = _two_tone(5.0)
    a_s = float(alpha_ratio(short, SR))
    a_l = float(alpha_ratio(long, SR))
    assert abs(a_s - a_l) < 0.5


def test_hammarberg_truncation_invariance():
    short = _two_tone(0.2)
    long = _two_tone(5.0)
    h_s = float(hammarberg_index(short, SR))
    h_l = float(hammarberg_index(long, SR))
    assert abs(h_s - h_l) < 0.5


def test_nasality_is_percentage():
    torch.manual_seed(0)
    sr = SR
    # Equal-power nasal and oral within the 300-800 Hz band -> nasalance ~50%.
    band_tone = _sine(500.0, dur=1.0, sr=sr)
    nas = nasality_index(band_tone.clone(), band_tone.clone(), fs=sr)
    assert 0.0 <= float(nas) <= 100.0
    assert float(nas) == pytest.approx(50.0, abs=1.0)
    # All energy nasal -> approaches 100%.
    nas_hi = nasality_index(band_tone, torch.zeros_like(band_tone), fs=sr)
    assert float(nas_hi) > 99.0


def test_alpha_hammarberg_nasality_silence_finite():
    sil = torch.zeros(SR)
    assert torch.isfinite(alpha_ratio(sil, SR))
    assert torch.isfinite(hammarberg_index(sil, SR))
    assert torch.isfinite(nasality_index(sil, sil, SR))


# --------------------------------------------------------------------------- #
# 7. vocal_tract_length: canonical L = c/(4*F1)                                #
# --------------------------------------------------------------------------- #
def test_vocal_tract_length_canonical():
    # F1 = 500 Hz, c = 35000 cm/s -> 17.5 cm (~17.2 cm textbook adult VTL).
    L = vocal_tract_length(500.0)
    assert isinstance(L, float)
    assert L == pytest.approx(17.5, abs=0.5)


def test_vocal_tract_length_guard_and_f2_optional():
    with pytest.raises(ValueError):
        vocal_tract_length(0.0)
    with pytest.raises(ValueError):
        vocal_tract_length(-100.0)
    # F2 optional; still returns a sane positive length.
    L = vocal_tract_length(500.0, 1500.0)
    assert L > 0


# --------------------------------------------------------------------------- #
# 8. jitter_rap / shimmer_apq5 / shimmer_apq11: known closed-form              #
# --------------------------------------------------------------------------- #
def test_jitter_rap_constant_and_alternating():
    # Constant periods -> zero perturbation.
    assert float(jitter_rap(torch.full((20,), 0.01))) == pytest.approx(0.0, abs=1e-4)

    # Alternating T0 +/- d. For the 3-point RAP the centre is T0+/-d and the
    # mean of three consecutive periods is T0 -/+ d/3, so |centre - mean| = 4d/3
    # for every interior point => RAP = (4d/3)/T0.
    T0, d = 0.01, 0.0005
    alt = torch.tensor([T0 + (d if i % 2 == 0 else -d) for i in range(20)])
    expected_pct = (4.0 * d / 3.0) / T0 * 100.0  # 6.667 %
    assert float(jitter_rap(alt)) == pytest.approx(expected_pct, abs=1e-2)


def test_shimmer_apq5_apq11_constant():
    assert float(shimmer_apq5(torch.full((20,), 1.0))) == pytest.approx(0.0, abs=1e-3)
    assert float(shimmer_apq11(torch.full((20,), 1.0))) == pytest.approx(0.0, abs=1e-3)


def test_shimmer_apq_short_input_guard():
    # Below window length -> 0.0, no crash.
    assert float(shimmer_apq5(torch.ones(3))) == 0.0
    assert float(shimmer_apq11(torch.ones(8))) == 0.0
    assert float(jitter_rap(torch.ones(2))) == 0.0


# --------------------------------------------------------------------------- #
# 9. semitone_sd: single voiced frame -> 0 (was NaN)                          #
# --------------------------------------------------------------------------- #
def test_semitone_sd_single_voiced_frame():
    out = semitone_sd(torch.tensor([100.0, 0.0, 0.0]))
    assert torch.isfinite(out)
    assert float(out) == 0.0


def test_semitone_sd_no_voiced_frames():
    out = semitone_sd(torch.zeros(5))
    assert torch.isfinite(out)
    assert float(out) == 0.0


# --------------------------------------------------------------------------- #
# 10. YIN: synthetic sines -> median f0 within 1% (parabolic interpolation)    #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("freq", [100.0, 220.0, 440.0])
def test_yin_accuracy(freq):
    sr = SR
    x = _sine(freq, dur=1.0, sr=sr)
    f0 = fundamental_frequency_yin(
        x, fs=sr, frame_length=2048, hop_length=512, fmin=50, fmax=600
    )
    median = float(f0.median())
    assert abs(median - freq) / freq < 0.01  # within 1 %


def test_yin_matches_librosa():
    librosa = pytest.importorskip("librosa")
    import numpy as np

    sr = SR
    x = _sine(220.0, dur=1.0, sr=sr)
    ours = float(
        fundamental_frequency_yin(
            x, fs=sr, frame_length=2048, hop_length=512, fmin=50, fmax=600
        ).median()
    )
    ref = float(
        np.median(librosa.yin(x.numpy(), fmin=50, fmax=600, sr=sr,
                              frame_length=2048, hop_length=512))
    )
    assert abs(ours - ref) < 1.0  # agree to ~1 Hz


def test_yin_silence_finite():
    out = fundamental_frequency_yin(
        torch.zeros(SR), fs=SR, frame_length=2048, hop_length=512
    )
    assert torch.isfinite(out).all()
