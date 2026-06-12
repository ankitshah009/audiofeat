"""Mathematical-correctness tests for the spectral module.

These tests cover three categories:

1. librosa-parity: ``spectral_rolloff``, ``spectral_flatness`` and
   ``spectral_bandwidth`` are checked against ``librosa`` on a fixed seeded
   noise signal and a pure tone.  ``rolloff``/``flatness`` use the project's
   non-centered, unwindowed framing, so the librosa reference is computed with
   ``window='boxcar', center=False``.  ``spectral_bandwidth`` wraps torchaudio's
   (Hann-windowed, centered) spectrogram, so its reference uses
   ``window='hann', center=True, pad_mode='reflect'``.
2. Known-answer: flatness of white noise vs. a pure tone, crest factor of a
   tone, and the sign of the spectral slope on an exponentially-decaying
   spectrum.
3. Edge cases: every fixed feature must return a finite value on silence
   (``torch.zeros(2048)``) and on a length-1 input (``torch.tensor([1.0])``)
   instead of NaN / inf / a divide-by-(L-1) crash.
"""

import numpy as np
import pytest
import torch

from audiofeat.spectral.bandwidth import spectral_bandwidth
from audiofeat.spectral.crest import spectral_crest_factor
from audiofeat.spectral.deviation import spectral_deviation
from audiofeat.spectral.energy_ratio import low_high_energy_ratio
from audiofeat.spectral.entropy import spectral_entropy
from audiofeat.spectral.flatness import spectral_flatness
from audiofeat.spectral.irregularity import spectral_irregularity
from audiofeat.spectral.moments import spectral_skewness, spectral_spread
from audiofeat.spectral.rolloff import spectral_rolloff
from audiofeat.spectral.roughness import spectral_roughness
from audiofeat.spectral.sharpness import spectral_sharpness
from audiofeat.spectral.sibilance import sibilant_spectral_peak_frequency
from audiofeat.spectral.slope import spectral_slope
from audiofeat.spectral.tonality import spectral_tonality

SR = 22050


@pytest.fixture
def noise_signal():
    """2 seconds of reproducible noise at 22050 Hz."""
    torch.manual_seed(42)
    return torch.randn(SR * 2), SR


@pytest.fixture
def tone_signal():
    """3 seconds of a 440 Hz tone at 22050 Hz."""
    t = torch.arange(SR * 3, dtype=torch.float32) / SR
    return torch.sin(2 * torch.pi * 440.0 * t), SR


# ── librosa parity ──────────────────────────────────────────────────


def test_spectral_rolloff_matches_librosa_noise(noise_signal):
    librosa = pytest.importorskip("librosa")
    x, sr = noise_signal
    # Project framing applies no window and no centering -> boxcar, center=False.
    s = np.abs(librosa.stft(x.numpy(), n_fft=2048, hop_length=512,
                            center=False, window="boxcar"))
    ref = librosa.feature.spectral_rolloff(S=s, sr=sr, roll_percent=0.85)[0]
    ours = spectral_rolloff(x, frame_length=2048, hop_length=512,
                            rolloff_percent=0.85, sample_rate=sr).numpy()
    np.testing.assert_allclose(ours, ref, atol=1e-4)


def test_spectral_rolloff_matches_librosa_tone(tone_signal):
    librosa = pytest.importorskip("librosa")
    x, sr = tone_signal
    s = np.abs(librosa.stft(x.numpy(), n_fft=2048, hop_length=512,
                            center=False, window="boxcar"))
    ref = librosa.feature.spectral_rolloff(S=s, sr=sr, roll_percent=0.85)[0]
    ours = spectral_rolloff(x, frame_length=2048, hop_length=512,
                            rolloff_percent=0.85, sample_rate=sr).numpy()
    np.testing.assert_allclose(ours, ref, atol=1e-4)


def test_spectral_flatness_matches_librosa_noise(noise_signal):
    librosa = pytest.importorskip("librosa")
    x, sr = noise_signal
    s = np.abs(librosa.stft(x.numpy(), n_fft=2048, hop_length=512,
                            center=False, window="boxcar"))
    ref = librosa.feature.spectral_flatness(S=s)[0]  # power=2.0, amin=1e-10
    ours = spectral_flatness(x, frame_length=2048, hop_length=512).numpy()
    np.testing.assert_allclose(ours, ref, atol=1e-5)


def test_spectral_flatness_matches_librosa_tone(tone_signal):
    librosa = pytest.importorskip("librosa")
    x, sr = tone_signal
    s = np.abs(librosa.stft(x.numpy(), n_fft=2048, hop_length=512,
                            center=False, window="boxcar"))
    ref = librosa.feature.spectral_flatness(S=s)[0]
    ours = spectral_flatness(x, frame_length=2048, hop_length=512).numpy()
    np.testing.assert_allclose(ours, ref, atol=1e-5)


def test_spectral_bandwidth_matches_librosa_noise(noise_signal):
    librosa = pytest.importorskip("librosa")
    x, sr = noise_signal
    # spectral_bandwidth wraps torchaudio.transforms.Spectrogram
    # (Hann window, center=True, reflect padding).
    s = np.abs(librosa.stft(x.numpy(), n_fft=2048, hop_length=512,
                            center=True, window="hann", pad_mode="reflect"))
    ref = librosa.feature.spectral_bandwidth(S=s, sr=sr)[0]
    ours = spectral_bandwidth(x, sample_rate=sr, n_fft=2048, hop_length=512).numpy()
    n = min(len(ours), len(ref))
    np.testing.assert_allclose(ours[:n], ref[:n], atol=1e-2)


def test_spectral_bandwidth_matches_librosa_tone(tone_signal):
    librosa = pytest.importorskip("librosa")
    x, sr = tone_signal
    s = np.abs(librosa.stft(x.numpy(), n_fft=2048, hop_length=512,
                            center=True, window="hann", pad_mode="reflect"))
    ref = librosa.feature.spectral_bandwidth(S=s, sr=sr)[0]
    ours = spectral_bandwidth(x, sample_rate=sr, n_fft=2048, hop_length=512).numpy()
    n = min(len(ours), len(ref))
    np.testing.assert_allclose(ours[:n], ref[:n], atol=0.2)


# ── known-answer ────────────────────────────────────────────────────


def test_flatness_white_noise_high_tone_low(noise_signal, tone_signal):
    noise, _ = noise_signal
    tone, _ = tone_signal
    noise_flat = float(spectral_flatness(noise).median())
    tone_flat = float(spectral_flatness(tone).median())
    # White noise has a near-flat spectrum (power-spectrum flatness ~0.56);
    # a pure tone has a single dominant line -> flatness ~0.
    assert noise_flat > 0.45
    assert tone_flat < 0.05
    assert noise_flat > 10 * tone_flat


def test_crest_factor_tone_large_noise_small(noise_signal, tone_signal):
    noise, _ = noise_signal
    tone, _ = tone_signal
    crest_tone = float(spectral_crest_factor(tone, n_fft=2048))
    crest_noise = float(spectral_crest_factor(noise[:2048], n_fft=2048))
    # max/mean is >= 1 by construction; a tone concentrates energy in one bin.
    assert crest_tone >= 1.0
    assert crest_noise >= 1.0
    assert crest_tone > 50.0
    assert crest_tone > 10 * crest_noise


def test_slope_negative_on_exponential_decay():
    # Construct a signal whose magnitude spectrum decays as exp(-alpha * f).
    n_fft = 2048
    alpha = 1e-3
    freqs = torch.fft.rfftfreq(n_fft, d=1.0 / SR)
    mag = torch.exp(-alpha * freqs)
    sig = torch.fft.irfft(mag.to(torch.complex64), n=n_fft)
    slope = float(spectral_slope(sig, n_fft=n_fft, sample_rate=SR))
    # log|X| ~ -alpha * f  =>  slope ~ -alpha < 0.
    assert slope < 0.0
    # Rising spectrum -> positive slope (sign sanity check).
    mag_rise = torch.exp(alpha * freqs)
    sig_rise = torch.fft.irfft(mag_rise.to(torch.complex64), n=n_fft)
    assert float(spectral_slope(sig_rise, n_fft=n_fft, sample_rate=SR)) > 0.0


def test_slope_is_per_hz_not_per_bin():
    # The slope must scale with sample_rate (Hz axis), proving it is no longer
    # expressed per FFT bin.
    n_fft = 2048
    alpha = 1e-3
    freqs = torch.fft.rfftfreq(n_fft, d=1.0 / SR)
    mag = torch.exp(-alpha * freqs)
    sig = torch.fft.irfft(mag.to(torch.complex64), n=n_fft)
    slope_22k = float(spectral_slope(sig, n_fft=n_fft, sample_rate=22050))
    slope_44k = float(spectral_slope(sig, n_fft=n_fft, sample_rate=44100))
    assert abs(slope_22k - slope_44k) > 1e-9


# ── edge cases: silence + length-1 ──────────────────────────────────

_SILENCE = torch.zeros(2048)
_LEN1 = torch.tensor([1.0])

_EDGE_CASES = {
    "rolloff": lambda s: spectral_rolloff(s),
    "flatness": lambda s: spectral_flatness(s),
    "crest": lambda s: spectral_crest_factor(s, n_fft=2048),
    "slope": lambda s: spectral_slope(s, n_fft=2048),
    "skewness": lambda s: spectral_skewness(s, n_fft=2048)[0],
    "kurtosis": lambda s: spectral_skewness(s, n_fft=2048)[1],
    "spread": lambda s: spectral_spread(s, n_fft=2048, sample_rate=SR),
    "entropy": lambda s: spectral_entropy(s, n_fft=2048),
    "deviation": lambda s: spectral_deviation(s, n_fft=2048),
    "energy_ratio": lambda s: low_high_energy_ratio(s, fs=SR),
    "sibilance": lambda s: sibilant_spectral_peak_frequency(s, fs=SR),
    "sharpness": lambda s: spectral_sharpness(s, sample_rate=SR),
    "roughness": lambda s: spectral_roughness(s, sample_rate=SR),
    "tonality": lambda s: spectral_tonality(s),
    "irregularity": lambda s: spectral_irregularity(s),
}


@pytest.mark.parametrize("name", sorted(_EDGE_CASES))
def test_silence_is_finite(name):
    result = _EDGE_CASES[name](_SILENCE)
    assert isinstance(result, torch.Tensor)
    assert torch.isfinite(result).all(), f"{name} produced non-finite output on silence"


@pytest.mark.parametrize("name", sorted(_EDGE_CASES))
def test_length_one_is_finite(name):
    # length-1 input previously crashed via hann_window(1) -> divide by (L-1).
    result = _EDGE_CASES[name](_LEN1)
    assert isinstance(result, torch.Tensor)
    assert torch.isfinite(result).all(), f"{name} produced non-finite output on length-1 input"


def test_skewness_finite_on_pure_tone(tone_signal):
    # Pure tone -> (near) zero spectral variance previously produced NaN via
    # division by var.
    tone, _ = tone_signal
    skew, kurt = spectral_skewness(tone[:2048], n_fft=2048)
    assert torch.isfinite(skew)
    assert torch.isfinite(kurt)


def test_spread_finite_on_pure_tone(tone_signal):
    tone, _ = tone_signal
    spread = spectral_spread(tone[:2048], n_fft=2048, sample_rate=SR)
    assert torch.isfinite(spread)
    assert float(spread) >= 0.0


def test_bandwidth_accepts_1xN_input(noise_signal):
    # A (1, N) input must be squeezed to 1-D mono, yielding a 1-D bandwidth
    # series (not a 3-D tensor from a 2-D torchaudio spectrogram).
    x, sr = noise_signal
    out = spectral_bandwidth(x.unsqueeze(0), sample_rate=sr, n_fft=2048, hop_length=512)
    assert out.ndim == 1
    assert out.numel() > 0
    assert torch.isfinite(out).all()
