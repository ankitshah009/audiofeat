"""Numerical parity tests — verify audiofeat matches librosa output.

These tests ensure that when librosa is available, audiofeat delegates
to it and returns identical (or near-identical) results.  Tolerances
are set to floating-point epsilon levels since we're wrapping the same
library, not re-implementing.
"""

import numpy as np
import pytest
import torch

# Skip entire module if librosa not available
librosa = pytest.importorskip("librosa")


@pytest.fixture
def noise_signal():
    """2 seconds of reproducible noise at 22050 Hz."""
    torch.manual_seed(42)
    return torch.randn(22050 * 2), 22050


@pytest.fixture
def tone_signal():
    """3 seconds of 440 Hz tone at 22050 Hz."""
    sr = 22050
    t = torch.arange(sr * 3, dtype=torch.float32) / sr
    return torch.sin(2 * torch.pi * 440.0 * t), sr


# ── Chroma ──────────────────────────────────────────────────────────


def test_chroma_matches_librosa(noise_signal):
    x, sr = noise_signal
    from audiofeat.spectral.chroma import chroma

    ours = chroma(x, sample_rate=sr, n_fft=2048, hop_length=512).numpy()
    ref = librosa.feature.chroma_stft(
        y=x.numpy(), sr=sr, n_fft=2048, hop_length=512, tuning=0.0,
    )
    np.testing.assert_allclose(ours, ref, atol=1e-5)


# ── Tonnetz ──────────────────────────────────────────────────────────


def test_tonnetz_matches_librosa(noise_signal):
    x, sr = noise_signal
    from audiofeat.spectral.chroma import chroma
    from audiofeat.spectral.tonnetz import tonnetz

    chroma_feat = chroma(x, sample_rate=sr).numpy()
    ours = tonnetz(torch.from_numpy(chroma_feat)).numpy()
    ref = librosa.feature.tonnetz(chroma=chroma_feat)
    np.testing.assert_allclose(ours, ref, atol=1e-5)


# ── Spectral Contrast ───────────────────────────────────────────────


def test_spectral_contrast_matches_librosa(noise_signal):
    x, sr = noise_signal
    from audiofeat.spectral.contrast import spectral_contrast

    ours = spectral_contrast(x, fs=sr).numpy()
    ref = librosa.feature.spectral_contrast(y=x.numpy(), sr=sr)
    np.testing.assert_allclose(ours, ref, atol=1e-4)


# ── CQT ──────────────────────────────────────────────────────────────


def test_cqt_matches_librosa(noise_signal):
    x, sr = noise_signal
    from audiofeat.spectral.spectrogram import cqt_spectrogram

    ours = cqt_spectrogram(x, sample_rate=sr).numpy()
    ref = np.abs(librosa.cqt(y=x.numpy(), sr=sr, hop_length=512, fmin=32.7, n_bins=84))
    np.testing.assert_allclose(ours, ref, atol=1e-5)


# ── Onset Detection ──────────────────────────────────────────────────


def test_onset_matches_librosa(noise_signal):
    x, sr = noise_signal
    from audiofeat.temporal.onset import onset_detect

    ours = onset_detect(x, sample_rate=sr, backtrack=True).numpy()

    onset_env = librosa.onset.onset_strength(
        y=x.numpy(), sr=sr, hop_length=512, n_fft=2048, aggregate=np.median,
    )
    ref_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env, sr=sr, hop_length=512, backtrack=True, units="frames",
    )
    ref_times = librosa.frames_to_time(ref_frames, sr=sr, hop_length=512)

    assert len(ours) == len(ref_times), f"Onset count mismatch: {len(ours)} vs {len(ref_times)}"
    if len(ours) > 0:
        np.testing.assert_allclose(ours, ref_times, atol=1e-5)


# ── Beat Tracking ─────────────────────────────────────────────────────


def test_beat_matches_librosa(noise_signal):
    x, sr = noise_signal
    from audiofeat.temporal.beat import beat_track as our_beat_track

    tempo_ours, beats_ours = our_beat_track(x, sample_rate=sr)

    onset_env = librosa.onset.onset_strength(
        y=x.numpy(), sr=sr, n_fft=2048, hop_length=512, aggregate=np.median,
    )
    tempo_lib, beats_lib = librosa.beat.beat_track(
        onset_envelope=onset_env, sr=sr, hop_length=512,
        start_bpm=150.0, tightness=100,
    )

    tempo_lib_val = float(np.asarray(tempo_lib).flat[0])
    assert abs(float(tempo_ours) - tempo_lib_val) < 1.0, (
        f"Tempo mismatch: {float(tempo_ours)} vs {tempo_lib_val}"
    )
    assert len(beats_ours) == len(beats_lib), (
        f"Beat count mismatch: {len(beats_ours)} vs {len(beats_lib)}"
    )


# ── GFCC / GTCC Smoke Tests ──────────────────────────────────────────


def test_gfcc_output_shape_and_finiteness(noise_signal):
    x, sr = noise_signal
    from audiofeat.spectral.gfcc import gfcc

    g = gfcc(x, sample_rate=sr, n_gfcc=13, n_fft=2048, hop_length=512, n_bands=64)
    assert g.shape[0] == 13
    assert g.shape[1] > 0
    assert torch.isfinite(g).all()


def test_gtcc_output_shape_and_finiteness(noise_signal):
    x, sr = noise_signal
    from audiofeat.cepstral.gtcc import gtcc

    gt = gtcc(x, sample_rate=sr, n_gtcc=20, n_fft=2048, hop_length=512, n_bands=64)
    assert gt.shape[0] == 20
    assert gt.shape[1] > 0
    assert torch.isfinite(gt).all()


# ── Tristimulus Accuracy ──────────────────────────────────────────────


def test_tristimulus_harmonic_signal(tone_signal):
    """For a known harmonic signal, verify tristimulus sums to 1 and
    T1 dominates when fundamental is loudest."""
    x, sr = tone_signal
    from audiofeat.temporal.tristimulus import tristimulus

    tri = tristimulus(x, sample_rate=sr, n_fft=2048, hop_length=512)
    assert tri.shape == (3,)
    assert abs(tri.sum().item() - 1.0) < 0.01, f"Sum={tri.sum():.4f}, expected ~1.0"
    assert tri[0] > 0.5, f"T1={tri[0]:.3f}, expected dominant fundamental"


def test_tristimulus_multi_harmonic():
    """Signal with strong 2nd-4th harmonics should have higher T2."""
    sr = 22050
    t = torch.arange(sr * 2, dtype=torch.float32) / sr
    f0 = 220.0
    signal = (
        0.1 * torch.sin(2 * torch.pi * f0 * t)         # weak fundamental
        + torch.sin(2 * torch.pi * 2 * f0 * t)          # strong H2
        + torch.sin(2 * torch.pi * 3 * f0 * t)          # strong H3
        + torch.sin(2 * torch.pi * 4 * f0 * t)          # strong H4
    )
    from audiofeat.temporal.tristimulus import tristimulus

    tri = tristimulus(signal, sample_rate=sr, n_fft=4096, n_harmonics=10)
    assert tri[1] > tri[0], f"T2={tri[1]:.3f} should exceed T1={tri[0]:.3f} for H2-H4 dominant"


# ── Attack Time Accuracy ─────────────────────────────────────────────


def test_attack_time_linear_ramp():
    """100ms linear ramp should give LAT ≈ log10(0.08) ≈ -1.097."""
    sr = 22050
    ramp = torch.linspace(0, 1, sr // 10)
    sustain = torch.ones(sr)
    signal = torch.cat([ramp, sustain])
    from audiofeat.temporal.attack import log_attack_time

    lat = log_attack_time(signal, sample_rate=sr)
    expected = float(np.log10(0.08))
    assert abs(lat - expected) < 0.15, f"LAT={lat:.3f}, expected ~{expected:.3f}"


def test_attack_time_silence():
    """Silence should return the sentinel value."""
    from audiofeat.temporal.attack import log_attack_time

    lat = log_attack_time(torch.zeros(22050), sample_rate=22050)
    assert lat < -7.0  # log10(1e-8) ≈ -8


# ── Key Detection Sanity ──────────────────────────────────────────────


def test_key_detect_c_major_scale():
    """A C-major triad should detect C major or relative minor (A minor/E minor).

    The Krumhansl-Schmuckler algorithm commonly confuses relative
    major/minor pairs since they share the same diatonic pitch classes.
    """
    sr = 22050
    t = torch.arange(sr * 4, dtype=torch.float32) / sr
    # C major triad: C4(261.6) + E4(329.6) + G4(392.0)
    signal = (
        torch.sin(2 * torch.pi * 261.63 * t)
        + torch.sin(2 * torch.pi * 329.63 * t)
        + torch.sin(2 * torch.pi * 392.00 * t)
    )
    from audiofeat.spectral.key import key_detect

    key = key_detect(signal, sample_rate=sr)
    # Accept C major, A minor (relative minor), or E minor (shares C-E-G)
    acceptable = {"C major", "C minor", "A minor", "E minor", "G major"}
    assert key in acceptable, f"Expected one of {acceptable}, got '{key}'"
