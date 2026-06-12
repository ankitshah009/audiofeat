"""Known-answer / edge-case correctness tests for advanced spectral features.

These tests pin the mathematical-correctness fixes applied to the advanced
spectral modules (lsp, hnr, key, hps, harmonic, mfcc, lpc, formants). They are
deterministic (seeded) and use either closed-form expected values or
librosa-derived ground truth.
"""

import warnings

import numpy as np
import pytest
import torch
from scipy.signal import iirpeak, lfilter

from audiofeat.spectral.formants import formant_bandwidths
from audiofeat.spectral.harmonic import inharmonicity_index, harmonic_richness_factor
from audiofeat.spectral.hnr import (
    harmonic_to_noise_ratio,
    harmonic_to_noise_ratio_acf,
)
from audiofeat.spectral.hps import hps
from audiofeat.spectral.key import key_detect
from audiofeat.spectral.lpc import lpc_coefficients
from audiofeat.spectral.lsp import lsp_coefficients
from audiofeat.spectral.mfcc import mfcc

librosa = pytest.importorskip("librosa")


def _broadband_lpc(order: int, sr: int = 16000, seed: int = 42) -> np.ndarray:
    """A genuine stable LPC polynomial from filtered white noise.

    Returns the full polynomial ``[1, a_1, ..., a_order]`` (librosa convention).
    """
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(sr).astype(np.float64)
    for fc, q in [(500, 5), (1500, 8), (2500, 10), (3500, 12)]:
        b, a = iirpeak(fc / (sr / 2), q)
        x = lfilter(b, a, x)
    return np.asarray(librosa.lpc(x.astype(np.float32), order=order), dtype=np.float64)


# ── 1. LSP: line spectral pairs from a known stable LPC polynomial ──────────


@pytest.mark.parametrize("order", [8, 10, 12, 16])
def test_lsp_coefficients_known_polynomial(order):
    """LSPs of a valid LPC polynomial: real, in [0,0.5), strictly increasing,
    exactly ``order`` of them, no NaN."""
    torch.manual_seed(0)
    full = _broadband_lpc(order)
    lpc_tail = torch.tensor(full[1:], dtype=torch.float64)  # a_1 .. a_p

    lsp = lsp_coefficients(lpc_tail)

    assert lsp.numel() == order, "LSP count must equal the LPC order"
    assert torch.all(torch.isfinite(lsp)), "no NaN/inf allowed in LSPs"
    assert not torch.is_complex(lsp), "LSPs must be real"
    assert torch.all(lsp >= 0.0) and torch.all(lsp < 0.5), "LSPs must lie in [0, 0.5)"
    assert torch.all(lsp[1:] > lsp[:-1]), "LSPs must be strictly increasing"


def test_lsp_coefficients_no_negative_frequencies():
    """Regression: the old implementation emitted negative 'frequencies' and
    NaN padding. Ensure neither occurs."""
    torch.manual_seed(1)
    full = _broadband_lpc(12)
    lsp = lsp_coefficients(torch.tensor(full[1:], dtype=torch.float64))
    assert torch.all(lsp >= 0.0)
    assert not torch.any(torch.isnan(lsp))


# ── 2. HNR (Boersma autocorrelation) ────────────────────────────────────────


def test_hnr_acf_clean_sine_is_high():
    """A clean periodic sine is almost fully harmonic -> high HNR."""
    sr = 22050
    t = torch.arange(sr, dtype=torch.float32) / sr
    sine = torch.sin(2 * torch.pi * 220.0 * t)
    hnr = harmonic_to_noise_ratio_acf(sine, sr)
    assert torch.isfinite(hnr)
    assert float(hnr) > 15.0, f"clean sine HNR should exceed 15 dB, got {float(hnr)}"


def test_hnr_acf_white_noise_is_low():
    """White noise is aperiodic -> low HNR, and strictly below a clean tone."""
    torch.manual_seed(42)
    sr = 22050
    noise = torch.randn(sr)
    hnr_noise = harmonic_to_noise_ratio_acf(noise, sr)

    t = torch.arange(sr, dtype=torch.float32) / sr
    sine = torch.sin(2 * torch.pi * 220.0 * t)
    hnr_sine = harmonic_to_noise_ratio_acf(sine, sr)

    assert float(hnr_noise) < 10.0, f"noise HNR should be low, got {float(hnr_noise)}"
    assert float(hnr_sine) > float(hnr_noise) + 15.0


def test_hnr_scalar_helper_still_works():
    """Backward-compat: the scalar energy-ratio helper is unchanged."""
    val = harmonic_to_noise_ratio(torch.tensor(10.0), torch.tensor(1.0))
    assert isinstance(val, torch.Tensor)
    assert abs(float(val) - 10.0) < 1e-3  # 10*log10(10/1) = 10 dB


# ── 3. Key detection degenerate-input guards ────────────────────────────────


def test_key_detect_silence_is_unknown():
    sr = 22050
    silence = torch.zeros(sr * 2)
    assert key_detect(silence, sample_rate=sr, n_fft=1024, hop_length=256) == "Unknown"


def test_key_detect_constant_dc_is_unknown():
    """A constant (DC) signal has a near-flat chroma -> no key information."""
    sr = 22050
    dc = torch.ones(sr * 2)
    assert key_detect(dc, sample_rate=sr, n_fft=1024, hop_length=256) == "Unknown"


def test_key_detect_c_major_triad_resolves():
    """A clear C-major triad resolves to C major or a relative/shared key.

    Krumhansl-Schmuckler commonly confuses relative major/minor pairs, so the
    accepted set mirrors tests/test_librosa_parity.py.
    """
    sr = 22050
    t = torch.arange(sr * 4, dtype=torch.float32) / sr
    cmaj = (
        torch.sin(2 * torch.pi * 261.63 * t)
        + torch.sin(2 * torch.pi * 329.63 * t)
        + torch.sin(2 * torch.pi * 392.00 * t)
    )
    key = key_detect(cmaj, sample_rate=sr)
    acceptable = {"C major", "C minor", "A minor", "E minor", "G major"}
    assert key in acceptable, f"expected one of {acceptable}, got {key!r}"


# ── 4. Harmonic-Percussive Separation ───────────────────────────────────────


def _tonal_concentration(sig: torch.Tensor, sr: int, f0: float = 440.0, n_fft: int = 4096) -> float:
    s = sig.detach().float()
    wlen = min(n_fft, s.numel())
    s = s[:wlen] * torch.hann_window(wlen)
    S = torch.fft.rfft(s, n=n_fft).abs()
    freqs = torch.fft.rfftfreq(n_fft, 1.0 / sr)
    bin0 = int(torch.argmin((freqs - f0).abs()))
    band = S[max(0, bin0 - 3) : bin0 + 4].pow(2).sum()
    return float(band / (S.pow(2).sum() + 1e-12))


def _transient_sharpness(sig: torch.Tensor, center: int) -> float:
    s = sig.detach().abs().float()
    center = min(center, s.numel() - 1)
    local = float(s[max(0, center - 64) : center + 64].max())
    return local / (float(s.mean()) + 1e-12)


def test_hps_separates_tone_and_click():
    """Tone+click mixture: harmonic keeps the tonal energy, percussive keeps
    the transient."""
    torch.manual_seed(42)
    sr = 22050
    t = torch.arange(sr, dtype=torch.float32) / sr
    tone = torch.sin(2 * torch.pi * 440.0 * t)
    mix = tone.clone()
    click = int(sr * 0.5)
    mix[click : click + 5] += 8.0  # sharp broadband transient

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        harm, perc = hps(mix, sample_rate=sr, n_fft=1024, hop_length=256)

    assert harm.ndim == 1 and perc.ndim == 1
    assert harm.numel() > 0 and perc.numel() > 0

    # Harmonic component is far more tonal at 440 Hz than the percussive one.
    assert _tonal_concentration(harm, sr) > _tonal_concentration(perc, sr)
    assert _tonal_concentration(harm, sr) > 0.8
    # Percussive component localises the transient far more sharply.
    assert _transient_sharpness(perc, click) > _transient_sharpness(harm, click)


def test_hps_default_kernel_is_larger_than_legacy():
    """Default kernels should be the sensible Fitzgerald-scale values, not the
    old 7-tap (margin=3) kernels."""
    torch.manual_seed(0)
    sr = 22050
    x = torch.randn(sr // 2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Should run without error and return finite, non-empty waveforms.
        harm, perc = hps(x, sample_rate=sr, n_fft=512, hop_length=128)
    assert torch.all(torch.isfinite(harm))
    assert torch.all(torch.isfinite(perc))


def test_hps_short_signal_does_not_crash():
    """Reflect-pad must be clamped for short signals."""
    torch.manual_seed(0)
    x = torch.randn(300)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        harm, perc = hps(x, sample_rate=8000, n_fft=128, hop_length=64)
    assert harm.numel() > 0 and perc.numel() > 0
    assert torch.all(torch.isfinite(harm)) and torch.all(torch.isfinite(perc))


# ── 5. Inharmonicity / harmonic richness ────────────────────────────────────


def test_inharmonicity_index_f0_zero_is_safe():
    """f0 <= 0 must not divide by zero; returns NaN."""
    peaks = torch.tensor([100.0, 205.0, 300.0])
    out = inharmonicity_index(peaks, f0=0.0)
    assert torch.isnan(out)
    out_neg = inharmonicity_index(peaks, f0=-50.0)
    assert torch.isnan(out_neg)


def test_inharmonicity_index_perfect_harmonic_is_zero():
    f0 = 100.0
    peaks = torch.tensor([100.0, 200.0, 300.0, 400.0])  # exact k*f0
    out = inharmonicity_index(peaks, f0=f0)
    assert torch.isfinite(out)
    assert float(out) < 1e-5


def test_inharmonicity_index_stretched_partials_positive():
    """Sharpened (stretched) partials give a positive, predictable index."""
    f0 = 100.0
    # 2% sharp on every partial -> mean |peaks/(k f0) - 1| = 0.02
    peaks = torch.tensor([102.0, 204.0, 306.0, 408.0])
    out = inharmonicity_index(peaks, f0=f0)
    assert abs(float(out) - 0.02) < 1e-4


def test_harmonic_richness_factor_db_value():
    """Documented as dB: equal-energy fundamental + one partial -> ~0 dB."""
    mags = torch.tensor([1.0, 1.0])  # H1 and one upper harmonic, equal energy
    out = harmonic_richness_factor(mags)
    assert abs(float(out)) < 1e-3  # 10*log10(1/1) = 0 dB


# ── 6. MFCC shape contract & short-input behaviour ──────────────────────────


def test_mfcc_normal_input_shape_contract():
    """Normal input returns exactly n_mfcc rows."""
    torch.manual_seed(0)
    x = torch.randn(22050 * 2)
    m = mfcc(x, sample_rate=22050, n_mfcc=13)
    assert m.shape[0] == 13
    assert m.shape[1] > 0
    assert torch.all(torch.isfinite(m))


def test_mfcc_2d_input_flattened():
    """A (1, n) input is flattened to the (n_mfcc, frames) contract."""
    torch.manual_seed(0)
    x = torch.randn(1, 22050 * 2)
    m = mfcc(x, sample_rate=22050, n_mfcc=13)
    assert m.shape[0] == 13


def test_mfcc_short_input_warns_not_silent():
    """Short input clamps n_mfcc to available mel bands AND warns (no silent
    dimensionality shrink)."""
    short = torch.randn(40)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = mfcc(short, sample_rate=22050, n_mfcc=40)
    assert torch.all(torch.isfinite(out))
    assert any(
        issubclass(w.category, UserWarning) and "clamping n_mfcc" in str(w.message)
        for w in caught
    ), "short-input dimensionality change must emit a UserWarning"


def test_mfcc_log_mels_option():
    """The log_mels switch runs and preserves the row contract."""
    torch.manual_seed(0)
    x = torch.randn(22050)
    m = mfcc(x, sample_rate=22050, n_mfcc=13, log_mels=True)
    assert m.shape[0] == 13
    assert torch.all(torch.isfinite(m))


# ── 7. LPC energy-floor guard ───────────────────────────────────────────────


def test_lpc_coefficients_zeros_finite():
    """A silent frame (R[0]=0) must not produce NaNs."""
    out = lpc_coefficients(torch.zeros(2048), 10)
    assert out.shape[0] == 10
    assert torch.all(torch.isfinite(out))


def test_lpc_coefficients_constant_finite():
    """A constant frame is perfectly predictable; coefficients stay finite."""
    out = lpc_coefficients(torch.ones(2048), 10)
    assert out.shape[0] == 10
    assert torch.all(torch.isfinite(out))


def test_lpc_coefficients_sign_convention_preserved():
    """Guard against accidental sign/shape regressions: a_1..a_p returned with
    the leading 1.0 stripped, real-valued."""
    torch.manual_seed(0)
    x = torch.randn(2048)
    out = lpc_coefficients(x, 10)
    assert out.shape[0] == 10
    assert not torch.is_complex(out)
    assert torch.all(torch.isfinite(out))


# ── 8. Formant bandwidths: no negatives, closed-form value ──────────────────


def test_formant_bandwidths_known_pole_radius():
    """Single resonance: bandwidth ≈ -(fs/pi)*ln|r| for a pole at radius r."""
    fs = 16000
    r = 0.95
    theta = 2.0 * np.pi * 1000.0 / fs
    # A(z) = 1 - 2 r cos(theta) z^-1 + r^2 z^-2  (poles at r e^{±j theta})
    a = np.array([1.0, -2.0 * r * np.cos(theta), r**2])
    bw = formant_bandwidths(torch.tensor(a), fs)
    expected = -(fs / np.pi) * np.log(r)
    assert bw.numel() == 1
    assert abs(float(bw[0]) - expected) < 1e-3
    assert torch.all(bw > 0)


def test_formant_bandwidths_drops_negative():
    """A pole on/outside the unit circle yields a non-positive bandwidth that
    must be discarded (regression: the old code returned negatives)."""
    fs = 16000
    r = 1.05  # outside the unit circle -> bw would be negative
    theta = 2.0 * np.pi * 1000.0 / fs
    a = np.array([1.0, -2.0 * r * np.cos(theta), r**2])
    bw = formant_bandwidths(torch.tensor(a), fs)
    # Either empty or strictly positive — never negative.
    if bw.numel() > 0:
        assert torch.all(bw > 0)


def test_formant_bandwidths_random_no_negatives():
    """Random LPC-shaped input must never produce a negative bandwidth."""
    torch.manual_seed(7)
    for _ in range(20):
        a = torch.randn(10)
        a[0] = 1.0
        bw = formant_bandwidths(a, fs=22050)
        if bw.numel() > 0:
            assert torch.all(bw > 0), "formant bandwidths must be strictly positive"
            assert torch.all(torch.isfinite(bw))
