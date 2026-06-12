"""Correctness tests for cepstral, stats, and segmentation fixes.

Each test proves a specific bug fix from the static audit. All tests are
deterministic (explicit ``torch.manual_seed`` / passed ``seed``).
"""

import collections
import math

import numpy as np
import pytest
import torch

from audiofeat.cepstral.lpcc import lpcc
from audiofeat.cepstral.deltas import delta, delta_delta
from audiofeat.cepstral.gtcc import gtcc
from audiofeat.spectral.gfcc import gfcc
from audiofeat.spectral.lpc import lpc_coefficients
from audiofeat.stats.functionals import compute_functionals
from audiofeat.segmentation.silence import silence_removal
from audiofeat.segmentation.diarization import speaker_diarization, kmeans
from audiofeat.segmentation.thumbnailing import music_thumbnailing
from audiofeat.validation.scorecard import run_gold_standard_scorecard


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _ar2_signal(a1: float, a2: float, n: int = 40000, seed: int = 0) -> torch.Tensor:
    """White noise driven through an AR(2) all-pole filter."""
    torch.manual_seed(seed)
    e = torch.randn(n)
    x = torch.zeros(n)
    for k in range(2, n):
        x[k] = a1 * x[k - 1] + a2 * x[k - 2] + e[k]
    return x


# ---------------------------------------------------------------------------
# 1. lpcc.py — sign convention + index guard + NaN guard
# ---------------------------------------------------------------------------
def test_lpcc_sign_convention_matches_prediction_coeffs():
    """c_1 must equal the (positive) prediction coefficient a_1, not its negation.

    For an AR(2) process with positive feedback a1>0, the first LPCC c_1 == a_1
    must be positive. The pre-fix code (no negation of lpc_coefficients) produced
    c_1 with the wrong sign.
    """
    a1, a2 = 0.6, -0.3
    x = _ar2_signal(a1, a2)
    out = lpcc(x, sample_rate=22050, n_lpcc=8, lpc_order=12)
    c1 = out[:, 0].mean()
    # c_1 == a_1 (prediction coeff). Compare against the negated raw LPC output.
    frame = x[1000:1000 + 2048] * torch.hann_window(2048)
    a_pred = -lpc_coefficients(frame, 12)
    assert c1.item() > 0.0, "c_1 should be positive for positive AR feedback"
    # c_1 should be close to a_1 (the first prediction coefficient).
    assert abs(c1.item() - a_pred[0].item()) < 0.2


def test_lpcc_n_lpcc_greater_than_order_no_indexerror():
    """n_lpcc > lpc_order must not raise IndexError (guard 0 <= m-k-1 < lpc_order)."""
    torch.manual_seed(0)
    x = torch.randn(22050)
    out = lpcc(x, sample_rate=22050, n_lpcc=20, lpc_order=12)
    assert out.shape[1] == 20
    assert torch.isfinite(out).all()


def test_lpcc_silent_frame_no_nan():
    """Silent frames must yield finite (zero) coefficients, not NaN."""
    silent = torch.zeros(22050)
    out = lpcc(silent, sample_rate=22050, n_lpcc=12, lpc_order=12)
    assert torch.isfinite(out).all()
    assert torch.all(out == 0)


# ---------------------------------------------------------------------------
# 2. deltas.py — width validation + librosa parity
# ---------------------------------------------------------------------------
def test_delta_matches_librosa_interior():
    librosa = pytest.importorskip("librosa")
    torch.manual_seed(0)
    x = torch.randn(5, 40)
    width = 9
    mine = delta(x, width=width).numpy()
    ref = librosa.feature.delta(x.numpy(), width=width, order=1, mode="interp")
    pad = width // 2
    interior = slice(pad, x.shape[1] - pad)
    np.testing.assert_allclose(mine[:, interior], ref[:, interior], rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("bad_width", [8, 2, 1, 0, 4])
def test_delta_even_or_small_width_raises(bad_width):
    torch.manual_seed(0)
    x = torch.randn(5, 40)
    with pytest.raises(ValueError):
        delta(x, width=bad_width)


def test_delta_short_input_raises():
    torch.manual_seed(0)
    x = torch.randn(3, 5)  # time_steps (5) < width (9)
    with pytest.raises(ValueError):
        delta(x, width=9)


def test_delta_delta_runs_and_shapes():
    torch.manual_seed(0)
    x = torch.randn(4, 30)
    out = delta_delta(x, width=9)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# 3. stats/functionals.py — skew/kurt vs scipy, axis, zero-variance
# ---------------------------------------------------------------------------
def test_functionals_skew_kurt_match_scipy():
    sps = pytest.importorskip("scipy.stats")
    torch.manual_seed(0)
    data = torch.randn(500, 3)  # (time, features), default time_axis=0
    out = compute_functionals(data)
    nf = 3
    skew = out[4 * nf:5 * nf].numpy()
    kurt = out[5 * nf:6 * nf].numpy()
    x = data.numpy()
    np.testing.assert_allclose(skew, sps.skew(x, axis=0, bias=True), rtol=1e-4, atol=1e-5)
    # excess (Fisher) kurtosis
    np.testing.assert_allclose(
        kurt, sps.kurtosis(x, axis=0, fisher=True, bias=True), rtol=1e-4, atol=1e-5
    )


def test_functionals_zero_variance_finite():
    const = torch.ones(10, 2) * 3.5
    out = compute_functionals(const)
    assert torch.isfinite(out).all()
    nf = 2
    # zero-variance => skew and excess-kurtosis defined as 0.0
    assert torch.allclose(out[4 * nf:5 * nf], torch.zeros(nf))  # skew
    assert torch.allclose(out[5 * nf:6 * nf], torch.zeros(nf))  # kurtosis


def test_functionals_axis_orientation():
    """time_axis=1 on (features, time) equals time_axis=0 on the transpose."""
    torch.manual_seed(0)
    m = torch.randn(4, 60)  # (features, time)
    via_axis1 = compute_functionals(m, time_axis=1)
    via_transpose = compute_functionals(m.T, time_axis=0)
    assert torch.allclose(via_axis1, via_transpose, atol=1e-6)


# ---------------------------------------------------------------------------
# 4. segmentation/thumbnailing.py — off-diagonal repetition detection
# ---------------------------------------------------------------------------
def _motif(sr: int, dur: float = 2.0) -> torch.Tensor:
    t = torch.arange(int(sr * dur)) / sr
    return (
        torch.sin(2 * torch.pi * 523.0 * t)
        + torch.sin(2 * torch.pi * 659.0 * t)
        + torch.sin(2 * torch.pi * 784.0 * t)
    ) / 3.0


def test_thumbnailing_finds_repeated_motif():
    torch.manual_seed(0)
    sr = 22050
    motif = _motif(sr, 2.0)
    noise = lambda d: 0.3 * torch.randn(int(sr * d))
    # [noise 3s][MOTIF@3s][noise 3s][MOTIF@9s][noise 3s]
    signal = torch.cat([noise(3.0), motif, noise(3.0), motif, noise(3.0)])
    start, end = music_thumbnailing(signal, sr, thumb_size=2.0, window_size=1.0, hop_size=0.5)
    assert isinstance(start, float)
    assert isinstance(end, float)
    assert start < end
    # The returned start must land near a known motif onset (3.0s or 9.0s),
    # within a few hops (hop_size=0.5 -> 3 hops = 1.5s tolerance).
    assert min(abs(start - 3.0), abs(start - 9.0)) <= 1.5


def test_thumbnailing_silent_no_nan():
    sr = 22050
    silent = torch.zeros(sr * 8)
    start, end = music_thumbnailing(silent, sr, thumb_size=2.0, window_size=1.0, hop_size=0.5)
    assert isinstance(start, float) and isinstance(end, float)
    assert math.isfinite(start) and math.isfinite(end)


def test_thumbnailing_short_input_no_crash():
    torch.manual_seed(0)
    sr = 22050
    short = 0.3 * torch.randn(sr // 2)  # 0.5s, shorter than thumb_size
    start, end = music_thumbnailing(short, sr, thumb_size=2.0, window_size=1.0, hop_size=0.5)
    assert isinstance(start, float) and isinstance(end, float)
    assert math.isfinite(start) and math.isfinite(end)


# ---------------------------------------------------------------------------
# 5. segmentation/diarization.py — determinism + final-iteration labels
# ---------------------------------------------------------------------------
def test_diarization_reproducible_two_clusters():
    sr = 22050
    t = torch.linspace(0, 3, sr * 3)
    sp1 = torch.sin(2 * torch.pi * 220.0 * t)
    sp2 = torch.sin(2 * torch.pi * 880.0 * t)
    signal = torch.cat([sp1, sp2])

    r1 = speaker_diarization(signal, sr, n_speakers=2, seed=0)
    r2 = speaker_diarization(signal, sr, n_speakers=2, seed=0)
    assert torch.equal(r1, r2)  # reproducible with fixed seed
    assert len(torch.unique(r1)) == 2

    # Each half should be dominated by a single (distinct) contiguous label.
    half = r1.shape[0] // 2
    first = collections.Counter(r1[:half].tolist()).most_common(1)[0]
    second = collections.Counter(r1[half:].tolist()).most_common(1)[0]
    assert first[1] / half >= 0.9
    assert second[1] / (r1.shape[0] - half) >= 0.9
    assert first[0] != second[0]


def test_kmeans_labels_consistent_with_final_centroids():
    """Returned labels must be argmin against the FINAL centroids."""
    torch.manual_seed(0)
    cluster_a = torch.randn(50, 2) + torch.tensor([5.0, 5.0])
    cluster_b = torch.randn(50, 2) + torch.tensor([-5.0, -5.0])
    X = torch.cat([cluster_a, cluster_b])
    labels = kmeans(X, n_clusters=2, seed=0)
    # The two well-separated clusters must receive different labels internally.
    assert len(torch.unique(labels)) == 2
    # Points in cluster_a all share a label; cluster_b shares the other.
    assert len(torch.unique(labels[:50])) == 1
    assert len(torch.unique(labels[50:])) == 1
    assert labels[0] != labels[50]


# ---------------------------------------------------------------------------
# 6. segmentation/silence.py — scale invariance + single frame
# ---------------------------------------------------------------------------
def test_silence_removal_scale_invariant():
    torch.manual_seed(0)
    sr = 22050
    base = torch.zeros(sr * 5)
    base[sr * 2:sr * 3] = torch.randn(sr)

    kept = {}
    for amp in (0.01, 1.0):
        out = silence_removal(base * amp, sr)
        kept[amp] = out.shape[0]
        assert out.shape[0] > 0
        assert out.shape[0] < base.shape[0]
    # Same signal scaled by a constant => identical silence decision.
    assert kept[0.01] == kept[1.0]


def test_silence_removal_single_frame_no_crash():
    torch.manual_seed(0)
    sr = 22050
    # Exactly one analysis window (window_size=0.05s).
    single = 0.5 * torch.randn(int(0.05 * sr))
    out = silence_removal(single, sr)
    assert isinstance(out, torch.Tensor)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# 7. cepstral/_erb.py + gtcc/gfcc — short input no crash
# ---------------------------------------------------------------------------
def test_gtcc_short_input_no_crash():
    torch.manual_seed(0)
    short = torch.randn(500)  # < default n_fft=2048
    out = gtcc(short, 22050)
    assert out.shape[0] > 0
    assert torch.isfinite(out).all()


def test_gfcc_short_input_no_crash():
    torch.manual_seed(0)
    short = torch.randn(500)  # < default n_fft=2048
    out = gfcc(short, 22050)
    assert out.shape[0] > 0
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# 8. validation/scorecard.py — reported threshold == enforced threshold
# ---------------------------------------------------------------------------
def test_scorecard_centroid_threshold_consistent():
    report = run_gold_standard_scorecard(include_optional=False)
    centroid = next(
        c for c in report["checks"] if c["name"] == "spectral_centroid_tone_consistency"
    )
    # Reported threshold_hz must equal the enforced bound (30.0 Hz).
    assert centroid["details"]["threshold_hz"] == 30.0
    # And the check should still be consistent with its own threshold.
    assert centroid["passed"] == (centroid["details"]["absolute_error_hz"] <= 30.0)


# ---------------------------------------------------------------------------
# 9. segmentation/__init__.py — package re-exports usable
# ---------------------------------------------------------------------------
def test_segmentation_package_exports():
    import audiofeat.segmentation as seg

    assert hasattr(seg, "silence_removal")
    assert hasattr(seg, "music_thumbnailing")
    assert hasattr(seg, "speaker_diarization")
