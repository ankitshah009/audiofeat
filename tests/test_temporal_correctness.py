"""Correctness, librosa-parity, and edge-case tests for the temporal module.

Each test targets a specific bug fixed in audiofeat/temporal/ (and the
related export wiring). Tests are deterministic (``torch.manual_seed``) and
use known-answer / librosa-parity / edge-case (silence, length-1, short)
checks. librosa-dependent parity tests are skipped if librosa is absent.
"""

import inspect
import warnings

import numpy as np
import pytest
import torch

warnings.filterwarnings("ignore")


# ── Export wiring (locks the temporal/__init__.py collision fix) ──────────


def test_temporal_exports_reachable():
    """onset_detect, tristimulus, temporal_centroid_framewise,
    beat_track_with_tempo (and friends) must all be importable from the
    temporal package."""
    import audiofeat.temporal as tmp

    for name in (
        "onset_detect",
        "tristimulus",
        "temporal_centroid_framewise",
        "beat_track_with_tempo",
        "tempo",
        "energy",
        "temporal_centroid",
        "beat_track",
        "decay_time",
        "entropy_of_energy",
        "loudness",
    ):
        assert hasattr(tmp, name), f"audiofeat.temporal.{name} is not reachable"


def test_temporal_centroid_export_is_mpeg7():
    """The package-level ``temporal_centroid`` must resolve to the MPEG-7
    whole-signal version (signature uses ``sample_rate``, not
    ``frame_length``) and return a scalar in seconds."""
    import audiofeat.temporal as tmp

    params = list(inspect.signature(tmp.temporal_centroid).parameters)
    assert "sample_rate" in params
    assert "frame_length" not in params

    sr = 22050
    t = torch.arange(int(sr * 2), dtype=torch.float32) / sr
    tone = torch.sin(2 * torch.pi * 220.0 * t)
    cent = tmp.temporal_centroid(tone, sample_rate=sr)
    assert cent.ndim == 0
    # Uniform-energy tone over 2 s -> centroid near the midpoint (1.0 s).
    assert abs(float(cent) - 1.0) < 0.05


def test_temporal_centroid_framewise_is_frame_based():
    """``temporal_centroid_framewise`` returns one value per frame (and the
    backward-compat ``rhythm.temporal_centroid`` still points at it)."""
    import audiofeat.temporal as tmp
    from audiofeat.temporal.rhythm import temporal_centroid as rhythm_tc

    x = torch.randn(5000)
    fw = tmp.temporal_centroid_framewise(x, 2048, 512)
    assert fw.ndim == 1 and fw.numel() > 1
    assert torch.allclose(fw, rhythm_tc(x, 2048, 512))


def test_beat_track_export_returns_times_and_with_tempo_returns_tuple():
    """Package-level ``beat_track`` returns beat *times* (1-D tensor);
    ``beat_track_with_tempo`` returns ``(tempo, frames)``."""
    import audiofeat.temporal as tmp

    torch.manual_seed(0)
    x = torch.randn(22050 * 3)
    times = tmp.beat_track(x, sample_rate=22050)
    assert isinstance(times, torch.Tensor) and times.ndim == 1

    out = tmp.beat_track_with_tempo(x, sample_rate=22050)
    assert isinstance(out, tuple) and len(out) == 2
    tempo_t, frames = out
    assert torch.isfinite(tempo_t).all()
    assert frames.ndim == 1


# ── Zero-crossing rate (signbit / librosa parity) ─────────────────────────


def test_zcr_known_answer_sine():
    """A sine of f Hz framed over one frame has ~2f/sr zero-crossing rate."""
    from audiofeat.temporal.zcr import zero_crossing_rate

    sr = 22050
    f = 100.0
    n = 2048
    t = torch.arange(n, dtype=torch.float32) / sr
    x = torch.sin(2 * torch.pi * f * t)
    zcr = zero_crossing_rate(x, frame_length=n, hop_length=n)
    expected = 2.0 * f / sr
    assert abs(float(zcr[0]) - expected) < 0.002, f"zcr={float(zcr[0])}, expected~{expected}"


def test_zcr_zero_touch_uses_signbit_convention():
    """A signal that *touches* zero but does not change sign must not be
    counted as a crossing. The signbit convention (``>= 0``) treats +0.0 as
    non-negative, so ``+, 0, +`` yields 0 crossings; the old ``torch.sign``
    convention gives 0 a dedicated sign and spuriously counts a crossing."""
    from audiofeat.temporal.zcr import zero_crossing_count

    # +, 0, + : sign-bit (>=0) sequence T,T,T -> XOR adjacent = 0,0 -> 0 crossings
    x = torch.tensor([1.0, 0.0, 1.0])
    count = zero_crossing_count(x, frame_length=x.numel(), hop_length=x.numel())
    assert int(count[0]) == 0, "a zero touch without a sign change is not a crossing"

    # The old torch.sign-based formula DID count a spurious crossing here.
    old = float(torch.sum(torch.abs(torch.diff(torch.sign(x)))) / 2)
    assert old == 1.0, "sanity: old convention spuriously counted the zero touch"

    # A genuine sign change (+ -> -) is still counted.
    y = torch.tensor([1.0, -1.0, 1.0])
    cnt = zero_crossing_count(y, frame_length=y.numel(), hop_length=y.numel())
    assert int(cnt[0]) == 2


def test_zcr_matches_librosa():
    librosa = pytest.importorskip("librosa")
    from audiofeat.temporal.zcr import zero_crossing_rate

    torch.manual_seed(42)
    x = torch.randn(22050)
    fl, hl = 2048, 512
    ours = zero_crossing_rate(x, frame_length=fl, hop_length=hl).numpy()
    ref = librosa.feature.zero_crossing_rate(
        x.numpy(), frame_length=fl, hop_length=hl, center=False
    ).ravel()
    n = min(len(ours), len(ref))
    np.testing.assert_allclose(ours[:n], ref[:n], atol=1e-6)


# ── RMS (rectangular window achieves librosa parity) ──────────────────────


def test_rms_rectangular_matches_librosa():
    librosa = pytest.importorskip("librosa")
    from audiofeat.temporal.rms import rms

    torch.manual_seed(42)
    x = torch.randn(22050)
    fl, hl = 2048, 512
    ours = rms(x, fl, hl, window="rect").numpy()
    ref = librosa.feature.rms(
        y=x.numpy(), frame_length=fl, hop_length=hl, center=False
    ).ravel()
    n = min(len(ours), len(ref))
    np.testing.assert_allclose(ours[:n], ref[:n], atol=1e-5)


def test_rms_default_is_hann_backward_compatible():
    """Default rms() keeps the historical Hann-windowed behavior."""
    from audiofeat.temporal.rms import rms, frame_signal, hann_window

    torch.manual_seed(0)
    x = torch.randn(8000)
    fl, hl = 2048, 512
    out = rms(x, fl, hl)  # default window="hann"
    frames = frame_signal(x, fl, hl)
    w = hann_window(fl)
    expected = torch.sqrt(torch.sum((frames * w) ** 2, dim=1) / torch.sum(w ** 2))
    assert torch.allclose(out, expected, atol=1e-6)
    assert out.shape[0] > 0


def test_rms_boxcar_alias():
    from audiofeat.temporal.rms import rms

    x = torch.randn(4096)
    a = rms(x, 2048, 512, window="rect")
    b = rms(x, 2048, 512, window="boxcar")
    assert torch.allclose(a, b)


# ── short_time_energy / energy (no crash on short input) ──────────────────


def test_energy_short_input_does_not_crash():
    """energy() must not raise when the signal is shorter than one window
    (the old unfold-based implementation raised RuntimeError)."""
    from audiofeat.temporal.energy import energy

    short = torch.randn(100)  # well below 0.05 s * 22050 = 1102 samples
    out = energy(short, sample_rate=22050)
    assert out.ndim == 2 and out.shape[0] == 1
    assert torch.isfinite(out).all()


def test_energy_length_one_does_not_crash():
    from audiofeat.temporal.energy import energy

    out = energy(torch.tensor([0.5]), sample_rate=22050)
    assert torch.isfinite(out).all()


def test_energy_device_safety_note_cpu():
    """The window is built on the signal's device; on CPU the output stays
    on CPU (GPU path is exercised on CUDA hardware)."""
    from audiofeat.temporal.energy import energy

    x = torch.randn(22050)
    out = energy(x, sample_rate=22050)
    assert out.device == x.device


# ── entropy_of_energy (known answers + vectorized + device) ───────────────


def test_entropy_flat_equals_log2_n_sub_frames():
    from audiofeat.temporal.energy_entropy import entropy_of_energy

    n_sub = 10
    fl = 100  # sub_frame_length = 10, evenly divisible
    x = torch.ones(fl)  # every sub-frame carries equal energy
    ent = entropy_of_energy(x, frame_length=fl, hop_length=fl, n_sub_frames=n_sub)
    assert abs(float(ent[0]) - np.log2(n_sub)) < 1e-4


def test_entropy_impulse_is_zero():
    from audiofeat.temporal.energy_entropy import entropy_of_energy

    n_sub = 10
    fl = 100
    x = torch.zeros(fl)
    x[5] = 1.0  # all energy in a single sub-frame
    ent = entropy_of_energy(x, frame_length=fl, hop_length=fl, n_sub_frames=n_sub)
    assert abs(float(ent[0])) < 1e-5


def test_entropy_silence_is_zero_and_finite():
    from audiofeat.temporal.energy_entropy import entropy_of_energy

    ent = entropy_of_energy(torch.zeros(100), frame_length=100, hop_length=100, n_sub_frames=10)
    assert torch.isfinite(ent).all()
    assert float(ent[0]) == 0.0


def test_entropy_device_preserved():
    from audiofeat.temporal.energy_entropy import entropy_of_energy

    x = torch.randn(2048)
    ent = entropy_of_energy(x, frame_length=2048, hop_length=512, n_sub_frames=10)
    assert ent.device == x.device


# ── decay_time (smoothed envelope; tracks tau monotonically) ──────────────


def _decaying_sine(tau: float, sr: int = 22050, dur: float = 1.0, f: float = 440.0):
    t = torch.arange(int(sr * dur), dtype=torch.float32) / sr
    return torch.exp(-t / tau) * torch.sin(2 * torch.pi * f * t)


def test_decay_time_tracks_tau_monotonically():
    from audiofeat.temporal.decay import decay_time

    sr = 22050
    taus = [0.05, 0.1, 0.2, 0.4]
    decays = [float(decay_time(_decaying_sine(tau, sr), sample_rate=sr, threshold_db=-20.0)) for tau in taus]
    assert all(decays[i] < decays[i + 1] for i in range(len(decays) - 1)), decays


def test_decay_time_matches_theory_for_envelope():
    """For an envelope exp(-t/tau), the -20 dB time is tau*ln(10)."""
    from audiofeat.temporal.decay import decay_time

    sr = 22050
    for tau in (0.1, 0.2):
        t = torch.arange(int(sr * 1.5), dtype=torch.float32) / sr
        env = torch.exp(-t / tau)
        d = float(decay_time(env, sample_rate=sr, threshold_db=-20.0))
        theory = tau * float(np.log(10.0))
        assert abs(d - theory) < 0.03, f"tau={tau}: decay={d}, theory={theory}"


def test_decay_time_bare_sine_not_spurious_zero():
    """A decaying tonal signal must NOT return a near-zero decay time (the
    old raw-|x| implementation triggered on the first per-period trough)."""
    from audiofeat.temporal.decay import decay_time

    sr = 22050
    d = float(decay_time(_decaying_sine(0.1, sr), sample_rate=sr, threshold_db=-20.0))
    assert d > 0.1, f"decay={d} is a spurious near-zero value"


def test_decay_time_existing_contract_nonnegative():
    """0.5 s sine then 0.5 s silence -> a meaningful, non-negative decay."""
    from audiofeat.temporal.decay import decay_time

    sr = 22050
    t = torch.linspace(0, 0.5, int(sr * 0.5))
    sig = torch.sin(2 * torch.pi * 440 * t)
    x = torch.cat([sig, torch.zeros(int(sr * 0.5))])
    val = decay_time(x, sample_rate=sr, threshold_db=-20)
    assert isinstance(val, torch.Tensor)
    assert float(val) >= 0.0
    assert float(val) > 0.1  # crosses only in the silent tail


def test_decay_time_silence_returns_zero():
    from audiofeat.temporal.decay import decay_time

    assert float(decay_time(torch.zeros(22050), sample_rate=22050)) == 0.0


def test_decay_time_length_one_does_not_crash():
    from audiofeat.temporal.decay import decay_time

    val = decay_time(torch.tensor([1.0]), sample_rate=22050)
    assert torch.isfinite(val)


# ── breath_group_duration / speech_rate (no crash on edge inputs) ─────────


def test_breath_group_duration_single_below_threshold_sample():
    """Exactly one below-threshold sample previously crashed via
    .squeeze() -> 0-d tensor indexing."""
    from audiofeat.temporal.rhythm import breath_group_duration

    env = torch.ones(5) * 10.0
    env[2] = 0.0  # single sample below the (mean*0.25) threshold
    out = breath_group_duration(env, fs=22050)
    assert isinstance(out, torch.Tensor)  # no IndexError


def test_breath_group_duration_length_one():
    from audiofeat.temporal.rhythm import breath_group_duration

    out = breath_group_duration(torch.tensor([1.0]), fs=22050)
    assert isinstance(out, torch.Tensor)


def test_speech_rate_single_peak_no_crash():
    from audiofeat.temporal.rhythm import speech_rate

    fs = 1000
    t = torch.arange(fs, dtype=torch.float32) / fs
    x = torch.exp(-((t - 0.5) ** 2) / (2 * 0.01 ** 2))  # one clear syllable nucleus
    rate = speech_rate(x, fs=fs)
    assert isinstance(rate, float)
    assert rate >= 0.0


def test_speech_rate_length_one_and_two():
    from audiofeat.temporal.rhythm import speech_rate

    assert speech_rate(torch.tensor([1.0]), fs=22050) == 0.0
    assert speech_rate(torch.tensor([1.0, -1.0]), fs=22050) == 0.0


# ── amplitude_modulation_depth (recovers modulation index m) ──────────────


@pytest.mark.parametrize("m", [0.2, 0.5, 0.8])
def test_amplitude_modulation_depth_recovers_m(m):
    """For an envelope 1 + m*sin(2*pi*f_m*t) with a block spanning a full
    modulation period, the depth approaches m."""
    from audiofeat.temporal.amplitude import amplitude_modulation_depth

    fs = 22050
    fm = 5.0
    t = torch.arange(int(fs * 4), dtype=torch.float32) / fs
    env = 1.0 + m * torch.sin(2 * torch.pi * fm * t)
    win = int(fs / fm)  # one modulation period
    depth = float(amplitude_modulation_depth(env, window=win))
    assert abs(depth - m) < 0.01, f"depth={depth}, expected~{m}"


def test_amplitude_modulation_depth_short_returns_zero():
    from audiofeat.temporal.amplitude import amplitude_modulation_depth

    out = amplitude_modulation_depth(torch.ones(10), window=512)
    assert float(out) == 0.0


# ── loudness (LUFS; doc accuracy + short/silent guards) ───────────────────


def test_loudness_tone_is_finite_and_numeric_stable():
    from audiofeat.temporal.loudness import loudness

    sr = 22050
    t = torch.arange(int(sr * 1.5), dtype=torch.float32) / sr
    x = torch.sin(2 * torch.pi * 220.0 * t)
    loud = loudness(x, sample_rate=sr)
    assert torch.isfinite(loud).all()
    assert float(loud) < 0.0  # a -1..1 sine sits well below 0 LUFS


def test_loudness_docstring_describes_lufs_not_zwicker():
    from audiofeat.temporal.loudness import loudness

    doc = loudness.__doc__ or ""
    assert "LUFS" in doc
    assert "BS.1770" in doc or "R 128" in doc
    # The inaccurate psychoacoustic claims must be gone.
    assert "Sone" not in doc or "not" in doc  # only mentioned to disclaim
    assert "Zwicker" not in doc or "not" in doc


def test_loudness_silence_does_not_crash():
    from audiofeat.temporal.loudness import loudness

    out = loudness(torch.zeros(22050), sample_rate=22050)
    # Silence is mathematically -inf LUFS (or nan in some torchaudio builds);
    # the contract is only that it returns without raising.
    assert isinstance(out, torch.Tensor)


def test_loudness_empty_and_scalar_guards():
    from audiofeat.temporal.loudness import loudness

    with pytest.raises(ValueError):
        loudness(torch.zeros(0), sample_rate=22050)
    with pytest.raises(ValueError):
        loudness(torch.tensor(1.0), sample_rate=22050)
