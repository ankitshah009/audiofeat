"""Tests for Praat alignment of pitch and voice quality features."""
import pytest
import torch
import numpy as np


def _sine_wave(sample_rate: int = 22050, frequency_hz: float = 180.0, duration_sec: float = 2.0):
    """Generate a pure sine wave test signal."""
    t = torch.arange(int(sample_rate * duration_sec), dtype=torch.float32) / sample_rate
    return torch.sin(2 * torch.pi * frequency_hz * t)


def _speech_like_signal(sample_rate: int = 22050, duration_sec: float = 1.8) -> torch.Tensor:
    """Generate a speech-like signal with harmonic structure."""
    t = torch.arange(int(sample_rate * duration_sec), dtype=torch.float32) / sample_rate
    source = torch.zeros_like(t)
    f0 = 120.0
    for k in range(1, 15):
        source += torch.sin(2 * torch.pi * (k * f0) * t) / k
    return source


class TestPraatPitchExtraction:
    """Tests for Praat-based pitch extraction."""

    @pytest.fixture
    def parselmouth_available(self):
        pytest.importorskip("parselmouth")
        return True

    def test_fundamental_frequency_praat_runs_on_tone(self, parselmouth_available):
        """Test that Praat pitch extraction runs on a pure tone."""
        from audiofeat.pitch.pitch_praat import fundamental_frequency_praat

        sr = 22050
        expected_f0 = 180.0
        sine = _sine_wave(sample_rate=sr, frequency_hz=expected_f0)

        f0 = fundamental_frequency_praat(
            sine,
            fs=sr,
            pitch_floor=75.0,
            pitch_ceiling=600.0,
        )

        assert isinstance(f0, torch.Tensor)
        assert f0.numel() > 0

        voiced = f0[f0 > 0]
        if voiced.numel() > 0:
            measured = float(voiced.median().item())
            rel_err = abs(measured - expected_f0) / expected_f0
            assert rel_err < 0.1, f"Expected ~{expected_f0} Hz, got {measured} Hz"

    def test_fundamental_frequency_praat_cc_runs(self, parselmouth_available):
        """Test that Praat cross-correlation pitch extraction runs."""
        from audiofeat.pitch.pitch_praat import fundamental_frequency_praat_cc

        sr = 22050
        sine = _sine_wave(sample_rate=sr, frequency_hz=200.0)

        f0 = fundamental_frequency_praat_cc(
            sine,
            fs=sr,
            pitch_floor=75.0,
            pitch_ceiling=600.0,
        )

        assert isinstance(f0, torch.Tensor)
        assert f0.numel() > 0

    def test_pitch_strength_praat_runs(self, parselmouth_available):
        """Test that pitch strength extraction runs."""
        from audiofeat.pitch.pitch_praat import pitch_strength_praat

        sr = 22050
        sine = _sine_wave(sample_rate=sr, frequency_hz=180.0)

        strength = pitch_strength_praat(
            sine,
            fs=sr,
            pitch_floor=75.0,
            pitch_ceiling=600.0,
        )

        assert isinstance(strength, torch.Tensor)
        assert strength.numel() > 0
        # Strength should be between 0 and 1
        assert strength.max() <= 1.0
        assert strength.min() >= 0.0

    def test_praat_pitch_matches_praat_reference(self, parselmouth_available):
        """Test that Praat pitch method produces very close results to Praat reference."""
        from audiofeat.pitch.pitch_praat import fundamental_frequency_praat
        from audiofeat.validation.praat import extract_praat_reference
        import tempfile
        import torchaudio
        import os

        sr = 22050
        signal = _speech_like_signal(sample_rate=sr)

        # Save to temp file for Praat reference extraction
        fd, temp_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            torchaudio.save(temp_path, signal.unsqueeze(0), sr)

            # Extract Praat reference
            praat_ref = extract_praat_reference(
                temp_path,
                speaker_profile="neutral",
                pitch_floor=75.0,
                pitch_ceiling=300.0,
            )

            # Extract using our Praat method
            f0 = fundamental_frequency_praat(
                signal,
                fs=sr,
                pitch_floor=75.0,
                pitch_ceiling=300.0,
            )

            voiced = f0[f0 > 0]
            if voiced.numel() > 0:
                our_mean = float(voiced.mean().item())
                praat_mean = praat_ref["pitch"]["mean_hz"]
                
                if np.isfinite(praat_mean) and np.isfinite(our_mean):
                    rel_err = abs(our_mean - praat_mean) / max(praat_mean, 1e-8)
                    # Should have very close parity (< 5% error)
                    assert rel_err < 0.05, f"Our mean: {our_mean}, Praat mean: {praat_mean}"
        finally:
            os.unlink(temp_path)


class TestPraatVoiceQuality:
    """Tests for Praat-based voice quality metrics."""

    @pytest.fixture
    def parselmouth_available(self):
        pytest.importorskip("parselmouth")
        return True

    def test_jitter_shimmer_praat_runs_on_tensor(self, parselmouth_available):
        """Test that Praat jitter/shimmer extraction runs on a tensor."""
        from audiofeat.voice.praat_voice import jitter_shimmer_praat

        sr = 22050
        signal = _speech_like_signal(sample_rate=sr)

        metrics = jitter_shimmer_praat(
            signal,
            fs=sr,
            pitch_floor=75.0,
            pitch_ceiling=300.0,
        )

        assert isinstance(metrics, dict)
        # Check that expected keys are present
        expected_keys = [
            "jitter_local_percent",
            "jitter_local_abs_sec",
            "jitter_rap_percent",
            "jitter_ppq5_percent",
            "shimmer_local_percent",
            "shimmer_local_db",
            "shimmer_apq3_percent",
            "shimmer_apq5_percent",
            "hnr_db",
            "num_periods",
        ]
        for key in expected_keys:
            assert key in metrics, f"Missing key: {key}"

    def test_jitter_shimmer_praat_runs_on_file(self, parselmouth_available, tmp_path):
        """Test that Praat jitter/shimmer extraction runs on a file."""
        from audiofeat.voice.praat_voice import jitter_shimmer_praat
        import torchaudio

        sr = 22050
        signal = _speech_like_signal(sample_rate=sr)
        audio_path = tmp_path / "test.wav"
        torchaudio.save(str(audio_path), signal.unsqueeze(0), sr)

        metrics = jitter_shimmer_praat(audio_path)

        assert isinstance(metrics, dict)
        assert "jitter_local_percent" in metrics
        assert "shimmer_local_percent" in metrics

    def test_hnr_praat_runs(self, parselmouth_available):
        """Test that Praat HNR extraction runs."""
        from audiofeat.voice.praat_voice import hnr_praat

        sr = 22050
        signal = _speech_like_signal(sample_rate=sr)

        hnr = hnr_praat(signal, fs=sr, pitch_floor=75.0)

        assert isinstance(hnr, float)
        # HNR should be finite for speech-like signal
        if np.isfinite(hnr):
            # Typical speech HNR is 5-25 dB, but synthetic signals can be much higher
            assert -10 < hnr < 100, f"Unexpected HNR value: {hnr}"


class TestValidationPraatPitchMethod:
    """Tests for Praat pitch method in validation module."""

    @pytest.fixture
    def parselmouth_available(self):
        pytest.importorskip("parselmouth")
        return True

    def test_validation_praat_pitch_method(self, parselmouth_available, tmp_path):
        """Test that validation module supports pitch_method='praat'."""
        from audiofeat.validation.praat import compare_audio_to_praat_reference, extract_praat_reference
        import torchaudio

        sr = 22050
        signal = _speech_like_signal(sample_rate=sr)
        audio_path = tmp_path / "test.wav"
        torchaudio.save(str(audio_path), signal.unsqueeze(0), sr)

        praat_ref = extract_praat_reference(audio_path, speaker_profile="neutral")

        report = compare_audio_to_praat_reference(
            audio_path,
            praat_ref,
            sample_rate=sr,
            pitch_method="praat",  # Use Praat method
            formant_method="praat",
        )

        assert "relative_error" in report
        rel_err = report["relative_error"]

        # With Praat backend on both sides, errors should be minimal
        if np.isfinite(rel_err.get("pitch_mean", float("nan"))):
            assert rel_err["pitch_mean"] < 0.05, f"Pitch mean error: {rel_err['pitch_mean']}"
        if np.isfinite(rel_err.get("pitch_median", float("nan"))):
            assert rel_err["pitch_median"] < 0.05, f"Pitch median error: {rel_err['pitch_median']}"
