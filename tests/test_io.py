from pathlib import Path

import pytest
import torch
import torchaudio

from audiofeat.io.features import (
    extract_core_features,
    extract_features_from_file,
    iter_audio_files,
    load_audio,
    write_feature_rows_to_csv,
)


def _sine_wave(sample_rate: int = 22050, frequency_hz: float = 220.0, duration_sec: float = 1.0):
    t = torch.arange(int(sample_rate * duration_sec), dtype=torch.float32) / sample_rate
    return torch.sin(2 * torch.pi * frequency_hz * t)


def test_load_audio_reports_placeholder_hint(tmp_path: Path):
    bad_audio = tmp_path / "bad.wav"
    bad_audio.write_text("404: Not Found")
    with pytest.raises(RuntimeError, match="placeholder"):
        load_audio(bad_audio)


def test_extract_features_from_file_and_write_csv(tmp_path: Path):
    waveform = _sine_wave().unsqueeze(0)
    audio_path = tmp_path / "tone.wav"
    torchaudio.save(str(audio_path), waveform, 22050)

    features = extract_features_from_file(audio_path)
    assert features["path"] == str(audio_path)
    assert features["sample_rate"] == 22050
    assert "f0_mean_hz" in features
    assert "mfcc_mean_0" in features

    out_csv = tmp_path / "features.csv"
    write_feature_rows_to_csv([features], out_csv)
    assert out_csv.exists()

    files = iter_audio_files(tmp_path)
    assert audio_path in files


def test_extract_core_features_handles_short_input():
    waveform = torch.randn(64)
    features = extract_core_features(
        waveform,
        sample_rate=22050,
        frame_length=512,
        hop_length=256,
    )
    assert features["num_samples"] == 64
    assert features["duration_sec"] > 0
