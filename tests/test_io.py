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


def test_load_audio_falls_back_to_soundfile(tmp_path: Path, monkeypatch):
    soundfile = pytest.importorskip("soundfile")

    audio_path = tmp_path / "tone.wav"
    waveform = _sine_wave(sample_rate=22050)
    soundfile.write(str(audio_path), waveform.numpy(), 22050)

    def _raise_torchcodec(*_args, **_kwargs):
        raise ImportError(
            "TorchCodec is required to decode audio with this version of torchaudio."
        )

    monkeypatch.setattr(torchaudio, "load", _raise_torchcodec)

    loaded, sr = load_audio(audio_path, target_sample_rate=22050)
    assert sr == 22050
    assert loaded.dtype == torch.float32
    assert loaded.dim() == 1  # mono downmix preserved
    assert loaded.numel() == waveform.numel()
    torch.testing.assert_close(loaded, waveform, atol=1e-4, rtol=0)


def test_load_audio_soundfile_fallback_preserves_resampling(tmp_path: Path, monkeypatch):
    soundfile = pytest.importorskip("soundfile")

    audio_path = tmp_path / "tone_44k.wav"
    waveform = _sine_wave(sample_rate=44100)
    soundfile.write(str(audio_path), waveform.numpy(), 44100)

    monkeypatch.setattr(
        torchaudio,
        "load",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("Couldn't find appropriate backend")),
    )

    loaded, sr = load_audio(audio_path, target_sample_rate=22050)
    assert sr == 22050
    # Resampling 44.1k -> 22.05k should roughly halve the sample count.
    assert abs(loaded.numel() - waveform.numel() // 2) <= 2


def test_load_audio_raises_when_all_backends_fail(tmp_path: Path, monkeypatch):
    import audiofeat.io.features as features_mod

    audio_path = tmp_path / "tone.wav"
    audio_path.write_bytes(b"RIFF....WAVEfmt ")  # not a placeholder, just undecodable

    monkeypatch.setattr(
        torchaudio,
        "load",
        lambda *a, **k: (_ for _ in ()).throw(ImportError("TorchCodec is required")),
    )

    def _no_soundfile(_path):
        raise ImportError("No module named 'soundfile'")

    monkeypatch.setattr(features_mod, "_read_with_soundfile", _no_soundfile)

    with pytest.raises(RuntimeError) as excinfo:
        load_audio(audio_path)
    message = str(excinfo.value)
    assert "audiofeat[io]" in message
    assert "torchcodec" in message


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
