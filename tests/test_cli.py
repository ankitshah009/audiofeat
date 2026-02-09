import json
import sys
from pathlib import Path

import torch
import torchaudio

from audiofeat.cli import main


def _run_cli(monkeypatch, args: list[str]) -> int:
    monkeypatch.setattr(sys, "argv", ["audiofeat", *args])
    return int(main())


def _tone(sample_rate: int = 22050, frequency_hz: float = 180.0, duration_sec: float = 1.0):
    t = torch.arange(int(sample_rate * duration_sec), dtype=torch.float32) / sample_rate
    return torch.sin(2 * torch.pi * frequency_hz * t)


def test_cli_extract_and_doctor(monkeypatch, tmp_path: Path):
    audio_path = tmp_path / "tone.wav"
    torchaudio.save(str(audio_path), _tone().unsqueeze(0), 22050)

    extract_out = tmp_path / "extract.json"
    code = _run_cli(monkeypatch, ["extract", str(audio_path), "--output", str(extract_out)])
    assert code == 0
    payload = json.loads(extract_out.read_text())
    assert payload["sample_rate"] == 22050
    assert "f0_mean_hz" in payload

    doctor_out = tmp_path / "doctor.json"
    code = _run_cli(
        monkeypatch,
        ["doctor", "--audio-dir", str(tmp_path), "--output", str(doctor_out)],
    )
    assert code == 0
    doctor = json.loads(doctor_out.read_text())
    assert "audio_diagnostics" in doctor
    assert doctor["audio_diagnostics"]["files_checked"] >= 1


def test_cli_catalog_and_gold_standard(monkeypatch, tmp_path: Path):
    catalog_out = tmp_path / "catalog.json"
    code = _run_cli(
        monkeypatch,
        ["list-features", "--format", "json", "--output", str(catalog_out)],
    )
    assert code == 0
    payload = json.loads(catalog_out.read_text())
    assert "summary" in payload
    assert payload["summary"]["total_features"] > 0

    score_out = tmp_path / "scorecard.json"
    code = _run_cli(
        monkeypatch,
        [
            "gold-standard",
            "--no-optional",
            "--min-score",
            "0",
            "--output",
            str(score_out),
        ],
    )
    assert code == 0
    score = json.loads(score_out.read_text())
    assert 0.0 <= float(score["score"]) <= 100.0
