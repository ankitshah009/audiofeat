import json
from pathlib import Path

import torch

from audiofeat.validation.praat import (
    apply_speaker_profile,
    build_praat_comparison_report,
    compute_audiofeat_reference,
    evaluate_praat_report,
    load_praat_reference,
    save_json,
)


def _sine_wave(sample_rate: int = 22050, frequency_hz: float = 180.0, duration_sec: float = 2.0):
    t = torch.arange(int(sample_rate * duration_sec), dtype=torch.float32) / sample_rate
    return torch.sin(2 * torch.pi * frequency_hz * t)


def test_praat_report_passes_when_reference_matches_audiofeat():
    waveform = _sine_wave()
    sr = 22050
    frame_length = 1024
    hop_length = 256

    af = compute_audiofeat_reference(
        waveform,
        sr,
        frame_length=frame_length,
        hop_length=hop_length,
    )
    praat_reference = {
        "pitch": {
            "mean_hz": af["pitch"]["mean_hz"],
            "median_hz": af["pitch"]["median_hz"],
        },
        "formants": {
            "f1_median_hz": af["formants"]["f1_median_hz"],
            "f2_median_hz": af["formants"]["f2_median_hz"],
        },
    }
    report = build_praat_comparison_report(
        waveform,
        sr,
        praat_reference,
        frame_length=frame_length,
        hop_length=hop_length,
    )
    evaluation = evaluate_praat_report(report)
    assert evaluation["passed"] is True


def test_praat_report_fails_when_errors_exceed_tolerances():
    waveform = _sine_wave()
    report = build_praat_comparison_report(
        waveform,
        22050,
        {
            "pitch": {"mean_hz": 40.0, "median_hz": 40.0},
            "formants": {"f1_median_hz": 5000.0, "f2_median_hz": 9000.0},
        },
        frame_length=1024,
        hop_length=256,
    )
    evaluation = evaluate_praat_report(report)
    assert evaluation["passed"] is False


def test_load_praat_reference_supports_nested_structure(tmp_path: Path):
    ref_path = tmp_path / "praat_reference.json"
    ref_path.write_text(
        json.dumps(
            {
                "praat": {
                    "pitch": {"mean_hz": 120.0, "median_hz": 118.0},
                    "formants": {"f1_median_hz": 500.0, "f2_median_hz": 1500.0},
                }
            }
        )
    )
    loaded = load_praat_reference(ref_path)
    assert loaded["pitch"]["mean_hz"] == 120.0
    assert loaded["formants"]["f2_median_hz"] == 1500.0


def test_apply_speaker_profile_defaults():
    male = apply_speaker_profile(speaker_profile="male")
    female = apply_speaker_profile(speaker_profile="female")
    assert male["max_formant"] == 5000.0
    assert female["max_formant"] == 5500.0


def test_save_json_sanitizes_nan_to_null(tmp_path: Path):
    out = tmp_path / "report.json"
    save_json({"a": float("nan"), "nested": {"b": float("inf")}}, out)
    text = out.read_text()
    assert '"a": null' in text
    assert '"b": null' in text
