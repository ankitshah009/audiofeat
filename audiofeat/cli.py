from __future__ import annotations

import argparse
import csv
import json
import platform
import shutil
from pathlib import Path

from ._version import __version__
from .catalog import build_feature_catalog, catalog_to_markdown, summarize_catalog
from .io.features import (
    DEFAULT_FRAME_LENGTH,
    DEFAULT_HOP_LENGTH,
    DEFAULT_SAMPLE_RATE,
    extract_features_for_directory,
    extract_features_from_file,
    load_audio,
    iter_audio_files,
    write_feature_rows_to_csv,
)
from .standards import (
    extract_opensmile_features,
)
from .validation.praat import (
    SPEAKER_PROFILES,
    apply_speaker_profile,
    compare_audio_to_praat_reference,
    evaluate_praat_report,
    extract_praat_reference,
    load_praat_reference,
    save_json,
)
from .validation.scorecard import run_gold_standard_scorecard


def _print_json(data: dict) -> None:
    print(json.dumps(data, indent=2))


def _add_shared_audio_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE)
    parser.add_argument("--frame-length", type=int, default=DEFAULT_FRAME_LENGTH)
    parser.add_argument("--hop-length", type=int, default=DEFAULT_HOP_LENGTH)


def _cmd_extract(args: argparse.Namespace) -> int:
    features = extract_features_from_file(
        args.audio_path,
        sample_rate=args.sample_rate,
        frame_length=args.frame_length,
        hop_length=args.hop_length,
    )
    if args.output:
        save_json(features, args.output)
        print(f"Wrote feature summary to {args.output}")
    else:
        _print_json(features)
    return 0


def _cmd_batch_extract(args: argparse.Namespace) -> int:
    errors: list[str] = []
    rows = extract_features_for_directory(
        args.input_dir,
        sample_rate=args.sample_rate,
        frame_length=args.frame_length,
        hop_length=args.hop_length,
        skip_errors=not args.strict,
        errors=errors,
    )
    if not rows:
        raise SystemExit("No supported audio files found.")
    out = write_feature_rows_to_csv(rows, args.output_csv)
    print(f"Wrote {len(rows)} rows to {out}")
    if errors:
        print(f"Skipped {len(errors)} file(s) due to decode/extraction errors.")
        for msg in errors:
            print(f"- {msg}")
    return 0


def _cmd_extract_opensmile(args: argparse.Namespace) -> int:
    try:
        features = extract_opensmile_features(
            args.audio_path,
            feature_set=args.feature_set,
            feature_level=args.feature_level,
            flatten=not args.no_flatten,
        )
    except ModuleNotFoundError as exc:
        raise SystemExit(str(exc)) from exc
    if args.output:
        if isinstance(features, dict):
            if args.output.suffix.lower() == ".csv":
                args.output.parent.mkdir(parents=True, exist_ok=True)
                with args.output.open("w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=sorted(features.keys()))
                    writer.writeheader()
                    writer.writerow(features)
            else:
                save_json(features, args.output)
        else:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            features.to_csv(args.output, index=True)
        print(f"Wrote openSMILE features to {args.output}")
    else:
        if isinstance(features, dict):
            _print_json(features)
        else:
            print(features.to_csv(index=True))
    return 0


def _cmd_list_features(args: argparse.Namespace) -> int:
    catalog = build_feature_catalog()
    if args.format == "markdown":
        text = catalog_to_markdown(catalog)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text)
            print(f"Wrote feature catalog to {args.output}")
        else:
            print(text)
        return 0

    payload = {
        "summary": summarize_catalog(catalog),
        "components": catalog,
    }
    if args.output:
        save_json(payload, args.output)
        print(f"Wrote feature catalog to {args.output}")
    else:
        _print_json(payload)
    return 0


def _cmd_gold_standard(args: argparse.Namespace) -> int:
    report = run_gold_standard_scorecard(
        sample_rate=args.sample_rate,
        frame_length=args.frame_length,
        hop_length=args.hop_length,
        praat_audio_path=args.audio_path,
        include_optional=not args.no_optional,
    )

    if args.output:
        save_json(report, args.output)
        print(f"Wrote gold-standard scorecard to {args.output}")
    else:
        _print_json(report)

    if report["score"] < args.min_score:
        return 1
    if args.fail_on_any and not report["passed"]:
        return 1
    return 0


def _cmd_doctor(args: argparse.Namespace) -> int:
    report: dict[str, object] = {}
    report["python"] = platform.python_version()
    report["platform"] = platform.platform()
    report["ffmpeg_in_path"] = shutil.which("ffmpeg") is not None
    try:
        import torch
        import torchaudio

        report["torch"] = str(torch.__version__)
        report["torchaudio"] = str(torchaudio.__version__)
        report["torch_torchaudio_major_minor_match"] = (
            ".".join(str(torch.__version__).split(".")[:2])
            == ".".join(str(torchaudio.__version__).split(".")[:2])
        )
    except Exception as exc:  # pragma: no cover
        report["torch_error"] = str(exc)

    module_versions: dict[str, str | None] = {}
    for module_name in ["librosa", "praat-parselmouth", "opensmile"]:
        try:
            imported = __import__(module_name if module_name != "praat-parselmouth" else "parselmouth")
            report[f"{module_name}_installed"] = True
            module_versions[module_name] = str(getattr(imported, "__version__", "unknown"))
        except Exception:
            report[f"{module_name}_installed"] = False
            module_versions[module_name] = None
    report["module_versions"] = module_versions

    sample_dir = Path(args.audio_dir)
    files = iter_audio_files(sample_dir) if sample_dir.exists() else []
    diagnostics = {"checked_dir": str(sample_dir), "files_checked": len(files), "errors": []}
    for path in files:
        try:
            waveform, sr = load_audio(path, target_sample_rate=None, mono=True)
            diagnostics[path.name] = {"ok": True, "sample_rate": sr, "num_samples": int(waveform.numel())}
        except Exception as exc:
            diagnostics["errors"].append(f"{path}: {exc}")
            diagnostics[path.name] = {"ok": False, "error": str(exc)}
    report["audio_diagnostics"] = diagnostics
    recommendations: list[str] = []
    if not report["ffmpeg_in_path"]:
        recommendations.append("Install ffmpeg to improve decoding support for mp3/m4a/ogg.")
    if not report.get("torch_torchaudio_major_minor_match", True):
        recommendations.append("Use matching torch/torchaudio major.minor versions.")
    if not report.get("praat-parselmouth_installed", False):
        recommendations.append("Install `audiofeat[validation]` for direct Praat extraction.")
    if not report.get("opensmile_installed", False):
        recommendations.append("Install `audiofeat[standards]` for eGeMAPS/ComParE extraction.")
    if recommendations:
        report["recommendations"] = recommendations

    if args.output:
        save_json(report, args.output)
        print(f"Wrote doctor report to {args.output}")
    else:
        _print_json(report)

    return 0


def _cmd_validate_praat(args: argparse.Namespace) -> int:
    profile_settings = apply_speaker_profile(
        speaker_profile=args.speaker_profile,
        pitch_floor=args.pitch_floor,
        pitch_ceiling=args.pitch_ceiling,
        max_formant=args.max_formant,
    )

    if args.praat_json:
        praat_reference = load_praat_reference(args.praat_json)
    else:
        praat_reference = extract_praat_reference(
            args.audio_path,
            speaker_profile=args.speaker_profile,
            pitch_floor=profile_settings["pitch_floor"],
            pitch_ceiling=profile_settings["pitch_ceiling"],
            num_formants=args.num_formants,
            max_formant=profile_settings["max_formant"],
            time_step=args.time_step,
            formant_window_length=args.formant_window_length,
            pre_emphasis_from_hz=args.pre_emphasis_from_hz,
        )
        if args.save_praat_reference:
            save_json(praat_reference, args.save_praat_reference)

    report = compare_audio_to_praat_reference(
        args.audio_path,
        praat_reference,
        sample_rate=args.sample_rate,
        frame_length=args.frame_length,
        hop_length=args.hop_length,
        speaker_profile=args.speaker_profile,
        pitch_floor=profile_settings["pitch_floor"],
        pitch_ceiling=profile_settings["pitch_ceiling"],
        pitch_method=args.pitch_method,
        yin_threshold=args.yin_threshold,
        formant_order=args.formant_order,
        num_formants=args.num_formants,
        max_formant=profile_settings["max_formant"],
        formant_method=args.formant_method,
        time_step_sec=args.time_step,
        formant_window_length_sec=args.formant_window_length,
        pre_emphasis_from_hz=args.pre_emphasis_from_hz,
    )
    evaluation = evaluate_praat_report(report)
    report["tolerance_check"] = evaluation

    if args.output:
        save_json(report, args.output)
        print(f"Wrote Praat validation report to {args.output}")
    else:
        _print_json(report)

    if args.fail_on_tolerance and not evaluation["passed"]:
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="audiofeat",
        description="Audio feature extraction and Praat validation toolkit.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract = subparsers.add_parser(
        "extract",
        help="Extract core audio features from one file and emit JSON.",
    )
    extract.add_argument("audio_path", type=Path, help="Path to input audio file.")
    _add_shared_audio_args(extract)
    extract.add_argument("--output", type=Path, default=None, help="Optional output JSON file.")
    extract.set_defaults(func=_cmd_extract)

    batch = subparsers.add_parser(
        "batch-extract",
        help="Extract core audio features for every file in a directory.",
    )
    batch.add_argument("input_dir", type=Path, help="Directory containing audio files.")
    batch.add_argument("output_csv", type=Path, help="Output CSV path.")
    _add_shared_audio_args(batch)
    batch.add_argument(
        "--strict",
        action="store_true",
        help="Fail immediately if any file cannot be decoded or processed.",
    )
    batch.set_defaults(func=_cmd_batch_extract)

    opensmile = subparsers.add_parser(
        "extract-opensmile",
        help="Extract standardized openSMILE feature sets (eGeMAPS/ComParE).",
    )
    opensmile.add_argument("audio_path", type=Path, help="Path to input audio file.")
    opensmile.add_argument("--feature-set", type=str, default="eGeMAPSv02")
    opensmile.add_argument("--feature-level", type=str, default="Functionals")
    opensmile.add_argument(
        "--no-flatten",
        action="store_true",
        help="Output full frame-level table instead of flattened first-row JSON.",
    )
    opensmile.add_argument("--output", type=Path, default=None, help="Optional output (.json or .csv).")
    opensmile.set_defaults(func=_cmd_extract_opensmile)

    catalog = subparsers.add_parser(
        "list-features",
        help="List discovered functions/components directly from the installed codebase.",
    )
    catalog.add_argument("--format", choices=["json", "markdown"], default="json")
    catalog.add_argument("--output", type=Path, default=None, help="Optional output path.")
    catalog.set_defaults(func=_cmd_list_features)

    doctor = subparsers.add_parser(
        "doctor",
        help="Diagnose environment dependencies and audio-file decodability.",
    )
    doctor.add_argument(
        "--audio-dir",
        type=Path,
        default=Path("examples"),
        help="Directory to scan for decodable audio assets (default: examples).",
    )
    doctor.add_argument("--output", type=Path, default=None, help="Optional JSON output report.")
    doctor.set_defaults(func=_cmd_doctor)

    gold = subparsers.add_parser(
        "gold-standard",
        help="Run strict validation checks and produce a score out of 100.",
    )
    gold.add_argument("--audio-path", type=Path, default=None, help="Optional audio path for Praat checks.")
    gold.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE)
    gold.add_argument("--frame-length", type=int, default=1024)
    gold.add_argument("--hop-length", type=int, default=256)
    gold.add_argument("--output", type=Path, default=None, help="Optional JSON output report.")
    gold.add_argument("--min-score", type=float, default=95.0, help="Return non-zero if score is lower.")
    gold.add_argument(
        "--no-optional",
        action="store_true",
        help="Skip optional checks that require extra dependencies.",
    )
    gold.add_argument(
        "--fail-on-any",
        action="store_true",
        help="Return non-zero if any non-skipped check fails.",
    )
    gold.set_defaults(func=_cmd_gold_standard)

    validate = subparsers.add_parser(
        "validate-praat",
        help="Compare audiofeat pitch/formants against Praat references.",
    )
    validate.add_argument("audio_path", type=Path, help="Path to input audio file.")
    source = validate.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--praat-json",
        type=Path,
        default=None,
        help="Existing Praat reference JSON.",
    )
    source.add_argument(
        "--extract-praat",
        action="store_true",
        help="Extract references directly with parselmouth.",
    )
    validate.add_argument(
        "--save-praat-reference",
        type=Path,
        default=None,
        help="When extracting via parselmouth, save the generated Praat JSON here.",
    )
    validate.add_argument("--output", type=Path, default=None, help="Optional output report path.")
    validate.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE)
    validate.add_argument("--frame-length", type=int, default=None)
    validate.add_argument("--hop-length", type=int, default=None)
    validate.add_argument(
        "--speaker-profile",
        choices=sorted(SPEAKER_PROFILES.keys()),
        default="neutral",
        help="Praat-style defaults for pitch/formant ranges.",
    )
    validate.add_argument("--pitch-floor", type=float, default=None)
    validate.add_argument("--pitch-ceiling", type=float, default=None)
    validate.add_argument("--pitch-method", choices=["autocorr", "yin", "pyin"], default="autocorr")
    validate.add_argument("--yin-threshold", type=float, default=0.1)
    validate.add_argument("--formant-order", type=int, default=None)
    validate.add_argument("--num-formants", type=int, default=5)
    validate.add_argument("--max-formant", type=float, default=None)
    validate.add_argument(
        "--formant-method",
        choices=["burg", "praat"],
        default="burg",
        help="audiofeat formant backend for the comparison side.",
    )
    validate.add_argument("--time-step", type=float, default=0.01)
    validate.add_argument(
        "--formant-window-length",
        type=float,
        default=0.025,
        help="Formant analysis window length in seconds.",
    )
    validate.add_argument(
        "--pre-emphasis-from-hz",
        type=float,
        default=50.0,
        help="Praat pre-emphasis frequency in Hz.",
    )
    validate.add_argument(
        "--fail-on-tolerance",
        action="store_true",
        help="Return non-zero exit code when any default tolerance check fails.",
    )
    validate.set_defaults(func=_cmd_validate_praat)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
