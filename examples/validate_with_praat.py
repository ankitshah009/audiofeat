"""
Validate audiofeat pitch/formants against Praat references.

Usage examples:
  python examples/validate_with_praat.py path/to/audio.wav --praat-json examples/praat_reference.json
  python examples/validate_with_praat.py path/to/audio.wav --extract-praat --save-praat-reference outputs/praat_ref.json
"""

import argparse
import json
from pathlib import Path

from audiofeat.validation.praat import (
    SPEAKER_PROFILES,
    apply_speaker_profile,
    compare_audio_to_praat_reference,
    evaluate_praat_report,
    extract_praat_reference,
    load_praat_reference,
    save_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare audiofeat pitch/formants with Praat reference values.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("audio_path", type=Path, help="Path to input audio file.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--praat-json", type=Path, help="Path to existing Praat reference JSON.")
    source.add_argument(
        "--extract-praat",
        action="store_true",
        help="Extract Praat references directly (requires `pip install audiofeat[validation]`).",
    )
    parser.add_argument("--save-praat-reference", type=Path, default=None)
    parser.add_argument("--sample-rate", type=int, default=22050)
    parser.add_argument("--frame-length", type=int, default=None)
    parser.add_argument("--hop-length", type=int, default=None)
    parser.add_argument(
        "--speaker-profile",
        choices=sorted(SPEAKER_PROFILES.keys()),
        default="neutral",
        help="Praat-style defaults for pitch and formant ranges.",
    )
    parser.add_argument("--pitch-floor", type=float, default=None)
    parser.add_argument("--pitch-ceiling", type=float, default=None)
    parser.add_argument("--pitch-method", choices=["autocorr", "yin", "pyin"], default="autocorr")
    parser.add_argument("--yin-threshold", type=float, default=0.1)
    parser.add_argument("--formant-order", type=int, default=None)
    parser.add_argument("--num-formants", type=int, default=5)
    parser.add_argument("--max-formant", type=float, default=None)
    parser.add_argument("--formant-method", choices=["burg", "praat"], default="burg")
    parser.add_argument("--time-step", type=float, default=0.01, help="Analysis hop in seconds.")
    parser.add_argument("--formant-window-length", type=float, default=0.025)
    parser.add_argument("--pre-emphasis-from-hz", type=float, default=50.0)
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path.")
    parser.add_argument(
        "--fail-on-tolerance",
        action="store_true",
        help="Return non-zero exit code when default tolerance checks fail.",
    )
    args = parser.parse_args()

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
    report["tolerance_check"] = evaluate_praat_report(report)

    if args.output:
        save_json(report, args.output)
        print(f"Wrote report to {args.output}")
    else:
        print(json.dumps(report, indent=2))

    if args.fail_on_tolerance and not report["tolerance_check"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
