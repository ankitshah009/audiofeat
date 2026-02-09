import argparse
import json
from pathlib import Path

from audiofeat.io.features import (
    DEFAULT_FRAME_LENGTH,
    DEFAULT_HOP_LENGTH,
    DEFAULT_SAMPLE_RATE,
    extract_features_from_file,
)
from audiofeat.validation.praat import save_json


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract a compact audiofeat feature summary from one file."
    )
    parser.add_argument("audio_path", type=Path, help="Path to input audio file.")
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE)
    parser.add_argument("--frame-length", type=int, default=DEFAULT_FRAME_LENGTH)
    parser.add_argument("--hop-length", type=int, default=DEFAULT_HOP_LENGTH)
    parser.add_argument("--output", type=Path, default=None, help="Optional output JSON path.")
    args = parser.parse_args()

    features = extract_features_from_file(
        args.audio_path,
        sample_rate=args.sample_rate,
        frame_length=args.frame_length,
        hop_length=args.hop_length,
    )

    if args.output:
        save_json(features, args.output)
    else:
        print(json.dumps(features, indent=2))


if __name__ == "__main__":
    main()
