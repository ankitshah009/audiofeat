import argparse
from pathlib import Path

from audiofeat.io.features import (
    DEFAULT_FRAME_LENGTH,
    DEFAULT_HOP_LENGTH,
    DEFAULT_SAMPLE_RATE,
    extract_features_for_directory,
    write_feature_rows_to_csv,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch extract audiofeat features and save to CSV.")
    parser.add_argument("input_dir", type=Path, help="Directory with audio files.")
    parser.add_argument("output_csv", type=Path, help="Path to output CSV file.")
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE)
    parser.add_argument("--frame-length", type=int, default=DEFAULT_FRAME_LENGTH)
    parser.add_argument("--hop-length", type=int, default=DEFAULT_HOP_LENGTH)
    args = parser.parse_args()

    errors = []
    rows = extract_features_for_directory(
        args.input_dir,
        sample_rate=args.sample_rate,
        frame_length=args.frame_length,
        hop_length=args.hop_length,
        errors=errors,
    )
    if not rows:
        raise SystemExit("No supported audio files found.")

    output_path = write_feature_rows_to_csv(rows, args.output_csv)
    print(f"Wrote {len(rows)} rows to {output_path}")
    if errors:
        print(f"Skipped {len(errors)} file(s) that could not be decoded.")


if __name__ == "__main__":
    main()
