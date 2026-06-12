"""Parallel audio feature extraction with audiofeat.

Extracts features from many files concurrently using a process pool. Each worker
loads a file with ``audiofeat.load_audio`` and computes real ``audiofeat``
features, then saves them as a ``.pt`` tensor dict.

Run it against the bundled samples:

    python examples/parallel_extraction.py examples

Requires the ``io`` extra for file decoding: ``pip install "audiofeat[io]"``.
"""

from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import torch

import audiofeat


def extract_features_for_file(
    file_path: str,
    output_dir: str,
    sample_rate: int = 16000,
) -> str:
    """Extract a small set of real ``audiofeat`` features from one file.

    Runs inside a worker process. Returns a human-readable status string.
    """
    try:
        # audiofeat.load_audio handles mono conversion + resampling and falls
        # back to soundfile when torchaudio has no decoder available.
        waveform, sr = audiofeat.load_audio(file_path, target_sample_rate=sample_rate)
    except Exception as exc:  # noqa: BLE001 - report and continue with other files
        return f"Failed to load {file_path}: {exc}"

    try:
        rms = audiofeat.rms(waveform, frame_length=2048, hop_length=512)
        zcr = audiofeat.zero_crossing_rate(waveform, frame_length=2048, hop_length=512)
        centroid = audiofeat.spectral_centroid(
            waveform, frame_length=2048, hop_length=512, sample_rate=sr
        )
        mfccs = audiofeat.mfcc(waveform, sr)  # (n_mfcc, frames)
        f0 = audiofeat.fundamental_frequency_yin(
            waveform, fs=sr, frame_length=2048, hop_length=512
        )
        voiced = f0[f0 > 0]

        features = {
            "rms_mean": rms.mean(),
            "zcr_mean": zcr.mean(),
            "spectral_centroid_mean": centroid.mean(),
            "mfcc_mean": mfccs.mean(dim=1),  # per-coefficient mean
            "f0_mean_hz": voiced.mean() if voiced.numel() else torch.tensor(float("nan")),
        }

        os.makedirs(output_dir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(file_path))[0]
        output_filename = os.path.join(output_dir, f"{stem}.pt")
        torch.save(features, output_filename)
        return f"Successfully processed {file_path} -> {output_filename}"
    except Exception as exc:  # noqa: BLE001 - report and continue with other files
        return f"Error extracting features from {file_path}: {exc}"


def parallel_feature_extraction(
    audio_files: list[str],
    output_dir: str,
    num_processes: int | None = None,
    sample_rate: int = 16000,
) -> None:
    """Extract features from ``audio_files`` concurrently into ``output_dir``."""
    os.makedirs(output_dir, exist_ok=True)
    if num_processes is None:
        num_processes = os.cpu_count() or 1

    print(f"Starting parallel feature extraction with {num_processes} processes...")
    start_time = time.time()

    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        future_to_file = {
            executor.submit(extract_features_for_file, fp, output_dir, sample_rate): fp
            for fp in audio_files
        }
        for future in as_completed(future_to_file):
            file_path = future_to_file[future]
            try:
                print(future.result())
            except Exception as exc:  # noqa: BLE001
                print(f"{file_path} generated an exception: {exc}")

    elapsed = time.time() - start_time
    print(f"Parallel feature extraction completed in {elapsed:.2f} seconds.")


def _discover_wavs(audio_dir: str) -> list[str]:
    return sorted(
        os.path.join(audio_dir, name)
        for name in os.listdir(audio_dir)
        if name.lower().endswith(".wav")
    )


if __name__ == "__main__":
    # Default to the bundled sample directory next to this script.
    audio_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
    files = _discover_wavs(audio_dir)
    if not files:
        print(f"No .wav files found in {audio_dir!r}.")
        raise SystemExit(0)

    output_features_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "extracted_features"
    )
    print(f"Found {len(files)} file(s) in {audio_dir!r}.")
    parallel_feature_extraction(files, output_features_dir, num_processes=4)
    print(f"Extracted features saved to {output_features_dir}/")
