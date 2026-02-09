# audiofeat: Comprehensive Audio + Voice Feature Extraction in Python

`audiofeat` is a PyTorch-first toolkit for extracting temporal, spectral, cepstral, pitch, rhythm, and voice-quality features from audio.

It is designed for two usage modes:

- low-level feature engineering (function-by-function DSP)
- production-style extraction workflows (single file, batch CSV, diagnostics, validation)

The package includes direct Praat comparison tooling and standardized openSMILE feature-set wrappers for reproducible speech research pipelines.

## Reliability upgrades in this version

- Added a strict `gold-standard` scorecard command that returns a numeric score out of 100.
- Added CI gate that requires `gold-standard` checks to pass.
- Added robust CLI diagnostics (`doctor`) with dependency recommendations.
- Added code-discovered feature catalog (`list-features`) to keep docs aligned with code.
- Added sample-rate correctness fixes for spectral centroid and rolloff.
- Added stronger test coverage for CLI, catalog, scorecard, standards, and spectral sample-rate behavior.

## Why Praat comparisons can fail

Common causes of mismatch between repo output and Praat references:

1. Invalid `.wav` assets (placeholder text files, missing LFS objects, corrupt audio).
2. Mismatched analysis settings (pitch floor/ceiling, max formant, time step, window length, pre-emphasis).
3. Backend differences (`burg` approximation vs direct Praat/parselmouth backend).
4. Algorithmic differences (YIN produces flatter trajectories than Praat's autocorrelation with pathfinding).

### Getting exact Praat parity

For research workflows requiring exact Praat parity, use the Praat backends directly:

```python
# Pitch extraction with exact Praat parity
from audiofeat.pitch import fundamental_frequency_praat

f0 = fundamental_frequency_praat(
    waveform,
    fs=sample_rate,
    pitch_floor=75.0,
    pitch_ceiling=600.0,
)

# Voice quality metrics with exact Praat parity
from audiofeat.voice import jitter_shimmer_praat

metrics = jitter_shimmer_praat(
    waveform,
    fs=sample_rate,
    pitch_floor=75.0,
    pitch_ceiling=300.0,
)
print(metrics["jitter_local_percent"], metrics["shimmer_local_percent"])
```

### Validation workflow

Use `audiofeat doctor` to check audio assets, then run validation:

```bash
audiofeat doctor --audio-dir examples
```

Validate with Praat backend for maximum parity:

```bash
audiofeat validate-praat path/to/audio.wav \
  --extract-praat \
  --speaker-profile neutral \
  --pitch-method praat \
  --formant-method praat \
  --output outputs/praat_report.json
```

## Installation (environment-first)

Python `>=3.8` is required.

### Option A: `venv` (recommended)

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install audiofeat
```

### Option B: Conda

```bash
conda create -n audiofeat python=3.11 -y
conda activate audiofeat
pip install audiofeat
```

### Option C: `uv`

```bash
uv venv
source .venv/bin/activate
uv pip install audiofeat
```

### Install from source

```bash
git clone https://github.com/ankitshah009/audiofeat.git
cd audiofeat
pip install -e .
```

### Optional dependency extras

- `audiofeat[dev]`: tests, linting, type tooling
- `audiofeat[examples]`: plotting + librosa-style examples
- `audiofeat[validation]`: Praat/parselmouth validation backend
- `audiofeat[standards]`: openSMILE eGeMAPS/ComParE wrappers
- `audiofeat[models]`: heavy wrapper modules (ASR/diarization/etc.)
- `audiofeat[full]`: examples + validation + standards in one install

Examples:

```bash
pip install "audiofeat[validation]"
pip install "audiofeat[full]"
```

## Quick start

### Python API

```python
import torch
import audiofeat

sr = 22050
x = torch.randn(sr * 3)

rms = audiofeat.rms(x, frame_length=2048, hop_length=512)
zcr = audiofeat.zero_crossing_rate(x, frame_length=2048, hop_length=512)
centroid = audiofeat.spectral_centroid(x, frame_length=2048, hop_length=512, sample_rate=sr)
f0 = audiofeat.fundamental_frequency_yin(x, fs=sr, frame_length=2048, hop_length=512)
```

### File extraction workflow

```python
from audiofeat.io.features import extract_features_from_file

features = extract_features_from_file("path/to/audio.wav")
print(features["f0_mean_hz"], features["rms_mean"])
```

## CLI commands

```bash
audiofeat --help
```

### `extract`

```bash
audiofeat extract path/to/audio.wav --output outputs/features.json
```

### `batch-extract`

```bash
audiofeat batch-extract path/to/audio_dir outputs/batch_features.csv
```

### `validate-praat`

```bash
# validate against existing reference
audiofeat validate-praat path/to/audio.wav \
  --praat-json examples/praat_reference.json \
  --output outputs/praat_report.json

# extract reference directly with parselmouth
audiofeat validate-praat path/to/audio.wav \
  --extract-praat \
  --save-praat-reference outputs/praat_reference.json \
  --output outputs/praat_report.json
```

### `extract-opensmile`

```bash
audiofeat extract-opensmile path/to/audio.wav \
  --feature-set eGeMAPSv02 \
  --feature-level Functionals \
  --output outputs/egemaps.json
```

### `doctor`

```bash
audiofeat doctor --audio-dir examples --output outputs/doctor_report.json
```

### `gold-standard` (objective score out of 100)

```bash
audiofeat gold-standard --no-optional --min-score 100 --fail-on-any
```

### `list-features` (code-discovered feature catalog)

```bash
audiofeat list-features --format json --output outputs/feature_catalog.json
audiofeat list-features --format markdown --output outputs/FEATURE_CATALOG.md
```

## Package component map

### `audiofeat.temporal`

Frame-level energy and dynamics descriptors: RMS, ZCR, rhythm features, attack, decay, loudness, modulation-depth style helpers.

### `audiofeat.spectral`

Frequency-domain descriptors: centroid, rolloff, flux, flatness, entropy, moments, contrast, slope, crest, chroma, tonnetz, MFCC/GFCC, LPC/LSP, spectrograms, and formant pipelines.

### `audiofeat.pitch`

Pitch and pitch-derived descriptors: autocorrelation F0, YIN, optional pYIN, pitch strength, semitone variation.

### `audiofeat.voice`

Voice-quality helpers: jitter/shimmer pipelines, CPP, alpha ratio, Hammarberg index, harmonic difference and quality descriptors.

### `audiofeat.cepstral` + `audiofeat.stats`

Cepstral families (LPCC/GTCC, deltas) and statistical functionals (mean/std/min/max/skewness/kurtosis).

### `audiofeat.io`

Robust loading (`load_audio`), single-file summaries, batch extraction, CSV writing, and placeholder-audio detection.

### `audiofeat.validation`

Praat reference extraction/comparison and the `gold-standard` quality scorecard.

### `audiofeat.standards`

openSMILE wrappers for standardized feature sets (eGeMAPS/ComParE) for benchmarking parity.

## Keep README aligned with code automatically

Generate a full feature catalog directly from code:

```bash
python scripts/generate_feature_catalog.py
```

Outputs:

- `docs/FEATURE_CATALOG.md`
- `docs/FEATURE_CATALOG.json`

You can regenerate these whenever modules/functions change.

## Validation methodology

See `docs/VALIDATION.md` for check definitions, scoring details, and CI gating behavior.

## Testing

```bash
pytest -q
```

With coverage:

```bash
pytest --cov=audiofeat --cov-report=term-missing -q
```

## Troubleshooting

- `Failed to decode audio file`: run `audiofeat doctor`; verify file is real audio, not text placeholder.
- Large Praat mismatch: align `--speaker-profile`, `--pitch-floor`, `--pitch-ceiling`, `--max-formant`, `--time-step`.
- Need closest parity run: use `--formant-method praat` with `audiofeat[validation]` installed.
- Missing openSMILE support: install `audiofeat[standards]`.

## External references used for alignment

- Praat manual (Pitch): <https://praat.org/manual/Sound__To_Pitch__ac____.html>
- Praat manual (Formant Burg): <https://praat.org/manual/Sound__To_Formant__burg____.html>
- Praat manual (Formant robust): <https://praat.org/manual/Sound__To_Formant__robust____.html>
- Parselmouth API docs: <https://parselmouth.readthedocs.io/en/stable/api_reference.html>
- librosa `pyin`: <https://librosa.org/doc/main/generated/librosa.pyin.html>
- openSMILE Python docs: <https://audeering.github.io/opensmile-python/>
- torchaudio transforms docs: <https://pytorch.org/audio/stable/transforms.html>

## Citation

```bibtex
@phdthesis{shah2024computational,
  title={Computational Audition with Imprecise Labels},
  author={Shah, Ankit Parag},
  year={2024},
  school={Carnegie Mellon University Pittsburgh, PA}
}
```
