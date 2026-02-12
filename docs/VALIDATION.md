# Validation & Praat Comparison

This document covers audiofeat's validation infrastructure: the gold-standard scorecard, Praat reference comparison, openSMILE integration, and troubleshooting.

## Gold-Standard Scorecard

The `gold-standard` command runs deterministic checks and computes a quality score normalized to 100:

```bash
audiofeat gold-standard --no-optional --min-score 100 --fail-on-any
```

### What is checked

Required checks:

- RMS theoretical consistency on a constant signal.
- ZCR theoretical consistency on a pure tone.
- Spectral centroid consistency on a pure tone.
- Spectral rolloff consistency on a pure tone.
- Pitch accuracy (autocorrelation) on a pure tone.
- Pitch accuracy (YIN) on a pure tone.
- Formant plausibility and monotonicity on a deterministic speech-like signal.

Optional checks (enabled by default; skipped if dependencies are not installed):

- pYIN pitch accuracy (`librosa`).
- Praat native alignment (`praat-parselmouth`) with tolerance evaluation.
- Praat backend parity (`method='praat'`) for near-reference equivalence.

### Score model

- Every check has a fixed point weight.
- Skipped checks are excluded from the denominator.
- Score = `(earned_points / available_points) * 100`.
- `--min-score` enforces an exit code for CI gates.
- `--fail-on-any` enforces strict pass for all non-skipped checks.

## Praat Comparison

### Why Praat comparisons can fail

Common causes of mismatch between audiofeat output and Praat references:

1. **Invalid audio assets** — placeholder text files, missing LFS objects, corrupt audio.
2. **Mismatched analysis settings** — pitch floor/ceiling, max formant, time step, window length, pre-emphasis.
3. **Backend differences** — Burg approximation vs direct Praat/parselmouth backend.
4. **Algorithmic differences** — YIN produces flatter trajectories than Praat's autocorrelation with pathfinding.

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

### Praat validation workflow

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

Validate against an existing reference:

```bash
audiofeat validate-praat path/to/audio.wav \
  --praat-json examples/praat_reference.json \
  --output outputs/praat_report.json
```

Extract and save a Praat reference for later use:

```bash
audiofeat validate-praat path/to/audio.wav \
  --extract-praat \
  --save-praat-reference outputs/praat_reference.json \
  --output outputs/praat_report.json
```

## openSMILE Integration

Extract standardized feature sets for benchmarking:

```bash
pip install "audiofeat[standards]"

audiofeat extract-opensmile path/to/audio.wav \
  --feature-set eGeMAPSv02 \
  --feature-level Functionals \
  --output outputs/egemaps.json
```

Supported feature sets: ComParE_2016, GeMAPSv01a/b, eGeMAPSv01a/b, eGeMAPSv02.

## Troubleshooting

- **`Failed to decode audio file`**: Run `audiofeat doctor`; verify the file is real audio, not a text placeholder.
- **Large Praat mismatch**: Align `--speaker-profile`, `--pitch-floor`, `--pitch-ceiling`, `--max-formant`, `--time-step`.
- **Need closest parity**: Use `--formant-method praat` with `audiofeat[validation]` installed.
- **Missing openSMILE support**: Install `audiofeat[standards]`.

## External references

- [Praat manual: Pitch (autocorrelation)](https://praat.org/manual/Sound__To_Pitch__ac____.html)
- [Praat manual: Formant (Burg)](https://praat.org/manual/Sound__To_Formant__burg____.html)
- [Praat manual: Formant (robust)](https://praat.org/manual/Sound__To_Formant__robust____.html)
- [Parselmouth API docs](https://parselmouth.readthedocs.io/en/stable/api_reference.html)
- [librosa pyin](https://librosa.org/doc/main/generated/librosa.pyin.html)
- [openSMILE Python docs](https://audeering.github.io/opensmile-python/)
- [torchaudio transforms](https://pytorch.org/audio/stable/transforms.html)
