# audiofeat

[![PyPI version](https://img.shields.io/pypi/v/audiofeat.svg)](https://pypi.org/project/audiofeat/)
[![Python](https://img.shields.io/pypi/pyversions/audiofeat.svg)](https://pypi.org/project/audiofeat/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE.md)

A comprehensive PyTorch-based audio feature extraction library for speech research, music analysis, and audio ML pipelines. Extract 130+ features across temporal, spectral, cepstral, pitch, voice-quality, and rhythm domains — from a single `pip install`.

```python
import audiofeat

features = audiofeat.extract_features_from_file("recording.wav")
print(features["f0_mean_hz"], features["rms_mean"], features["mfcc_mean_0"])
```

> Reading `.wav`/`.flac`/... files needs an audio backend. Install one with
> `pip install "audiofeat[io]"` (soundfile + torchcodec). Pure in-memory tensor
> workflows work with the base install alone.

## Why audiofeat?

- **One library, all features.** Temporal, spectral, cepstral, pitch, voice quality, rhythm, formants, and tonal features in a single package.
- **PyTorch-first.** Most feature functions return a `torch.Tensor` you can plug straight into a training loop. A few summary descriptors return plain Python scalars/strings where that is the natural type (`tempo()` → `float`, `key_detect()` → `str`, `log_attack_time()` → `float`, `voice_onset_time()` → `float`).
- **Librosa-grade accuracy.** Primary paths delegate to librosa when available for bit-exact parity; pure-PyTorch fallbacks when it's not installed.
- **Beginner to production.** Use individual functions for exploration, or the built-in CLI and batch extraction for production pipelines.
- **Validated.** Built-in Praat comparison tooling and a gold-standard scorecard for reproducible research.

## Features

### Temporal Features

| Feature | Function | Description |
|---------|----------|-------------|
| RMS | `rms()` | Root-mean-square amplitude per frame |
| Short-Time Energy | `short_time_energy()` | Sum of squared signal values in each frame |
| Zero-Crossing Rate | `zero_crossing_rate()` | Rate at which the signal changes sign |
| Zero-Crossing Count | `zero_crossing_count()` | Number of zero-crossings per frame |
| Loudness | `loudness()` | Perceptual loudness estimation |
| Log Attack Time | `log_attack_time()` | MPEG-7 style attack time (10%–90% rise); returns `float` |
| Decay Time | `decay_time()` | Time for envelope to decay from peak |
| Temporal Centroid | `temporal_centroid()` | MPEG-7 energy-weighted center of time, in **seconds** |
| Temporal Centroid (framewise) | `temporal_centroid_framewise()` | Per-frame temporal centroid contour |
| Amplitude Modulation | `amplitude_modulation_depth()` | Depth of amplitude modulation over a sliding window |
| Entropy of Energy | `entropy_of_energy()` | Abrupt changes in energy within a frame |
| Teager Energy | `teager_energy_operator()` | Teager-Kaiser energy for amplitude/frequency tracking |
| Breath Group Duration | `breath_group_duration()` | Estimated duration of breath groups |
| Speech Rate | `speech_rate()` | Syllables per second estimation |
| Tristimulus | `tristimulus()` | T1/T2/T3 timbre ratios from harmonic amplitudes |

### Spectral Features

| Feature | Function | Description |
|---------|----------|-------------|
| Spectral Centroid | `spectral_centroid()` | Center of mass of the spectrum |
| Spectral Rolloff | `spectral_rolloff()` | Frequency below which X% of energy is concentrated |
| Spectral Flux | `spectral_flux()` | Rate of change of the power spectrum |
| Spectral Flatness | `spectral_flatness()` | How noise-like a sound is (Wiener entropy) |
| Spectral Entropy | `spectral_entropy()` | Randomness of the spectral distribution |
| Spectral Bandwidth | `spectral_bandwidth()` | Spread of the spectrum around the centroid |
| Spectral Spread | `spectral_spread()` | Standard deviation of the spectral distribution |
| Spectral Slope | `spectral_slope()` | Linear regression slope fitted to the spectrum |
| Spectral Skewness | `spectral_skewness()` | Asymmetry of the spectral distribution |
| Spectral Crest Factor | `spectral_crest_factor()` | Peak-to-average ratio (peakiness) |
| Spectral Contrast | `spectral_contrast()` | Peak-valley amplitude difference across sub-bands |
| Spectral Deviation | `spectral_deviation()` | Jaggedness of the spectral envelope |
| Spectral Sharpness | `spectral_sharpness()` | Perceived sharpness (Zwicker model) |
| Spectral Roughness | `spectral_roughness()` | Sensory dissonance measure |
| Spectral Tonality | `spectral_tonality()` | Tonal vs. noise-like character |
| Spectral Irregularity | `spectral_irregularity()` | Irregularity of the spectral envelope |
| Low-High Energy Ratio | `low_high_energy_ratio()` | Energy below 1 kHz vs. above 3 kHz |
| HNR (energy ratio) | `harmonic_to_noise_ratio()` | dB ratio of two pre-computed energy scalars |
| HNR (from waveform) | `harmonic_to_noise_ratio_acf()` | Boersma autocorrelation HNR estimate (dB) from a signal frame |
| Harmonic Richness | `harmonic_richness_factor()` | Richness of harmonic content |
| Inharmonicity | `inharmonicity_index()` | Inharmonicity of the spectrum |
| Phase Coherence | `phase_coherence()` | Phase coherence across frequency bins |
| Sibilant Peak | `sibilant_spectral_peak_frequency()` | Peak frequency in the sibilant region |

### Spectrograms & Transforms

| Feature | Function | Description |
|---------|----------|-------------|
| Linear Spectrogram | `linear_spectrogram()` | STFT magnitude spectrogram |
| Mel Spectrogram | `mel_spectrogram()` | Mel-scaled frequency spectrogram |
| Log Mel Spectrogram | `log_mel_spectrogram()` | Log-scaled Mel spectrogram |
| CQT Spectrogram | `cqt_spectrogram()` | Constant-Q transform (log-frequency bins) |
| MFCCs | `mfcc()` | Mel-Frequency Cepstral Coefficients |
| Chroma | `chroma()` | 12-bin pitch class intensity (chromagram) |
| Tonnetz | `tonnetz()` | 6D tonal centroid features |

### Formant Analysis

| Feature | Function | Description |
|---------|----------|-------------|
| Formant Frequencies | `formant_frequencies()` | Extract F1, F2, F3, ... via Burg LPC |
| Formant Contours | `formant_contours()` | Time-varying formant trajectories |
| Formant Bandwidths | `formant_bandwidths()` | Bandwidth of each formant |
| Formant Dispersion | `formant_dispersion()` | Average spacing between formants |

### Linear Prediction

| Feature | Function | Description |
|---------|----------|-------------|
| LPC | `lpc_coefficients()` | Linear Prediction Coefficients (Burg method) |
| LSP | `lsp_coefficients()` | Line Spectral Pairs from LPC |

### Cepstral Features

| Feature | Function | Description |
|---------|----------|-------------|
| LPCC | `lpcc()` | Linear Predictive Cepstral Coefficients |
| GTCC | `gtcc()` | Gammatone Cepstral Coefficients |
| GFCC | `gfcc()` | Gammatone Frequency Cepstral Coefficients |
| ERB Cepstral | `erb_cepstral_coefficients()` | ERB-scale cepstral coefficients |
| Delta | `delta()` | First-order derivative of a feature contour |
| Delta-Delta | `delta_delta()` | Second-order derivative (acceleration) |

### Pitch Features

| Feature | Function | Description |
|---------|----------|-------------|
| F0 (Autocorrelation) | `fundamental_frequency_autocorr()` | F0 via autocorrelation |
| F0 (YIN) | `fundamental_frequency_yin()` | F0 via YIN algorithm |
| F0 (pYIN) | `fundamental_frequency_pyin()` | Probabilistic YIN (requires librosa) |
| F0 (Praat) | `fundamental_frequency_praat()` | Exact Praat parity (requires parselmouth) |
| Pitch Strength | `pitch_strength()` | Strength of periodicity |
| Semitone Std Dev | `semitone_sd()` | F0 variation in semitones |

### Voice Quality Features

The jitter/shimmer family operates on **sequences**, not raw audio: jitter functions
take a tensor of glottal **periods** (seconds) and shimmer functions take a tensor of
per-cycle **amplitudes**. Derive periods from an F0 contour
(`periods = 1.0 / f0[f0 > 0]`). For end-to-end Praat parity from a waveform, use
`audiofeat.voice.praat_voice.jitter_shimmer_praat` (requires the `validation` extra).

| Feature | Function | Description |
|---------|----------|-------------|
| Jitter | `jitter(periods)` | Cycle-to-cycle F0 variation (alias of `jitter_local`, %) |
| Jitter (local) | `jitter_local(periods)` | Average absolute period difference (%) |
| Jitter (RAP) | `jitter_rap(periods)` | Three-point Relative Average Perturbation (eGeMAPS) |
| Jitter (PPQ5) | `jitter_ppq5(periods)` | Five-point Period Perturbation Quotient |
| Jitter (DDP) | `jitter_ddp(periods)` | Difference of Differences of Periods |
| Shimmer | `shimmer(amplitudes)` | Cycle-to-cycle amplitude variation (alias of `shimmer_local`, %) |
| Shimmer (local) | `shimmer_local(amplitudes)` | Local shimmer (%) |
| Shimmer (dB) | `shimmer_local_db(amplitudes)` | Shimmer in decibels |
| Shimmer (APQ3) | `shimmer_apq3(amplitudes)` | Three-point Amplitude Perturbation Quotient |
| Shimmer (APQ5) | `shimmer_apq5(amplitudes)` | Five-point Amplitude Perturbation Quotient |
| Shimmer (APQ11) | `shimmer_apq11(amplitudes)` | Eleven-point Amplitude Perturbation Quotient |
| Shimmer (DDA) | `shimmer_dda(amplitudes)` | Difference of Differences of Amplitudes |
| CPP | `cepstral_peak_prominence()` | Cepstral Peak Prominence for dysphonia detection |
| Alpha Ratio | `alpha_ratio()` | Energy ratio: 50–1000 Hz vs 1–5 kHz |
| Hammarberg Index | `hammarberg_index()` | Max energy ratio: 0–2 kHz vs 2–5 kHz |
| Harmonic Differences | `harmonic_differences()` | H1-H2, H1-A3, and other harmonic ratios |
| SHR | `subharmonic_to_harmonic_ratio()` | Subharmonic-to-harmonic power ratio |
| NAQ | `normalized_amplitude_quotient()` | Normalized Amplitude Quotient |
| Closed Quotient | `closed_quotient()` | Closed phase ratio from EGG |
| Soft Phonation Index | `soft_phonation_index()` | Low/high band energy ratio |
| GNE | `glottal_to_noise_excitation()` | Glottal-to-Noise Excitation ratio |
| MFDR | `maximum_flow_declination_rate()` | Maximum Flow Declination Rate |
| Vocal Fry Index | `vocal_fry_index()` | Ratio of fry frames to voiced frames |
| VOT | `voice_onset_time()` | Voice Onset Time estimation |
| Vocal Tract Length | `vocal_tract_length()` | Estimated from F1 and F2 |
| Nasality Index | `nasality_index()` | Nasal vs. oral microphone energy |

### Rhythm Features

| Feature | Function | Description |
|---------|----------|-------------|
| Tempo | `tempo()` | BPM estimation from onset autocorrelation; returns `float` |
| Beat Tracking | `beat_track()` | Beat **times** (seconds) as a 1-D `torch.Tensor` |
| Beat Tracking (+tempo) | `beat_track_with_tempo()` | Returns `(tempo_tensor, beat_frames_tensor)` |
| Onset Detection | `onset_detect()` | Transient event detection (frame indices) |

### Statistical Functionals

Aggregate any time-series feature via `compute_functionals()`. It takes a **2-D**
tensor and returns a **flat 1-D `torch.Tensor`** of length `6 * num_features`,
concatenating `[mean, std, min, max, skewness, kurtosis]` (excess/Fisher kurtosis).
The default `time_axis=0` treats the input as `(time, features)`; for this library's
`(features, time)` matrices (e.g. `mfcc()`), pass `time_axis=1`.

## Architecture

Importing `audiofeat` eagerly loads only the lightweight, dependency-free
feature subpackages below — these form the flat public API
(`from audiofeat import *`, 131 names):

```
audiofeat/
├── temporal/         # (pkg) RMS, ZCR, energy, attack, loudness, beat, tristimulus, ...
├── spectral/         # (pkg) Centroid, rolloff, flux, MFCCs, chroma, formants, key, ...
├── cepstral/         # (pkg) LPCC, GTCC, GFCC, ERB cepstral, deltas
├── pitch/            # (pkg) Autocorrelation, YIN, pYIN, Praat backends
├── voice/            # (pkg) Jitter, shimmer, CPP, harmonic ratios, glottal flow
├── rhythm/           # (pkg) Beat detection (beat_detection)
├── segmentation/     # (pkg) Silence detection, thumbnailing, diarization helpers
├── stats/            # (pkg) Statistical functionals
├── io/               # (pkg) Audio loading, single-file & batch extraction, CSV export
├── validation/       # (pkg) Praat comparison, gold-standard scorecard
├── preprocessing/    # (pkg) reserved namespace (currently empty)
├── standards.py      # (module) openSMILE eGeMAPS/ComParE wrappers
├── catalog.py        # (module) auto-discovered feature catalog
└── cli.py            # (module) `audiofeat` command-line interface
```

Heavy, optional ML-model wrappers are shipped as **separately importable modules**
and are intentionally **not** imported by `import audiofeat`, so the base import
never pulls in large dependencies. Import them explicitly and install the matching
extra, e.g. `from audiofeat.asr import transcribe` (needs `audiofeat[asr]`). These
include `asr`, `vad`, `diarization`, `embeddings`, `ssl`, `emotion`, `scene`,
`spatial`/`separation`, `denoise`, and `beat_madmom`.

**How it works:** Each feature function checks if librosa is available. If so, it delegates to librosa's implementation for bit-exact parity with the research standard. If librosa is not installed, a pure-PyTorch fallback computes the same feature. Most feature functions return a `torch.Tensor`; a few summary descriptors return plain Python scalars/strings (`tempo`, `log_attack_time`, `voice_onset_time` → `float`; `key_detect` → `str`).

## Installation

Python `>=3.14` is required. We recommend creating a virtual environment first.

### pip (from PyPI)

```bash
pip install audiofeat
```

The base install pulls in only `numpy`, `scipy`, `torch`, and `torchaudio`. To
**load audio files** (`.wav`, `.flac`, ...) you need a decode backend — install
the `io` extra, which adds `soundfile` and `torchcodec`:

```bash
pip install "audiofeat[io]"
```

This is required because `torchaudio >= 2.1` no longer bundles a default audio
decoder. If a file fails to decode, `load_audio` raises a `RuntimeError` telling
you to install `audiofeat[io]`. The library ships a `py.typed` marker, so type
checkers (mypy/pyright) pick up its annotations out of the box.

### From source

```bash
git clone https://github.com/ankitshah009/audiofeat.git
cd audiofeat
pip install -e .
```

### With a virtual environment

```bash
# Option A: venv
python -m venv .venv
source .venv/bin/activate
pip install audiofeat

# Option B: conda
conda create -n audiofeat python=3.14 -y
conda activate audiofeat
pip install audiofeat

# Option C: uv
uv venv && source .venv/bin/activate
uv pip install audiofeat
```

### Optional extras

Extras are **feature-scoped**: install only what a given capability needs. Heavy
pretrained-model backends each get their own extra and are imported lazily, so
they never load unless you explicitly import the matching module.

| Extra | What it adds | Enables |
|-------|--------------|---------|
| `io` | soundfile, torchcodec | Loading/saving audio files (`load_audio`, file extraction) |
| `examples` | matplotlib, librosa, soundfile | Running the scripts in `examples/` |
| `validation` | praat-parselmouth | Praat parity (`*_praat`), gold-standard scorecard |
| `standards` | opensmile | openSMILE eGeMAPS/ComParE (`audiofeat.standards`) |
| `dev` | pytest, pytest-cov, black, isort, flake8, mypy | Development & tests |
| `asr` | openai-whisper | `audiofeat.asr` (transcription) |
| `vad` | silero-vad | `audiofeat.vad` (voice activity detection) |
| `diarization` | pyannote.audio, huggingface-hub | `audiofeat.diarization` |
| `embeddings` | speechbrain | `audiofeat.embeddings` (ECAPA speaker embeddings) |
| `ssl` | transformers | `audiofeat.ssl` (wav2vec2 / HuBERT) |
| `emotion` | transformers | `audiofeat.emotion_ssl` |
| `scene` | panns_inference | `audiofeat.scene` (PANNs CNN14) |
| `separation` | asteroid | `audiofeat.spatial` (ConvTasNet) |
| `denoise` | noisereduce, rnnoise-torch | `audiofeat.denoise` |
| `beat` | madmom | `audiofeat.beat_madmom` |
| `models` | union of all model backends above | Every ML wrapper |
| `full` | examples + validation + standards + io + models | Everything |

```bash
# A couple of capabilities at once
pip install "audiofeat[io,validation]"

# Everything
pip install "audiofeat[full]"
```

> **Note on heavy ML models.** Modules like `asr`, `vad`, `diarization`,
> `embeddings`, `ssl`, `scene`, `separation`, and `denoise` wrap large pretrained
> models. They are **not** imported by `import audiofeat` and load their
> dependencies only when you import the specific submodule. Core DSP features need
> none of them.

## Quick Start

> Every snippet below is executed in CI against the sample files in `examples/`.

### Extract features from a file

The simplest way to get started. This extracts a compact set of core features and
returns a **flat dictionary of summary statistics** (requires the `io` extra for
file decoding):

```python
import audiofeat

features = audiofeat.extract_features_from_file("path/to/audio.wav")

# What you get back (keys are "<feature>_<stat>" and "<feature>_<stat>_<index>"):
print(features["f0_mean_hz"])              # Mean voiced fundamental frequency
print(features["rms_mean"])                # Mean RMS energy
print(features["spectral_centroid_mean"])  # Mean spectral centroid
print(features["mfcc_mean_0"])             # Mean of the first MFCC coefficient
```

### Compute individual features

For fine-grained control, call feature functions directly. Most accept a 1-D
`torch.Tensor` waveform; check each signature for the right keyword (some use
`sample_rate=`, some use `fs=`):

```python
import torch
import audiofeat

# Use a test signal (or load a real file with audiofeat.load_audio)
sr = 22050
waveform = torch.randn(sr * 3)  # 3 seconds of noise

# Temporal features
rms = audiofeat.rms(waveform, frame_length=2048, hop_length=512)
zcr = audiofeat.zero_crossing_rate(waveform, frame_length=2048, hop_length=512)

# Spectral features  (note: spectral_contrast takes `fs`, the others `sample_rate`)
centroid = audiofeat.spectral_centroid(waveform, frame_length=2048, hop_length=512, sample_rate=sr)
rolloff = audiofeat.spectral_rolloff(waveform, frame_length=2048, hop_length=512, sample_rate=sr)
contrast = audiofeat.spectral_contrast(waveform, fs=sr)

# Cepstral features
mfccs = audiofeat.mfcc(waveform, sr)
chroma = audiofeat.chroma(waveform, sr)

# Pitch (YIN needs frame_length and hop_length)
f0 = audiofeat.fundamental_frequency_yin(waveform, fs=sr, frame_length=2048, hop_length=512)

print(f"RMS shape: {rms.shape}")
print(f"Contrast shape: {contrast.shape}")  # (n_bands + 1, frames)
print(f"MFCCs shape: {mfccs.shape}")         # (n_mfcc, frames)
print(f"F0 shape: {f0.shape}")
```

### Voice quality: jitter and shimmer

Jitter functions take a tensor of glottal **periods** (seconds) and shimmer
functions take a tensor of per-cycle **amplitudes** — not a raw waveform. Derive
periods from an F0 contour and use a frame-amplitude series for shimmer:

```python
import audiofeat

waveform, sr = audiofeat.load_audio("path/to/voiced.wav", target_sample_rate=22050)

# Periods (seconds) from voiced F0 frames
f0 = audiofeat.fundamental_frequency_yin(waveform, fs=sr, frame_length=2048, hop_length=512)
periods = 1.0 / f0[f0 > 0]

# Per-cycle amplitude proxy: frame RMS
amplitudes = audiofeat.rms(waveform, frame_length=2048, hop_length=512)

jit = audiofeat.jitter(periods)        # local jitter, percent  (== jitter_local)
rap = audiofeat.jitter_rap(periods)    # 3-point RAP (eGeMAPS)
shim = audiofeat.shimmer(amplitudes)   # local shimmer, percent (== shimmer_local)
print(f"jitter={float(jit):.4f}%  rap={float(rap):.4f}%  shimmer={float(shim):.4f}%")
```

For exact Praat parity straight from a waveform (handles its own period
detection), use the parselmouth-backed helper (needs `pip install "audiofeat[validation]"`):

```python
from audiofeat.voice.praat_voice import jitter_shimmer_praat

metrics = jitter_shimmer_praat("path/to/voiced.wav")  # or (tensor, fs=sr)
print(metrics["jitter_local_percent"], metrics["shimmer_local_percent"])
```

### Load a real audio file

```python
import audiofeat

# target_sample_rate is keyword-only; pass None to keep the native rate
waveform, sr = audiofeat.load_audio("path/to/audio.wav", target_sample_rate=16000)
# waveform is a 1D torch.Tensor (mono), sr is an int
```

### Beat tracking

```python
import audiofeat

waveform, sr = audiofeat.load_audio("path/to/music.wav", target_sample_rate=22050)

beat_times = audiofeat.beat_track(waveform, sr)             # 1-D tensor of beat times (s)
tempo, beat_frames = audiofeat.beat_track_with_tempo(waveform, sr)  # (tensor, tensor)
print(f"tempo={float(tempo):.1f} BPM, {beat_times.numel()} beats")
```

### Aggregate over time with statistical functionals

`compute_functionals` takes a **2-D** tensor and returns a **flat 1-D tensor** of
length `6 * num_features`: `[mean, std, min, max, skewness, kurtosis]` per feature.
Use `time_axis=1` for this library's `(features, time)` matrices:

```python
import audiofeat
from audiofeat import compute_functionals

mfccs = audiofeat.mfcc(waveform, sr)          # (n_mfcc, frames)
stats = compute_functionals(mfccs, time_axis=1)
# flat tensor of length 6 * n_mfcc, ordered [means..., stds..., mins..., ...]
n = mfccs.shape[0]
means, stds = stats[:n], stats[n:2 * n]
print(stats.shape, means.shape)

# For a single 1-D contour, add a feature axis first:
rms = audiofeat.rms(waveform, frame_length=2048, hop_length=512)  # (frames,)
rms_stats = compute_functionals(rms.unsqueeze(0), time_axis=1)    # length 6
mean, std, mn, mx, skew, kurt = rms_stats.tolist()
```

### Batch extraction to CSV

`extract_features_for_directory` returns a list of per-file dicts; write them with
`write_feature_rows_to_csv` (or use the CLI `batch-extract` command below):

```python
from audiofeat.io.features import (
    extract_features_for_directory,
    write_feature_rows_to_csv,
)

rows = extract_features_for_directory("audio_folder/")
write_feature_rows_to_csv(rows, "output.csv")
```

## CLI

audiofeat includes a command-line interface for common workflows.

```bash
audiofeat --help
```

### Extract features from a single file

```bash
audiofeat extract recording.wav --output features.json
```

### Batch extract an entire directory

```bash
audiofeat batch-extract audio_folder/ output.csv
```

### Diagnose your environment

```bash
audiofeat doctor --audio-dir examples
```

Checks installed dependencies, verifies audio files are valid, and reports any issues.

### Browse available features

```bash
audiofeat list-features
audiofeat list-features --format markdown --output FEATURES.md
```

## Advanced Topics

For Praat validation, openSMILE integration, the gold-standard scorecard, and troubleshooting, see [docs/VALIDATION.md](docs/VALIDATION.md).

For the auto-generated feature catalog, see [docs/FEATURE_CATALOG.md](docs/FEATURE_CATALOG.md).

## Testing

```bash
pytest -q
```

With coverage:

```bash
pytest --cov=audiofeat --cov-report=term-missing -q
```

## Contributing

We welcome contributions! If you have new features, bug fixes, or improvements, please open a pull request on [GitHub](https://github.com/ankitshah009/audiofeat).

## Citation

If you use `audiofeat` in your research, please cite:

```bibtex
@phdthesis{shah2024computational,
  title={Computational Audition with Imprecise Labels},
  author={Shah, Ankit Parag},
  year={2024},
  school={Carnegie Mellon University Pittsburgh, PA}
}
```
