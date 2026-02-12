# audiofeat

[![PyPI version](https://img.shields.io/pypi/v/audiofeat.svg)](https://pypi.org/project/audiofeat/)
[![Python](https://img.shields.io/pypi/pyversions/audiofeat.svg)](https://pypi.org/project/audiofeat/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE.md)

A comprehensive PyTorch-based audio feature extraction library for speech research, music analysis, and audio ML pipelines. Extract 140+ features across temporal, spectral, cepstral, pitch, voice-quality, and rhythm domains — from a single `pip install`.

```python
import audiofeat

features = audiofeat.extract_features_from_file("recording.wav")
print(features["f0_mean_hz"], features["rms_mean"], features["mfcc_0_mean"])
```

## Why audiofeat?

- **One library, all features.** Temporal, spectral, cepstral, pitch, voice quality, rhythm, formants, and tonal features in a single package.
- **PyTorch-first.** Every feature returns a `torch.Tensor`. Plug directly into your training loop — no numpy-to-tensor conversion needed.
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
| Log Attack Time | `log_attack_time()` | MPEG-7 style attack time (10%–90% rise) |
| Decay Time | `decay_time()` | Time for envelope to decay from peak |
| Temporal Centroid | `temporal_centroid()` | Center of gravity of the amplitude envelope |
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
| HNR | `harmonic_to_noise_ratio()` | Harmonic-to-noise ratio |
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

| Feature | Function | Description |
|---------|----------|-------------|
| Jitter | `jitter()` | Cycle-to-cycle F0 variation |
| Jitter (local) | `jitter_local()` | Average absolute period difference (%) |
| Jitter (PPQ5) | `jitter_ppq5()` | Five-point Period Perturbation Quotient |
| Jitter (DDP) | `jitter_ddp()` | Difference of Differences of Periods |
| Shimmer | `shimmer()` | Cycle-to-cycle amplitude variation |
| Shimmer (local) | `shimmer_local()` | Local shimmer (%) |
| Shimmer (dB) | `shimmer_local_db()` | Shimmer in decibels |
| Shimmer (APQ3) | `shimmer_apq3()` | Three-point Amplitude Perturbation Quotient |
| Shimmer (DDA) | `shimmer_dda()` | Difference of Differences of Amplitudes |
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
| Tempo | `tempo()` | BPM estimation from onset autocorrelation |
| Beat Tracking | `beat_track()` | Beat positions in the audio signal |
| Onset Detection | `onset_detect()` | Transient event detection |

### Statistical Functionals

Apply to any time-series feature via `compute_functionals()`: mean, standard deviation, min, max, skewness, kurtosis.

## Architecture

```
audiofeat
├── temporal/      # RMS, ZCR, energy, attack, loudness, rhythm, ...
├── spectral/      # Centroid, rolloff, flux, MFCCs, chroma, formants, ...
├── cepstral/      # LPCC, GTCC, ERB cepstral, deltas
├── pitch/         # Autocorrelation, YIN, pYIN, Praat backends
├── voice/         # Jitter, shimmer, CPP, harmonic ratios, glottal flow
├── rhythm/        # Beat detection
├── stats/         # Statistical functionals
├── io/            # Audio loading, single-file & batch extraction, CSV export
├── validation/    # Praat comparison, gold-standard scorecard
├── standards/     # openSMILE eGeMAPS/ComParE wrappers
└── catalog/       # Auto-discovered feature catalog
```

**How it works:** Each feature function checks if librosa is available. If so, it delegates to librosa's implementation for bit-exact parity with the research standard. If librosa is not installed, a pure-PyTorch fallback computes the same feature. Either way, you always get a `torch.Tensor` back.

## Installation

Python `>=3.8` is required. We recommend creating a virtual environment first.

### pip (from PyPI)

```bash
pip install audiofeat
```

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
conda create -n audiofeat python=3.11 -y
conda activate audiofeat
pip install audiofeat

# Option C: uv
uv venv && source .venv/bin/activate
uv pip install audiofeat
```

### Optional extras

| Extra | What it adds | Install command |
|-------|-------------|-----------------|
| `dev` | pytest, black, mypy, flake8 | `pip install "audiofeat[dev]"` |
| `examples` | matplotlib, librosa, soundfile | `pip install "audiofeat[examples]"` |
| `validation` | Praat/parselmouth backend | `pip install "audiofeat[validation]"` |
| `standards` | openSMILE eGeMAPS/ComParE | `pip install "audiofeat[standards]"` |
| `models` | ASR, diarization, VAD, denoising | `pip install "audiofeat[models]"` |
| `full` | examples + validation + standards | `pip install "audiofeat[full]"` |

## Quick Start

### Extract features from a file

The simplest way to get started. This extracts all core features and returns a flat dictionary of summary statistics:

```python
from audiofeat.io.features import extract_features_from_file

features = extract_features_from_file("path/to/audio.wav")

# What you get back:
print(features["f0_mean_hz"])        # Mean fundamental frequency
print(features["rms_mean"])          # Mean RMS energy
print(features["spectral_centroid_mean"])  # Mean spectral centroid
print(features["mfcc_0_mean"])       # Mean of first MFCC coefficient
```

### Compute individual features

For fine-grained control, call feature functions directly. Every function accepts a 1D `torch.Tensor` waveform:

```python
import torch
import audiofeat

# Load your audio (or use a test signal)
sr = 22050
waveform = torch.randn(sr * 3)  # 3 seconds of noise

# Temporal features
rms = audiofeat.rms(waveform, frame_length=2048, hop_length=512)
zcr = audiofeat.zero_crossing_rate(waveform, frame_length=2048, hop_length=512)

# Spectral features
centroid = audiofeat.spectral_centroid(waveform, frame_length=2048, hop_length=512, sample_rate=sr)
rolloff = audiofeat.spectral_rolloff(waveform, frame_length=2048, hop_length=512, sample_rate=sr)
contrast = audiofeat.spectral_contrast(waveform, sample_rate=sr)

# Cepstral features
mfccs = audiofeat.mfcc(waveform, sr)
chroma = audiofeat.chroma(waveform, sr)

# Pitch
f0 = audiofeat.fundamental_frequency_yin(waveform, fs=sr, frame_length=2048, hop_length=512)

# Voice quality
jit = audiofeat.jitter(waveform, fs=sr)
shim = audiofeat.shimmer(waveform, fs=sr)

# Every result is a torch.Tensor
print(f"RMS shape: {rms.shape}")
print(f"MFCCs shape: {mfccs.shape}")
print(f"F0 shape: {f0.shape}")
```

### Load a real audio file

```python
from audiofeat.io import load_audio

waveform, sr = load_audio("path/to/audio.wav", target_sr=16000)
# waveform is a 1D torch.Tensor, sr is an int
```

### Aggregate over time with statistical functionals

```python
from audiofeat import compute_functionals

rms = audiofeat.rms(waveform, frame_length=2048, hop_length=512)
stats = compute_functionals(rms)
# {'mean': tensor(...), 'std': tensor(...), 'min': tensor(...),
#  'max': tensor(...), 'skewness': tensor(...), 'kurtosis': tensor(...)}
```

### Batch extraction to CSV

```python
from audiofeat.io.features import extract_features_for_directory

extract_features_for_directory("audio_folder/", "output.csv")
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
