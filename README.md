# audiofeat: A Comprehensive Audio Feature Extraction Library

`audiofeat` is designed to be the most comprehensive publicly available Python library for audio feature extraction. It provides a wide range of temporal, spectral, pitch, and voice-related features, along with various spectrogram representations, all implemented using `torch` for efficient computation.

## Features

### Temporal Features
- **RMS (Root Mean Square):** Measures the loudness or power of an audio signal.
- **Short-Time Energy (STE):** The sum of squared signal values in a frame.
- **Zero-Crossing Rate (ZCR):** Indicates the rate at which the signal changes its sign.
- **Amplitude Modulation Depth:** Measures the depth of amplitude modulation over a sliding window.
- **Breath Group Duration:** Estimates the duration of breath groups from the audio envelope.
- **Speech Rate:** Estimates speech rate in syllables per second.
- **Log Attack Time:** Measures the time for a signal's envelope to rise to its peak.
- **Temporal Centroid:** The "center of gravity" of the signal's amplitude envelope.
- **Entropy of Energy:** Measures abrupt changes in energy within a frame.
- **Decay Time:** Measures the time for a signal's envelope to decay from its peak.

### Spectral Features
- **Spectral Centroid:** Represents the "center of mass" of the spectrum, indicating dominant frequencies.
- **Spectral Rolloff:** The frequency below which a certain percentage of the total spectral energy is concentrated. Configurable `rolloff_percent` (e.g., 0.85, 0.90, 0.95).
- **Spectral Flux:** Measures the rate of change of the power spectrum.
- **Spectral Flatness:** Quantifies how noise-like a sound is, using a `torch`-native geometric mean.
- **Spectral Entropy:** Measures the randomness or unpredictability of the spectrum.
- **Spectral Skewness:** Describes the asymmetry of the spectral distribution.
- **Spectral Spread (Bandwidth):** Measures the bandwidth of the spectrum, or how "spread out" it is around the centroid.
- **Spectral Slope:** The slope of a linear regression fitted to the spectrum.
- **Spectral Crest Factor:** Ratio of the max spectral magnitude to the sum of magnitudes; measures "peakiness".
- **Spectral Contrast:** Measures the amplitude difference between spectral peaks and valleys across several frequency sub-bands, calculated as `(peak - valley) / (peak + valley)`.
- **Harmonic-to-Noise Ratio (HNR):** Ratio of energy in harmonic components to noise components.
- **Spectral Deviation:** Quantifies the "jaggedness" of the local spectrum.
- **Low-High Energy Ratio:** Ratio of energy below 1 kHz to that above 3 kHz.
- **LPC (Linear Prediction Coefficients):** Coefficients representing the spectral envelope of a signal.
- **LSP (Line Spectral Pairs):** Robust and compact representation of the LPC filter.
- **MFCCs (Mel-Frequency Cepstral Coefficients):** Compact representation of the spectral envelope, based on the Mel scale.
- **Linear Spectrogram (STFT):** Visual representation of the spectrum of frequencies over time.
- **Mel Spectrogram:** Spectrogram with a Mel-scaled frequency axis, mimicking human auditory perception.
- **CQT Spectrogram (Constant-Q Transform):** Spectrogram with logarithmically spaced frequency bins. (Note: This is a simplified `torch`-native implementation and not a full, optimized CQT).
- **Chroma Features:** Represents the intensity of the 12 different pitch classes of the Western musical scale.
- **Spectral Sharpness (Zwicker Model):** Measures the perceived sharpness of a sound based on the Zwicker model.
- **Spectral Tonality:** Quantifies the tonal characteristics of a sound using the spectral crest factor.

### Cepstral Features
- **LPCC (Linear Predictive Cepstral Coefficients):** Cepstral coefficients derived from Linear Predictive Coding (LPC) analysis.
- **GTCC (Gammatone Cepstral Coefficients):** Cepstral coefficients derived from a Gammatone filterbank.
- **Delta Coefficients:** First-order derivative of a feature contour over time.
- **Delta-Delta Coefficients:** Second-order derivative of a feature contour over time.

### Pitch Features
- **Fundamental Frequency (F0) Autocorrelation:** Estimates F0 via autocorrelation.
- **Fundamental Frequency (F0) YIN:** Estimates F0 using the YIN algorithm.
- **Semitone Standard Deviation:** Standard deviation of F0 in semitones.
- **Pitch Strength:** Measures the strength of periodicity in a signal.

### Voice Features

### Advanced ML Wrappers (install with `pip install audiofeat[models]`)
- **Speaker Diarization** — `audiofeat.diarization.diarize()` for *who-spoke-when* using pyannote-audio.
- **Speaker Embeddings (ECAPA)** — `audiofeat.embeddings.extract_speaker_embedding()` for voice similarity.
- **ASR (Whisper)** — `audiofeat.asr.transcribe()` to convert speech to text.
- **Environmental Sound Classification** — `audiofeat.scene.classify_scene()` returns AudioSet tags.
- **VAD (Silero)** — `audiofeat.vad.is_speech()` for robust voice activity detection.
- **Noise Suppression (RNNoise)** — `audiofeat.denoise.denoise_rnn()` for real-time cleanup.
- **SSL Embeddings (Wav2Vec2 / HuBERT / AST)** — `audiofeat.ssl.embed()` to get 768-D contextual features.
- **Jitter:** Cycle-to-cycle F0 variation.
- **Shimmer:** Cycle-to-cycle amplitude variation.
- **Subharmonic to Harmonic Ratio:** Ratio of subharmonic power to harmonic power.
- **Normalized Amplitude Quotient (NAQ):** Computed from peak glottal flow, MFDR, and period.
- **Closed Quotient:** Derived from EGG timings per cycle.
- **Glottal Closure Time:** Average relative glottal closure time.
- **Soft Phonation Index:** Derived from low/high band energies.
- **Speed Quotient:** From glottal flow opening and closing times.
- **Vocal Fry Index:** Ratio of fry frames to voiced frames.
- **Voice Onset Time (VOT):** Simplified estimation of voice onset time.
- **Glottal to Noise Excitation (GNE):** Approximate GNE using band cross-correlations.
- **Maximum Flow Declination Rate (MFDR):** Approximate MFDR from differentiated glottal flow.
- **Nasality Index:** Computed from nasal and oral microphone signals.
- **Vocal Tract Length:** Estimated from the first two formants.
- **Alpha Ratio:** Ratio of low-frequency energy (50-1k Hz) to high-frequency energy (1-5k Hz).
- **Hammarberg Index:** Ratio of max energy in 0-2k Hz band to max energy in 2-5k Hz band.
- **Harmonic Differences (e.g., H1-H2, H1-A3):** Ratios between the amplitudes of specific harmonics.

### Tonal and Musical Features
- **Tonnetz (Tonal Centroid Features):** A 6-dimensional representation of tonal space based on music theory.

### Rhythm Features
- **Tempo:** Estimates the tempo (BPM) of an audio signal.
- **Beat Tracking:** Performs simple beat tracking on an audio signal.

### Statistical Functionals
- **Mean:** Average value of a feature over time.
- **Standard Deviation:** Variability of a feature over time.
- **Min:** Minimum value of a feature over time.
- **Max:** Maximum value of a feature over time.
- **Skewness:** Asymmetry of the feature distribution over time.
- **Kurtosis:** Peakiness of the feature distribution over time.

## Installation

### Install from PyPI (Recommended)

```bash
pip install audiofeat
```

### Install from Source

To install `audiofeat` from source, clone the repository and install it in editable mode:

```bash
git clone https://github.com/ankitshah009/audiofeat.git
cd audiofeat
pip install -e .
```

### Optional Dependencies

For development and examples:

```bash
# For development
pip install audiofeat[dev]

# For running examples
pip install audiofeat[examples]
```

## Usage

Here's a basic example of how to use `audiofeat` to extract various features:

```python
import torch
import audiofeat

# Create a dummy audio signal
sample_rate = 22050
duration = 5
audio_data = torch.randn(sample_rate * duration)

# Compute features
rms = audiofeat.rms(audio_data, frame_length=2048, hop_length=512)
zcr = audiofeat.zero_crossing_rate(audio_data)
spectral_centroid = audiofeat.spectral_centroid(audio_data)
mel_spec = audiofeat.mel_spectrogram(audio_data, sample_rate)
mfccs = audiofeat.mfcc(audio_data, sample_rate)

print(f"RMS: {rms.shape}")
print(f"ZCR: {zcr.shape}")
print(f"Spectral Centroid: {spectral_centroid.shape}")
print(f"Mel Spectrogram: {mel_spec.shape}")
print(f"MFCCs: {mfccs.shape}")
```

For more detailed examples, refer to the `examples/compute_features.py` file.

## Contributing

We welcome contributions to `audiofeat`! If you have new features to add, bug fixes, or improvements, please feel free to open a pull request.

## Citation

If you use `audiofeat` in your research, please cite the following Ph.D. thesis:

```bibtex
@phdthesis{shah2024computational,
  title={Computational Audition with Imprecise Labels},
  author={Shah, Ankit Parag},
  year={2024},
  school={Carnegie Mellon University Pittsburgh, PA}
}
```