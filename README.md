# audiofeat: A Comprehensive Audio Feature Extraction Library

`audiofeat` is designed to be the most comprehensive publicly available Python library for audio feature extraction. It provides a wide range of temporal, spectral, pitch, and voice-related features, along with various spectrogram representations, all implemented using `torch` for efficient computation.

## Features

### Temporal Features
- **RMS (Root Mean Square):** Measures the loudness or power of an audio signal.
- **Zero-Crossing Rate (ZCR):** Indicates the rate at which the signal changes its sign.
- **Amplitude Modulation Depth:** Measures the depth of amplitude modulation over a sliding window.
- **Breath Group Duration:** Estimates the duration of breath groups from the audio envelope.
- **Speech Rate:** Estimates speech rate in syllables per second.

### Spectral Features
- **Spectral Centroid:** Represents the "center of mass" of the spectrum, indicating dominant frequencies.
- **Spectral Rolloff:** The frequency below which a certain percentage of the total spectral energy is concentrated.
- **Spectral Flux:** Measures the rate of change of the power spectrum.
- **Spectral Flatness:** Quantifies how noise-like a sound is.
- **Spectral Entropy:** Measures the randomness or unpredictability of the spectrum.
- **Spectral Skewness:** Describes the asymmetry of the spectral distribution.
- **Low-High Energy Ratio:** Ratio of energy below 1 kHz to that above 3 kHz.
- **Harmonic Richness Factor:** Measures the richness of harmonics in a sound.
- **Inharmonicity Index:** Quantifies the deviation of a sound's partials from a perfect harmonic series.
- **Phase Coherence:** Measures the consistency of phase relationships across different frequencies.
- **Formant Frequencies:** Estimates the resonant frequencies of the vocal tract.
- **Formant Bandwidths:** Estimates the bandwidths of the formants.
- **Formant Dispersion:** Average spacing between the first five formants.
- **Sibilant Spectral Peak Frequency:** Peak frequency of sibilant energy between 3 and 12 kHz.
- **MFCCs (Mel-Frequency Cepstral Coefficients):** Compact representation of the spectral envelope, based on the Mel scale.
- **Linear Spectrogram (STFT):** Visual representation of the spectrum of frequencies over time.
- **Mel Spectrogram:** Spectrogram with a Mel-scaled frequency axis, mimicking human auditory perception.

### Pitch Features
- **Fundamental Frequency (F0) Autocorrelation:** Estimates F0 via autocorrelation.
- **Fundamental Frequency (F0) YIN:** Estimates F0 using the YIN algorithm.
- **Semitone Standard Deviation:** Standard deviation of F0 in semitones.

### Voice Features
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

## Installation

To install `audiofeat`, clone the repository and install it in editable mode:

```bash
git clone https://github.com/ankitshah009/audiofeat.git
cd audiofeat
pip install -e .
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