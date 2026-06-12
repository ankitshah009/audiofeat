# audiofeat — Release Notes

A comprehensive PyTorch-based audio feature extraction library for speech
research, music analysis, and audio ML pipelines.

- **PyPI:** https://pypi.org/project/audiofeat/
- **Full changelog:** see [CHANGELOG.md](CHANGELOG.md)

## Installation

```bash
pip install audiofeat            # core DSP features (numpy, scipy, torch, torchaudio)
pip install "audiofeat[io]"      # + audio file loading (soundfile, torchcodec)
pip install "audiofeat[full]"    # + examples, validation, standards, all ML backends
```

`torchaudio >= 2.1` ships without a default audio decoder, so loading `.wav`/etc.
files requires the `io` extra. Heavy pretrained-model backends (`asr`, `vad`,
`diarization`, `embeddings`, `ssl`, `scene`, `separation`, `denoise`, ...) are
feature-scoped extras and are imported lazily — `import audiofeat` never pulls
them in.

## 1.2.0 (current)

API-consistency and accuracy release. Highlights:

- `from audiofeat import *` works; 131 public names are exported.
- New functions: `jitter_rap`, `shimmer_apq5`, `shimmer_apq11`,
  `harmonic_to_noise_ratio_acf`, `temporal_centroid_framewise`,
  `beat_track_with_tempo`.
- Energy-fallback VAD, a soundfile-based audio I/O fallback, a `py.typed` marker,
  and feature-scoped optional dependencies.
- DSP corrections for librosa/Praat-grade accuracy: `spectral_rolloff`
  (magnitude-based), `spectral_flatness` (power-based), `spectral_crest_factor`
  (`max/mean`), YIN parabolic interpolation, `voice_onset_time`, CPP power
  cepstrum, `vocal_tract_length` quarter-wave, LSP extraction, LPCC sign, and
  full-frame `alpha_ratio` / `hammarberg_index` / `nasality_index`.
- Every README and `examples/` snippet now runs as written.

> **Breaking (numeric):** several features now return different, *correct* values
> (notably `temporal_centroid`, which is now MPEG-7 seconds, and `beat_track`,
> which now returns beat times). Re-baseline downstream pipelines. The complete
> list is in [CHANGELOG.md](CHANGELOG.md).

## Feature categories

Temporal, spectral, spectrograms/transforms, formants, linear prediction,
cepstral, pitch, voice quality, rhythm, and statistical functionals. Browse the
full, code-aligned catalog with `audiofeat list-features` or in
[docs/FEATURE_CATALOG.md](docs/FEATURE_CATALOG.md).

## Citation

```bibtex
@phdthesis{shah2024computational,
  title={Computational Audition with Imprecise Labels},
  author={Shah, Ankit Parag},
  year={2024},
  school={Carnegie Mellon University Pittsburgh, PA}
}
```
