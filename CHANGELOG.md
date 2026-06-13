# Changelog

All notable changes to **audiofeat** are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.3.0] - 2026-06-12

Raises the supported Python floor to 3.14.

### Changed

- **Python 3.14 is now required** (`requires-python = ">=3.14"`). Earlier
  versions (3.9–3.13) are no longer supported; `pip` will keep serving 1.2.0 to
  those interpreters, so existing installs are unaffected. The full test suite
  (352 passed, 1 skipped) and the gold-standard score gate pass on 3.14, and the
  CI test/lint matrices and the `black` formatting target now pin 3.14.

### Internal

- Added `from __future__ import annotations` to the few modules that use PEP 604
  (`X | None`) and builtin-generic annotations, so annotation evaluation is
  deferred and import time is marginally lower.

## [1.2.0] - 2026-06-12

This release makes the public API consistent and importable, corrects several
DSP implementations for research-grade accuracy, and fixes every documented
example so it runs as written. **Some features now return different (correct)
numeric values** — see *Changed* below and re-baseline any downstream pipelines.

### Added

- **Public star import.** `from audiofeat import *` now works and exports a flat,
  auto-assembled `__all__` of 131 public names (plus `__version__`). Names
  reachable only from submodules (e.g. `key_detect`) are bound at package level so
  both `audiofeat.<name>` and the star import resolve them.
- **New voice-quality functions:**
  - `jitter_rap` — 3-point Relative Average Perturbation (required by eGeMAPS).
  - `shimmer_apq5` — 5-point Amplitude Perturbation Quotient.
  - `shimmer_apq11` — 11-point Amplitude Perturbation Quotient.
- **`harmonic_to_noise_ratio_acf`** — HNR estimated directly from a waveform frame
  via Boersma's autocorrelation method (returns dB). Complements the existing
  `harmonic_to_noise_ratio`, which only converts two pre-computed energy scalars.
- **`temporal_centroid_framewise`** — per-frame temporal-centroid contour
  alongside the global MPEG-7 `temporal_centroid`.
- **`beat_track_with_tempo`** — returns `(tempo, beat_frames)`; the rich beat
  tracker that complements `beat_track` (which returns beat *times*).
- **Energy-fallback VAD** — voice-activity detection degrades to an energy-based
  estimator when the optional Silero model is not installed.
- **Audio I/O backend fallback** — `load_audio` / file extraction try
  `torchaudio.load` first and fall back to `soundfile`, with a clear `RuntimeError`
  (pointing to `pip install audiofeat[io]`) when no decoder is available. Needed
  because `torchaudio >= 2.1` no longer bundles a default decoder.
- **`py.typed` marker** — the package now ships typing information for
  mypy/pyright consumers.
- **Feature-scoped extras** — per-backend optional dependencies: `io`, `asr`,
  `vad`, `diarization`, `embeddings`, `ssl`, `emotion`, `scene`, `separation`,
  `denoise`, `beat`, plus the convenience unions `models` and `full`.
- **Reachable rhythm/segmentation/preprocessing** — these subpackages are imported
  into the public API; rhythm (`beat_detection`) and segmentation helpers are now
  exposed.

### Fixed

- **`spectral_rolloff`** — now accumulates the **magnitude** spectrum by default
  (`power=1.0`) for librosa parity, instead of the power spectrum.
- **`spectral_flatness`** — computed on the **power** spectrum (`power=2.0`,
  Wiener-entropy convention) with an `amin` floor.
- **`spectral_crest_factor`** — now `max(magnitude) / mean(magnitude)` (>= 1 by
  construction), the standard peak-to-average definition.
- **`fundamental_frequency_yin`** — adds **parabolic interpolation** around the
  chosen lag for sub-sample period (pitch) accuracy.
- **`voice_onset_time`** — corrected autocorrelation-based onset estimation.
- **`cepstral_peak_prominence` (CPP)** — uses a proper power cepstrum
  (next-power-of-two FFT) and regression-line prominence.
- **`vocal_tract_length`** — quarter-wave model `L = c / (4 * F1)` (tube closed at
  one end), with an optional formant-spacing estimate from `F2 - F1`.
- **LSP extraction (`lsp_coefficients`)** — corrected Line Spectral Pair
  computation from LPC.
- **`lpcc`** — corrected coefficient **sign** convention.
- **`alpha_ratio`, `hammarberg_index`, `nasality_index`** — no longer truncate
  frames; they analyse the full framed signal.
- **Markdown feature catalog** — `catalog_to_markdown` now escapes `|` (and
  newlines) in signatures/descriptions, so modern union type hints such as
  `int | None` / `str | Path` no longer corrupt the GitHub-Flavored-Markdown
  table columns. `docs/FEATURE_CATALOG.md` regenerated accordingly.
- **Documentation examples** — every README/quick-start snippet and every script
  in `examples/` was corrected to the real API and is executed in CI. Notable
  corrections: headline key `mfcc_mean_0` (was `mfcc_0_mean`); `spectral_contrast`
  takes `fs=` (not `sample_rate=`); `load_audio(..., target_sample_rate=...)`
  (keyword-only); `jitter`/`shimmer` take period/amplitude **sequences**, not a
  waveform; `compute_functionals` takes a 2-D tensor and returns a flat tensor;
  batch extraction returns rows written via `write_feature_rows_to_csv`.

### Changed

> **BREAKING (numeric).** The following features now produce different values than
> in 1.1.x. They are now *correct*, but any cached features, thresholds, or trained
> models that depend on the old outputs must be re-baselined.

- **`temporal_centroid`** now follows MPEG-7: it is the **energy-weighted center of
  time in seconds** (previously an amplitude-weighted index in samples that ignored
  `sample_rate`). Magnitude and units both change.
- **`spectral_rolloff`** values shift because accumulation is now magnitude-based
  (was power-based).
- **`spectral_flatness`** values shift because it is now power-spectrum based.
- **`spectral_crest_factor`** values shift to the `max/mean` definition.
- **`fundamental_frequency_yin`** F0 estimates shift slightly due to parabolic
  interpolation (sub-sample precision).
- **`vocal_tract_length`** values change to the quarter-wave (and formant-spacing)
  model.
- **`cepstral_peak_prominence`**, **`lpcc`**, **`lsp_coefficients`**,
  **`voice_onset_time`**, **`alpha_ratio`**, **`hammarberg_index`**, and
  **`nasality_index`** outputs change as a result of the correctness fixes above.
- **`beat_track`** now returns beat **times** (seconds); use
  `beat_track_with_tempo` for the `(tempo, beat_frames)` tuple.
- **Minimum Python is now 3.9** (was advertised as 3.8).

## [1.1.1] - 2026-01-25

- Documentation refresh and PyPI metadata update.

## [1.1.0]

- librosa parity for spectral, temporal, and cepstral features.
- Rewrote the Burg LPC algorithm using Marple's method.
- Added CLI, auto-discovered feature catalog, and Praat validation infrastructure.

## [1.0.0]

- Initial public release on PyPI: PyTorch-based temporal, spectral, pitch, voice,
  cepstral, and statistical audio features.

[1.2.0]: https://github.com/ankitshah009/audiofeat/releases/tag/v1.2.0
[1.1.1]: https://github.com/ankitshah009/audiofeat/releases/tag/v1.1.1
[1.1.0]: https://github.com/ankitshah009/audiofeat/releases/tag/v1.1.0
[1.0.0]: https://github.com/ankitshah009/audiofeat/releases/tag/v1.0.0
