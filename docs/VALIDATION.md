# Validation Methodology

This project now exposes a reproducible quality gate via:

```bash
audiofeat gold-standard --no-optional --min-score 100 --fail-on-any
```

The command runs deterministic checks and computes a score normalized to 100.

## What is checked

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

## Score model

- Every check has a fixed point weight.
- Skipped checks are excluded from the denominator.
- Score = `(earned_points / available_points) * 100`.
- `--min-score` enforces an exit code for CI gates.
- `--fail-on-any` enforces strict pass for all non-skipped checks.

## Why this is useful

- Turns "confidence" into a machine-checkable contract.
- Catches math regressions in core DSP functions.
- Surfaces backend/dependency drift early in CI.
