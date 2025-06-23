#audiofeat

Utility functions for computing various speech features using PyTorch.

The `audiofeat.features` module provides implementations of several
commonly used acoustic measurements such as RMS energy, spectral
entropy and rolloff, fundamental frequency estimation via autocorrelation or
the YIN algorithm, and more.

## Example

An example script `examples/compute_features.py` shows how to load a WAV
file and compute a selection of features:

```bash
python examples/compute_features.py path/to/file.wav
```

This will print values for features such as RMS energy, F0, spectral
entropy, spectral rolloff and additional measures derived from the
signal.
