"""Chord recognition via chroma template matching.

A lightweight, dependency-free chord estimator: it computes a 12-bin chromagram
from the magnitude STFT and scores each frame against the 24 major/minor triad
binary templates, returning the best match per frame.

Both the chroma binning and the template scoring are vectorized:

* a precomputed ``(12, n_freq)`` 0/1 mapping ``M`` collapses spectrum bins to
  pitch classes in one ``M @ magnitudes`` matmul;
* a ``(24, 12)`` template matrix scores all chords at once with a single matmul.
"""
from __future__ import annotations

from functools import lru_cache
from typing import List, Tuple

import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T

# Pitch classes, C first (matches the chroma-bin convention C, C#, …, B).
_PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Triad masks at root C. Major = root + major third + perfect fifth.
# Minor = root + minor third + perfect fifth.
_MAJOR_MASK = [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0]
_MINOR_MASK = [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0]


def _build_templates() -> Tuple[torch.Tensor, List[str]]:
    """Return a ``(24, 12)`` template matrix and the matching chord labels.

    The 24 rows are the 12 major triads followed by the 12 minor triads,
    obtained by rolling the C-rooted masks across all 12 roots.
    """
    rows: List[List[int]] = []
    labels: List[str] = []
    for root in range(12):
        rows.append(np.roll(_MAJOR_MASK, root).tolist())
        labels.append(f"{_PITCH_CLASSES[root]}")
    for root in range(12):
        rows.append(np.roll(_MINOR_MASK, root).tolist())
        labels.append(f"{_PITCH_CLASSES[root]}m")
    templates = torch.tensor(rows, dtype=torch.float32)  # (24, 12)
    return templates, labels


# Public-ish constant retained for backward compatibility / introspection:
# a dict mapping every chord name to its 12-bin binary template.
_TEMPLATE_MATRIX, _CHORD_LABELS = _build_templates()
_CHORD_TEMPLATES = {
    label: _TEMPLATE_MATRIX[i].to(torch.int64).tolist()
    for i, label in enumerate(_CHORD_LABELS)
}


@lru_cache(maxsize=8)
def _chroma_map(n_freq: int, sr: int, n_fft: int, fmin: float) -> torch.Tensor:
    """Build the ``(12, n_freq)`` pitch-class binning matrix.

    Each frequency bin above *fmin* is assigned to exactly one pitch class; bins
    at or below *fmin* (and DC) get no class, suppressing sub-bass rumble.
    """
    freqs = torch.linspace(0, sr / 2, n_freq)
    mapping = torch.zeros(12, n_freq)
    for i in range(n_freq):
        f = float(freqs[i])
        if f < fmin:
            continue
        pitch_class = int(round(12 * np.log2(f / 440.0) + 69)) % 12
        mapping[pitch_class, i] = 1.0
    return mapping


@torch.inference_mode()
def detect_chords(
    path: str,
    hop_length: int = 2048,
    n_fft: int = 4096,
    fmin: float = 65.0,
) -> List[Tuple[float, str]]:
    """Estimate the chord at each analysis frame of an audio file.

    Parameters
    ----------
    path : str
        Path to an audio file (loaded via torchaudio, mixed to mono).
    hop_length : int, default 2048
        STFT hop in samples.
    n_fft : int, default 4096
        STFT window / FFT size.
    fmin : float, default 65.0
        Frequencies below this (Hz) are ignored when building the chroma,
        removing sub-bass energy that would otherwise smear the templates
        (~65 Hz is roughly C2).

    Returns
    -------
    list of (time_seconds, chord_label)
        One ``(time, label)`` pair per frame; labels are like ``"C"`` (major)
        or ``"Am"`` (minor).
    """
    wav, sr = torchaudio.load(path)
    if wav.ndim == 2 and wav.size(0) > 1:
        wav = wav.mean(0, keepdim=True)

    # power=1.0 → magnitude spectrum directly (no sqrt round-trip).
    magnitudes = T.Spectrogram(n_fft=n_fft, hop_length=hop_length, power=1.0)(wav)
    magnitudes = magnitudes.squeeze(0)  # (n_freq, n_frames)
    n_freq, n_frames = magnitudes.shape

    # Vectorized chroma: (12, n_freq) @ (n_freq, n_frames) -> (12, n_frames).
    chroma_map = _chroma_map(n_freq, sr, n_fft, fmin).to(magnitudes.dtype)
    chroma = chroma_map @ magnitudes  # (12, n_frames)

    # Vectorized scoring: (24, 12) @ (12, n_frames) -> (24, n_frames).
    templates = _TEMPLATE_MATRIX.to(magnitudes.dtype)
    scores = templates @ chroma  # (24, n_frames)
    best = torch.argmax(scores, dim=0)  # (n_frames,)

    times = torch.arange(n_frames) * hop_length / sr
    return [(float(times[t]), _CHORD_LABELS[int(best[t])]) for t in range(n_frames)]
