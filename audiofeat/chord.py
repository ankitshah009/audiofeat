"""Chord recognition using simple chroma template matching (fallback if deep model missing)."""
from __future__ import annotations
from typing import List, Tuple

import torch
import torchaudio
import torchaudio.transforms as T
import numpy as np

_CHORD_TEMPLATES = {
    "C": [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
    "G": [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0],
    # ... extend for all 24 major/minor chords
}


@torch.inference_mode()
def detect_chords(path: str, hop_length: int = 2048) -> List[Tuple[float, str]]:
    wav, sr = torchaudio.load(path)
    if wav.ndim == 2 and wav.size(0) > 1:
        wav = wav.mean(0, keepdim=True)

    stft = T.Spectrogram(n_fft=4096, hop_length=hop_length, power=2.0)(wav)
    magnitudes = torch.sqrt(stft + 1e-9)
    freqs = torch.linspace(0, sr / 2, magnitudes.size(1))

    # crude chroma
    chroma = torch.zeros(12, magnitudes.size(-1))
    for i, f in enumerate(freqs):
        if f.item() <= 0:
            continue
        bin_idx = int(round(12 * np.log2(f.item() / 440.0) + 69)) % 12
        chroma[bin_idx] += magnitudes[:, i, :].squeeze(0)

    times = torch.arange(chroma.size(1)) * hop_length / sr
    chords = []
    for t in range(chroma.size(1)):
        v = chroma[:, t]
        best, score = None, -float("inf")
        for chord, template in _CHORD_TEMPLATES.items():
            s = torch.dot(v, torch.tensor(template, dtype=v.dtype, device=v.device))
            if s > score:
                best, score = chord, s
        chords.append((times[t].item(), best))
    return chords
