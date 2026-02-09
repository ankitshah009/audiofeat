"""
Generate deterministic demo WAV assets for examples.

This avoids broken placeholder files (e.g., text content in .wav files)
and makes example scripts runnable out of the box.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torchaudio
from scipy.signal import iirpeak, lfilter


ROOT = Path(__file__).resolve().parent


def _sine_mix(sr: int, duration: float, freqs: list[float]) -> torch.Tensor:
    t = torch.arange(int(sr * duration), dtype=torch.float32) / sr
    y = torch.zeros_like(t)
    for i, f in enumerate(freqs, start=1):
        y += torch.sin(2 * torch.pi * f * t) / i
    return y


def _chirp(sr: int, duration: float, f0: float, f1: float) -> torch.Tensor:
    t = torch.arange(int(sr * duration), dtype=torch.float32) / sr
    k = (f1 - f0) / duration
    phase = 2 * torch.pi * (f0 * t + 0.5 * k * t * t)
    return torch.sin(phase)


def _speech_like_vowel(sr: int, duration: float, f0: float = 120.0) -> torch.Tensor:
    t = torch.arange(int(sr * duration), dtype=torch.float32) / sr
    source = torch.zeros_like(t)
    for k in range(1, 20):
        source += torch.sin(2 * torch.pi * (k * f0) * t) / k
    y = source.numpy()
    for center_hz, q in [(500.0, 3.0), (1500.0, 4.0), (2500.0, 5.0)]:
        b, a = iirpeak(center_hz / (sr / 2), q)
        y = lfilter(b, a, y)
    y = torch.from_numpy(y.astype("float32"))
    y = y / (y.abs().max() + 1e-8)
    return y


def _beep(sr: int, duration: float = 1.0) -> torch.Tensor:
    y = torch.zeros(int(sr * duration), dtype=torch.float32)
    on = int(0.1 * sr)
    off = int(0.2 * sr)
    tone = torch.sin(2 * torch.pi * 1000 * torch.arange(off - on, dtype=torch.float32) / sr)
    y[on:off] = tone
    return y


def _write(path: Path, waveform: torch.Tensor, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(path), waveform.unsqueeze(0), sr)
    print(f"wrote {path} ({waveform.numel()/sr:.2f}s @ {sr} Hz)")


def main() -> None:
    _write(ROOT / "sample_yesno.wav", _speech_like_vowel(22050, 1.6, f0=120.0), 22050)
    _write(ROOT / "sample_test1_22050.wav", _sine_mix(22050, 1.0, [220, 440, 880]), 22050)
    _write(ROOT / "sample_beep.wav", _beep(22050, 1.0), 22050)
    _write(ROOT / "sample_libri.wav", _sine_mix(16000, 2.0, [110, 330, 550, 770]), 16000)
    _write(ROOT / "sample_pydub2.wav", _chirp(16000, 1.5, 80.0, 400.0), 16000)
    _write(ROOT / "sample_pytorch.wav", _chirp(22050, 1.2, 120.0, 1200.0), 22050)


if __name__ == "__main__":
    main()
