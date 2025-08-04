"""Streaming feature extraction utilities."""
from __future__ import annotations
from typing import Callable, Dict
import torch


class StreamingFeatureExtractor:
    """Process audio in chunks and emit features online."""

    def __init__(self, feature_fn: Callable[[torch.Tensor, int], torch.Tensor], sample_rate: int, frame_ms: int = 25, hop_ms: int = 10):
        self.fn = feature_fn
        self.sr = sample_rate
        self.frame = int(self.sr * frame_ms / 1000)
        self.hop = int(self.sr * hop_ms / 1000)
        self.buffer = torch.zeros(0)

    def push(self, chunk: torch.Tensor) -> Dict[str, torch.Tensor]:
        self.buffer = torch.cat([self.buffer, chunk])
        feats = {}
        while self.buffer.size(0) >= self.frame:
            frame = self.buffer[: self.frame]
            self.buffer = self.buffer[self.hop :]
            feats.setdefault("frames", []).append(self.fn(frame, self.sr))
        return feats
