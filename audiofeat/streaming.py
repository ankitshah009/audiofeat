"""Streaming feature extraction utilities."""
from __future__ import annotations

from typing import Callable, Dict, List

import torch


class StreamingFeatureExtractor:
    """Process audio in chunks and emit per-frame features online.

    Call :meth:`push` repeatedly with successive audio chunks; each call returns
    the features for any *complete* frames that became available. When the stream
    ends, call :meth:`flush` to emit the trailing partial frame. Use
    :meth:`reset` to clear state and reuse the extractor for a new stream.
    """

    def __init__(
        self,
        feature_fn: Callable[[torch.Tensor, int], torch.Tensor],
        sample_rate: int,
        frame_ms: int = 25,
        hop_ms: int = 10,
    ):
        self.fn = feature_fn
        self.sr = sample_rate
        self.frame = int(self.sr * frame_ms / 1000)
        self.hop = int(self.sr * hop_ms / 1000)
        # Lazily initialised from the first chunk so we inherit its dtype/device.
        self.buffer: torch.Tensor | None = None

    def reset(self) -> None:
        """Clear the internal buffer so the extractor can process a new stream."""
        self.buffer = None

    def push(self, chunk: torch.Tensor) -> Dict[str, List[torch.Tensor]]:
        """Append *chunk* and return features for any complete frames."""
        if self.buffer is None:
            # Inherit dtype/device from the first chunk we see.
            self.buffer = torch.zeros(0, dtype=chunk.dtype, device=chunk.device)
        self.buffer = torch.cat([self.buffer, chunk])

        feats: Dict[str, List[torch.Tensor]] = {}
        while self.buffer.size(0) >= self.frame:
            frame = self.buffer[: self.frame]
            self.buffer = self.buffer[self.hop :]
            feats.setdefault("frames", []).append(self.fn(frame, self.sr))
        return feats

    def flush(self) -> Dict[str, List[torch.Tensor]]:
        """Emit the trailing partial frame (if any) and clear the buffer.

        Returns the same ``{"frames": [...]}`` shape as :meth:`push`. If no
        residual samples remain, returns an empty dict.
        """
        feats: Dict[str, List[torch.Tensor]] = {}
        if self.buffer is not None and self.buffer.numel() > 0:
            feats["frames"] = [self.fn(self.buffer, self.sr)]
        self.buffer = None
        return feats
