"""Environmental sound / acoustic-scene classification.

Uses CNN14 (PANNs) via the `panns_inference` helper package.
Returns top-k AudioSet class predictions with probabilities.
Requires optional deps (`models`).
"""
from __future__ import annotations
from typing import List, Tuple

import torch

try:
    from panns_inference import AudioTagging, labels
except ModuleNotFoundError:
    AudioTagging = None  # type: ignore
    labels = []  # type: ignore


@torch.inference_mode()
def classify_scene(waveform: torch.Tensor, sample_rate: int, top_k: int = 5) -> List[Tuple[str, float]]:
    if AudioTagging is None:
        raise ModuleNotFoundError(
            "`panns_inference` is required. Install with `pip install audiofeat[models]`."
        )

    if waveform.ndim == 2 and waveform.size(0) > 1:
        waveform = waveform.mean(0, keepdim=True)
    elif waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)

    model = AudioTagging(checkpoint_path=None, device="cpu")
    scores = model.inference(waveform, sample_rate)
    probs = torch.softmax(scores, -1)[0]
    topk = torch.topk(probs, top_k)
    return [(labels[i], float(probs[i])) for i in topk.indices]
