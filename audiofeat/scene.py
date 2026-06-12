"""Environmental sound / acoustic-scene classification.

Uses CNN14 (PANNs) via the ``panns_inference`` helper package.  Returns the
top-*k* AudioSet class predictions with their probabilities.  Requires the
optional dependency group ``scene`` (``pip install audiofeat[scene]``).

Notes on the PANNs contract (important for correctness):

* ``AudioTagging.inference`` takes a **single** NumPy array shaped
  ``(batch, samples)`` sampled at **32 kHz** — the sample rate is *not* an
  argument, it is fixed by the CNN14 checkpoint.
* It returns a ``(clipwise_output, embedding)`` **tuple** of NumPy arrays.
  ``clipwise_output`` is already passed through a sigmoid (per-class AudioSet
  probabilities), so it must **not** be re-softmaxed.
* ``labels`` is a plain Python ``list`` of 527 class names, so it must be
  indexed with a Python ``int`` (a tensor index raises ``TypeError``).
"""
from __future__ import annotations

from typing import List, Tuple

import torch

from ._optional import require

_PANNS_SAMPLE_RATE = 32000

# Module-level cache: building CNN14 + loading its checkpoint is expensive.
_TAGGER = None


def _load_tagger():
    """Import ``panns_inference`` (deferred) and build/cache the tagger.

    The import is wrapped in :func:`require`, which catches the broad
    :class:`ImportError` because ``panns_inference`` can fail at import time
    against newer NumPy/torch builds (not just when it is missing entirely).
    """
    global _TAGGER
    if _TAGGER is None:
        panns = require("panns_inference", extra="scene", pip_name="panns_inference")
        _TAGGER = (panns.AudioTagging(checkpoint_path=None, device="cpu"), panns.labels)
    return _TAGGER


@torch.inference_mode()
def classify_scene(
    waveform: torch.Tensor,
    sample_rate: int,
    top_k: int = 5,
) -> List[Tuple[str, float]]:
    """Classify the acoustic scene of *waveform* into AudioSet tags.

    Parameters
    ----------
    waveform : torch.Tensor
        Audio samples ``(samples,)`` or ``(channels, samples)``.
    sample_rate : int
        Sampling rate of *waveform*; it is resampled to 32 kHz for CNN14.
    top_k : int, default 5
        Number of highest-probability tags to return.

    Returns
    -------
    list of (label, probability)
        The *top_k* AudioSet tags, highest probability first.
    """
    model, labels = _load_tagger()

    # Collapse to mono (CNN14 expects a single channel).
    if waveform.ndim == 2 and waveform.size(0) > 1:
        waveform = waveform.mean(0, keepdim=True)
    elif waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)

    # PANNs CNN14 runs at a fixed 32 kHz.
    if sample_rate != _PANNS_SAMPLE_RATE:
        import torchaudio.transforms as T

        waveform = T.Resample(sample_rate, _PANNS_SAMPLE_RATE)(waveform)

    # inference() takes a single (batch, samples) NumPy array; no sample_rate.
    clipwise, _ = model.inference(waveform.cpu().numpy())
    # clipwise is (batch, 527), already sigmoid'd → DO NOT softmax again.
    probs = torch.from_numpy(clipwise[0])

    top_k = min(top_k, probs.numel())
    topk = torch.topk(probs, top_k)
    # ``labels`` is a Python list → index with int(i), never a tensor.
    return [(labels[int(i)], float(p)) for i, p in zip(topk.indices, topk.values)]
