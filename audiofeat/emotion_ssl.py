"""Emotion & stress detection via SSL fine-tuned models."""
from __future__ import annotations

from functools import lru_cache

import torch

# Default SER model on the HuggingFace hub (overridable per call).
_DEFAULT_MODEL = "YKwok1/wav2vec2-base-ks-emotion"


@lru_cache(maxsize=4)
def _load(model_id: str):
    """Load and cache a HuggingFace audio-classification processor + model.

    Cached via :func:`lru_cache` keyed on *model_id* so repeated calls do not
    reload the model. The broad ``ImportError`` catch also covers
    ``transformers`` failing to import against incompatible builds.
    """
    try:
        from transformers import (  # type: ignore
            AutoModelForAudioClassification,
            AutoProcessor,
        )
    except ImportError as exc:
        raise ModuleNotFoundError(
            "`transformers` is required for SSL emotion detection. "
            "Install with `pip install audiofeat[emotion]`."
        ) from exc

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForAudioClassification.from_pretrained(model_id)
    model.eval()
    return processor, model


@torch.inference_mode()
def detect_emotion_ssl(
    waveform: torch.Tensor,
    sample_rate: int,
    model: str = _DEFAULT_MODEL,
) -> str:
    """Predict an emotion label for *waveform* using a pretrained SSL model.

    Parameters
    ----------
    waveform : torch.Tensor
        Audio samples ``(samples,)`` or ``(channels, samples)``.
    sample_rate : int
        Sampling rate of *waveform* (typically 16 kHz).
    model : str, default ``"YKwok1/wav2vec2-base-ks-emotion"``
        HuggingFace hub id of the audio-classification model to use.
    """
    processor, classifier = _load(model)
    if waveform.ndim == 2 and waveform.size(0) > 1:
        waveform = waveform.mean(0)
    waveform = waveform.reshape(-1)

    # HuggingFace feature extractors expect a NumPy array, not a torch.Tensor.
    inputs = processor(
        waveform.cpu().numpy(),
        sampling_rate=sample_rate,
        return_tensors="pt",
    )
    logits = classifier(**inputs).logits
    pred = logits.argmax(-1).item()
    return classifier.config.id2label[pred]
