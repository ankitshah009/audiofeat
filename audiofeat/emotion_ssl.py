"""Emotion & stress detection via SSL fine-tuned models."""
from __future__ import annotations
import torch

_MODEL = "YKwok1/wav2vec2-base-ks-emotion"  # example SER model on HF hub


def _load():
    try:
        from transformers import AutoModelForAudioClassification, AutoProcessor  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Install transformers via `pip install transformers`. ") from exc

    processor = AutoProcessor.from_pretrained(_MODEL)
    model = AutoModelForAudioClassification.from_pretrained(_MODEL)
    model.eval()
    return processor, model


@torch.inference_mode()
def detect_emotion_ssl(waveform: torch.Tensor, sample_rate: int):
    processor, model = _load()
    if waveform.ndim == 2 and waveform.size(0) > 1:
        waveform = waveform.mean(0)
    inputs = processor(waveform, sampling_rate=sample_rate, return_tensors="pt")
    logits = model(**inputs).logits
    pred = logits.argmax(-1).item()
    label = model.config.id2label[pred]
    return label
