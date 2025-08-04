"""Self-supervised audio embeddings (Wav2Vec2, HuBERT, AST)."""
from __future__ import annotations
from typing import Literal

import torch

_MODEL_MAP = {
    "wav2vec2_base": "facebook/wav2vec2-base-960h",
    "hubert_base": "facebook/hubert-base-ls960",
    "ast": "MIT/ast-finetuned-audioset-10-10-0.4593",
}


def _load_transformer(model_name: str):
    try:
        from transformers import AutoModel, AutoProcessor  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "`transformers` is required for SSL embeddings. Install with `pip install transformers`."
        ) from exc

    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    return processor, model


@torch.inference_mode()
def embed(waveform: torch.Tensor, sample_rate: int, backend: Literal["wav2vec2_base", "hubert_base", "ast"] = "wav2vec2_base") -> torch.Tensor:
    model_name = _MODEL_MAP[backend]
    processor, model = _load_transformer(model_name)

    if waveform.ndim == 2 and waveform.size(0) > 1:
        waveform = waveform.mean(0)
    inputs = processor(waveform, sampling_rate=sample_rate, return_tensors="pt")
    outputs = model(**inputs)
    # take mean over time frames
    return outputs.last_hidden_state.mean(1).squeeze(0)
