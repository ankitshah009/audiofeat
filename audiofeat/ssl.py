"""Self-supervised audio embeddings (Wav2Vec2, HuBERT, AST)."""
from __future__ import annotations

from functools import lru_cache
from typing import Literal

import torch

_MODEL_MAP = {
    "wav2vec2_base": "facebook/wav2vec2-base-960h",
    "hubert_base": "facebook/hubert-base-ls960",
    "ast": "MIT/ast-finetuned-audioset-10-10-0.4593",
}


@lru_cache(maxsize=4)
def _load_transformer(model_name: str):
    """Load and cache a HuggingFace processor + model pair.

    Cached via :func:`lru_cache` so repeated calls with the same backend do not
    re-download / re-instantiate the model. The broad ``ImportError`` catch also
    covers ``transformers`` failing to import against incompatible builds.
    """
    try:
        from transformers import AutoModel, AutoProcessor  # type: ignore
    except ImportError as exc:
        raise ModuleNotFoundError(
            "`transformers` is required for SSL embeddings. "
            "Install with `pip install audiofeat[ssl]`."
        ) from exc

    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    return processor, model


@torch.inference_mode()
def embed(
    waveform: torch.Tensor,
    sample_rate: int,
    backend: Literal["wav2vec2_base", "hubert_base", "ast"] = "wav2vec2_base",
) -> torch.Tensor:
    """Return a mean-pooled SSL embedding for *waveform*.

    Parameters
    ----------
    waveform : torch.Tensor
        Audio samples ``(samples,)`` or ``(channels, samples)``.
    sample_rate : int
        Sampling rate of *waveform* (most SSL models expect 16 kHz).
    backend : {"wav2vec2_base", "hubert_base", "ast"}
        Which pretrained backbone to use.
    """
    model_name = _MODEL_MAP[backend]
    processor, model = _load_transformer(model_name)

    if waveform.ndim == 2 and waveform.size(0) > 1:
        waveform = waveform.mean(0)
    waveform = waveform.reshape(-1)

    # HuggingFace feature extractors expect a NumPy array (a torch.Tensor is not
    # accepted and raises). Convert explicitly.
    inputs = processor(
        waveform.cpu().numpy(),
        sampling_rate=sample_rate,
        return_tensors="pt",
    )
    outputs = model(**inputs)
    # mean over time frames -> (hidden,)
    return outputs.last_hidden_state.mean(1).squeeze(0)
