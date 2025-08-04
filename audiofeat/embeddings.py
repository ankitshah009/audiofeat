"""Speaker embedding utilities (ECAPA-TDNN).

This wrapper uses SpeechBrain's pre-trained ECAPA model to extract 192-dim speaker
embeddings (a.k.a. x-vectors).  Heavyweights are downloaded on first use and cached
via HuggingFace Hub.  Requires the optional dependency group `models`.

Example
-------
>>> import torch, torchaudio
>>> waveform, sr = torchaudio.load("speech.wav")
>>> from audiofeat.embeddings import extract_speaker_embedding
>>> emb = extract_speaker_embedding(waveform, sr)
>>> print(emb.shape)  # (192,)
"""
from __future__ import annotations
from pathlib import Path
from typing import Union

import torch

_WAV_CHANNEL_WARNING = (
    "`extract_speaker_embedding` expects a mono waveform. Using channel 0 of multi-channel input."
)


def _lazy_load_pipeline():
    try:
        from speechbrain.pretrained import EncoderClassifier  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "SpeechBrain is required for speaker embeddings. Install with `pip install audiofeat[models]`."
        ) from exc

    # official checkpoint for ECAPA-TDNN on VoxCeleb
    return EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")


@torch.inference_mode()
def extract_speaker_embedding(
    waveform: torch.Tensor | str | Path,
    sample_rate: int | None = None,
) -> torch.Tensor:
    """Return a 192-D speaker embedding.

    Parameters
    ----------
    waveform : torch.Tensor or str/Path
        Either a waveform tensor (*c x n*) or a path to an audio file.
    sample_rate : int, optional
        Required if *waveform* is a tensor. Ignored otherwise.
    """
    if not isinstance(waveform, torch.Tensor):
        import torchaudio  # local import to avoid hard dep in base install

        waveform, sample_rate = torchaudio.load(str(waveform))

    if waveform.ndim == 2 and waveform.size(0) > 1:
        import warnings

        warnings.warn(_WAV_CHANNEL_WARNING)
        waveform = waveform[:1]
    elif waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)

    if sample_rate != 16000:
        import torchaudio.transforms as T

        resampler = T.Resample(sample_rate, 16000)
        waveform = resampler(waveform)

    classifier = _lazy_load_pipeline()
    embeddings = classifier.encode_batch(waveform)  # (1, 192)
    return embeddings.squeeze(0).cpu()
