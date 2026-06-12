"""Speaker embedding utilities (ECAPA-TDNN).

This wrapper uses SpeechBrain's pretrained ECAPA model to extract 192-dim
speaker embeddings (a.k.a. x-vectors). Heavyweights are downloaded on first use
and cached via HuggingFace Hub. Requires the optional dependency group
``embeddings`` (``pip install audiofeat[embeddings]``).

Example
-------
>>> import torch, torchaudio
>>> waveform, sr = torchaudio.load("speech.wav")
>>> from audiofeat.embeddings import extract_speaker_embedding
>>> emb = extract_speaker_embedding(waveform, sr)
>>> print(emb.shape)  # (192,)
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Union

import torch

_WAV_CHANNEL_WARNING = (
    "`extract_speaker_embedding` expects a mono waveform. Using channel 0 of "
    "multi-channel input."
)

# Default on-disk cache for the downloaded checkpoint so we don't pollute CWD
# (SpeechBrain otherwise writes a ``pretrained_models/`` dir wherever you run).
_DEFAULT_SAVEDIR = str(Path(tempfile.gettempdir()) / "audiofeat_speechbrain_ecapa")
_ECAPA_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"

# Cache the loaded classifier across calls.
_CLASSIFIER = None


def _import_encoder_classifier():
    """Return SpeechBrain's ``EncoderClassifier`` across versions.

    SpeechBrain >= 1.0 moved the pretrained interfaces to
    ``speechbrain.inference`` and deprecated/removed ``speechbrain.pretrained``.
    Try the new location first, then fall back to the old one. The broad
    ``ImportError`` catch also handles SpeechBrain failing to import against
    incompatible NumPy/torch builds.
    """
    try:
        from speechbrain.inference import EncoderClassifier  # type: ignore

        return EncoderClassifier
    except ImportError:
        pass
    try:
        from speechbrain.pretrained import EncoderClassifier  # type: ignore

        return EncoderClassifier
    except ImportError as exc:
        raise ModuleNotFoundError(
            "SpeechBrain is required for speaker embeddings. "
            "Install with `pip install audiofeat[embeddings]`."
        ) from exc


def _lazy_load_pipeline(savedir: str | None = None):
    global _CLASSIFIER
    if _CLASSIFIER is None:
        EncoderClassifier = _import_encoder_classifier()
        # official checkpoint for ECAPA-TDNN on VoxCeleb; explicit savedir keeps
        # the download out of the current working directory.
        _CLASSIFIER = EncoderClassifier.from_hparams(
            source=_ECAPA_SOURCE,
            savedir=savedir or _DEFAULT_SAVEDIR,
        )
    return _CLASSIFIER


@torch.inference_mode()
def extract_speaker_embedding(
    waveform: torch.Tensor | str | Path,
    sample_rate: int | None = None,
    *,
    savedir: str | None = None,
) -> torch.Tensor:
    """Return a 192-D speaker embedding.

    Parameters
    ----------
    waveform : torch.Tensor or str/Path
        Either a waveform tensor (*c x n*) or a path to an audio file.
    sample_rate : int, optional
        Required if *waveform* is a tensor. Ignored otherwise.
    savedir : str, keyword-only, optional
        Directory to cache the downloaded checkpoint. Defaults to a temp dir so
        the current working directory stays clean.
    """
    if not isinstance(waveform, torch.Tensor):
        import torchaudio  # local import to avoid hard dep in base install

        waveform, sample_rate = torchaudio.load(str(waveform))
    elif sample_rate is None:
        raise ValueError("sample_rate is required when passing a tensor waveform.")

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

    classifier = _lazy_load_pipeline(savedir=savedir)
    embeddings = classifier.encode_batch(waveform)  # real ECAPA output: (1, 1, 192)
    # reshape(-1) robustly flattens whatever batch/time dims SpeechBrain returns.
    return embeddings.reshape(-1).cpu()
