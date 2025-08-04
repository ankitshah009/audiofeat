"""Automatic speech-recognition wrapper (OpenAI Whisper).

Lightweight helper around the `openai-whisper` package.  Supports all Whisper model sizes.
Requires extra dependency group `models`.
"""
from __future__ import annotations
from pathlib import Path
from typing import Literal, Union


def _lazy_load_model(model_size: str = "base"):
    try:
        import whisper  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "`openai-whisper` is required. Install with `pip install audiofeat[models]`."
        ) from exc

    return whisper.load_model(model_size)


def transcribe(
    audio: Union[str, Path],
    *,
    model_size: Literal[
        "tiny",
        "base",
        "small",
        "medium",
        "large",
    ] = "base",
    language: str | None = None,
    **whisper_kwargs,
):
    """Return Whisper transcription dict (segments, text, language)."""
    model = _lazy_load_model(model_size)
    result = model.transcribe(str(audio), language=language, **whisper_kwargs)
    return result
