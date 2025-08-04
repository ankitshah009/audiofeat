"""Voice Activity Detection wrapper (Silero-VAD)."""
from __future__ import annotations
import torch


def _lazy_model():
    try:
        import silero_vad  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Install silero-vad via `pip install audiofeat[models]`.") from exc

    return silero_vad.get_silero_vad_model()


@torch.inference_mode()
def is_speech(waveform: torch.Tensor, sample_rate: int, threshold: float = 0.5) -> bool:
    model = _lazy_model()
    speech_prob = model(waveform, sample_rate).item()
    return speech_prob >= threshold
