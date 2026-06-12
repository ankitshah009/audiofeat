"""Voice Activity Detection wrapper (Silero-VAD) with a pure-torch fallback.

Two code paths are offered:

* :func:`is_speech` — when ``silero-vad`` is installed it runs the Silero VAD
  model (loaded once and cached); otherwise it transparently falls back to a
  pure-torch RMS energy threshold so basic VAD works with **zero** optional
  dependencies.
* :func:`is_speech_energy` — the energy/RMS fallback exposed directly.

The Silero model is loaded via the real ``silero_vad.load_silero_vad`` entry
point (the older ``get_silero_vad_model`` name does **not** exist).
"""
from __future__ import annotations

import torch

from ._optional import require

# Module-level cache so the (non-trivial) Silero model is loaded at most once.
_MODEL = None


def _lazy_model():
    """Load and cache the Silero VAD model.

    Raises
    ------
    ModuleNotFoundError
        If ``silero-vad`` is not installed.  Callers that want a dependency-free
        path should use :func:`is_speech_energy` (or :func:`is_speech` with
        ``use_silero=False``) instead.
    """
    global _MODEL
    if _MODEL is None:
        silero_vad = require("silero_vad", extra="vad", pip_name="silero-vad")
        # Real Silero API (>=5.0). The legacy ``get_silero_vad_model`` is gone.
        _MODEL = silero_vad.load_silero_vad()
    return _MODEL


def _to_mono_1d(waveform: torch.Tensor) -> torch.Tensor:
    """Collapse any ``(..., samples)`` waveform to a mono 1-D tensor."""
    if waveform.ndim > 1:
        waveform = waveform.reshape(-1, waveform.shape[-1]).mean(0)
    return waveform


@torch.inference_mode()
def is_speech_energy(
    waveform: torch.Tensor,
    sample_rate: int,
    threshold: float = 0.01,
) -> bool:
    """Pure-torch RMS energy VAD fallback (no optional dependencies).

    Returns ``True`` when the root-mean-square amplitude of *waveform* exceeds
    *threshold*. This is a crude but dependency-free heuristic that reliably
    separates an audible tone from (near-)silence; it does not model speech
    structure the way Silero does.

    Parameters
    ----------
    waveform : torch.Tensor
        Audio samples ``(samples,)`` or ``(channels, samples)``.
    sample_rate : int
        Unused by the energy heuristic; kept for signature parity with
        :func:`is_speech`.
    threshold : float, default 0.01
        RMS amplitude above which the frame is considered speech/active.
    """
    del sample_rate  # not needed for an RMS threshold; kept for API parity
    mono = _to_mono_1d(waveform).float()
    if mono.numel() == 0:
        return False
    rms = torch.sqrt(torch.mean(mono**2))
    return bool(rms.item() > threshold)


@torch.inference_mode()
def is_speech(
    waveform: torch.Tensor,
    sample_rate: int,
    threshold: float = 0.5,
    *,
    use_silero: bool = True,
    energy_threshold: float = 0.01,
) -> bool:
    """Return ``True`` if *waveform* contains speech.

    Parameters
    ----------
    waveform : torch.Tensor
        Audio samples ``(samples,)`` or ``(channels, samples)``.
    sample_rate : int
        Sampling rate in Hz (Silero supports 8 kHz and 16 kHz).
    threshold : float, default 0.5
        Probability threshold applied to the Silero speech probability.
    use_silero : bool, keyword-only, default True
        If ``True`` (default) use the Silero model when available, otherwise
        raise.  If ``False``, always use the :func:`is_speech_energy` fallback.
        When ``True`` but ``silero-vad`` is **not** installed, the function does
        *not* raise — it gracefully degrades to the energy fallback so basic VAD
        works out of the box. Pass ``use_silero=True`` and install the extra for
        accurate, model-based VAD.
    energy_threshold : float, keyword-only, default 0.01
        RMS threshold forwarded to the energy fallback.
    """
    if not use_silero:
        return is_speech_energy(waveform, sample_rate, threshold=energy_threshold)

    try:
        model = _lazy_model()
    except ModuleNotFoundError:
        # No silero installed: degrade gracefully to the dependency-free path.
        return is_speech_energy(waveform, sample_rate, threshold=energy_threshold)

    speech_prob = model(waveform, sample_rate)
    if isinstance(speech_prob, torch.Tensor):
        speech_prob = speech_prob.item()
    return bool(speech_prob >= threshold)
