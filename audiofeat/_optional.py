"""Helpers for lazily importing optional, heavyweight ML dependencies.

The wrapper modules in :mod:`audiofeat` (``asr``, ``vad``, ``scene`` …) sit on
top of large third-party packages (Whisper, Silero-VAD, PANNs, asteroid,
SpeechBrain, madmom, transformers, pyannote, …) that are **not** installed in
the base distribution.  ``import audiofeat`` must never pull these in, so every
wrapper defers its import to call time and routes it through :func:`require`.

:func:`require` deliberately catches the broad :class:`ImportError` rather than
only :class:`ModuleNotFoundError`: several of these packages (madmom, pyannote,
asteroid, panns_inference) raise a plain ``ImportError`` or even
``AttributeError`` from *within* their own ``__init__`` against newer
NumPy/SciPy/torch builds, which would otherwise leak out as a confusing
traceback instead of an actionable "install the extra" message.
"""
from __future__ import annotations

import importlib
from types import ModuleType


def require(module_name: str, extra: str, pip_name: str | None = None) -> ModuleType:
    """Import *module_name* or raise an actionable :class:`ModuleNotFoundError`.

    Parameters
    ----------
    module_name : str
        The importable module path, e.g. ``"silero_vad"`` or ``"pyannote.audio"``.
    extra : str
        The ``audiofeat`` optional-dependency group that ships this backend,
        e.g. ``"vad"`` (used to build the ``pip install audiofeat[...]`` hint).
    pip_name : str, optional
        Human-facing distribution name to mention in the error when it differs
        from *module_name* (e.g. ``"openai-whisper"`` for the ``whisper`` module).

    Returns
    -------
    module
        The imported module.

    Raises
    ------
    ModuleNotFoundError
        If the module (or one of *its* imports) cannot be loaded.  The original
        :class:`ImportError` is chained via ``from`` for debuggability.
    """
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:  # broader than ModuleNotFoundError on purpose
        raise ModuleNotFoundError(
            f"`{pip_name or module_name}` is required for this feature. "
            f"Install with `pip install audiofeat[{extra}]`."
        ) from exc
