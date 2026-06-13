"""audiofeat: a PyTorch-based audio feature extraction library.

The public, flat API is assembled from the feature subpackages below so that
both attribute access (``audiofeat.rms``) and star imports
(``from audiofeat import *``) expose every documented feature plus
``__version__``.

``__all__`` is built dynamically (see ``_build_public_api`` below) rather than
hand-maintained: it walks the imported subpackages, honours each submodule's
explicit ``__all__`` when present, and otherwise falls back to the public,
locally-defined callables. This keeps the surface in sync with the codebase as
modules are added while excluding leaked third-party names (``torch``/``np``),
module objects, and internal frame/window/IO helpers.

Heavy/optional ML wrappers (``asr``, ``vad``, ``scene``, ``ssl``, ``denoise``,
``streaming``, ``spatial``, ``emotion``, ``embeddings``, ``diarization``,
``chord``, ``beat_madmom``, ``noise_reduction``, ...) are intentionally NOT
imported here so that ``import audiofeat`` never pulls in optional dependencies.
Access them explicitly, e.g. ``from audiofeat.asr import transcribe``.
"""

from __future__ import annotations

from ._version import __version__

# --- Core feature subpackages (eagerly imported, lightweight) -----------------
from .temporal import *
from .spectral import *
from .pitch import *
from .voice import *
from .cepstral import *
from .stats import *
from .io import *
from .validation import *
from .standards import *
from .catalog import *
from .rhythm import *
from .segmentation import *

# ``preprocessing`` is currently an empty stub. Import its public names only
# once it actually defines content; importing ``*`` from an empty package is a
# no-op today, so it is omitted to keep the namespace clean.
# from .preprocessing import *  # noqa: E501  (enable when preprocessing gains content)


def _build_public_api() -> list[str]:
    """Assemble the aggregated ``__all__`` from the feature subpackages.

    Strategy:
      * Walk every subpackage (and its submodules) that contributes to the flat
        API.
      * If a (sub)module defines ``__all__``, honour it verbatim (author intent,
        e.g. ``segmentation.diarization`` deliberately exports ``kmeans``).
      * Otherwise, include public names that are *defined in that module*
        (``__module__`` match) and are not module objects -- this naturally
        drops re-exported third-party names such as ``torch``/``np`` and typing
        helpers.
      * Apply a small denylist of internal helpers / re-exports that slip
        through modules lacking ``__all__`` (notably ``io.features``).
      * Bind any discovered name that is reachable from a submodule but not yet
        in this package's namespace (e.g. ``spectral.key.key_detect``, which is
        not re-exported by ``spectral/__init__``) so attribute access and star
        imports both work.

    Always returns a list that includes ``__version__``.
    """
    import importlib
    import pkgutil
    import types

    # Subpackages whose public functions form the flat ``audiofeat`` API.
    subpackages = (
        "temporal",
        "spectral",
        "pitch",
        "voice",
        "cepstral",
        "stats",
        "io",
        "validation",
        "standards",
        "rhythm",
        "segmentation",
    )

    # Names that leak from modules without an explicit ``__all__`` but are
    # internal helpers / framework re-exports, not user-facing features.
    deny = {
        # third-party / stdlib re-exports
        "torch",
        "torchaudio",
        "np",
        "numpy",
        "math",
        "warnings",
        "os",
        "sys",
        "csv",
        "json",
        "platform",
        "tempfile",
        "inspect",
        "importlib",
        "pkgutil",
        "annotations",
        # typing / pathlib helpers occasionally re-exported
        "Path",
        "Iterable",
        "Mapping",
        "Any",
        "Literal",
        # internal signal-processing helpers
        "frame_signal",
        "hann_window",
        # io.features helpers (module has no __all__)
        "to_mono",
        "resample_if_needed",
        "load_audio",
        "summarize_series",
        "summarize_matrix",
        "extract_core_features",
        "iter_audio_files",
        "write_feature_rows_to_csv",
    }

    g = globals()
    discovered: dict[str, object] = {}

    def _consider(name: str, obj: object) -> None:
        if name.startswith("_"):
            return
        if isinstance(obj, types.ModuleType):
            return
        if name in deny:
            return
        discovered[name] = obj

    for short in subpackages:
        base = f"{__name__}.{short}"
        try:
            pkg = importlib.import_module(base)
        except Exception:
            # A subpackage that cannot import shouldn't break the whole API.
            continue

        modules = [pkg]
        if hasattr(pkg, "__path__"):
            for info in pkgutil.iter_modules(pkg.__path__, prefix=f"{base}."):
                try:
                    modules.append(importlib.import_module(info.name))
                except Exception:
                    # Optional backends (parselmouth, crepe, ...) may be absent.
                    continue

        for module in modules:
            explicit = getattr(module, "__all__", None)
            if explicit is not None:
                for name in explicit:
                    obj = getattr(module, name, None)
                    if obj is not None:
                        _consider(name, obj)
            else:
                module_name = getattr(module, "__name__", None)
                for name, obj in vars(module).items():
                    if getattr(obj, "__module__", None) != module_name:
                        continue
                    _consider(name, obj)

    # Bind names reachable from submodules but not yet in this namespace so that
    # ``audiofeat.<name>`` and ``from audiofeat import *`` both resolve them.
    for name, obj in discovered.items():
        g.setdefault(name, obj)

    names = set(discovered) | {"__version__"}
    return sorted(names)


__all__ = _build_public_api()
