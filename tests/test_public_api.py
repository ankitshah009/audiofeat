"""Tests for the top-level ``audiofeat`` public API surface.

These lock in the contract that:

* ``import audiofeat`` works and exposes a real ``__version__``.
* The documented features resolve both as attributes and via
  ``from audiofeat import *``.
* ``from audiofeat import *`` exports the full feature set, not just
  ``__version__`` (regression lock for the ``__all__ = ["__version__"]`` bug
  that suppressed every feature).
* Heavy/optional ML wrappers are NOT auto-imported by ``import audiofeat``.
"""

from __future__ import annotations

import importlib.util
import sys
import types

import pytest


# Representative set drawn from what the README advertises, spanning every
# subpackage that contributes to the flat API (temporal, spectral, cepstral,
# pitch, stats, io, spectral.key, segmentation, rhythm).
REPRESENTATIVE_FEATURES = [
    "rms",
    "spectral_centroid",
    "spectral_bandwidth",
    "spectral_rolloff",
    "mfcc",
    "chroma",
    "zero_crossing_rate",
    "fundamental_frequency_yin",
    "onset_detect",
    "tristimulus",
    "temporal_centroid",
    "beat_track",
    "compute_functionals",
    "extract_features_from_file",
    "key_detect",
    "silence_removal",  # segmentation
    "beat_detection",  # rhythm
]

# Names that must never leak into the public surface (third-party re-exports,
# module objects, and internal helpers).
FORBIDDEN_NAMES = {
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
    "annotations",
    "frame_signal",
    "hann_window",
    "to_mono",
    "load_audio",
    "resample_if_needed",
    "summarize_series",
    "summarize_matrix",
}

# Heavy/optional dependencies pulled only by the lazy ML wrapper modules.
HEAVY_OPTIONAL_DEPS = [
    "whisper",
    "speechbrain",
    "panns_inference",
    "noisereduce",
    "madmom",
    "crepe",
    "resemblyzer",
    "pyannote",
]


def test_import_and_version():
    import audiofeat

    assert isinstance(audiofeat.__version__, str)
    assert audiofeat.__version__  # non-empty
    assert "__version__" in audiofeat.__all__


@pytest.mark.parametrize("name", REPRESENTATIVE_FEATURES)
def test_feature_resolves_as_attribute(name):
    import audiofeat

    assert hasattr(audiofeat, name), f"audiofeat.{name} should be accessible"
    assert callable(getattr(audiofeat, name)), f"audiofeat.{name} should be callable"


@pytest.mark.parametrize("name", REPRESENTATIVE_FEATURES)
def test_feature_is_exported_in_all(name):
    import audiofeat

    assert name in audiofeat.__all__, f"{name!r} should be listed in audiofeat.__all__"


def test_star_import_exposes_more_than_just_version():
    """Regression lock for the bug where ``__all__ = ['__version__']`` hid
    every feature from ``from audiofeat import *``."""
    import audiofeat

    assert len(audiofeat.__all__) > 30, (
        "audiofeat.__all__ must aggregate the feature set, not just __version__; "
        f"got {len(audiofeat.__all__)} names"
    )
    assert "rms" in audiofeat.__all__
    # __all__ should have no duplicates.
    assert len(audiofeat.__all__) == len(set(audiofeat.__all__))


def test_star_import_lands_names_in_clean_namespace():
    """Execute ``from audiofeat import *`` in an isolated namespace and verify
    the representative names (and ``__version__``) actually bind, with no
    crash."""
    namespace: dict[str, object] = {}
    exec("from audiofeat import *", namespace)

    assert "__version__" in namespace
    assert isinstance(namespace["__version__"], str) and namespace["__version__"]

    for name in REPRESENTATIVE_FEATURES:
        assert name in namespace, f"{name!r} should land via `from audiofeat import *`"
        assert callable(namespace[name])

    # The star import must expose substantially more than just the version.
    feature_names = [k for k in namespace if not k.startswith("_")]
    assert len(feature_names) > 30


def test_public_surface_has_no_leaked_helpers_or_modules():
    import audiofeat

    leaked = sorted(set(audiofeat.__all__) & FORBIDDEN_NAMES)
    assert not leaked, f"forbidden names leaked into __all__: {leaked}"

    module_objs = [
        name
        for name in audiofeat.__all__
        if isinstance(getattr(audiofeat, name, None), types.ModuleType)
    ]
    assert not module_objs, f"module objects leaked into __all__: {module_objs}"


@pytest.mark.parametrize("dep", HEAVY_OPTIONAL_DEPS)
def test_heavy_optional_wrappers_not_auto_imported(dep):
    """``import audiofeat`` must not pull in heavy/optional ML deps.

    Guarded: if the dependency happens to already be installed/imported in the
    environment for other reasons, skip rather than produce a false failure.
    """
    # If the package isn't even installed it obviously isn't imported; if it is
    # installed but already in sys.modules before we import audiofeat, skip.
    if importlib.util.find_spec(dep) is None:
        already_present = False
    else:
        already_present = dep in sys.modules

    # Importing audiofeat (idempotent if already imported) must not add it.
    import audiofeat  # noqa: F401

    if already_present:
        pytest.skip(f"{dep} was already imported by the environment")

    assert dep not in sys.modules, (
        f"importing audiofeat must not import the heavy optional dependency {dep!r}"
    )
