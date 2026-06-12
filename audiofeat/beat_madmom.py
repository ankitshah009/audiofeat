"""Advanced Beat & Downbeat tracking via madmom."""
from __future__ import annotations
from typing import List, Tuple

# madmom can raise a plain ImportError/AttributeError from inside its own
# __init__ against newer NumPy/SciPy, so catch ImportError broadly (not just
# ModuleNotFoundError) and degrade to the "install the extra" error below.
try:
    from madmom.features.beats import RNNBeatProcessor, DBNBeatTrackingProcessor
    from madmom.features.downbeats import RNNDownBeatProcessor, DBNDownBeatTrackingProcessor
except (ModuleNotFoundError, ImportError):
    RNNBeatProcessor = None  # type: ignore
    DBNBeatTrackingProcessor = None  # type: ignore
    RNNDownBeatProcessor = None  # type: ignore
    DBNDownBeatTrackingProcessor = None  # type: ignore


def beat_track(path: str) -> List[float]:
    if RNNBeatProcessor is None:
        raise ModuleNotFoundError(
            "`madmom` is required for beat tracking. Install with `pip install audiofeat[beat]`."
        )

    proc = RNNBeatProcessor()
    act = proc(path)
    tracker = DBNBeatTrackingProcessor(fps=100)
    beats = tracker(act)
    return beats.tolist()


def downbeat_track(path: str) -> List[Tuple[float, int]]:
    if RNNDownBeatProcessor is None:
        raise ModuleNotFoundError(
            "`madmom` is required for downbeat tracking. Install with `pip install audiofeat[beat]`."
        )

    proc = RNNDownBeatProcessor()
    act = proc(path)
    tracker = DBNDownBeatTrackingProcessor(beats_per_bar=[3, 4], fps=100)
    return [(float(t), int(b)) for t, b in tracker(act)]
