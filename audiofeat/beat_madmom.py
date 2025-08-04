"""Advanced Beat & Downbeat tracking via madmom."""
from __future__ import annotations
from typing import List, Tuple

try:
    from madmom.features.beats import RNNBeatProcessor, DBNBeatTrackingProcessor
    from madmom.features.downbeats import RNNDownBeatProcessor, DBNDownBeatTrackingProcessor
except ModuleNotFoundError:
    RNNBeatProcessor = None  # type: ignore


def beat_track(path: str) -> List[float]:
    if RNNBeatProcessor is None:
        raise ModuleNotFoundError("madmom not installed. Use `pip install audiofeat[models]`.")

    proc = RNNBeatProcessor()
    act = proc(path)
    tracker = DBNBeatTrackingProcessor(fps=100)
    beats = tracker(act)
    return beats.tolist()


def downbeat_track(path: str) -> List[Tuple[float, int]]:
    if RNNDownBeatProcessor is None:
        raise ModuleNotFoundError("madmom not installed. Use `pip install audiofeat[models]`.")

    proc = RNNDownBeatProcessor()
    act = proc(path)
    tracker = DBNDownBeatTrackingProcessor(beats_per_bar=[3, 4], fps=100)
    return [(float(t), int(b)) for t, b in tracker(act)]
