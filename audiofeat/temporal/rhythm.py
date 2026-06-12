import torch
from ..temporal.rms import frame_signal

# NOTE: ``temporal_centroid`` is intentionally NOT in ``__all__``. It stays
# importable as ``audiofeat.temporal.rhythm.temporal_centroid`` (frame-based,
# backward compat) but is excluded from ``from .rhythm import *`` so it does
# not shadow the canonical MPEG-7 ``temporal_centroid`` from ``centroid.py``
# at the package level.
__all__ = [
    "breath_group_duration",
    "speech_rate",
    "temporal_centroid_framewise",
]


def breath_group_duration(env: torch.Tensor, fs: int):
    """Estimate breath-group durations from an amplitude envelope.

    Heuristic: contiguous low-energy regions (below 25 % of the mean
    envelope) are treated as pauses; the spacing between pause onsets that
    are at least 250 ms apart is reported as breath-group durations. This
    is a coarse prosodic heuristic, not a calibrated measure.

    Returns an empty tensor when fewer than two breath groups are found.
    """
    threshold = env.mean() * 0.25
    below = (env < threshold).float()
    # ``.reshape(-1)`` (not ``.squeeze()``) keeps a 1-D tensor even when a
    # single sample is below threshold; ``.squeeze()`` would yield a 0-D
    # tensor and crash on the subsequent slice indexing.
    indices = torch.nonzero(below).reshape(-1)
    if indices.numel() < 2:
        return torch.tensor([], device=env.device)
    diffs = indices[1:] - indices[:-1]
    starts = indices[:-1][diffs > int(0.25 * fs)]
    if starts.numel() < 2:
        return torch.tensor([], device=env.device)
    durations = (starts[1:] - starts[:-1]).float() / fs
    return durations


def speech_rate(x: torch.Tensor, fs: int, threshold_ratio: float = 0.3, min_gap: float = 0.1):
    """Estimate speech rate in syllables per second.

    Heuristic: smooth ``|x|`` with a 20 ms moving average, count envelope
    peaks above ``threshold_ratio`` of the mean envelope that are at least
    ``min_gap`` seconds apart, and divide by the signal duration. This is a
    rough syllable-nucleus proxy, not a validated syllabification.

    Returns ``0.0`` when no qualifying peaks are found.
    """
    env = torch.abs(x)
    win_len = max(1, int(0.02 * fs))
    kernel = torch.ones(win_len, device=x.device) / win_len
    env = torch.nn.functional.conv1d(
        env.view(1, 1, -1), kernel.view(1, 1, -1), padding=win_len // 2
    ).reshape(-1)
    if env.numel() < 3:
        return 0.0
    threshold = env.mean() * threshold_ratio
    peaks = (env[1:-1] > env[:-2]) & (env[1:-1] > env[2:]) & (env[1:-1] > threshold)
    # ``.reshape(-1)`` guards the single-peak case (a 0-D tensor from
    # ``.squeeze()`` would break the ``indices[1:]`` slice below).
    indices = torch.nonzero(peaks).reshape(-1) + 1
    if indices.numel() == 0:
        return 0.0
    if indices.numel() == 1:
        return float(1) / (x.numel() / fs)
    keep = torch.cat([
        torch.tensor([True], device=x.device),
        (indices[1:] - indices[:-1]) > int(min_gap * fs),
    ])
    syllables = indices[keep]
    return float(syllables.numel()) / (x.numel() / fs)

def temporal_centroid_framewise(audio_data: torch.Tensor, frame_length: int, hop_length: int):
    """
    Computes the per-frame temporal centroid of an audio signal.

    Unlike the MPEG-7 whole-signal :func:`audiofeat.temporal.centroid.temporal_centroid`
    (energy-weighted mean time in seconds), this returns one centroid per
    frame, expressed as a (energy-weighted) sample index *within* each frame
    (range ``[0, frame_length)``).

    Args:
        audio_data (torch.Tensor): The audio signal.
        frame_length (int): The length of each frame in samples.
        hop_length (int): The number of samples to slide the window.

    Returns:
        torch.Tensor: The temporal centroid for each frame.
    """
    frames = frame_signal(audio_data, frame_length, hop_length)
    sample_energy = frames**2

    time_indices = torch.arange(0, frame_length, device=audio_data.device, dtype=torch.float32)

    numerator = torch.sum(sample_energy * time_indices, dim=1)
    denominator = torch.sum(sample_energy, dim=1)

    temporal_centroids = torch.where(denominator != 0, numerator / denominator, torch.zeros_like(numerator))

    return temporal_centroids


# Backward-compatible name. ``temporal_centroid`` has historically referred to
# the frame-based descriptor when imported directly from this module
# (``audiofeat.temporal.rhythm.temporal_centroid``); keep that contract. The
# top-level ``temporal_centroid`` export resolves to the canonical MPEG-7
# whole-signal version in ``centroid.py`` (see temporal/__init__.py import
# ordering). ``temporal_centroid`` is intentionally excluded from this
# module's ``__all__`` so ``from .rhythm import *`` does not shadow the
# MPEG-7 export, while direct imports still work.
temporal_centroid = temporal_centroid_framewise