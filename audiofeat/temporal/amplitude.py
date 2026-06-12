
import torch
from ..temporal.rms import frame_signal

__all__ = ["amplitude_modulation_depth"]


def amplitude_modulation_depth(env: torch.Tensor, window: int):
    """Amplitude modulation depth over a sliding window.

    For each non-overlapping block of ``window`` samples, the modulation
    depth is computed as ``(max - min) / (max + min)`` and the result is
    averaged across blocks.

    Parameters
    ----------
    env : torch.Tensor
        A **smoothed amplitude envelope** (non-negative), e.g. the output
        of an RMS/Hilbert envelope follower — *not* the raw waveform.
        Passing a raw (zero-mean) signal makes ``min`` approach
        ``-max`` and the ratio degenerates; this function assumes the
        per-block extrema describe the envelope's modulation, not the
        carrier's oscillation.
    window : int
        Block length in samples. To recover the true modulation index of a
        sinusoidally modulated envelope, each block must span at least one
        full modulation period so both the envelope peak and trough fall
        inside it.

    Returns
    -------
    torch.Tensor
        Scalar mean modulation depth. For an envelope of the form
        ``1 + m * sin(2*pi*f_m*t)`` with ``0 <= m <= 1`` and a block long
        enough to contain a full period, the per-block depth is
        ``(max - min)/(max + min) = ((1+m) - (1-m)) / ((1+m) + (1-m)) = m``,
        so the mean approaches ``m``. Returns ``0.0`` if the envelope is
        shorter than ``window``.
    """
    if env.numel() < window:
        return torch.tensor(0.0, device=env.device)
    frames = frame_signal(env, window, window)
    max_e = frames.max(dim=1).values
    min_e = frames.min(dim=1).values
    return ((max_e - min_e) / (max_e + min_e + 1e-8)).mean()
