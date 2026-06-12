import torch
import torchaudio

__all__ = ["loudness"]


def loudness(waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """
    Compute integrated loudness in LUFS (ITU-R BS.1770 / EBU R 128).

    This measures program loudness using the BS.1770 algorithm: a
    K-weighting pre-filter followed by mean-square gating, yielding
    Loudness Units relative to Full Scale (LUFS, a.k.a. LKFS). It is a
    broadcast loudness standard, **not** a psychoacoustic loudness model
    such as Sone / Stevens' or Zwicker loudness (those integrate specific
    loudness across critical bands and are not what this function returns).

    Parameters
    ----------
    waveform : torch.Tensor
        Mono audio waveform. Shape ``(num_samples,)`` or ``(1, num_samples)``
        (the first channel is used for multi-channel input).
    sample_rate : int
        Sampling rate of the waveform, in Hz.

    Returns
    -------
    torch.Tensor
        Integrated loudness in LUFS.

    Notes
    -----
    - Backed by :class:`torchaudio.transforms.Loudness` (BS.1770 / EBU R 128).
    - For digital silence (or signal below the absolute gating threshold)
      the integrated loudness is mathematically ``-inf`` LUFS; depending on
      the torchaudio build the transform may return ``-inf`` or ``nan``.
      This function returns that value unchanged. Callers that need a finite,
      non-``nan`` floor should sanitize the result (e.g.
      ``torch.nan_to_num(loud, neginf=-70.0, nan=-70.0)``).
    - The numeric result is identical to prior versions; only the
      documentation and short/empty-input guards changed.
    """
    if waveform.ndim == 0:
        raise ValueError("Input waveform cannot be a scalar.")
    if waveform.ndim > 1 and waveform.shape[0] > 1:
        waveform = waveform[0]

    if waveform.numel() == 0:
        raise ValueError("Input waveform must be non-empty.")

    # Ensure waveform is float32.
    waveform = waveform.to(torch.float32)

    # torchaudio's Loudness expects (batch, channel, samples).
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0).unsqueeze(0)  # add batch and channel dims
    elif waveform.ndim == 2 and waveform.shape[0] == 1:  # (1, samples)
        waveform = waveform.unsqueeze(0)  # add batch dim

    # Create the Loudness transform (BS.1770 / EBU R 128).
    loudness_transform = torchaudio.transforms.Loudness(sample_rate=sample_rate)

    integrated_loudness = loudness_transform(waveform)
    return integrated_loudness
