import torch

__all__ = ["temporal_centroid"]


def temporal_centroid(waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """
    Compute the MPEG-7 temporal centroid of an audio waveform.

    The temporal centroid is the energy-weighted mean **time** (the
    "center of mass" of the signal-energy envelope along the time axis),
    returned in **seconds**:

    ``TC = sum_n( t[n] * env[n] ) / sum_n( env[n] )``

    where ``t[n] = n / sample_rate`` and ``env[n]`` is the local energy
    (``waveform[n] ** 2``). A higher value means the energy is concentrated
    later in the sound. This matches the MPEG-7 ``TemporalCentroid``
    descriptor, which is defined in seconds using the sampling rate.

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
        A scalar tensor: the temporal centroid in seconds (0.0 for a
        silent / zero-energy signal).

    Notes
    -----
    The previous implementation ignored ``sample_rate`` and weighted by
    ``|waveform|`` (amplitude) in **samples**; this version follows MPEG-7
    by weighting with energy (``waveform ** 2``) and scaling the time axis
    by ``1 / sample_rate`` so the result is in seconds.
    """
    if waveform.ndim > 1 and waveform.shape[0] > 1:
        # Multi-channel: use the first channel.
        waveform = waveform[0]
    elif waveform.ndim == 0:
        raise ValueError("Input waveform cannot be a scalar.")

    waveform = waveform.flatten().float()
    if sample_rate <= 0:
        raise ValueError("sample_rate must be > 0.")

    # Signal energy envelope (MPEG-7 weights by energy, not |amplitude|).
    energy_env = waveform ** 2

    total_energy = torch.sum(energy_env)
    if total_energy == 0:
        return torch.tensor(0.0, device=waveform.device)  # Avoid division by zero.

    # Time index in SECONDS (this is the MPEG-7 definition; uses sample_rate).
    time_seconds = torch.arange(
        waveform.numel(), dtype=torch.float32, device=waveform.device
    ) / float(sample_rate)

    temporal_c = torch.sum(time_seconds * energy_env) / total_energy
    return temporal_c
