import torch
import torchaudio

def temporal_centroid(waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """
    Computes the temporal centroid of an audio waveform.

    The temporal centroid is a measure of the "center of mass" of the
    signal's amplitude envelope. A higher value indicates that the
    energy of the sound is concentrated towards the end of the sound.

    Parameters
    ----------
    waveform : torch.Tensor
        Mono audio waveform tensor. Expected shape: (num_samples,) or (1, num_samples).
    sample_rate : int
        Sampling rate of the waveform.

    Returns
    -------
    torch.Tensor
        A scalar tensor representing the temporal centroid.

    Notes
    -----
    This implementation assumes the input waveform is a single channel.
    For multi-channel audio, you might need to process each channel
    separately or average them.
    Requires 'torch' and 'torchaudio' to be installed.
    """
    if waveform.ndim > 1 and waveform.shape[0] > 1:
        # Assuming mono or taking the first channel if multi-channel
        waveform = waveform[0]
    elif waveform.ndim == 0:
        raise ValueError("Input waveform cannot be a scalar.")

    # Calculate the amplitude envelope (absolute value of the waveform)
    amplitude_envelope = torch.abs(waveform)

    # Create a time index tensor
    time_indices = torch.arange(len(amplitude_envelope), dtype=torch.float32, device=waveform.device)

    # Calculate the weighted sum of time indices by amplitude
    weighted_sum = torch.sum(time_indices * amplitude_envelope)

    # Calculate the sum of amplitudes
    sum_amplitudes = torch.sum(amplitude_envelope)

    if sum_amplitudes == 0:
        return torch.tensor(0.0, device=waveform.device) # Avoid division by zero

    # Temporal Centroid = (sum of (time_index * amplitude)) / (sum of amplitudes)
    temporal_c = weighted_sum / sum_amplitudes

    # Convert to seconds if desired, by dividing by sample_rate,
    # but typically it's returned in samples or normalized.
    # For consistency with some definitions, we return in samples.
    return temporal_c
