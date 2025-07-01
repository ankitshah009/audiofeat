import torch
import torchaudio

def loudness(waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """
    Computes the perceived loudness of an audio waveform.

    Loudness is a psychoacoustic measure that describes the subjective
    intensity of a sound. It is often measured in Sone or LUFS.

    Parameters
    ----------
    waveform : torch.Tensor
        Mono audio waveform tensor. Expected shape: (num_samples,) or (1, num_samples).
    sample_rate : int
        Sampling rate of the waveform.

    Returns
    -------
    torch.Tensor
        A scalar tensor representing the integrated loudness in LUFS.

    Notes
    -----
    This implementation uses torchaudio's `Loudness` transform (EBU R 128).
    Requires 'torch' and 'torchaudio' to be installed.
    """
    if waveform.ndim > 1 and waveform.shape[0] > 1:
        waveform = waveform[0]
    elif waveform.ndim == 0:
        raise ValueError("Input waveform cannot be a scalar.")

    # Ensure waveform is float32
    waveform = waveform.to(torch.float32)

    # Torchaudio's Loudness expects a batch dimension and channel dimension
    # (batch, channel, samples)
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0).unsqueeze(0) # Add batch and channel dims
    elif waveform.ndim == 2 and waveform.shape[0] == 1: # (1, samples)
        waveform = waveform.unsqueeze(0) # Add batch dim

    # Create the Loudness transform (EBU R 128)
    # The `Loudness` transform computes integrated loudness in LUFS.
    loudness_transform = torchaudio.transforms.Loudness(sample_rate=sample_rate)

    # Compute loudness
    integrated_loudness = loudness_transform(waveform)

    return integrated_loudness
