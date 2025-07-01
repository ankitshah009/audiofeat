import torch

def rms(waveform: torch.Tensor, frame_length: int = 2048, hop_length: int = 512) -> torch.Tensor:
    """
    Computes the Root Mean Square (RMS) energy of an audio waveform.

    RMS energy is a measure of the loudness of an audio signal. It is
    calculated over short frames of the audio.

    Parameters
    ----------
    waveform : torch.Tensor
        Mono audio waveform tensor. Expected shape: (num_samples,) or (1, num_samples).
    frame_length : int
        Length of the analysis frame in samples.
    hop_length : int
        Number of samples between successive frames.

    Returns
    -------
    torch.Tensor
        RMS energy per frame.

    Notes
    -----
    This implementation uses torch for calculations.
    """
    if waveform.ndim > 1 and waveform.shape[0] > 1:
        waveform = waveform[0]
    elif waveform.ndim == 0:
        raise ValueError("Input waveform cannot be a scalar.")

    # Ensure waveform is float for calculations
    waveform = waveform.to(torch.float32)

    # Pad the waveform to ensure full frames
    pad_amount = frame_length - (len(waveform) - frame_length) % hop_length
    if pad_amount > 0 and pad_amount != frame_length:
        waveform = torch.nn.functional.pad(waveform, (0, pad_amount))

    # Unfold the waveform into frames
    frames = waveform.unfold(0, frame_length, hop_length)

    # Calculate RMS for each frame
    # RMS = sqrt(mean(x^2))
    rms_energy = torch.sqrt(torch.mean(frames**2, dim=1))

    return rms_energy
