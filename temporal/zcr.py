import torch

def zcr(waveform: torch.Tensor, frame_length: int = 2048, hop_length: int = 512) -> torch.Tensor:
    """
    Computes the Zero-Crossing Rate (ZCR) of an audio waveform.

    ZCR is the rate at which the signal changes sign, i.e., the number of times
    the signal crosses the zero axis. It's a simple yet effective feature for
    distinguishing between voiced and unvoiced speech, or for characterizing
    percussive sounds.

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
        ZCR per frame.

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

    # Calculate ZCR for each frame
    # ZCR = 0.5 * sum(|sgn(x[n]) - sgn(x[n-1])|)
    # We use torch.sign to get the sign of each sample
    # and then count the differences.
    zcr_values = torch.sum(torch.abs(torch.sign(frames[:, 1:]) - torch.sign(frames[:, :-1])), dim=1) * 0.5

    return zcr_values
