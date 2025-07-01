import torch
import torchaudio
import torchaudio.transforms as T

def cqt(waveform: torch.Tensor, sample_rate: int, hop_length: int = 512, f_min: float = 32.70, n_bins: int = 84, bins_per_octave: int = 12) -> torch.Tensor:
    """
    Computes the Constant-Q Transform (CQT) of an audio waveform.

    The CQT is a time-frequency representation that uses a logarithmically spaced
    frequency axis, making it well-suited for musical analysis where pitch
    relationships are logarithmic.

    Parameters
    ----------
    waveform : torch.Tensor
        Mono audio waveform tensor. Expected shape: (num_samples,) or (1, num_samples).
    sample_rate : int
        Sampling rate of the waveform.
    hop_length : int
        Number of samples between successive frames.
    f_min : float
        Minimum frequency (Hz) for the lowest CQT bin (e.g., C1 = 32.70 Hz).
    n_bins : int
        Number of CQT bins.
    bins_per_octave : int
        Number of bins per octave.

    Returns
    -------
    torch.Tensor
        CQT magnitude spectrogram, shape (n_bins, num_frames).

    Notes
    -----
    This implementation uses torchaudio's `CQT` transform.
    Requires 'torch' and 'torchaudio' to be installed.
    """
    if waveform.ndim > 1 and waveform.shape[0] > 1:
        waveform = waveform[0]
    elif waveform.ndim == 0:
        raise ValueError("Input waveform cannot be a scalar.")

    # Ensure waveform is float32
    waveform = waveform.to(torch.float32)

    # Create the CQT transform
    cqt_transform = T.CQT(
        sample_rate=sample_rate,
        hop_length=hop_length,
        f_min=f_min,
        n_bins=n_bins,
        bins_per_octave=bins_per_octave,
        # For magnitude, we typically take the absolute value of the complex CQT output
        # The transform itself returns complex numbers, so we'll take abs later.
    )

    # Compute CQT
    cqt_output = cqt_transform(waveform)

    # Return magnitude spectrogram
    return torch.abs(cqt_output)
