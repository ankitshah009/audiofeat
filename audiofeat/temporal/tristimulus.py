import torch
import torchaudio
import torchaudio.transforms as T

def tristimulus(waveform: torch.Tensor, sample_rate: int, n_fft: int = 2048, hop_length: int = 512) -> torch.Tensor:
    """
    Computes the tristimulus of an audio waveform.

    Tristimulus is a measure of the spectral balance of a sound,
    often used in musical acoustics. It is based on the relative
    strengths of the fundamental, the second harmonic, and the
    higher harmonics.

    Parameters
    ----------
    waveform : torch.Tensor
        Mono audio waveform tensor. Expected shape: (num_samples,) or (1, num_samples).
    sample_rate : int
        Sampling rate of the waveform.
    n_fft : int
        Size of the FFT window.
    hop_length : int
        Number of samples between successive frames.

    Returns
    -------
    torch.Tensor
        A tensor of shape (3,) representing the tristimulus values (T1, T2, T3).

    Notes
    -----
    This implementation assumes the input waveform is a single channel.
    For multi-channel audio, you might need to process each channel
    separately or average them.
    Requires 'torch' and 'torchaudio' to be installed.
    """
    if waveform.ndim > 1 and waveform.shape[0] > 1:
        waveform = waveform[0]
    elif waveform.ndim == 0:
        raise ValueError("Input waveform cannot be a scalar.")

    # Compute the Short-Time Fourier Transform (STFT)
    # Using torchaudio's Spectrogram for convenience
    spectrogram_transform = T.Spectrogram(
        n_fft=n_fft,
        hop_length=hop_length,
        power=2,  # Power spectrogram
    )
    spec = spectrogram_transform(waveform)

    # Get the magnitudes (amplitude) from the power spectrogram
    magnitudes = torch.sqrt(spec)

    # Use average spectrum over time to estimate the dominant frequency bin.
    mean_spectrum = magnitudes.mean(dim=1)
    fundamental_bin = int(torch.argmax(mean_spectrum).item())

    # Calculate the energy of the fundamental (T1)
    # Sum of magnitudes in the fundamental frequency bin
    T1 = magnitudes[fundamental_bin, :].sum()

    # Calculate the energy of the second harmonic (T2)
    # Assuming the second harmonic is at 2 * fundamental_bin
    second_harmonic_bin = min(2 * fundamental_bin, magnitudes.shape[0] - 1)
    T2 = magnitudes[second_harmonic_bin, :].sum()

    # Calculate the energy of the remaining harmonics (T3)
    # Sum of magnitudes of all other harmonics
    # This is a very simplified approach. A proper tristimulus calculation
    # would involve summing specific harmonic regions.
    total_energy = magnitudes.sum()
    T3 = total_energy - T1 - T2

    # Normalize the tristimulus values
    total_tristimulus = T1 + T2 + T3
    if total_tristimulus == 0:
        return torch.zeros(3, device=waveform.device)
    else:
        return torch.tensor([T1, T2, T3], device=waveform.device) / total_tristimulus
