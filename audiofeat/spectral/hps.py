import torch
import torchaudio
import torchaudio.transforms as T

def hps(waveform: torch.Tensor, sample_rate: int, n_fft: int = 2048, hop_length: int = 512,
        margin_h: float = 3.0, margin_p: float = 3.0) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Performs Harmonic-Percussive Separation (HPS) on an audio waveform.

    Separates an audio signal into its harmonic and percussive components
    using median filtering on the magnitude spectrogram.

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
    margin_h : float
        Margin for harmonic median filter (frequency axis).
    margin_p : float
        Margin for percussive median filter (time axis).

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        A tuple containing:
        - harmonic_waveform (torch.Tensor): The separated harmonic component.
        - percussive_waveform (torch.Tensor): The separated percussive component.

    Notes
    -----
    This implementation uses median filtering on the magnitude spectrogram.
    The median filter is applied along the frequency axis for harmonic
    components and along the time axis for percussive components.
    Requires 'torch' and 'torchaudio' to be installed.
    The median filtering is a basic, loop-based implementation for clarity
    and to avoid complex `torch.nn.functional.unfold` patterns for 1D median
    filtering without external dependencies.
    """
    if waveform.ndim > 1 and waveform.shape[0] > 1:
        waveform = waveform[0]
    elif waveform.ndim == 0:
        raise ValueError("Input waveform cannot be a scalar.")

    # Compute STFT
    stft_transform = T.Spectrogram(
        n_fft=n_fft,
        hop_length=hop_length,
        return_complex=True
    )
    stft_matrix = stft_transform(waveform)
    magnitude_spectrogram = torch.abs(stft_matrix)
    phase_spectrogram = torch.angle(stft_matrix)

    # Apply median filtering for harmonic and percussive masks

    # Harmonic mask (median filter along frequency axis)
    harmonic_mask = torch.zeros_like(magnitude_spectrogram)
    for i in range(magnitude_spectrogram.shape[1]): # Iterate over time frames
        kernel_size_h = int(margin_h * 2 + 1) # Odd kernel size
        if kernel_size_h > magnitude_spectrogram.shape[0]:
            kernel_size_h = magnitude_spectrogram.shape[0]
        if kernel_size_h % 2 == 0: # Ensure odd
            kernel_size_h += 1

        padded_freq = torch.nn.functional.pad(
            magnitude_spectrogram[:, i].unsqueeze(0).unsqueeze(0),
            (kernel_size_h // 2, kernel_size_h // 2),
            mode='reflect',
        ).squeeze(0).squeeze(0)
        for j in range(magnitude_spectrogram.shape[0]): # Iterate over frequency bins
            window = padded_freq[j : j + kernel_size_h]
            harmonic_mask[j, i] = torch.median(window)

    # Percussive mask (median filter along time axis)
    percussive_mask = torch.zeros_like(magnitude_spectrogram)
    for i in range(magnitude_spectrogram.shape[0]): # Iterate over frequency bins
        kernel_size_p = int(margin_p * 2 + 1) # Odd kernel size
        if kernel_size_p > magnitude_spectrogram.shape[1]:
            kernel_size_p = magnitude_spectrogram.shape[1]
        if kernel_size_p % 2 == 0: # Ensure odd
            kernel_size_p += 1

        padded_time = torch.nn.functional.pad(
            magnitude_spectrogram[i, :].unsqueeze(0).unsqueeze(0),
            (kernel_size_p // 2, kernel_size_p // 2),
            mode='reflect',
        ).squeeze(0).squeeze(0)
        for j in range(magnitude_spectrogram.shape[1]): # Iterate over time frames
            window = padded_time[j : j + kernel_size_p]
            percussive_mask[i, j] = torch.median(window)

    # Soft masking (power law)
    harmonic_median = harmonic_mask
    percussive_median = percussive_mask
    denom = harmonic_median + percussive_median + 1e-8
    harmonic_mask = (harmonic_median / denom) ** 2
    percussive_mask = (percussive_median / denom) ** 2

    # Apply masks to the original STFT magnitude
    harmonic_spectrogram = magnitude_spectrogram * harmonic_mask
    percussive_spectrogram = magnitude_spectrogram * percussive_mask

    # Reconstruct complex spectrograms
    harmonic_stft = harmonic_spectrogram * torch.exp(1j * phase_spectrogram)
    percussive_stft = percussive_spectrogram * torch.exp(1j * phase_spectrogram)

    # Inverse STFT to get time-domain waveforms
    istft_transform = T.InverseSpectrogram(
        n_fft=n_fft,
        hop_length=hop_length,
    )
    harmonic_waveform = istft_transform(harmonic_stft)
    percussive_waveform = istft_transform(percussive_stft)

    return harmonic_waveform, percussive_waveform
