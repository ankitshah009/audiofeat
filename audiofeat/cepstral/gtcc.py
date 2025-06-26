
import torch
import torchaudio.transforms as T

def _gammatone_filterbank(n_filters: int, sample_rate: int, n_fft: int, fmin: float = 50.0, fmax: float = 8000.0):
    """
    Creates a Gammatone-like filterbank.
    This is a simplified approximation and not a full, psychoacoustically accurate Gammatone filterbank.
    It creates triangular filters on a linear frequency scale, similar to Mel but with Gammatone-like spacing.
    """
    # Calculate center frequencies on a logarithmic scale (approximating ERB scale)
    # Using a simple log spacing for demonstration
    min_log_freq = torch.log(torch.tensor(fmin))
    max_log_freq = torch.log(torch.tensor(fmax))
    center_freqs = torch.exp(torch.linspace(min_log_freq, max_log_freq, n_filters))

    # Convert center frequencies to FFT bin numbers
    fft_bins = torch.linspace(0, sample_rate / 2, n_fft // 2 + 1)
    
    filter_bank = torch.zeros(n_filters, n_fft // 2 + 1)

    for i in range(n_filters):
        # Define the width of the filter (simplified)
        # A more accurate Gammatone filter would have a specific bandwidth relationship
        if i == 0:
            left_freq = fmin
        else:
            left_freq = (center_freqs[i-1] + center_freqs[i]) / 2
        
        if i == n_filters - 1:
            right_freq = fmax
        else:
            right_freq = (center_freqs[i] + center_freqs[i+1]) / 2

        # Find the corresponding FFT bin indices
        left_bin = torch.argmin(torch.abs(fft_bins - left_freq))
        center_bin = torch.argmin(torch.abs(fft_bins - center_freqs[i]))
        right_bin = torch.argmin(torch.abs(fft_bins - right_freq))

        # Create triangular filter shape
        if center_bin > left_bin:
            filter_bank[i, left_bin:center_bin+1] = torch.linspace(0, 1, center_bin - left_bin + 1)
        if right_bin > center_bin:
            filter_bank[i, center_bin:right_bin+1] = torch.linspace(1, 0, right_bin - center_bin + 1)

    return filter_bank

def _dct(x: torch.Tensor, n_coeffs: int, norm: str = 'ortho'):
    """
    Computes the Discrete Cosine Transform (DCT) of a tensor.
    Equivalent to torchaudio.transforms.DCT(n_coeffs, norm=norm)
    """
    # Type II DCT
    N = x.shape[-1]
    n = torch.arange(N, device=x.device)
    k = torch.arange(n_coeffs, device=x.device).unsqueeze(-1)
    
    # DCT matrix
    basis = torch.cos(torch.pi / (2 * N) * (2 * n + 1) * k)
    
    if norm == 'ortho':
        basis[0, :] *= 1 / torch.sqrt(torch.tensor(2.0))
        basis *= torch.sqrt(torch.tensor(2.0 / N)) # Should be sqrt(2/N) for ortho
    
    return torch.matmul(x, basis.T)


def gtcc(audio_data: torch.Tensor, sample_rate: int, n_gtcc: int = 20, n_fft: int = 2048, hop_length: int = 512, n_bands: int = 128):
    """
    Computes the Gammatone Cepstral Coefficients (GTCCs) of an audio signal.

    Args:
        audio_data (torch.Tensor): The audio signal.
        sample_rate (int): The sample rate of the audio.
        n_gtcc (int): The number of GTCCs to compute.
        n_fft (int): The number of FFT points.
        hop_length (int): The number of samples to slide the window.
        n_bands (int): The number of Gammatone filterbank bands.

    Returns:
        torch.Tensor: The GTCCs.
    """
    # Compute STFT magnitude spectrogram
    spectrogram_transform = T.Spectrogram(n_fft=n_fft, hop_length=hop_length, window_fn=torch.hann_window, return_complex=True)
    magnitude_spectrogram = torch.abs(spectrogram_transform(audio_data))

    # Create Gammatone-like filterbank
    gammatone_filter_bank = _gammatone_filterbank(n_bands, sample_rate, n_fft).to(audio_data.device)

    # Apply filterbank to magnitude spectrogram
    filtered_spectrogram = torch.matmul(gammatone_filter_bank, magnitude_spectrogram)

    # Take logarithm of energies
    log_energies = torch.log(filtered_spectrogram + 1e-8)

    # Apply Discrete Cosine Transform (DCT)
    gtccs = _dct(log_energies.T, n_gtcc) # Transpose to (time_frames, n_bands) for DCT

    return gtccs
