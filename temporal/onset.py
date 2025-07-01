import torch
import torchaudio
import torchaudio.transforms as T

def onset_detect(waveform: torch.Tensor, sample_rate: int, n_fft: int = 2048, hop_length: int = 512,
                 backtrack: bool = True) -> torch.Tensor:
    """
    Computes an onset detection function (ODF) and optionally backtracks to find precise onset times.

    Onset detection is the task of identifying the precise time points at which
    new musical events (e.g., notes, drum hits) occur in an audio signal.

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
    backtrack : bool
        If True, backtracks from peaks in the ODF to find the precise onset time.

    Returns
    -------
    torch.Tensor
        A tensor of onset times in seconds.

    Notes
    -----
    This implementation uses a basic spectral flux onset detection function.
    More advanced ODFs exist (e.g., complex domain, phase-based).
    The backtracking mechanism is a simple search for the local minimum before a peak.
    Requires 'torch' and 'torchaudio' to be installed.
    """
    if waveform.ndim > 1 and waveform.shape[0] > 1:
        waveform = waveform[0]
    elif waveform.ndim == 0:
        raise ValueError("Input waveform cannot be a scalar.")

    # Compute the STFT magnitude spectrogram
    spectrogram_transform = T.Spectrogram(
        n_fft=n_fft,
        hop_length=hop_length,
        power=2.0, # Power spectrogram
    )
    spec = spectrogram_transform(waveform)
    magnitudes = torch.sqrt(spec) # Amplitude spectrogram

    # Compute spectral flux (a common ODF)
    # Spectral flux is the L2 norm of the difference between successive frames.
    spectral_flux = torch.cat([
        torch.zeros(1, magnitudes.shape[0], device=waveform.device),
        torch.norm(magnitudes[:, 1:] - magnitudes[:, :-1], dim=0, p=2).unsqueeze(0)
    ], dim=1).squeeze(0)

    # Normalize spectral flux (optional, but can help with peak picking)
    spectral_flux = (spectral_flux - spectral_flux.min()) / (spectral_flux.max() - spectral_flux.min() + 1e-8)

    # Find peaks in the ODF
    # A simple peak picking: find points where the ODF is greater than its neighbors.
    # This can be improved with thresholding and local maxima finding algorithms.
    onsets_frames = []
    for i in range(1, len(spectral_flux) - 1):
        if spectral_flux[i] > spectral_flux[i-1] and spectral_flux[i] > spectral_flux[i+1] and spectral_flux[i] > 0.1: # Simple threshold
            onsets_frames.append(i)

    onsets_seconds = []
    for frame_idx in onsets_frames:
        time_in_seconds = frame_idx * hop_length / sample_rate

        if backtrack:
            # Backtrack to find the local minimum before the peak
            start_search_idx = max(0, frame_idx - int(sample_rate * 0.1 / hop_length)) # Search 100ms before
            min_val = spectral_flux[start_search_idx]
            min_idx = start_search_idx
            for i in range(start_search_idx + 1, frame_idx):
                if spectral_flux[i] < min_val:
                    min_val = spectral_flux[i]
                    min_idx = i
            time_in_seconds = min_idx * hop_length / sample_rate

        onsets_seconds.append(time_in_seconds)

    return torch.tensor(onsets_seconds, dtype=torch.float32, device=waveform.device)
