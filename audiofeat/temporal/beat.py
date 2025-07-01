import torch
import torchaudio
import torchaudio.transforms as T

def beat_track(waveform: torch.Tensor, sample_rate: int, n_fft: int = 2048, hop_length: int = 512,
               tempo_min: float = 60.0, tempo_max: float = 240.0) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Performs beat tracking on an audio waveform to estimate tempo and beat times.

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
    tempo_min : float
        Minimum tempo to consider (BPM).
    tempo_max : float
        Maximum tempo to consider (BPM).

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        A tuple containing:
        - estimated_tempo (torch.Tensor): Estimated tempo in BPM.
        - beat_frames (torch.Tensor): Frame indices of detected beats.

    Notes
    -----
    This is a simplified beat tracking algorithm based on onset detection and
    autocorrelation of the onset detection function (ODF).
    More robust beat tracking algorithms often involve complex comb filtering
    and dynamic programming (e.g., DP-based beat tracking in librosa).
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

    # Compute spectral flux (as an ODF)
    spectral_flux = torch.cat([
        torch.zeros(1, magnitudes.shape[0], device=waveform.device),
        torch.norm(magnitudes[:, 1:] - magnitudes[:, :-1], dim=0, p=2).unsqueeze(0)
    ], dim=1).squeeze(0)

    # Normalize spectral flux
    spectral_flux = (spectral_flux - spectral_flux.min()) / (spectral_flux.max() - spectral_flux.min() + 1e-8)

    # Compute autocorrelation of the ODF to find periodicity (tempo)
    # This is a simplified autocorrelation. For better results, consider
    # using a dedicated autocorrelation function or onset auto-correlation.
    autocorr = torch.nn.functional.conv1d(
        spectral_flux.unsqueeze(0).unsqueeze(0),
        spectral_flux.flip(dims=[0]).unsqueeze(0).unsqueeze(0),
        padding='full'
    ).squeeze()

    # Find peaks in the autocorrelation function within the tempo range
    # Convert tempo (BPM) to frames per beat
    min_frames_per_beat = 60.0 / tempo_max * (sample_rate / hop_length)
    max_frames_per_beat = 60.0 / tempo_min * (sample_rate / hop_length)

    # Search for the strongest peak in the autocorrelation within the valid range
    peak_val = 0.0
    estimated_tempo = 0.0
    best_lag = 0

    for lag in range(int(min_frames_per_beat), int(max_frames_per_beat) + 1):
        if lag < autocorr.shape[0] and autocorr[lag] > peak_val:
            peak_val = autocorr[lag]
            best_lag = lag

    if best_lag > 0:
        estimated_tempo = 60.0 / (best_lag * hop_length / sample_rate)

    # Beat tracking: find peaks in the ODF that align with the estimated tempo
    beat_frames = []
    if estimated_tempo > 0:
        # Simple peak picking with periodicity constraint
        interval_frames = int(round(60.0 / estimated_tempo * (sample_rate / hop_length)))
        if interval_frames > 0:
            for i in range(0, len(spectral_flux), interval_frames):
                # Find local maximum around the expected beat position
                search_window_start = max(0, i - interval_frames // 4)
                search_window_end = min(len(spectral_flux), i + interval_frames // 4)
                if search_window_start < search_window_end:
                    max_val, max_idx = torch.max(spectral_flux[search_window_start:search_window_end], dim=0)
                    beat_frames.append(search_window_start + max_idx.item())

    return torch.tensor(estimated_tempo, dtype=torch.float32, device=waveform.device), torch.tensor(beat_frames, dtype=torch.int64, device=waveform.device)
