import torch
import torchaudio.transforms as T
from ..temporal.rms import frame_signal, hann_window

def _onset_detection_function(audio_data: torch.Tensor, n_fft: int, hop_length: int):
    """
    Computes a simple onset detection function (ODF) based on spectral flux.
    """
    spectrogram_transform = T.Spectrogram(n_fft=n_fft, hop_length=hop_length, window_fn=torch.hann_window, return_complex=True)
    magnitude_spectrogram = torch.abs(spectrogram_transform(audio_data))

    # Spectral flux (half-wave rectified difference)
    # Ensure magnitude_spectrogram has at least 2 time frames for diff
    if magnitude_spectrogram.shape[1] < 2:
        return torch.zeros(magnitude_spectrogram.shape[1], device=audio_data.device)

    flux = torch.max(
        torch.zeros_like(magnitude_spectrogram[:, 1:]),
        magnitude_spectrogram[:, 1:] - magnitude_spectrogram[:, :-1]
    )
    
    return torch.sum(flux, dim=0) # Sum across frequency bins

def tempo(audio_data: torch.Tensor, sample_rate: int, n_fft: int = 2048, hop_length: int = 512):
    """
    Estimates the tempo (BPM) of an audio signal.

    Args:
        audio_data (torch.Tensor): The audio signal.
        sample_rate (int): The sample rate of the audio.
        n_fft (int): The number of FFT points.
        hop_length (int): The number of samples to slide the window.

    Returns:
        float: The estimated tempo in BPM.
    """
    odf = _onset_detection_function(audio_data, n_fft, hop_length)

    if odf.numel() < 2: # Need at least two points for autocorrelation
        return 0.0

    # Autocorrelation of the ODF
    # Pad ODF to avoid circular convolution issues for autocorrelation
    padded_odf = torch.nn.functional.pad(odf, (0, odf.numel()))
    ODF_fft = torch.fft.fft(padded_odf)
    autocorr_odf = torch.fft.ifft(ODF_fft * torch.conj(ODF_fft)).real

    # Find peaks in autocorrelation to identify periodicities
    # Search for peaks in a reasonable tempo range (e.g., 60-240 BPM)
    min_lag_samples = int(sample_rate * 60 / 240 / hop_length) # Max BPM = 240
    max_lag_samples = int(sample_rate * 60 / 60 / hop_length)  # Min BPM = 60

    # Ensure lag ranges are valid
    if max_lag_samples >= autocorr_odf.numel():
        max_lag_samples = autocorr_odf.numel() - 1
    if min_lag_samples >= max_lag_samples or min_lag_samples < 0:
        return 0.0 # Cannot determine tempo

    # Find the strongest peak in the relevant lag range
    relevant_autocorr = autocorr_odf[min_lag_samples:max_lag_samples]
    if relevant_autocorr.numel() == 0:
        return 0.0

    peak_idx = torch.argmax(relevant_autocorr) + min_lag_samples

    # Convert lag (in frames) to BPM
    tempo_bpm = (sample_rate * 60) / (peak_idx * hop_length)

    return tempo_bpm.item()

def beat_track(audio_data: torch.Tensor, sample_rate: int, n_fft: int = 2048, hop_length: int = 512):
    """
    Performs simple beat tracking on an audio signal.

    Args:
        audio_data (torch.Tensor): The audio signal.
        sample_rate (int): The sample rate of the audio.
        n_fft (int): The number of FFT points.
        hop_length (int): The number of samples to slide the window.

    Returns:
        torch.Tensor: A tensor of beat times in seconds.
    """
    odf = _onset_detection_function(audio_data, n_fft, hop_length)
    estimated_tempo = tempo(audio_data, sample_rate, n_fft, hop_length)

    if estimated_tempo == 0.0 or odf.numel() == 0:
        return torch.tensor([])

    # Simple beat tracking: find peaks in ODF that align with the estimated tempo
    # This is a very basic beat tracking and can be improved with dynamic programming
    # or comb filtering.

    beat_interval_frames = (sample_rate * 60 / estimated_tempo) / hop_length

    beat_times = []
    current_time_frame = 0
    while current_time_frame < odf.numel():
        beat_times.append(current_time_frame * hop_length / sample_rate)
        current_time_frame += int(beat_interval_frames)

    return torch.tensor(beat_times)