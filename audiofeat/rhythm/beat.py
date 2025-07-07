
import torch
import torchaudio
from ..spectral.spectrogram import linear_spectrogram as spectrogram

def beat_detection(signal: torch.Tensor, sample_rate: int, window_size: float = 0.05, hop_size: float = 0.025):
    """
    Estimates the beat rate of a music signal.

    Args:
        signal (torch.Tensor): The input audio signal.
        sample_rate (int): The sample rate of the audio signal.
        window_size (float, optional): The size of the analysis window in seconds. Defaults to 0.05.
        hop_size (float, optional): The hop size between consecutive windows in seconds. Defaults to 0.025.

    Returns:
        Tuple[float, float]: A tuple containing the estimated BPM and a confidence measure.
    """
    # 1. Get the spectrogram
    n_fft = int(window_size * sample_rate)
    hop_length = int(hop_size * sample_rate)
    spec = spectrogram(signal, n_fft=n_fft, hop_length=hop_length)
    
    # 2. Onset detection (simplified version using spectral flux)
    onset_env = torch.sum(torch.diff(spec, dim=1), dim=0)
    onset_env = torch.cat((torch.tensor([0.0]), onset_env)) # Pad with a zero at the beginning

    # 3. Find peaks in the onset detection function
    # A simple peak picking logic
    peaks = (onset_env > torch.roll(onset_env, 1, 0)) & (onset_env > torch.roll(onset_env, -1, 0))
    peak_indices = torch.where(peaks)[0]

    if len(peak_indices) < 2:
        return 0.0, 0.0

    # 4. Calculate inter-onset intervals (IOIs)
    iois = torch.diff(peak_indices) * hop_size

    # 5. Create a histogram of IOIs to find the dominant tempo
    # A simple approach is to find the median IOI
    median_ioi = torch.median(iois)
    
    if median_ioi == 0:
        return 0.0, 0.0

    bpm = 60.0 / median_ioi.item()

    # Confidence can be calculated based on the variance of IOIs
    confidence = 1.0 - torch.std(iois) / torch.mean(iois)

    return bpm, confidence.item()
