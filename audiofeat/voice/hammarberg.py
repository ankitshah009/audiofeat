
import torch
from ..temporal.rms import frame_signal, hann_window

def hammarberg_index(x: torch.Tensor, fs: int, n_fft: int = 2048,
                     frame_length_ms: float = 60.0, hop_length_ms: float = 20.0):
    """
    Computes the Hammarberg Index: ratio of the highest energy peak in 0-2 kHz
    to the highest energy peak in 2-5 kHz (dB).

    The signal is framed (default 60 ms / 20 ms hop) and the average power
    spectrum over frames is used. This avoids the previous behaviour of
    windowing the whole signal and truncating it to the first ``n_fft`` samples.

    Args:
        x (torch.Tensor): The audio signal.
        fs (int): The sample rate of the audio.
        n_fft (int): The number of FFT points (>= frame length).
        frame_length_ms (float): Analysis frame length in milliseconds.
        hop_length_ms (float): Hop length in milliseconds.

    Returns:
        torch.Tensor: The Hammarberg Index (dB).
    """
    x = x.flatten().float()
    frame_length = max(int(fs * frame_length_ms / 1000), 1)
    hop_length = max(int(fs * hop_length_ms / 1000), 1)
    n_fft = max(n_fft, frame_length)

    frames = frame_signal(x, frame_length, hop_length)
    window = hann_window(frame_length).to(x.device)
    X = torch.fft.rfft(frames * window, n=n_fft, dim=1)
    P = (X.abs() ** 2).mean(dim=0)  # average power spectrum over frames
    freqs = torch.linspace(0, fs / 2, P.numel(), device=x.device)

    low_band_mask = (freqs >= 0) & (freqs < 2000)
    high_band_mask = (freqs >= 2000) & (freqs < 5000)

    low_band_peak = torch.max(P[low_band_mask]) if low_band_mask.sum() > 0 else torch.tensor(1e-8, device=x.device)
    high_band_peak = torch.max(P[high_band_mask]) if high_band_mask.sum() > 0 else torch.tensor(1e-8, device=x.device)

    return 10 * torch.log10((low_band_peak + 1e-8) / (high_band_peak + 1e-8))
