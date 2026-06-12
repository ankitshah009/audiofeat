
import torch
from ..temporal.rms import frame_signal, hann_window

def alpha_ratio(x: torch.Tensor, fs: int, n_fft: int = 2048,
                frame_length_ms: float = 60.0, hop_length_ms: float = 20.0):
    """
    Computes the Alpha Ratio: ratio of energy in 50-1000 Hz to 1000-5000 Hz (dB).

    The signal is framed (default 60 ms / 20 ms hop) and band energies are summed
    over frames before the ratio is taken. This avoids the previous behaviour of
    windowing the whole signal with a full-length Hann window and then truncating
    it to the first ``n_fft`` samples (which silently discarded all but the first
    frame's worth of, taper-attenuated, samples).

    Args:
        x (torch.Tensor): The audio signal.
        fs (int): The sample rate of the audio.
        n_fft (int): The number of FFT points (>= frame length).
        frame_length_ms (float): Analysis frame length in milliseconds.
        hop_length_ms (float): Hop length in milliseconds.

    Returns:
        torch.Tensor: The Alpha Ratio (dB).
    """
    x = x.flatten().float()
    frame_length = max(int(fs * frame_length_ms / 1000), 1)
    hop_length = max(int(fs * hop_length_ms / 1000), 1)
    n_fft = max(n_fft, frame_length)

    frames = frame_signal(x, frame_length, hop_length)
    window = hann_window(frame_length).to(x.device)
    X = torch.fft.rfft(frames * window, n=n_fft, dim=1)
    P = (X.abs() ** 2).sum(dim=0)  # average (sum) power spectrum over frames
    freqs = torch.linspace(0, fs / 2, P.numel(), device=x.device)

    low_band_mask = (freqs >= 50) & (freqs < 1000)
    high_band_mask = (freqs >= 1000) & (freqs < 5000)

    low_band_energy = P[low_band_mask].sum()
    high_band_energy = P[high_band_mask].sum()

    return 10 * torch.log10((low_band_energy + 1e-8) / (high_band_energy + 1e-8))
