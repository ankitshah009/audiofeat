
import torch
from ..temporal.rms import frame_signal, hann_window

def nasality_index(nasal: torch.Tensor, oral: torch.Tensor, fs: int, n_fft: int = 1024,
                   frame_length_ms: float = 60.0, hop_length_ms: float = 20.0):
    """Compute the nasalance score from nasal and oral microphone signals.

    Nasalance is defined as ``100 * N / (N + O)`` where ``N`` and ``O`` are the
    nasal and oral band energies (a percentage in ``[0, 100]``), not a dB ratio.

    Both signals are framed (default 60 ms / 20 ms hop) and band energies are
    summed over frames; this avoids the previous behaviour of windowing the whole
    signal and truncating it to the first ``n_fft`` samples.

    Args:
        nasal (torch.Tensor): Nasal microphone signal.
        oral (torch.Tensor): Oral microphone signal.
        fs (int): Sample rate.
        n_fft (int): Number of FFT points (>= frame length).
        frame_length_ms (float): Analysis frame length in milliseconds.
        hop_length_ms (float): Hop length in milliseconds.

    Returns:
        torch.Tensor: Nasalance in percent ``[0, 100]``.
    """
    nasal = nasal.flatten().float()
    oral = oral.flatten().float()
    frame_length = max(int(fs * frame_length_ms / 1000), 1)
    hop_length = max(int(fs * hop_length_ms / 1000), 1)
    n_fft = max(n_fft, frame_length)
    window = hann_window(frame_length).to(nasal.device)

    n_frames = frame_signal(nasal, frame_length, hop_length)
    o_frames = frame_signal(oral, frame_length, hop_length)
    N = torch.fft.rfft(n_frames * window, n=n_fft, dim=1)
    O = torch.fft.rfft(o_frames * window, n=n_fft, dim=1)
    freqs = torch.linspace(0, fs / 2, N.shape[1], device=nasal.device)
    mask = (freqs >= 300) & (freqs <= 800)

    n_power = (N.abs() ** 2)[:, mask].sum()
    o_power = (O.abs() ** 2)[:, mask].sum()
    return 100.0 * n_power / (n_power + o_power + 1e-8)
