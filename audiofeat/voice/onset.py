import torch
from ..temporal.rms import frame_signal, hann_window

def voice_onset_time(x: torch.Tensor, fs: int, frame_length: int, hop_length: int):
    """
    Simplified voice onset time (VOT) estimation.

    The burst is taken as the first frame whose short-time energy exceeds 10% of
    the peak energy. Voicing onset is the first subsequent frame whose normalized
    autocorrelation, evaluated over the plausible pitch-period lag range
    (2-15 ms), exceeds a voicing threshold. VOT is the time between burst and
    voicing onset.

    Returns:
        torch.Tensor: VOT in seconds, or ``NaN`` when undefined (no burst or no
        voiced frame after the burst).
    """
    frames = frame_signal(x, frame_length, hop_length)
    energy = (frames ** 2).sum(dim=1)
    burst = (energy > energy.max() * 0.1).nonzero(as_tuple=False)
    if burst.numel() == 0:
        return torch.tensor(float("nan"), device=x.device)
    nb = int(burst[0, 0].item())

    # True autocorrelation via Wiener-Khinchin: irfft(|FFT|^2), NOT irfft(FFT).
    spec = torch.fft.rfft(frames, n=2 * frame_length, dim=1)
    autocorr = torch.fft.irfft(spec * spec.conj(), n=2 * frame_length, dim=1)
    autocorr = autocorr[:, :frame_length]

    lo = max(int(0.002 * fs), 1)
    hi = min(int(0.015 * fs), frame_length)
    nv = None
    for i in range(nb, frames.size(0)):
        ac = autocorr[i]
        if hi <= lo or ac[0] <= 0:
            continue
        r = (ac[lo:hi].max() / ac[0]).item()
        if r > 0.3:
            nv = i
            break
    if nv is None:
        return torch.tensor(float("nan"), device=x.device)
    return torch.tensor((nv - nb) * hop_length / fs, device=x.device)
