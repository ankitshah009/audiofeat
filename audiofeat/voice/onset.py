import torch
from ..temporal.rms import frame_signal, hann_window

def voice_onset_time(x: torch.Tensor, fs: int, frame_length: int, hop_length: int):
    """
    Simplified voice onset time estimation."""
    frames = frame_signal(x, frame_length, hop_length)
    energy = (frames ** 2).sum(dim=1)
    burst = (energy > energy.max() * 0.1).nonzero(as_tuple=False)
    if burst.numel() == 0:
        return 0.0
    nb = burst[0,0].item()
    autocorr = torch.fft.irfft(torch.fft.rfft(frames, n=2*frame_length), n=2*frame_length)
    nv = None
    for i in range(nb, frames.size(0)):
        ac = autocorr[i]
        r = (ac[int(0.002*fs):int(0.015*fs)].max() / ac[0]).item()
        if r > 0.3:
            nv = i
            break
    if nv is None:
        return 0.0
    return float((nv - nb) * hop_length / fs)