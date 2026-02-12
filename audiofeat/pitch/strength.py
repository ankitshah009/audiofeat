
import torch
from ..temporal.rms import frame_signal, hann_window

def pitch_strength(x: torch.Tensor, fs: int, frame_length: int, hop_length: int, fmin: int = 50, fmax: int = 600):
    """
    Computes the pitch strength using the autocorrelation method.

    Args:
        x (torch.Tensor): The audio signal.
        fs (int): The sample rate of the audio.
        frame_length (int): The length of each frame in samples.
        hop_length (int): The number of samples to slide the window.
        fmin (int): Minimum fundamental frequency to consider.
        fmax (int): Maximum fundamental frequency to consider.

    Returns:
        torch.Tensor: The pitch strength for each frame.
    """
    frames = frame_signal(x, frame_length, hop_length)
    w = hann_window(frame_length).to(x.device)
    win = frames * w
    
    # Compute autocorrelation via Wiener-Khinchin theorem
    spec = torch.fft.rfft(win, n=2 * frame_length)
    autocorr = torch.fft.irfft(spec * torch.conj(spec), n=2 * frame_length)
    
    min_lag = int(fs / fmax)
    max_lag = int(fs / fmin)
    
    # Find the peak in the relevant lag range
    pitch_strengths = []
    for ac_frame in autocorr:
        if ac_frame.numel() == 0 or max_lag >= ac_frame.numel() or min_lag >= max_lag:
            pitch_strengths.append(torch.tensor(0.0, device=x.device))
            continue
        
        ac_segment = ac_frame[min_lag:max_lag]
        if ac_segment.numel() == 0:
            pitch_strengths.append(torch.tensor(0.0, device=x.device))
            continue
            
        # Normalize autocorrelation by the value at lag 0 (energy)
        # Avoid division by zero if ac_frame[0] is very small
        if ac_frame[0] == 0:
            pitch_strengths.append(torch.tensor(0.0, device=x.device))
        else:
            pitch_strengths.append(ac_segment.max() / ac_frame[0])
            
    return torch.stack(pitch_strengths)
