
import torch
from ..temporal.rms import frame_signal, hann_window
from ..spectral.lpc import lpc_coefficients

def lpcc(audio_data: torch.Tensor, sample_rate: int, n_lpcc: int = 12, n_fft: int = 2048, hop_length: int = 512, lpc_order: int = 12):
    """
    Computes the Linear Predictive Cepstral Coefficients (LPCCs) of an audio signal.

    Args:
        audio_data (torch.Tensor): The audio signal.
        sample_rate (int): The sample rate of the audio.
        n_lpcc (int): The number of LPCCs to compute.
        n_fft (int): The number of FFT points.
        hop_length (int): The number of samples to slide the window.
        lpc_order (int): The order of the LPC analysis.

    Returns:
        torch.Tensor: The LPCCs.
    """
    # Frame the signal
    frames = frame_signal(audio_data, n_fft, hop_length)
    
    # Apply Hann window
    window = hann_window(n_fft).to(audio_data.device)
    windowed_frames = frames * window

    lpccs_list = []
    for frame in windowed_frames:
        # Compute LPC coefficients (a_1 to a_p)
        a_coeffs = lpc_coefficients(frame, lpc_order)
        
        # Initialize LPCCs for the current frame
        c = torch.zeros(n_lpcc, device=audio_data.device)

        # Recursive formula for LPCCs
        for m in range(n_lpcc):
            sum_val = torch.zeros(1, device=audio_data.device)
            for k in range(m): # k goes from 0 to m-1
                if k < lpc_order: # Ensure a_coeffs index is valid
                    sum_val += (k + 1) * c[k] * a_coeffs[m - k - 1] # c_k is c[k], a_{m-k} is a_coeffs[m-k-1]
            
            if m < lpc_order:
                c[m] = a_coeffs[m] + sum_val / (m + 1)
            else:
                c[m] = sum_val / (m + 1)
        lpccs_list.append(c)

    return torch.stack(lpccs_list)
