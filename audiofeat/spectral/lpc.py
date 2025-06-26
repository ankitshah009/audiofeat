
import torch
from ..temporal.rms import hann_window

def _autocorrelation(x: torch.Tensor, max_lag: int):
    """
    Computes the autocorrelation of a 1D signal using FFT.
    """
    n = x.shape[-1]
    # Pad the signal to avoid circular convolution issues
    padded_x = torch.nn.functional.pad(x, (0, n))
    
    # Compute autocorrelation using FFT
    X = torch.fft.fft(padded_x)
    autocorr = torch.fft.ifft(X * torch.conj(X))
    return autocorr.real[:max_lag]

def lpc_coefficients(audio_frame: torch.Tensor, order: int):
    """
    Computes Linear Prediction Coefficients (LPC) for a single audio frame
    using the Levinson-Durbin algorithm.

    Args:
        audio_frame (torch.Tensor): A single frame of audio data (1D tensor).
        order (int): The LPC order.

    Returns:
        torch.Tensor: The LPC coefficients.
    """
    if order >= audio_frame.numel():
        raise ValueError("LPC order must be less than the frame length.")

    # Compute autocorrelation coefficients
    R = _autocorrelation(audio_frame, order + 1)

    a = torch.zeros(order + 1, dtype=audio_frame.dtype, device=audio_frame.device)
    a[0] = 1.0
    E = R[0]

    for i in range(1, order + 1):
        k = 0.0
        for j in range(1, i):
            k -= a[j] * R[i - j]
        k -= R[i]
        k /= E

        a[i] = k

        for j in range(1, (i // 2) + 1):
            temp1 = a[j]
            temp2 = a[i - j]
            a[j] = temp1 + k * temp2
            a[i - j] = temp2 + k * temp1

        E *= (1 - k * k)

    return a[1:] # Return a_1 to a_p
