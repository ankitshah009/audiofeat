
import torch
from ..temporal.rms import frame_signal, hann_window
from ..spectral.lpc import lpc_coefficients

__all__ = ["lpcc"]

def lpcc(audio_data: torch.Tensor, sample_rate: int, n_lpcc: int = 12, n_fft: int = 2048, hop_length: int = 512, lpc_order: int = 12):
    """
    Computes the Linear Predictive Cepstral Coefficients (LPCCs) of an audio signal.

    The LPCCs are derived from the LPC coefficients via the standard recursion

        c_m = a_m + (1/m) * sum_{k=1}^{m-1} k * c_k * a_{m-k}     (1 <= m <= p)
        c_m =       (1/m) * sum_{k=m-p}^{m-1} k * c_k * a_{m-k}   (m > p)

    where ``a_k`` are *prediction* coefficients in the convention
    ``A(z) = 1 - sum_k a_k z^-k``. ``lpc_coefficients`` (from
    ``spectral.lpc``) returns coefficients in the analysis convention
    ``A(z) = 1 + sum_k a_k z^-k`` (Levinson-Durbin output), so the
    coefficients are negated here before the cepstral recursion.

    Args:
        audio_data (torch.Tensor): The audio signal.
        sample_rate (int): The sample rate of the audio.
        n_lpcc (int): The number of LPCCs to compute. May exceed ``lpc_order``.
        n_fft (int): The number of FFT points (frame length).
        hop_length (int): The number of samples to slide the window.
        lpc_order (int): The order of the LPC analysis.

    Returns:
        torch.Tensor: The LPCCs of shape ``(n_frames, n_lpcc)``.
    """
    # Frame the signal
    frames = frame_signal(audio_data, n_fft, hop_length)

    # Apply Hann window
    window = hann_window(n_fft).to(audio_data.device)
    windowed_frames = frames * window

    lpccs_list = []
    for frame in windowed_frames:
        # Silent / near-silent frames have no meaningful LPC spectrum and would
        # otherwise produce NaNs (Levinson-Durbin divides by the energy).
        if not torch.isfinite(frame).all() or float(torch.sum(frame * frame)) <= 1e-12:
            lpccs_list.append(torch.zeros(n_lpcc, device=audio_data.device))
            continue

        # Compute LPC coefficients and convert from the analysis convention
        # A(z) = 1 + sum a_k z^-k to prediction coefficients A(z) = 1 - sum a_k z^-k.
        a_coeffs = -lpc_coefficients(frame, lpc_order)

        if not torch.isfinite(a_coeffs).all():
            lpccs_list.append(torch.zeros(n_lpcc, device=audio_data.device))
            continue

        # Initialize LPCCs for the current frame
        c = torch.zeros(n_lpcc, device=audio_data.device)

        # Recursive formula for LPCCs
        for m in range(n_lpcc):
            sum_val = torch.zeros(1, device=audio_data.device)
            for k in range(m):  # k goes from 0 to m-1; c_k is c[k]
                idx = m - k - 1  # a_{m-k} is a_coeffs[m-k-1]
                if 0 <= idx < lpc_order:
                    sum_val += (k + 1) * c[k] * a_coeffs[idx]

            if m < lpc_order:
                c[m] = a_coeffs[m] + sum_val / (m + 1)
            else:
                c[m] = sum_val / (m + 1)
        lpccs_list.append(c)

    return torch.stack(lpccs_list)
