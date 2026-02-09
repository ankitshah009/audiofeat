import torch
from ..temporal.rms import frame_signal, hann_window

def cepstral_peak_prominence(
    waveform: torch.Tensor,
    sample_rate: int,
    f0_min: float = 60.0,
    f0_max: float = 333.3,
    frame_length_ms: float = 40.0,
    hop_length_ms: float = 10.0
) -> torch.Tensor:
    """
    Compute Cepstral Peak Prominence (CPP) for voice quality assessment.

    CPP serves as a robust measure of breathiness and dysphonia. It allows for 
    differentiation between modal speech and various voice pathologies.
    The algorithm computes the prominence of the cepstral peak corresponding 
    to the fundamental frequency relative to a linear regression line 
    fitted over the cepstrum.

    Args:
        waveform (torch.Tensor): Mono audio waveform.
        sample_rate (int): Sampling rate in Hz.
        f0_min (float): Minimum expected periodicity (Hz). Default 60Hz.
        f0_max (float): Maximum expected periodicity (Hz). Default 333Hz.
                        (Defines the search range for the cepstral peak).
        frame_length_ms (float): Frame length in milliseconds.
        hop_length_ms (float): Hop length in milliseconds.

    Returns:
        torch.Tensor: CPP values in dB for each frame.
    """
    if waveform.ndim > 1:
        waveform = waveform.squeeze()

    frame_length = int(sample_rate * frame_length_ms / 1000)
    hop_length = int(sample_rate * hop_length_ms / 1000)
    n_fft = 2 ** ((frame_length * 2 - 1).bit_length())  # Next power of 2

    # 1. Framing and Windowing
    frames = frame_signal(waveform, frame_length, hop_length)
    window = hann_window(frame_length).to(frames.device)
    windowed_frames = frames * window

    # 2. Compute Real Cepstrum
    # C[n] = Real(IFFT(log(|FFT(x)|)))
    spectrum = torch.fft.rfft(windowed_frames, n=n_fft)
    log_magnitude = torch.log(torch.abs(spectrum) + 1e-9)
    cepstrum = torch.fft.irfft(log_magnitude, n=n_fft)
    
    # We only care about positive quefrency
    cepstrum = cepstrum[:, :n_fft//2]
    
    # Convert quefrency indices to frequency/time
    # Quefrency q (samples) corresponds to Period T = q/fs
    # Frequency f = fs/q
    
    # Define valid quefrency range for F0 search
    # min_period = 1/max_f0, max_period = 1/min_f0
    q_min = int(sample_rate / f0_max)
    q_max = int(sample_rate / f0_min)
    
    # Clamp to valid range
    q_min = max(q_min, 1)
    q_max = min(q_max, cepstrum.shape[1] - 1)
    if q_max <= q_min:
        return torch.zeros(cepstrum.shape[0], device=waveform.device)

    cpp_values = []
    
    # 3. Processing per frame (some steps are hard to vectorize purely due to regression logic)
    # We can vectorize the finding of peaks, but regression is often done per frame or batched linear algebra.
    
    # Vectorized Regression
    # X axis: quefrency indices [1, ..., N/2]
    # Y axis: cepstral magnitude (dB)
    
    # Create quefrency axis (integers)
    quefrencies = torch.arange(cepstrum.shape[1], device=waveform.device, dtype=torch.float32)
    
    # We fit the regression line over the ENTIRE cepstrum (usually excluding minimal quefrencies)
    # Standard practice: Fit line over > 1ms to end of frame
    reg_start_q = int(sample_rate * 0.001) # 1 ms
    reg_end_q = cepstrum.shape[1] - 1
    
    # Prepare X matrix for regression: [1, q]
    x_reg = quefrencies[reg_start_q:reg_end_q]
    X_mat = torch.stack([torch.ones_like(x_reg), x_reg], dim=1) # (N_points, 2)
    
    # Pre-compute pseudo-inverse for linear regression: (X^T X)^-1 X^T
    # This assumes fixed frame size/range, which is true.
    # Result is (2, N_points)
    pinv = torch.linalg.pinv(X_mat)
    
    # Y data: cepstrum for all frames in the valid range
    # shape: (Num_frames, N_points)
    Y_mat = cepstrum[:, reg_start_q:reg_end_q]
    
    # Solve coeffs: Beta = Y * pinv^T  -> shape (Num_frames, 2) [intercept, slope]
    coeffs = torch.matmul(Y_mat, pinv.t())
    
    # 4. Find Peak in Valid F0 Range
    # We search for max in [q_min, q_max]
    search_region = cepstrum[:, q_min:q_max]
    peaks, peak_indices_rel = torch.max(search_region, dim=1)
    
    # Absolute quefrency of peaks
    peak_quefrencies = peak_indices_rel + q_min
    
    # 5. Compute Predicted Value at Peak Quefrency
    # Predicted = intercept + slope * peak_q
    intercepts = coeffs[:, 0]
    slopes = coeffs[:, 1]
    predicted_vals = intercepts + slopes * peak_quefrencies.float()
    
    # 6. Calculate CPP
    # CPP = Peak_Value - Predicted_Value
    # Factor is typically in dB if using log magnitude spectrum (which we are)
    # Note: Initial log was natural log. Convert to dB (20*log10) if standard scale required.
    # However, 'power cepstrum' vs 'real cepstrum' conventions vary. 
    # Usually CPP is reported in dB.
    # Our cepstrum is derived from ln(|X|). 
    # To get roughly dB-like scale: 20 * log10(e) * value approx 8.68 * value
    
    cpp = (peaks - predicted_vals)
    
    # Convert to standard dB scale often used in clinical tools
    cpp_db = cpp * 8.685889 
    
    return cpp_db

def delta_cpp(cpp: torch.Tensor):
    """Frame-wise difference of cepstral peak prominence."""
    return torch.cat([torch.tensor([0.0], device=cpp.device), cpp[1:] - cpp[:-1]])
