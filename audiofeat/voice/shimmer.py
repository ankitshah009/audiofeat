import torch

def shimmer_local(amplitudes: torch.Tensor) -> torch.Tensor:
    """
    Compute Shimmer (local): Average absolute difference between consecutive amplitudes,
    divided by the average amplitude.
    
    Formula:
        Shimmer = (1/(N-1) * sum(|A_i - A_{i+1}|)) / (1/N * sum(A_i))
    """
    if amplitudes.shape[-1] < 2:
        return torch.tensor(0.0, device=amplitudes.device)
        
    diffs = torch.abs(amplitudes[1:] - amplitudes[:-1])
    mean_amp = amplitudes.mean()
    
    if mean_amp == 0:
        return torch.tensor(0.0, device=amplitudes.device)
        
    shim = diffs.mean() / mean_amp * 100.0
    return shim

def shimmer_local_db(amplitudes: torch.Tensor) -> torch.Tensor:
    """
    Compute Shimmer (local, dB): Average absolute difference in log-amplitudes.
    
    Formula:
        Shimmer_dB = 1/(N-1) * sum(|20*log10(A_{i+1}/A_i)|)
    """
    if amplitudes.shape[-1] < 2:
        return torch.tensor(0.0, device=amplitudes.device)
    
    # Avoid log0
    safe_amps = amplitudes + 1e-9
    ratios = safe_amps[1:] / safe_amps[:-1]
    log_ratios = torch.abs(20 * torch.log10(ratios))
    
    return log_ratios.mean()

def shimmer_apq3(amplitudes: torch.Tensor) -> torch.Tensor:
    """
    Compute Shimmer (APQ3): Amplitude Perturbation Quotient (3-point).
    Average absolute difference between an amplitude and the average of its 
    neighbors.
    """
    N = amplitudes.shape[-1]
    if N < 3:
        return torch.tensor(0.0, device=amplitudes.device)

    # 3-point moving average
    kernel = torch.ones(3, device=amplitudes.device) / 3.0
    avg_3 = torch.nn.functional.conv1d(
        amplitudes.view(1, 1, -1),
        kernel.view(1, 1, -1),
        padding=0
    ).view(-1)
    
    # Corresponds to amplitudes[1:-1]
    center_amps = amplitudes[1:-1]
    
    numerator = torch.abs(center_amps - avg_3).mean()
    mean_amp = amplitudes.mean()
    
    if mean_amp == 0:
        return torch.tensor(0.0, device=amplitudes.device)
        
    return numerator / mean_amp * 100.0

def shimmer_dda(amplitudes: torch.Tensor) -> torch.Tensor:
    """
    Compute Shimmer (DDA): Difference of Differences of Amplitudes.
    """
    if amplitudes.shape[-1] < 3:
        return torch.tensor(0.0, device=amplitudes.device)
        
    diff1 = torch.abs(amplitudes[1:] - amplitudes[:-1])
    diff2 = torch.abs(diff1[1:] - diff1[:-1])
    
    numerator = diff2.mean()
    mean_amp = amplitudes.mean()
    
    if mean_amp == 0:
        return torch.tensor(0.0, device=amplitudes.device)
        
    return numerator / mean_amp * 100.0
