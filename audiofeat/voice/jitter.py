import torch

def _get_diffs(periods: torch.Tensor):
    """Compute first-order difference of periods."""
    return periods[1:] - periods[:-1]

def jitter_local(periods: torch.Tensor) -> torch.Tensor:
    """
    Compute Jitter (local): Average absolute difference between consecutive periods,
    divided by the average period.
    
    Formula:
        Jitter = (1/(N-1) * sum(|T_i - T_{i+1}|)) / (1/N * sum(T_i))
    """
    if periods.shape[-1] < 2:
        return torch.tensor(0.0, device=periods.device)
        
    diffs = torch.abs(_get_diffs(periods))
    mean_period = periods.mean()
    
    if mean_period == 0:
        return torch.tensor(0.0, device=periods.device)
        
    jitt = diffs.mean() / mean_period * 100.0
    return jitt

def jitter_ppq5(periods: torch.Tensor) -> torch.Tensor:
    """
    Compute Jitter (PPQ5): Five-point Period Perturbation Quotient.
    Average absolute difference between a period and the average of it and its 
    four nearest neighbors, divided by the average period.
    """
    N = periods.shape[-1]
    if N < 5:
        return torch.tensor(0.0, device=periods.device)

    # Moving average over 5 points
    # Avg_5[i] = 1/5 * sum(T_{i-2} ... T_{i+2})
    # We can use 1D convolution
    kernel = torch.ones(5, device=periods.device) / 5.0
    avg_5 = torch.nn.functional.conv1d(
        periods.view(1, 1, -1), 
        kernel.view(1, 1, -1), 
        padding=0
    ).view(-1)
    
    # Valid indices for comparison are 2 to N-3 (0-indexed: indices 2 to N-3 generally aligns)
    # The convolution reduces size by 4 (kernel 5).
    # Output length L_out = L_in - 5 + 1 = L_in - 4.
    # Corresponds to periods[2:-2]
    
    center_periods = periods[2:-2]
    
    numerator = torch.abs(center_periods - avg_5).mean()
    mean_period = periods.mean()

    if mean_period == 0:
        return torch.tensor(0.0, device=periods.device)

    return numerator / mean_period * 100.0

def jitter_ddp(periods: torch.Tensor) -> torch.Tensor:
    """
    Compute Jitter (DDP): Difference of Differences of Periods.
    Formula:
        DDP = 1/(N-2) * sum(|(T_{i+1} - T_i) - (T_i - T_{i-1})|) / mean(T)
    """
    if periods.shape[-1] < 3:
        return torch.tensor(0.0, device=periods.device)

    # First difference
    diff1 = periods[1:] - periods[:-1]
    # Second difference
    diff2 = diff1[1:] - diff1[:-1]
    
    numerator = torch.abs(diff2).mean()
    mean_period = periods.mean()
    
    if mean_period == 0:
        return torch.tensor(0.0, device=periods.device)
        
    return numerator / mean_period * 100.0
