import torch

def compute_functionals(feature_series: torch.Tensor):
    """
    Computes a set of statistical functionals (mean, std, min, max, skewness, kurtosis)
    for a given time-series of features.

    Args:
        feature_series (torch.Tensor): A 2D tensor of shape (time_frames, num_features).

    Returns:
        torch.Tensor: A 1D tensor containing the aggregated statistics.
    """
    if feature_series.dim() != 2:
        raise ValueError("Input feature_series must be a 2D tensor (time_frames, num_features).")

    if feature_series.shape[0] == 0: # Handle empty time_frames
        num_features = feature_series.shape[1]
        # Return a tensor of NaNs for undefined statistics
        return torch.full((num_features * 6,), float('nan'), device=feature_series.device)

    # Ensure float type for calculations
    feature_series = feature_series.float()

    # Mean
    mean_val = torch.mean(feature_series, dim=0)

    # Standard Deviation
    std_val = torch.std(feature_series, dim=0, unbiased=False)

    # Min
    min_val = torch.min(feature_series, dim=0).values

    # Max
    max_val = torch.max(feature_series, dim=0).values

    # Skewness
    # Skewness = E[((X - mu)/sigma)^3]
    # Using torch.mean for expectation
    diff = feature_series - mean_val
    skew_val = torch.mean(diff**3, dim=0) / (std_val**3 + 1e-8) # Add epsilon for stability

    # Kurtosis
    # Kurtosis = E[((X - mu)/sigma)^4] - 3
    kurt_val = torch.mean(diff**4, dim=0) / (std_val**4 + 1e-8) - 3 # Add epsilon for stability

    # Concatenate all statistics
    # Order: mean, std, min, max, skewness, kurtosis
    aggregated_stats = torch.cat([
        mean_val,
        std_val,
        min_val,
        max_val,
        skew_val,
        kurt_val
    ])

    return aggregated_stats
