import torch

__all__ = ["compute_functionals"]


def compute_functionals(feature_series: torch.Tensor, time_axis: int = 0):
    """
    Computes a set of statistical functionals (mean, std, min, max, skewness,
    excess kurtosis) for a given 2D time-series of features.

    The functionals are reduced over ``time_axis``. The returned vector is the
    concatenation, in this order, of the six per-feature statistics::

        [mean, std, min, max, skewness, kurtosis]

    so the output length is ``6 * num_features``.

    .. warning::
        The DEFAULT ``time_axis=0`` treats the input as ``(time_frames,
        num_features)``. This is the OPPOSITE of the ``(features, time)``
        convention used elsewhere in this library (e.g. ``mfcc`` returns
        ``(n_mfcc, n_frames)``). When aggregating a feature matrix produced by
        this library, pass ``time_axis=1`` (or transpose first) so that the
        statistics reduce over time rather than over the feature dimension.

    ``skewness`` and ``kurtosis`` use a multiplicative (clamped) guard on the
    standard deviation so the result is scale-invariant. For a constant
    (zero-variance) feature, skewness and kurtosis are defined to be ``0.0``
    (finite). ``kurtosis`` is **excess kurtosis** (Fisher), i.e. ``0`` for a
    normal distribution.

    Args:
        feature_series (torch.Tensor): A 2D tensor. With ``time_axis=0`` the
            shape is ``(time_frames, num_features)``; with ``time_axis=1`` it is
            ``(num_features, time_frames)``.
        time_axis (int): The axis to reduce over (``0`` or ``1``). Defaults to
            ``0`` for backward compatibility.

    Returns:
        torch.Tensor: A 1D tensor of length ``6 * num_features`` containing the
        aggregated statistics.
    """
    if feature_series.dim() != 2:
        raise ValueError("Input feature_series must be a 2D tensor (time_frames, num_features).")

    if time_axis not in (0, 1, -1, -2):
        raise ValueError("time_axis must be 0 or 1 (or the negative equivalents).")
    # Normalize so that we always reduce along dim=0 with features along dim=1.
    if time_axis in (1, -1):
        feature_series = feature_series.transpose(0, 1)

    if feature_series.shape[0] == 0:  # Handle empty time_frames
        num_features = feature_series.shape[1]
        # Return a tensor of NaNs for undefined statistics
        return torch.full((num_features * 6,), float('nan'), device=feature_series.device)

    # Ensure float type for calculations
    feature_series = feature_series.float()

    # Mean
    mean_val = torch.mean(feature_series, dim=0)

    # Standard Deviation (population / biased)
    std_val = torch.std(feature_series, dim=0, unbiased=False)

    # Min
    min_val = torch.min(feature_series, dim=0).values

    # Max
    max_val = torch.max(feature_series, dim=0).values

    # Skewness / excess kurtosis with a scale-invariant clamped guard.
    # Constant (zero-variance) features are defined to have skew = kurt = 0.
    diff = feature_series - mean_val
    nonzero = std_val > 0
    safe_std = torch.where(nonzero, std_val, torch.ones_like(std_val))

    skew_raw = torch.mean(diff ** 3, dim=0) / (safe_std ** 3)
    kurt_raw = torch.mean(diff ** 4, dim=0) / (safe_std ** 4) - 3.0  # excess kurtosis

    skew_val = torch.where(nonzero, skew_raw, torch.zeros_like(skew_raw))
    kurt_val = torch.where(nonzero, kurt_raw, torch.zeros_like(kurt_raw))

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
