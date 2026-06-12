import torch

__all__ = ["delta", "delta_delta"]

def delta(x: torch.Tensor, width: int = 9):
    """
    Computes the first-order derivative (delta) of a feature contour.

    Uses a symmetric linear-regression filter of length ``width`` (the same
    formulation as :func:`librosa.feature.delta` with ``order=1``). Edges are
    handled with ``'replicate'`` (edge) padding of ``width // 2`` frames on
    each side along the time axis, so interior frames match librosa's default
    ``mode='interp'`` / edge behavior closely.

    Args:
        x (torch.Tensor): The feature contour. Can be (time_steps,), (features, time_steps),
                          or (batch, features, time_steps).
        width (int): The width of the regression window. Must be an odd integer >= 3
                     (matching librosa's constraint).

    Returns:
        torch.Tensor: The delta features, same shape as ``x``.

    Raises:
        ValueError: If ``width`` is not an odd integer >= 3, or if the number of
            time steps is smaller than ``width``.
    """
    if not isinstance(width, int) or width < 3 or width % 2 != 1:
        raise ValueError(f"width must be an odd integer >= 3, got {width}.")

    original_dim = x.dim()

    if original_dim == 1:
        # (time_steps,) -> (1, 1, time_steps)
        x = x.unsqueeze(0).unsqueeze(0)
    elif original_dim == 2:
        # (features, time_steps) -> (1, features, time_steps)
        x = x.unsqueeze(0)

    # x is now (batch, features, time_steps)

    batch_size, num_features, time_steps = x.shape

    if time_steps < width:
        raise ValueError(
            f"Number of time steps ({time_steps}) must be >= width ({width})."
        )

    # Pad the input to handle edges along the time_steps dimension (edge padding).
    padding = width // 2
    padded_x = torch.nn.functional.pad(x, (padding, padding), mode='replicate')  # Pad L_in dimension

    # Create the regression coefficients
    denom_values = torch.arange(-padding, padding + 1, device=x.device, dtype=torch.float32)
    denom = torch.sum(denom_values**2)

    coeffs = denom_values / denom

    # Apply the convolution
    # conv1d expects (N, C_in, L_in) where C_in is features, L_in is time_steps
    # Our padded_x is (batch, features, time_steps)
    # We need to expand coeffs to (num_features, 1, width) for groups=num_features
    delta_x = torch.nn.functional.conv1d(padded_x, coeffs.view(1, 1, -1).repeat(num_features, 1, 1), padding=0, groups=num_features)

    # Reshape back to original dimensions if input was 1D or 2D
    if original_dim == 1:
        return delta_x.squeeze(0).squeeze(0)
    elif original_dim == 2:
        return delta_x.squeeze(0)
    else:
        return delta_x

def delta_delta(x: torch.Tensor, width: int = 9):
    """
    Computes the second-order derivative (delta-delta) of a feature contour.

    NOTE: This applies the first-order :func:`delta` operator twice (an
    *iterated* delta). This differs from ``librosa.feature.delta(..., order=2)``,
    which convolves with the analytic second-derivative Savitzky-Golay kernel.
    The two agree on the overall shape but not numerically.

    Args:
        x (torch.Tensor): The feature contour.
        width (int): The width of the regression window (odd, >= 3).

    Returns:
        torch.Tensor: The delta-delta features, same shape as ``x``.
    """
    return delta(delta(x, width), width)
