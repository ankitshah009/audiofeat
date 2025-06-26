import torch

def delta(x: torch.Tensor, width: int = 9):
    """
    Computes the first-order derivative (delta) of a feature contour.

    Args:
        x (torch.Tensor): The feature contour. Can be (time_steps,), (features, time_steps),
                          or (batch, features, time_steps).
        width (int): The width of the regression window.

    Returns:
        torch.Tensor: The delta features.
    """
    original_dim = x.dim()
    original_shape = x.shape

    if original_dim == 1:
        # (time_steps,) -> (1, 1, time_steps)
        x = x.unsqueeze(0).unsqueeze(0)
    elif original_dim == 2:
        # (features, time_steps) -> (1, features, time_steps)
        x = x.unsqueeze(0)

    # x is now (batch, features, time_steps)

    batch_size, num_features, time_steps = x.shape

    # Pad the input to handle edges along the time_steps dimension
    padding = width // 2
    padded_x = torch.nn.functional.pad(x, (padding, padding), mode='replicate') # Pad L_in dimension

    # Create the regression coefficients
    denom_values = torch.arange(-padding, padding + 1, device=x.device, dtype=torch.float32)
    denom = torch.sum(denom_values**2)

    if denom == 0: # Handle width = 1 case
        return torch.zeros_like(x)

    coeffs = denom_values / denom

    # Apply the convolution
    # conv1d expects (N, C_in, L_in) where C_in is features, L_in is time_steps
    # Our padded_x is (batch, features, time_steps)
    # coeffs.view(1, 1, -1) is (1, 1, width)
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

    Args:
        x (torch.Tensor): The feature contour.
        width (int): The width of the regression window.

    Returns:
        torch.Tensor: The delta-delta features.
    """
    return delta(delta(x, width), width)