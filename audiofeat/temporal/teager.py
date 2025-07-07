import torch


def teager_energy_operator(x: torch.Tensor) -> torch.Tensor:
    """Compute the average *Teager Energy* of a signal.

    The Teager-Kaiser Energy Operator (TEO) is defined for a discrete
    sample sequence *x[n]* as ::

        Ψ{x[n]} = x[n]^2 - x[n-1] * x[n+1]

    It provides a simple measure of the instantaneous energy of
    oscillatory systems and has been widely used in speech and music
    signal processing.

    Args:
        x (torch.Tensor): 1-D audio signal.

    Returns:
        torch.Tensor: Mean Teager energy across the signal.
    """
    if x.numel() < 3:
        return torch.tensor(0.0, device=x.device)

    # Compute TEO for samples 1 .. N-2
    teo = x[1:-1] ** 2 - x[:-2] * x[2:]
    return teo.mean().detach() 