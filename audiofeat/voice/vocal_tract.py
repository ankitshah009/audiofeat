
import torch

def vocal_tract_length(F1: float, F2: float, c: float = 34400):
    """Estimate vocal tract length from first two formants."""
    return c / 4 * (1 / F1 + 1 / (F2 - F1))
