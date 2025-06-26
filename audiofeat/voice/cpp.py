
import torch

def delta_cpp(cpp: torch.Tensor):
    """Frame-wise difference of cepstral peak prominence."""
    return cpp[1:] - cpp[:-1]
