
import torch

def maximum_flow_declination_rate(flow: torch.Tensor, fs: int):
    """Approximate MFDR from differentiated glottal flow."""
    dflow = torch.diff(flow) * fs
    return dflow.max()
