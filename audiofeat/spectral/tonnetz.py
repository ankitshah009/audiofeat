"""Tonal centroid (Tonnetz) features — Harte, Sandler & Gasser (2006).

Projects L1-normalized chroma onto a 6-D basis representing
the perfect fifth, minor third, and major third as 2-D coordinates.

Primary path delegates to ``librosa.feature.tonnetz``.  The fallback
reproduces librosa's exact transformation matrix so outputs match
regardless of which path is taken.
"""

from __future__ import annotations

import numpy as np
import torch


def _fallback_tonnetz(chroma_features: torch.Tensor) -> torch.Tensor:
    """Reproduce librosa's tonnetz transform exactly in PyTorch.

    librosa source (feature/spectral.py, ~line 1812):
        dim_map = linspace(0, 12, n_chroma, endpoint=False)
        scale   = [7/6, 7/6, 3/2, 3/2, 2/3, 2/3]
        V       = outer(scale, dim_map);  V[::2] -= 0.5
        R       = [1, 1, 1, 1, 0.5, 0.5]
        phi     = R[:, None] * cos(pi * V)
        tonnetz = phi @ normalize(chroma, L1, axis=0)
    """
    n_chroma = chroma_features.shape[0]
    chroma = torch.nn.functional.normalize(
        chroma_features.float(), p=1, dim=0, eps=1e-12
    )

    # numpy.linspace(0, 12, 12, endpoint=False) = [0, 1, ..., 11]
    dim_map = torch.arange(n_chroma, device=chroma.device, dtype=torch.float32) * (
        12.0 / n_chroma
    )

    scale = torch.tensor(
        [7.0 / 6, 7.0 / 6, 3.0 / 2, 3.0 / 2, 2.0 / 3, 2.0 / 3],
        device=chroma.device,
        dtype=torch.float32,
    )
    V = torch.outer(scale, dim_map)  # (6, n_chroma)
    V[::2] -= 0.5

    R = torch.tensor(
        [1.0, 1.0, 1.0, 1.0, 0.5, 0.5],
        device=chroma.device,
        dtype=torch.float32,
    )
    phi = R.unsqueeze(1) * torch.cos(torch.pi * V)  # (6, n_chroma)

    return phi @ chroma  # (6, frames)


def tonnetz(chroma_features: torch.Tensor):
    """Compute tonal centroid (Tonnetz), matching librosa when available.

    Parameters
    ----------
    chroma_features : torch.Tensor
        Shape ``(12, frames)`` — L1-normalized chroma energy per frame.

    Returns
    -------
    torch.Tensor
        Shape ``(6, frames)`` — tonal centroid features.
    """
    if chroma_features.dim() != 2:
        raise ValueError("chroma_features must be 2-D with shape (12, frames).")
    if chroma_features.shape[0] != 12:
        raise ValueError("chroma_features must have 12 pitch-class bins.")

    device = chroma_features.device

    try:
        import librosa  # type: ignore
    except ModuleNotFoundError:
        return _fallback_tonnetz(chroma_features)

    tonnetz_np = librosa.feature.tonnetz(
        chroma=chroma_features.detach().cpu().numpy().astype(np.float32, copy=False)
    )
    return torch.from_numpy(tonnetz_np.astype(np.float32, copy=False)).to(device=device)
