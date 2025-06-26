
import torch

def tonnetz(chroma_features: torch.Tensor):
    """
    Computes the Tonnetz (Tonal Centroid Features) from Chroma features.

    Args:
        chroma_features (torch.Tensor): The Chroma features (n_chroma, time_frames).
                                        Expected n_chroma = 12.

    Returns:
        torch.Tensor: The Tonnetz features (6, time_frames).
    """
    if chroma_features.shape[0] != 12:
        raise ValueError("Chroma features must have 12 bins for Tonnetz calculation.")

    # Tonnetz basis vectors for 6 dimensions:
    # 0: fifths (C-G axis)
    # 1: minor thirds (C-Eb axis)
    # 2: major thirds (C-E axis)
    # 3: fifths (sin component)
    # 4: minor thirds (sin component)
    # 5: major thirds (sin component)

    # These are derived from the circle of fifths and other musical intervals
    # and projected onto a 2D plane for each pair of intervals.
    # The values are typically normalized.

    # Cosine components (real part of complex exponential)
    cos_fifths = torch.cos(torch.arange(12) * 2 * torch.pi / 12)
    cos_minor_thirds = torch.cos(torch.arange(12) * 2 * torch.pi / 4) # 3 semitones per step
    cos_major_thirds = torch.cos(torch.arange(12) * 2 * torch.pi / 3) # 4 semitones per step

    # Sine components (imaginary part of complex exponential)
    sin_fifths = torch.sin(torch.arange(12) * 2 * torch.pi / 12)
    sin_minor_thirds = torch.sin(torch.arange(12) * 2 * torch.pi / 4)
    sin_major_thirds = torch.sin(torch.arange(12) * 2 * torch.pi / 3)

    # Combine into a 6x12 transformation matrix
    tonnetz_basis = torch.stack([
        cos_fifths,
        cos_minor_thirds,
        cos_major_thirds,
        sin_fifths,
        sin_minor_thirds,
        sin_major_thirds
    ], dim=0).to(chroma_features.device)

    # Apply the transformation
    tonnetz_features = torch.matmul(tonnetz_basis, chroma_features)

    # Normalize each Tonnetz dimension (optional, but common)
    # For example, normalize by the maximum value of each dimension
    # tonnetz_features = tonnetz_features / (tonnetz_features.abs().max(dim=1, keepdim=True).values + 1e-8)

    return tonnetz_features
