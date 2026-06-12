import torch
import numpy as np

def lsp_coefficients(lpc_coeffs: torch.Tensor):
    """
    Converts Linear Prediction Coefficients (LPC) to Line Spectral Pairs (LSP).

    The LPC analysis polynomial is ``A(z) = 1 + a_1 z^-1 + ... + a_p z^-p``.
    Two auxiliary polynomials of degree ``p + 1`` are formed::

        P(z) = A(z) + z^-(p+1) A(z^-1)
        Q(z) = A(z) - z^-(p+1) A(z^-1)

    whose roots lie on the unit circle and strictly interlace. The line
    spectral frequencies are the angles of those roots taken in the upper
    half plane ``[0, pi]`` (one per conjugate pair), sorted ascending and
    normalised by ``2*pi`` so the returned values fall in ``[0, 0.5)``.

    Args:
        lpc_coeffs (torch.Tensor): LPC coefficients ``a_1 .. a_p`` (1D tensor,
            *excluding* the leading 1.0).

    Returns:
        torch.Tensor: ``order`` line spectral frequencies in ascending order,
        each in ``[0, 0.5)``.
    """
    lpc_coeffs = lpc_coeffs.flatten()
    order = lpc_coeffs.numel()
    if order == 0:
        return torch.empty(0, dtype=torch.float64, device=lpc_coeffs.device)

    # A(z) = 1 + a_1 z^-1 + ... + a_p z^-p  (length p + 1).
    a = torch.cat(
        [torch.ones(1, dtype=lpc_coeffs.dtype, device=lpc_coeffs.device), lpc_coeffs]
    )
    a_np = a.detach().cpu().numpy().astype(np.float64)

    # Pad A to length p + 2 so the time-reversed (flipped) sequence represents
    # z^-(p+1) A(z^-1), i.e. the symmetric/antisymmetric extension of degree p+1.
    a_padded = np.concatenate([a_np, [0.0]])
    a_rev = a_padded[::-1]

    p_coeffs = a_padded + a_rev  # symmetric polynomial,  degree p + 1
    q_coeffs = a_padded - a_rev  # antisymmetric polynomial, degree p + 1

    p_roots = np.roots(p_coeffs)
    q_roots = np.roots(q_coeffs)

    def _upper_half_angles(roots: np.ndarray) -> np.ndarray:
        # Keep exactly one root per conjugate pair: the one in the open upper
        # half plane, i.e. angle strictly in (0, pi). This drops the conjugate
        # mirror (negative angle) AND the trivial roots at z = +/-1 (angle 0/pi).
        angles = np.angle(roots)
        mask = (angles > 1e-9) & (angles < np.pi - 1e-9)
        return angles[mask]

    angles = np.concatenate([_upper_half_angles(p_roots), _upper_half_angles(q_roots)])
    angles = np.sort(angles)

    lsp_freqs = angles / (2.0 * np.pi)  # normalise to [0, 0.5)

    # The construction yields exactly ``order`` interlaced frequencies for a
    # valid analysis polynomial. Guard against numerical drop-outs/extras so
    # the contract (length == order, no NaN) always holds.
    if lsp_freqs.shape[0] >= order:
        lsp_freqs = lsp_freqs[:order]
    else:
        pad = np.full(order - lsp_freqs.shape[0], lsp_freqs[-1] if lsp_freqs.size else 0.0)
        lsp_freqs = np.concatenate([lsp_freqs, pad])

    return torch.from_numpy(np.ascontiguousarray(lsp_freqs)).to(lpc_coeffs.device)
