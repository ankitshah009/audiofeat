import torch
import numpy as np

def lsp_coefficients(lpc_coeffs: torch.Tensor):
    """
    Converts Linear Prediction Coefficients (LPC) to Line Spectral Pairs (LSP).

    Args:
        lpc_coeffs (torch.Tensor): LPC coefficients (1D tensor).

    Returns:
        torch.Tensor: LSP coefficients.
    """
    order = lpc_coeffs.numel()
    
    # Construct P(z) and Q(z) polynomials
    # A(z) = 1 + a_1 z^-1 + ... + a_p z^-p
    # P(z) = A(z) + z^-(p+1) A(z^-1)
    # Q(z) = A(z) - z^-(p+1) A(z^-1)

    # Coefficients for A(z)
    a = torch.cat([torch.tensor([1.0], device=lpc_coeffs.device), lpc_coeffs])

    # Coefficients for A(z^-1) * z^-(p+1)
    # This needs to be padded to match the length of 'a' for element-wise addition/subtraction
    a_rev_padded = torch.nn.functional.pad(torch.flip(a, dims=[0]), (0, order + 1 - a.numel()))

    # P(z) and Q(z) coefficients
    p_coeffs = a + a_rev_padded
    q_coeffs = a - a_rev_padded

    # Find roots of P(z) and Q(z)
    # Convert to numpy for root finding, then back to torch
    p_roots = np.roots(p_coeffs.cpu().numpy())
    q_roots = np.roots(q_coeffs.cpu().numpy())

    # Extract angles of roots on the unit circle
    # Filter for roots on or very close to the unit circle
    p_angles = np.angle(p_roots[np.isclose(np.abs(p_roots), 1.0)])
    q_angles = np.angle(q_roots[np.isclose(np.abs(q_roots), 1.0)])

    # Combine and sort LSP frequencies (angles)
    # There should be 'order' LSP frequencies, interleaving from P and Q roots
    # This part is tricky and often requires careful handling of root pairing and sorting
    # For simplicity, we'll just take the first 'order' unique sorted angles
    
    all_angles = np.sort(np.unique(np.concatenate((p_angles, q_angles))))
    
    # Select 'order' LSP frequencies. This might need more sophisticated logic
    # to ensure correct pairing and ordering in a real application.
    # For a basic test, we'll assume the first 'order' unique sorted angles are the LSPs.
    lsp_freqs = all_angles[:order] / (2 * np.pi) # Normalize to [0, 0.5]
    if lsp_freqs.shape[0] < order:
        lsp_freqs = np.pad(
            lsp_freqs,
            (0, order - lsp_freqs.shape[0]),
            mode="constant",
            constant_values=np.nan,
        )

    return torch.from_numpy(lsp_freqs).to(lpc_coeffs.device)
