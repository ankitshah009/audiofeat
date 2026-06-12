
import torch
from ..spectral.chroma import chroma as chroma_stft

__all__ = ["music_thumbnailing"]


def music_thumbnailing(signal: torch.Tensor, sample_rate: int, thumb_size: float = 10.0, window_size: float = 1.0, hop_size: float = 0.5):
    """
    Detects the most representative (most-repeated) part of a music recording
    using Foote-style audio thumbnailing on a chroma self-similarity matrix.

    The thumbnail is the ``thumb_size``-second segment whose chroma sequence is
    most strongly repeated elsewhere in the recording. This is found by scoring
    *off-diagonal* diagonal stripes of the self-similarity matrix: the main
    diagonal (trivial self-similarity, lag 0) is masked out so the result
    reflects genuine repetition rather than a segment's similarity to itself.

    Args:
        signal (torch.Tensor): The input audio signal.
        sample_rate (int): The sample rate of the audio signal.
        thumb_size (float, optional): The desired thumbnail size in seconds. Defaults to 10.0.
        window_size (float, optional): The size of the analysis window in seconds. Defaults to 1.0.
        hop_size (float, optional): The hop size between consecutive windows in seconds. Defaults to 0.5.

    Returns:
        Tuple[float, float]: A tuple ``(start_time, end_time)`` in seconds (floats).
    """
    # 1. Extract Chroma features -> (n_frames, n_chroma)
    chroma = chroma_stft(
        signal,
        sample_rate,
        n_chroma=12,
        n_fft=int(window_size * sample_rate),
        hop_length=int(hop_size * sample_rate),
    )
    chroma = chroma.squeeze(0).T  # (n_frames, n_chroma)

    n_frames = chroma.shape[0]

    # 2. Cosine self-similarity matrix (eps guards zero/silent chroma frames).
    norms = torch.norm(chroma, dim=1, keepdim=True)
    chroma_n = chroma / (norms + 1e-8)
    sim_matrix = torch.matmul(chroma_n, chroma_n.T)

    # Desired thumbnail length in frames (hops).
    m_filter = int(round(thumb_size / hop_size))
    m_filter = max(1, min(m_filter, n_frames))

    # Short-input guard: not enough frames to score any off-diagonal stripe.
    if n_frames < 2 or m_filter < 1:
        start_time = 0.0
        end_time = min(float(thumb_size), float(n_frames * hop_size))
        return float(start_time), float(end_time)

    # 3. Score off-diagonal diagonal stripes.
    # For a starting frame ``i`` and a lag ``d > 0`` we measure how well the
    # window [i, i + m_filter) matches the window [i + d, i + d + m_filter) by
    # summing sim_matrix[i + k, i + d + k] for k in [0, m_filter). The best
    # starting frame over all valid lags is the thumbnail start.
    best_score = None
    best_start = 0

    # Minimum lag to be considered "off-diagonal" (avoid the self-similar band).
    min_lag = max(1, m_filter // 2)

    for d in range(min_lag, n_frames):
        # Diagonal at offset d: entries sim[i, i + d] for i in [0, n_frames - d).
        diag = torch.diagonal(sim_matrix, offset=d)  # length n_frames - d
        if diag.numel() < m_filter:
            break
        # Sliding sum of length m_filter over this diagonal (stripe energy).
        stripe = diag.unsqueeze(0).unsqueeze(0)
        kernel = torch.ones(1, 1, m_filter, device=diag.device, dtype=diag.dtype)
        scores = torch.nn.functional.conv1d(stripe, kernel).squeeze(0).squeeze(0)
        local_max_idx = int(torch.argmax(scores).item())
        local_max = float(scores[local_max_idx].item())
        if best_score is None or local_max > best_score:
            best_score = local_max
            best_start = local_max_idx

    # 4. Determine thumbnail start and end times.
    start_frame = max(0, min(best_start, n_frames - 1))
    start_time = float(start_frame) * float(hop_size)
    end_time = start_time + float(thumb_size)

    return float(start_time), float(end_time)
