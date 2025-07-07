
import torch
from ..spectral.chroma import chroma as chroma_stft

def music_thumbnailing(signal: torch.Tensor, sample_rate: int, thumb_size: float = 10.0, window_size: float = 1.0, hop_size: float = 0.5):
    """
    Detects the most representative part of a music recording.

    Args:
        signal (torch.Tensor): The input audio signal.
        sample_rate (int): The sample rate of the audio signal.
        thumb_size (float, optional): The desired thumbnail size in seconds. Defaults to 10.0.
        window_size (float, optional): The size of the analysis window in seconds. Defaults to 1.0.
        hop_size (float, optional): The hop size between consecutive windows in seconds. Defaults to 0.5.

    Returns:
        Tuple[float, float]: A tuple containing the start and end times of the thumbnail in seconds.
    """
    # 1. Extract Chroma features
    chroma = chroma_stft(signal, sample_rate, n_chroma=12, n_fft=int(window_size*sample_rate), hop_length=int(hop_size*sample_rate))
    chroma = chroma.squeeze(0).T # Reshape to (n_frames, n_chroma)

    # 2. Compute self-similarity matrix
    sim_matrix = torch.matmul(chroma, chroma.T)
    sim_matrix = sim_matrix / (torch.norm(chroma, dim=1).unsqueeze(1) * torch.norm(chroma, dim=1).unsqueeze(0)) # Cosine similarity

    # 3. Apply a diagonal mask for thumbnailing
    m_filter = int(round(thumb_size / hop_size))
    kernel = torch.eye(m_filter).unsqueeze(0).unsqueeze(0)
    sim_matrix_filtered = torch.nn.functional.conv2d(sim_matrix.unsqueeze(0).unsqueeze(0), kernel, padding='valid').squeeze()

    # 4. Find the max value in the filtered similarity matrix
    max_pos = torch.argmax(sim_matrix_filtered)
    row, col = max_pos // sim_matrix_filtered.shape[1], max_pos % sim_matrix_filtered.shape[1]

    # 5. Determine thumbnail start and end times
    start_time = row * hop_size
    end_time = start_time + thumb_size

    return start_time, end_time
