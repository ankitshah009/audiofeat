import torch
import torchaudio
from ..spectral.mfcc import mfcc

def kmeans(X, n_clusters, max_iters=100):
    """
    A simple implementation of K-Means clustering using PyTorch.

    Args:
        X (torch.Tensor): The input data (n_samples, n_features).
        n_clusters (int): The number of clusters.
        max_iters (int, optional): The maximum number of iterations. Defaults to 100.

    Returns:
        torch.Tensor: The cluster assignments for each data point.
    """
    # Randomly initialize centroids
    centroids = X[torch.randperm(X.size(0))[:n_clusters]]

    for _ in range(max_iters):
        # Assign each data point to the closest centroid
        dists = torch.cdist(X, centroids)
        labels = torch.argmin(dists, dim=1)

        # Update centroids
        new_centroids = torch.zeros_like(centroids)
        for i in range(n_clusters):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                new_centroids[i] = cluster_points.mean(dim=0)
            else:
                # Handle empty clusters by re-initializing the centroid
                new_centroids[i] = X[torch.randperm(X.size(0))[:1]].squeeze(0)

        # Check for convergence
        if torch.allclose(centroids, new_centroids):
            break
        centroids = new_centroids

    return labels

def speaker_diarization(signal: torch.Tensor, sample_rate: int, n_speakers: int, window_size: float = 0.05, hop_size: float = 0.025):
    """
    Performs speaker diarization on an audio signal.

    Args:
        signal (torch.Tensor): The input audio signal.
        sample_rate (int): The sample rate of the audio signal.
        n_speakers (int): The number of speakers.
        window_size (float, optional): The size of the analysis window in seconds. Defaults to 0.05.
        hop_size (float, optional): The hop size between consecutive windows in seconds. Defaults to 0.025.

    Returns:
        torch.Tensor: A tensor containing the speaker label for each segment.
    """
    # 1. Extract MFCCs
    if signal.shape[0] < int(window_size * sample_rate):
        raise ValueError("Input signal must be longer than window size.")

    # Ensure signal is 2D (batch, n_samples) for mfcc function
    if signal.dim() == 1:
        signal = signal.unsqueeze(0)

    mfccs = mfcc(signal, sample_rate, n_mfcc=13)

    # torchaudio versions >= 2.1 may return shape (batch, n_mels, n_mfcc, n_frames).
    # If we detect such a shape, average across the Mel dimension to obtain
    # the expected (batch, n_mfcc, n_frames) format.
    if mfccs.dim() == 4 and mfccs.shape[1] > mfccs.shape[2]:
        # (batch, n_mels, n_mfcc, n_frames) -> mean over n_mels
        mfccs = mfccs.mean(dim=1)

    # Now mfccs should be (batch, n_mfcc, n_frames)
    mfccs = mfccs.squeeze(0)

    if mfccs.dim() != 2:
        # As a fall-back, collapse any extra leading dimensions except feature/time.
        mfccs = mfccs.view(mfccs.shape[-2], mfccs.shape[-1])

    # Transpose to (n_frames, n_mfcc) for k-means
    mfccs = mfccs.permute(1, 0)

    # 2. Perform K-Means clustering
    labels = kmeans(mfccs, n_speakers)

    return labels
