
import torch

def log_attack_time(audio_data: torch.Tensor, sample_rate: int, threshold: float = 0.01):
    """
    Computes the log attack time of an audio signal.

    Args:
        audio_data (torch.Tensor): The audio signal.
        sample_rate (int): The sample rate of the audio.
        threshold (float): The threshold for attack detection.

    Returns:
        float: The log attack time.
    """
    # Simple attack time estimation (can be improved with more sophisticated methods)
    # Find the first point where the amplitude exceeds the threshold
    onset_idx = (audio_data.abs() > threshold * audio_data.abs().max()).nonzero(as_tuple=True)[0]
    if onset_idx.numel() == 0:
        return 0.0
    
    # Find the peak after the onset
    peak_idx = torch.argmax(audio_data.abs()[onset_idx[0]:]) + onset_idx[0]
    
    if peak_idx <= onset_idx[0]:
        return 0.0

    attack_time = (peak_idx - onset_idx[0]).float() / sample_rate
    return torch.log10(attack_time + 1e-8).item()
