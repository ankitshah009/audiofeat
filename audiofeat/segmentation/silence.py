
import torch
from ..temporal.energy import energy

__all__ = ["silence_removal"]


def silence_removal(signal: torch.Tensor, sample_rate: int, window_size: float = 0.05, hop_size: float = 0.025, threshold: float = 0.1):
    """
    Removes silent segments from an audio signal based on short-term energy.

    The energy threshold is applied **relative to the peak frame energy**, i.e.
    a frame is considered non-silent when its energy exceeds
    ``threshold * max(energies)``. This makes the decision amplitude- and
    window-length-robust: scaling the whole signal by a constant (or changing
    the window size) does not change which frames are kept.

    Args:
        signal (torch.Tensor): The input audio signal.
        sample_rate (int): The sample rate of the audio signal.
        window_size (float, optional): The size of the analysis window in seconds. Defaults to 0.05.
        hop_size (float, optional): The hop size between consecutive windows in seconds. Defaults to 0.025.
        threshold (float, optional): The energy threshold as a FRACTION of the
            peak frame energy (``0 < threshold < 1``). Defaults to 0.1.

    Returns:
        torch.Tensor: The audio signal with silent segments removed.
    """
    win_length = int(window_size * sample_rate)
    hop_length = int(hop_size * sample_rate)

    # 1. Calculate short-term energy. Use squeeze(0) so a single-frame result
    #    (shape (1, 1)) does not collapse to a 0-d tensor.
    energies = energy(signal, sample_rate, window_size, hop_size).squeeze(0)

    if energies.numel() == 0:
        return torch.tensor([], device=signal.device, dtype=signal.dtype)

    # 2. Scale-invariant threshold relative to the peak energy.
    max_energy = torch.max(energies)
    if float(max_energy) <= 0.0:
        return torch.tensor([], device=signal.device, dtype=signal.dtype)
    abs_threshold = threshold * max_energy

    non_silent_frames = torch.where(energies > abs_threshold)[0]

    if len(non_silent_frames) == 0:
        return torch.tensor([], device=signal.device, dtype=signal.dtype)

    # 3. Group consecutive non-silent frames into segments.
    segment_starts = [int(non_silent_frames[0].item())]
    segment_ends = []

    for i in range(1, len(non_silent_frames)):
        if int(non_silent_frames[i].item()) > int(non_silent_frames[i - 1].item()) + 1:
            segment_ends.append(int(non_silent_frames[i - 1].item()))
            segment_starts.append(int(non_silent_frames[i].item()))
    segment_ends.append(int(non_silent_frames[-1].item()))

    # 4. Convert frame indices to sample indices and concatenate segments.
    #    Frame f covers samples [f*hop_length, f*hop_length + win_length).
    #    Segments are separated by gaps in frame indices, so the sample ranges
    #    below never overlap; clamping prevents reading past the signal end.
    n = signal.shape[0]
    result_signal = []
    prev_end_sample = 0
    for start, end in zip(segment_starts, segment_ends):
        start_sample = max(0, start * hop_length)
        end_sample = min(n, end * hop_length + win_length)
        # Guard against any residual overlap with a previous segment.
        start_sample = max(start_sample, prev_end_sample)
        if end_sample <= start_sample:
            continue
        result_signal.append(signal[start_sample:end_sample])
        prev_end_sample = end_sample

    if not result_signal:
        return torch.tensor([], device=signal.device, dtype=signal.dtype)

    return torch.cat(result_signal)
