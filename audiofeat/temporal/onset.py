from __future__ import annotations

import numpy as np
import torch
import torchaudio.transforms as T


def _to_mono_tensor(waveform: torch.Tensor) -> torch.Tensor:
    if waveform.dim() == 1:
        return waveform.float()
    if waveform.dim() == 2:
        if waveform.shape[0] == 1:
            return waveform.squeeze(0).float()
        return waveform.mean(dim=0).float()
    raise ValueError("waveform must be 1-D or 2-D (channels, samples).")


def _fallback_onset_frames(
    waveform: torch.Tensor,
    sample_rate: int,
    n_fft: int,
    hop_length: int,
) -> torch.Tensor:
    mel = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=64,
        power=1.0,
    )(waveform)
    log_mel = torch.log(mel + 1e-8)
    flux = torch.relu(log_mel[:, 1:] - log_mel[:, :-1]).mean(dim=0)
    flux = torch.cat([flux.new_zeros(1), flux], dim=0)

    med = torch.median(flux)
    mad = torch.median(torch.abs(flux - med))
    threshold = med + 1.5 * mad

    peaks = []
    for i in range(1, flux.numel() - 1):
        if flux[i] > threshold and flux[i] >= flux[i - 1] and flux[i] > flux[i + 1]:
            peaks.append(i)
    if not peaks:
        return torch.zeros(0, dtype=torch.long, device=waveform.device)
    return torch.tensor(peaks, dtype=torch.long, device=waveform.device)


def onset_detect(
    waveform: torch.Tensor,
    sample_rate: int,
    n_fft: int = 2048,
    hop_length: int = 512,
    backtrack: bool = True,
) -> torch.Tensor:
    """
    Detect onset times, matching librosa onset behavior when available.
    """
    if sample_rate <= 0:
        raise ValueError("sample_rate must be > 0.")
    if n_fft <= 0 or hop_length <= 0:
        raise ValueError("n_fft and hop_length must be > 0.")

    x = _to_mono_tensor(waveform)
    device = waveform.device
    if x.numel() == 0:
        return torch.zeros(0, dtype=torch.float32, device=device)

    try:
        import librosa  # type: ignore
    except ModuleNotFoundError:
        frames = _fallback_onset_frames(x, sample_rate, n_fft, hop_length)
        times = frames.float() * float(hop_length) / float(sample_rate)
        return times.to(device=device)

    y = x.detach().cpu().numpy().astype(np.float32, copy=False)
    onset_env = librosa.onset.onset_strength(
        y=y,
        sr=int(sample_rate),
        hop_length=int(hop_length),
        n_fft=int(n_fft),
        aggregate=np.median,
    )
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env,
        sr=int(sample_rate),
        hop_length=int(hop_length),
        backtrack=bool(backtrack),
        units="frames",
    )
    onset_times = librosa.frames_to_time(
        onset_frames,
        sr=int(sample_rate),
        hop_length=int(hop_length),
    )
    return torch.from_numpy(onset_times.astype(np.float32, copy=False)).to(device=device)
