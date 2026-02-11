from __future__ import annotations

import numpy as np
import torch


def _to_mono_tensor(waveform: torch.Tensor) -> torch.Tensor:
    if waveform.dim() == 1:
        return waveform.float()
    if waveform.dim() == 2:
        if waveform.shape[0] == 1:
            return waveform.squeeze(0).float()
        return waveform.mean(dim=0).float()
    raise ValueError("waveform must be 1-D or 2-D (channels, samples).")


def _fallback_beat_track(
    waveform: torch.Tensor,
    sample_rate: int,
    hop_length: int,
    tempo_min: float,
    tempo_max: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = waveform - waveform.mean()
    n = x.numel()
    if n < 4:
        return (
            torch.tensor(0.0, dtype=torch.float32, device=waveform.device),
            torch.zeros(0, dtype=torch.int64, device=waveform.device),
        )
    onset_env = torch.relu(x[1:] - x[:-1])
    onset_env = torch.nn.functional.avg_pool1d(
        onset_env.view(1, 1, -1),
        kernel_size=max(1, hop_length),
        stride=max(1, hop_length),
    ).flatten()
    if onset_env.numel() < 4:
        return (
            torch.tensor(0.0, dtype=torch.float32, device=waveform.device),
            torch.zeros(0, dtype=torch.int64, device=waveform.device),
        )

    onset_env = onset_env - onset_env.mean()
    ac = torch.fft.irfft(torch.fft.rfft(onset_env) * torch.conj(torch.fft.rfft(onset_env)))

    min_lag = max(1, int(round((60.0 / tempo_max) * sample_rate / hop_length)))
    max_lag = min(ac.numel() - 1, int(round((60.0 / tempo_min) * sample_rate / hop_length)))
    if max_lag <= min_lag:
        return (
            torch.tensor(0.0, dtype=torch.float32, device=waveform.device),
            torch.zeros(0, dtype=torch.int64, device=waveform.device),
        )

    lag = int(torch.argmax(ac[min_lag:max_lag]).item()) + min_lag
    tempo = 60.0 * sample_rate / (hop_length * lag)
    beat_frames = torch.arange(0, onset_env.numel(), step=max(lag, 1), device=waveform.device)
    return (
        torch.tensor(float(tempo), dtype=torch.float32, device=waveform.device),
        beat_frames.long(),
    )


def beat_track(
    waveform: torch.Tensor,
    sample_rate: int,
    n_fft: int = 2048,
    hop_length: int = 512,
    tempo_min: float = 60.0,
    tempo_max: float = 240.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Beat tracking with librosa parity when available.
    """
    if sample_rate <= 0:
        raise ValueError("sample_rate must be > 0.")
    if hop_length <= 0:
        raise ValueError("hop_length must be > 0.")
    if tempo_min <= 0 or tempo_max <= tempo_min:
        raise ValueError("Expected 0 < tempo_min < tempo_max.")

    x = _to_mono_tensor(waveform)
    device = waveform.device

    try:
        import librosa  # type: ignore
    except ModuleNotFoundError:
        return _fallback_beat_track(x, sample_rate, hop_length, tempo_min, tempo_max)

    y = x.detach().cpu().numpy().astype(np.float32, copy=False)
    onset_env = librosa.onset.onset_strength(
        y=y,
        sr=int(sample_rate),
        n_fft=int(n_fft),
        hop_length=int(hop_length),
        aggregate=np.median,
    )
    tempo, beat_frames = librosa.beat.beat_track(
        onset_envelope=onset_env,
        sr=int(sample_rate),
        hop_length=int(hop_length),
        start_bpm=float(np.clip((tempo_min + tempo_max) / 2.0, 40.0, 300.0)),
        tightness=100,
    )
    tempo_val = float(np.asarray(tempo).flat[0])
    tempo_tensor = torch.tensor(tempo_val, dtype=torch.float32, device=device)
    beat_tensor = torch.from_numpy(np.asarray(beat_frames, dtype=np.int64)).to(device=device)
    return tempo_tensor, beat_tensor
