from __future__ import annotations

import warnings

import numpy as np
import torch


def fundamental_frequency_pyin(
    x: torch.Tensor,
    fs: int,
    frame_length: int = 2048,
    hop_length: int = 512,
    fmin: float = 50.0,
    fmax: float = 600.0,
    fill_unvoiced: float = 0.0,
) -> torch.Tensor:
    """
    Estimate F0 with probabilistic YIN (pYIN) via librosa.

    This is an optional high-robustness estimator that can outperform plain YIN
    in low-SNR/expressive speech settings due to Viterbi smoothing.
    """
    try:
        import librosa  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "librosa is required for pYIN pitch extraction. "
            "Install with `pip install \"audiofeat[examples]\"` or `pip install librosa`."
        ) from exc

    waveform = x.flatten().detach().cpu().numpy().astype(np.float32, copy=False)
    if waveform.size == 0:
        raise ValueError("Input waveform must be non-empty.")

    if fmin <= 0 or fmax <= fmin:
        raise ValueError("Expected 0 < fmin < fmax for pYIN.")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f0, voiced_flag, _voiced_prob = librosa.pyin(
            waveform,
            fmin=float(fmin),
            fmax=float(fmax),
            sr=int(fs),
            frame_length=int(frame_length),
            hop_length=int(hop_length),
        )

    if f0 is None:
        return torch.zeros(0, dtype=torch.float32, device=x.device)

    f0 = np.asarray(f0, dtype=np.float32)
    if np.isnan(fill_unvoiced):
        # Keep NaNs for unvoiced frames.
        pass
    else:
        if voiced_flag is not None:
            f0[~np.asarray(voiced_flag, dtype=bool)] = float(fill_unvoiced)
        else:
            f0[~np.isfinite(f0)] = float(fill_unvoiced)

    out = torch.from_numpy(f0)
    return out.to(device=x.device)
