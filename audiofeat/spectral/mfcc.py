
import warnings

import torch
import torchaudio.transforms as T

def mfcc(audio_data: torch.Tensor, sample_rate: int, n_mfcc: int = 40, n_fft: int = 2048,
         hop_length: int = 512, n_mels: int = 128, mel_scale: str = "htk",
         log_mels: bool = False):
    """
    Computes the Mel-Frequency Cepstral Coefficients (MFCCs) of an audio signal.

    Conventions (delegates to :class:`torchaudio.transforms.MFCC`):

    * **Mel scale**: ``mel_scale`` (default ``"htk"``). This differs from
      ``gfcc``/``gtcc`` in this library, which use a Slaney-style ERB/gammatone
      filterbank. Pass ``mel_scale="slaney"`` for a Slaney mel scale.
    * **Amplitude compression**: when ``log_mels=False`` (default) the mel
      spectrogram is converted to **decibels** via ``AmplitudeToDB`` (the
      torchaudio MFCC default); when ``log_mels=True`` a natural log is used,
      matching the ``log``-domain convention of ``gfcc``/``gtcc``.
    * **DCT**: type-II DCT with **orthonormal** (``norm="ortho"``) basis.

    Args:
        audio_data (torch.Tensor): The audio signal (flattened internally).
        sample_rate (int): The sample rate of the audio.
        n_mfcc (int): Number of MFCCs to return (output rows). See the
            short-input note below.
        n_fft (int): FFT size for the mel spectrogram.
        hop_length (int): Hop length for the mel spectrogram.
        n_mels (int): Number of mel filterbank bands.
        mel_scale (str): ``"htk"`` (default) or ``"slaney"``.
        log_mels (bool): If True use natural-log compression instead of dB.

    Returns:
        torch.Tensor: MFCCs of shape ``(n_mfcc, n_frames)`` for a 1D/normal
        input.

    Short-input behaviour:
        For clips shorter than ``n_fft`` the spectrogram parameters
        (``n_fft``, ``hop_length``, ``n_mels``) are reduced so an STFT is still
        well-defined. Because torchaudio requires ``n_mfcc <= n_mels``, if the
        reduced ``n_mels`` would be smaller than the requested ``n_mfcc`` the
        coefficient count is clamped to ``n_mels`` **and a UserWarning is
        emitted** (the count is never silently reduced).
    """
    audio_data = audio_data.flatten().float()
    if audio_data.numel() == 0:
        raise ValueError("audio_data must be non-empty.")

    # Keep MFCC extraction stable for short clips by adapting spectrogram params.
    effective_n_fft = int(min(n_fft, max(16, audio_data.numel())))
    effective_hop = int(min(hop_length, max(1, effective_n_fft // 2)))
    effective_n_mels = int(min(n_mels, max(8, effective_n_fft // 2)))

    # torchaudio constraint: n_mfcc must not exceed n_mels. Only clamp when the
    # (possibly reduced) filterbank cannot supply enough coefficients, and warn
    # so the dimensionality change is never silent.
    effective_n_mfcc = int(n_mfcc)
    if effective_n_mfcc > effective_n_mels:
        warnings.warn(
            f"n_mfcc={n_mfcc} exceeds available mel bands "
            f"({effective_n_mels}) for an input of length {audio_data.numel()}; "
            f"clamping n_mfcc to {effective_n_mels}.",
            UserWarning,
            stacklevel=2,
        )
        effective_n_mfcc = effective_n_mels

    mfcc_transform = T.MFCC(
        sample_rate=sample_rate,
        n_mfcc=effective_n_mfcc,
        log_mels=log_mels,
        melkwargs={
            "n_fft": effective_n_fft,
            "hop_length": effective_hop,
            "n_mels": effective_n_mels,
            "mel_scale": mel_scale,
            "center": False,
        },
    )
    return mfcc_transform(audio_data)
