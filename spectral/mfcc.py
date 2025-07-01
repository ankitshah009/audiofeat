import torch
import torchaudio
import torchaudio.transforms as T

def mfcc(waveform: torch.Tensor, sample_rate: int, n_mfcc: int = 40,
         mel_n_fft: int = 2048, mel_hop_length: int = 512, mel_n_mels: int = 128,
         mel_f_min: float = 0.0, mel_f_max: float = None) -> torch.Tensor:
    """
    Computes Mel-frequency cepstral coefficients (MFCCs) of an audio waveform.

    MFCCs are widely used features in speech recognition and music information
    retrieval. They represent the short-term power spectrum of a sound,
    based on a linear cosine transform of a log power spectrum on a nonlinear
    Mel scale.

    Parameters
    ----------
    waveform : torch.Tensor
        Mono audio waveform tensor. Expected shape: (num_samples,) or (1, num_samples).
    sample_rate : int
        Sampling rate of the waveform.
    n_mfcc : int
        Number of MFCCs to return.
    mel_n_fft : int
        Length of the FFT window for Mel spectrogram computation.
    mel_hop_length : int
        Number of samples between successive frames for Mel spectrogram computation.
    mel_n_mels : int
        Number of Mel bands to generate for Mel spectrogram computation.
    mel_f_min : float
        Minimum frequency (Hz) for the Mel filterbank.
    mel_f_max : float or None
        Maximum frequency (Hz) for the Mel filterbank. If None, defaults to sample_rate / 2.

    Returns
    -------
    torch.Tensor
        MFCC features, shape (n_mfcc, num_frames).

    Notes
    -----
    This implementation uses torchaudio's `MFCC` transform, which internally
    computes the Mel spectrogram and then applies the Discrete Cosine Transform (DCT).
    Requires 'torch' and 'torchaudio' to be installed.
    """
    if waveform.ndim > 1 and waveform.shape[0] > 1:
        # Assuming mono or taking the first channel if multi-channel
        waveform = waveform[0]
    elif waveform.ndim == 0:
        raise ValueError("Input waveform cannot be a scalar.")

    if mel_f_max is None:
        mel_f_max = float(sample_rate / 2)

    # Create the MFCC transform
    mfcc_transform = T.MFCC(
        sample_rate=sample_rate,
        n_mfcc=n_mfcc,
        mel_spectrogram_config={
            "n_fft": mel_n_fft,
            "hop_length": mel_hop_length,
            "n_mels": mel_n_mels,
            "f_min": mel_f_min,
            "f_max": mel_f_max,
            "power": 2.0, # Power spectrogram
        }
    )

    # Compute MFCCs
    mfccs = mfcc_transform(waveform)

    return mfccs
