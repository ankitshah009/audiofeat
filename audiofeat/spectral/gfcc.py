import torch
import torchaudio
import torchaudio.transforms as T

def gfcc(waveform: torch.Tensor, sample_rate: int, n_gfcc: int = 40, n_fft: int = 2048, hop_length: int = 512, n_bands: int = 128) -> torch.Tensor:
    """
    Computes Gammatone Frequency Cepstral Coefficients (GFCCs) of an audio waveform.

    GFCCs are similar to MFCCs but use a gammatone filterbank, which is thought
    to better model human auditory perception, especially at lower frequencies.

    Parameters
    ----------
    waveform : torch.Tensor
        Mono audio waveform tensor. Expected shape: (num_samples,) or (1, num_samples).
    sample_rate : int
        Sampling rate of the waveform.
    n_gfcc : int
        Number of GFCCs to return.
    n_fft : int
        Length of the FFT window for spectrogram computation.
    hop_length : int
        Number of samples between successive frames.
    n_bands : int
        Number of gammatone bands to generate.

    Returns
    -------
    torch.Tensor
        GFCC features, shape (n_gfcc, num_frames).

    Notes
    -----
    Torchaudio does not directly provide a Gammatone filterbank or GFCC transform.
    This implementation approximates GFCCs by using a MelSpectrogram (which is similar
    in concept to a gammatone filterbank in terms of logarithmic spacing) and then
    applying the DCT. For a true gammatone filterbank, a custom implementation
    or another library (e.g., librosa) would be needed.
    This serves as a placeholder and conceptual representation within torchaudio's capabilities.
    """
    if waveform.ndim > 1 and waveform.shape[0] > 1:
        waveform = waveform[0]
    elif waveform.ndim == 0:
        raise ValueError("Input waveform cannot be a scalar.")

    # Ensure waveform is float32
    waveform = waveform.to(torch.float32)

    # Approximate Gammatone filterbank with MelSpectrogram for torchaudio compatibility
    # A true GFCC would use a gammatone filterbank, which is not directly in torchaudio.
    # This is a conceptual approximation.
    mel_spectrogram_transform = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_bands, # Using n_bands for n_mels to approximate gammatone bands
        power=2.0, # Power spectrogram
    )
    mel_spec = mel_spectrogram_transform(waveform)

    # Apply log to the power spectrogram (kept here to mirror the intended pipeline).
    _ = torch.log(mel_spec + 1e-8) # Add epsilon to avoid log(0)

    # As torchaudio.transforms.MFCC applies DCT, we can use it conceptually
    # by setting n_mfcc to n_gfcc and using the mel_spectrogram_config.
    # This is a pragmatic approach given torchaudio's current API.
    mfcc_transform = T.MFCC(
        sample_rate=sample_rate,
        n_mfcc=n_gfcc,
        melkwargs={
            "n_fft": n_fft,
            "hop_length": hop_length,
            "n_mels": n_bands,
            "power": 2.0,
        }
    )
    gfccs = mfcc_transform(waveform)

    return gfccs
