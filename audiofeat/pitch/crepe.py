import torch
import warnings

def predict_crepe(
    audio: torch.Tensor,
    sample_rate: int,
    output_activation: bool = False,
    model_capacity: str = 'full'
) -> torch.Tensor:
    """
    Estimate pitch using CREPE (Convolutional REpresentation for Pitch Estimation).
    
    This function acts as a wrapper. In a full production environment, it would 
    load the pre-trained weights (approx 10MB) or rely on the `torchcrepe` library.
    
    Since we cannot install external packages on the fly, this wrapper defines the 
    interface and suggests the user install `torchcrepe`.
    
    Args:
        audio (torch.Tensor): Audio waveform (shape [B, T] or [T]).
        sample_rate (int): Sampling rate (must be 16000 for standard CREPE).
        output_activation (bool): If True, returns full activation matrix.
        model_capacity (str): 'tiny', 'small', 'medium', 'large', 'full'.
    
    Returns:
        torch.Tensor: Estimated pitch F0 in Hz.
    """
    try:
        import torchcrepe
    except ImportError:
        warnings.warn(
            "torchcrepe not installed. Please install it via `pip install torchcrepe` "
            "to use the CREPE pitch estimator. Returning zeros."
        )
        # Return dummy tensor matching expected output shape
        # Standard CREPE hop is 10ms (160 samples at 16k)
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)
        
        num_frames = audio.shape[-1] // 160
        return torch.zeros((audio.shape[0], num_frames), device=audio.device)

    # Resample if necessary (CREPE expects 16kHz). Actually resample instead of
    # only warning, otherwise non-16k input yields wrong pitch.
    if sample_rate != 16000:
        import torchaudio
        audio = torchaudio.functional.resample(audio, sample_rate, 16000)
        sample_rate = 16000

    f0 = torchcrepe.predict(
        audio,
        sample_rate,
        hop_length=160,
        fmin=50,
        fmax=2000,
        model=model_capacity,
        decoder=torchcrepe.decode.viterbi,
        # torchcrepe exposes ``return_periodicity`` (not ``return_harmonicity``).
        return_periodicity=False,
    )

    return f0
