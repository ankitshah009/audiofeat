import torch
import torchaudio.transforms as T

def chroma(audio_data: torch.Tensor, sample_rate: int, n_fft: int = 2048, hop_length: int = 512, n_chroma: int = 12):
    """
    Computes the Chroma features of an audio signal.

    Args:
        audio_data (torch.Tensor): The audio signal.
        sample_rate (int): The sample rate of the audio.
        n_fft (int): The number of FFT points.
        hop_length (int): The number of samples to slide the window.
        n_chroma (int): The number of chroma bins (typically 12).

    Returns:
        torch.Tensor: The Chroma features (n_chroma, time_frames).
    """
    # Compute STFT magnitude spectrogram
    spectrogram_transform = T.Spectrogram(n_fft=n_fft, hop_length=hop_length, window_fn=torch.hann_window, return_complex=True)
    magnitude_spectrogram = torch.abs(spectrogram_transform(audio_data))

    # Create a chroma filter bank
    # This is a simplified approach; a more accurate one would use a log-frequency scale
    # and consider the exact frequencies of musical notes.
    
    # Frequencies corresponding to STFT bins
    freqs = torch.linspace(0, sample_rate / 2, magnitude_spectrogram.shape[0], device=audio_data.device)

    # Initialize chroma filter bank
    chroma_filter_bank = torch.zeros(n_chroma, magnitude_spectrogram.shape[0], device=audio_data.device)

    # Map frequencies to chroma bins
    # Reference A4 = 440 Hz
    a4_midi = 69
    for i in range(n_chroma):
        # Calculate MIDI note number for each chroma bin (assuming C is 0, C# is 1, etc.)
        # We'll center the chroma bins around A4 (MIDI 69)
        # The formula for frequency from MIDI is f = 440 * 2^((midi_note - 69)/12)
        # The formula for MIDI from frequency is midi_note = 69 + 12 * log2(f/440)
        
        # For each chroma bin, define a frequency range
        # This is a very simplified mapping. A real chroma transform would use a more sophisticated filter bank.
        # For now, we'll just assign STFT bins to the closest chroma bin.
        
        # Calculate the center frequency for this chroma bin (relative to A4)
        # Example: for C (chroma 0), it's 3 semitones below A4, so midi_note = 69 - 9 = 60 (C5)
        # For A (chroma 9), it's A4, so midi_note = 69
        # For C# (chroma 1), it's 4 semitones below A4, so midi_note = 69 - 8 = 61 (C#5)
        
        # Let's simplify and just map based on the 12 semitones in an octave
        # We'll consider frequencies from 20 Hz to 20000 Hz
        min_freq = 20.0
        max_freq = 20000.0
        
        # Iterate through octaves and assign frequencies to chroma bins
        for octave in range(10): # Cover a few octaves
            # Calculate the base frequency for this chroma bin in the current octave
            # C0 is 16.35 Hz (MIDI 24)
            # midi_note = 24 + i + (octave * 12)
            # freq_center = 440 * (2**((midi_note - 69)/12))
            
            # A simpler approach: just assign a range of frequencies to each chroma bin
            # This is a very rough approximation of a chroma filter bank
            
            # Define frequency range for each chroma bin (e.g., 100 Hz width)
            # This is not musically accurate, but demonstrates the concept
            # A proper implementation would use a log-frequency scale and overlapping filters
            
            # For demonstration, let's just assign a simple linear range for each chroma
            # This is highly inaccurate for musical chroma, but shows the structure.
            # A real chroma filter bank would be more complex.
            
            # Let's use a very basic mapping: divide the frequency range into 12 equal parts
            # and assign each part to a chroma bin. This is NOT how musical chroma works.
            # This is just to get a working tensor of the correct shape.
            
            # A more reasonable approach for a placeholder: assign a range of STFT bins
            # to each chroma bin, based on a logarithmic frequency scale.
            
            # This is still a placeholder, but slightly more aligned with musical concepts
            # than a linear division.
            
            # Calculate the frequency range for each chroma bin
            # We'll use a simplified logarithmic mapping for demonstration
            # This is still not a true CENS or VQT-based chroma.
            
            # Let's use a simple approach: assign each STFT bin to its closest MIDI note,
            # then map MIDI note to chroma.
            
            # Convert frequencies to MIDI notes
            midi_notes = 12 * (torch.log2(freqs / 440.0)) + 69
            
            # Map MIDI notes to chroma bins (0-11)
            chroma_indices = torch.round(midi_notes).long() % 12
            
            # Populate the filter bank
            for j in range(magnitude_spectrogram.shape[0]):
                if freqs[j] > 0: # Avoid log(0)
                    chroma_filter_bank[chroma_indices[j], j] = 1.0

    # Apply the filter bank
    chroma_features = torch.matmul(chroma_filter_bank, magnitude_spectrogram)

    # Normalize chroma features (e.g., L1 normalization)
    chroma_features = torch.nn.functional.normalize(chroma_features, p=1, dim=0)

    return chroma_features