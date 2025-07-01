import torch
import torchaudio
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

# Assume audiofeat is installed or accessible in PYTHONPATH
# from audiofeat.spectral import cqt, gfcc
# from audiofeat.voice import jitter, shimmer
# from audiofeat.temporal import loudness

# For demonstration, we'll use dummy functions if audiofeat is not fully set up
# In a real scenario, you would import and use the actual functions.

def _load_audio(file_path: str, sample_rate: int = 16000):
    """Loads an audio file and resamples it to the target sample rate."""
    try:
        waveform, sr = torchaudio.load(file_path)
        if sr != sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)
            waveform = resampler(waveform)
        # Ensure mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        return waveform.squeeze(0) # Remove channel dimension
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def extract_features_for_file(file_path: str, output_dir: str, sample_rate: int = 16000):
    """
    Extracts a set of features from a single audio file.
    This function will be run in parallel.
    """
    print(f"Processing {file_path}...")
    waveform = _load_audio(file_path, sample_rate)
    if waveform is None:
        return f"Failed to process {file_path}"

    features = {}
    try:
        # Example feature extraction (replace with actual audiofeat calls)
        # For demonstration, using dummy values or simple torch operations
        features['rms'] = torch.sqrt(torch.mean(waveform**2)).item()
        features['zcr'] = torch.mean(torch.abs(torch.diff(torch.sign(waveform)))).item() / 2

        # If audiofeat functions were imported:
        # features['cqt'] = cqt(waveform, sample_rate).mean().item()
        # features['gfcc'] = gfcc(waveform, sample_rate).mean().item()
        # features['jitter'] = jitter(waveform, sample_rate)
        # features['shimmer'] = shimmer(waveform, sample_rate)
        # features['loudness'] = loudness(waveform, sample_rate).item()

        # Save features to a file (e.g., .pt for PyTorch tensor, or .json)
        output_filename = os.path.join(output_dir, os.path.basename(file_path).replace('.wav', '.pt'))
        torch.save(features, output_filename)
        return f"Successfully processed {file_path}"
    except Exception as e:
        return f"Error extracting features from {file_path}: {e}"

def parallel_feature_extraction(audio_files: list, output_dir: str, num_processes: int = None, sample_rate: int = 16000):
    """
    Orchestrates parallel feature extraction using a ProcessPoolExecutor.
    Leverages disk I/O by processing files independently.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if num_processes is None:
        num_processes = os.cpu_count() # Use all available CPU cores

    print(f"Starting parallel feature extraction with {num_processes} processes...")
    start_time = time.time()

    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        # Submit tasks to the executor
        future_to_file = {executor.submit(extract_features_for_file, file_path, output_dir, sample_rate): file_path for file_path in audio_files}

        for future in as_completed(future_to_file):
            file_path = future_to_file[future]
            try:
                result = future.result()
                print(result)
            except Exception as exc:
                print(f'{file_path} generated an exception: {exc}')

    end_time = time.time()
    print(f"Parallel feature extraction completed in {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    # Create some dummy audio files for demonstration
    dummy_audio_dir = "dummy_audio_files"
    if not os.path.exists(dummy_audio_dir):
        os.makedirs(dummy_audio_dir)

    num_dummy_files = 10
    dummy_files = []
    for i in range(num_dummy_files):
        dummy_file_path = os.path.join(dummy_audio_dir, f"audio_{i:03d}.wav")
        # Create a dummy waveform (e.g., 1 second of random noise)
        dummy_waveform = torch.randn(1, 16000) # 1 channel, 16000 samples (1 second at 16kHz)
        torchaudio.save(dummy_file_path, dummy_waveform, 16000)
        dummy_files.append(dummy_file_path)
    print(f"Created {num_dummy_files} dummy audio files in {dummy_audio_dir}/")

    output_features_dir = "extracted_features"
    parallel_feature_extraction(dummy_files, output_features_dir, num_processes=4)

    # Clean up dummy files and directory
    for f in dummy_files:
        os.remove(f)
    os.rmdir(dummy_audio_dir)
    print(f"Cleaned up dummy audio files and directory {dummy_audio_dir}/")

    # You can inspect the extracted_features directory to see the .pt files
    print(f"Extracted features saved to {output_features_dir}/")
    # Example of loading a saved feature file
    # if os.path.exists(os.path.join(output_features_dir, "audio_000.pt")):
    #     loaded_features = torch.load(os.path.join(output_features_dir, "audio_000.pt"))
    #     print(f"Example loaded features: {loaded_features}")
