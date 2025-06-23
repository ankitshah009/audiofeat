import argparse
import torch
import torchaudio

from audiofeat import (
    rms,
    fundamental_frequency_autocorr,
    fundamental_frequency_yin,
    spectral_entropy,
    spectral_rolloff,
    pitch_strength,
    spectral_skewness,
    low_high_energy_ratio,
    amplitude_modulation_depth,
    spectral_flux_frames,
    speech_rate,
)


def main(audio_path: str):
    # Load the waveform and convert to mono
    waveform, fs = torchaudio.load(audio_path)
    waveform = waveform.mean(dim=0)

    frame_length = int(0.03 * fs)
    hop_length = int(0.01 * fs)

    # Compute several acoustic features
    features = {}
    features["rms"] = rms(waveform, frame_length, hop_length)
    features["f0"] = fundamental_frequency_autocorr(waveform, fs, frame_length, hop_length)
    features["f0_yin"] = fundamental_frequency_yin(waveform, fs, frame_length, hop_length)
    features["spectral_entropy"] = spectral_entropy(waveform[:frame_length], 1024)
    features["spectral_rolloff"] = spectral_rolloff(waveform[:frame_length], fs)
    features["pitch_strength"] = pitch_strength(waveform, fs, frame_length, hop_length)
    skew, kurt = spectral_skewness(waveform[:frame_length], 1024)
    features["spectral_skewness"] = skew
    features["spectral_kurtosis"] = kurt
    features["low_high_energy_ratio"] = low_high_energy_ratio(waveform[:frame_length], fs)

    # Envelope based features
    env = torch.abs(waveform)
    win_env = torch.nn.functional.avg_pool1d(env.view(1,1,-1), kernel_size=int(0.01*fs), stride=1, padding=int(0.005*fs)).squeeze()
    features["amplitude_mod_depth"] = amplitude_modulation_depth(win_env, int(0.2*fs))

    # Spectral flux and speech rate
    features["spectral_flux"] = spectral_flux_frames(waveform, 1024, hop_length)
    features["speech_rate"] = speech_rate(waveform, fs)

    for name, value in features.items():
        print(f"{name}: {value}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute example speech features")
    parser.add_argument("audio", help="Path to a WAV file")
    args = parser.parse_args()
    main(args.audio)
