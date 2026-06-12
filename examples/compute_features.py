
import torch
import audiofeat

# Create a dummy audio signal
sample_rate = 22050
duration = 10 # Increased duration
audio_data = torch.randn(sample_rate * duration)

# Compute features
rms = audiofeat.rms(audio_data, frame_length=2048, hop_length=512)
ste = audiofeat.short_time_energy(audio_data, frame_length=2048, hop_length=512)
_zcr = audiofeat.zero_crossing_rate(audio_data)
_spectral_centroid = audiofeat.spectral_centroid(audio_data)
_spectral_rolloff = audiofeat.spectral_rolloff(audio_data)
_spectral_rolloff_90 = audiofeat.spectral_rolloff(audio_data, rolloff_percent=0.90)
_spectral_flux = audiofeat.spectral_flux(audio_data)
_spectral_flatness = audiofeat.spectral_flatness(audio_data)
_spectral_entropy = audiofeat.spectral_entropy(audio_data, n_fft=2048)
_spectral_skewness, _ = audiofeat.spectral_skewness(audio_data, n_fft=2048)
_spectral_spread = audiofeat.spectral_spread(audio_data, n_fft=2048, sample_rate=sample_rate)
_spectral_slope = audiofeat.spectral_slope(audio_data, n_fft=2048)
_spectral_crest_factor = audiofeat.spectral_crest_factor(audio_data, n_fft=2048)
_spectral_contrast = audiofeat.spectral_contrast(audio_data, fs=sample_rate)
_harmonic_to_noise_ratio = audiofeat.harmonic_to_noise_ratio(torch.tensor(10.0), torch.tensor(1.0))
_spectral_deviation = audiofeat.spectral_deviation(audio_data, n_fft=2048)
_low_high_energy_ratio = audiofeat.low_high_energy_ratio(audio_data, fs=sample_rate)
_amplitude_modulation_depth = audiofeat.amplitude_modulation_depth(audio_data, window=512)
_breath_group_duration = audiofeat.breath_group_duration(audio_data, fs=sample_rate)
_speech_rate = audiofeat.speech_rate(audio_data, fs=sample_rate)
_log_attack_time = audiofeat.log_attack_time(audio_data, sample_rate)
# Global MPEG-7 temporal centroid (in seconds); the framewise contour is separate.
_temporal_centroid = audiofeat.temporal_centroid(audio_data, sample_rate)
_temporal_centroid_framewise = audiofeat.temporal_centroid_framewise(
    audio_data, frame_length=2048, hop_length=512
)
_entropy_of_energy = audiofeat.entropy_of_energy(audio_data, frame_length=2048, hop_length=512)

# Pitch features
f0_autocorr = audiofeat.fundamental_frequency_autocorr(audio_data, fs=sample_rate, frame_length=2048, hop_length=512)
f0_yin = audiofeat.fundamental_frequency_yin(audio_data, fs=sample_rate, frame_length=2048, hop_length=512)
semitone_sd = audiofeat.semitone_sd(f0_autocorr)
pitch_strength = audiofeat.pitch_strength(audio_data, fs=sample_rate, frame_length=2048, hop_length=512)

# Voice features
_alpha_ratio = audiofeat.alpha_ratio(audio_data, sample_rate)
_hammarberg_index = audiofeat.hammarberg_index(audio_data, sample_rate)
_harmonic_differences = audiofeat.harmonic_differences(torch.randn(1025), f0_hz=100.0, fs=sample_rate)
# Jitter takes a sequence of glottal PERIODS (seconds); shimmer takes a sequence
# of per-cycle AMPLITUDES -- not a raw waveform. Derive periods from the F0
# contour and use the frame RMS as an amplitude proxy.
_voiced_f0 = f0_yin[f0_yin > 0]
_periods = (1.0 / _voiced_f0) if _voiced_f0.numel() > 1 else torch.linspace(0.009, 0.011, 20)
_amplitudes = rms
_jitter = audiofeat.jitter(_periods)                 # local jitter (%)
_jitter_rap = audiofeat.jitter_rap(_periods)          # 3-point RAP (eGeMAPS)
_shimmer = audiofeat.shimmer(_amplitudes)            # local shimmer (%)
_shimmer_apq5 = audiofeat.shimmer_apq5(_amplitudes)  # 5-point APQ
_vocal_fry_index = audiofeat.vocal_fry_index(f0_yin)
_voice_onset_time = audiofeat.voice_onset_time(
    audio_data, fs=sample_rate, frame_length=2048, hop_length=512
)
_vocal_tract_length = audiofeat.vocal_tract_length(500.0, 1500.0)  # F1, F2 in Hz
# HNR estimated directly from a waveform frame via Boersma autocorrelation (dB).
_hnr_acf = audiofeat.harmonic_to_noise_ratio_acf(audio_data[:4096], sample_rate)

# Spectral features (additional)
harmonic_richness_factor = audiofeat.harmonic_richness_factor(torch.randn(10))
inharmonicity_index = audiofeat.inharmonicity_index(torch.randn(10), f0=100)
phase_coherence = audiofeat.phase_coherence(torch.randn(10))
formant_frequencies = audiofeat.formant_frequencies(audio_data, fs=sample_rate, order=10)
formant_bandwidths = audiofeat.formant_bandwidths(torch.randn(10), fs=sample_rate)
formant_dispersion = audiofeat.formant_dispersion(torch.randn(10))
sibilant_spectral_peak_frequency = audiofeat.sibilant_spectral_peak_frequency(audio_data, fs=sample_rate)

# Spectrograms
linear_spec = audiofeat.linear_spectrogram(audio_data)
mel_spec = audiofeat.mel_spectrogram(audio_data, sample_rate)
cqt_spec = audiofeat.cqt_spectrogram(audio_data, sample_rate)

# MFCCs
mfccs = audiofeat.mfcc(audio_data, sample_rate)

# Cepstral features
lpc_coeffs = audiofeat.lpc_coefficients(audio_data[0:2048], order=12)
lsp_coeffs = audiofeat.lsp_coefficients(lpc_coeffs)
lpccs = audiofeat.lpcc(audio_data, sample_rate)
gtccs = audiofeat.gtcc(audio_data, sample_rate)
deltas = audiofeat.delta(mfccs)
delta_deltas = audiofeat.delta_delta(mfccs)

# Tonal and Musical features
chroma_features = audiofeat.chroma(audio_data, sample_rate)
tonnetz_features = audiofeat.tonnetz(chroma_features)

# Apply functionals to a feature series (e.g., RMS). compute_functionals expects a
# 2-D tensor and returns a flat tensor [mean, std, min, max, skew, kurt] per feature.
# Here we reduce a single (1, time) contour over the time axis (time_axis=1).
rms_functionals = audiofeat.compute_functionals(rms.unsqueeze(0), time_axis=1)

# Rhythm features
estimated_tempo = audiofeat.tempo(audio_data, sample_rate)
beat_times = audiofeat.beat_track(audio_data, sample_rate)


# Print the first 5 values of each feature
print("RMS:", rms[:5])
print("Short-Time Energy:", ste[:5])
print("Zero-Crossing Rate:", _zcr[:5])
print("Spectral Centroid:", _spectral_centroid[:5])
print("Spectral Rolloff (85%):", _spectral_rolloff[:5])
print("Spectral Rolloff (90%):", _spectral_rolloff_90[:5])
print("Spectral Flux:", _spectral_flux[:5])
print("Spectral Flatness:", _spectral_flatness[:5])
print("Spectral Entropy:", _spectral_entropy)
print("Spectral Skewness:", _spectral_skewness)
print("Spectral Spread:", _spectral_spread)
print("Spectral Slope:", _spectral_slope)
print("Spectral Crest Factor:", _spectral_crest_factor)
print("Spectral Contrast:", _spectral_contrast)
print("Harmonic to Noise Ratio:", _harmonic_to_noise_ratio)
print("Spectral Deviation:", _spectral_deviation)
print("Low-High Energy Ratio:", _low_high_energy_ratio)
print("Amplitude Modulation Depth:", _amplitude_modulation_depth)
print("Breath Group Duration:", _breath_group_duration)
print("Speech Rate:", _speech_rate)
print("Log Attack Time:", _log_attack_time)
print("Temporal Centroid (seconds):", _temporal_centroid)
print("Temporal Centroid (framewise):", _temporal_centroid_framewise[:5])
print("Entropy of Energy:", _entropy_of_energy[:5])
print("F0 Autocorrelation:", f0_autocorr[:5])
print("F0 YIN:", f0_yin[:5])
print("Semitone SD:", semitone_sd)
print("Pitch Strength:", pitch_strength[:5])
print("Harmonic Richness Factor:", harmonic_richness_factor)
print("Inharmonicity Index:", inharmonicity_index)
print("Phase Coherence:", phase_coherence)
print("Formant Frequencies:", formant_frequencies)
print("Formant Bandwidths:", formant_bandwidths)
print("Formant Dispersion:", formant_dispersion)
print("Sibilant Spectral Peak Frequency:", sibilant_spectral_peak_frequency)
print("Linear Spectrogram shape:", linear_spec.shape)
print("Mel Spectrogram shape:", mel_spec.shape)
print("CQT Spectrogram shape:", cqt_spec.shape)
print("MFCCs shape:", mfccs.shape)
print("LPC Coefficients shape:", lpc_coeffs.shape)
print("LSP Coefficients shape:", lsp_coeffs.shape)
print("LPCCs shape:", lpccs.shape)
print("GTCCs shape:", gtccs.shape)
print("Deltas shape:", deltas.shape)
print("Delta-Deltas shape:", delta_deltas.shape)
print("Chroma Features shape:", chroma_features.shape)
print("Tonnetz Features shape:", tonnetz_features.shape)
print("Alpha Ratio:", _alpha_ratio)
print("Hammarberg Index:", _hammarberg_index)
print("Harmonic Differences:", _harmonic_differences)
print("Jitter (local, %):", float(_jitter))
print("Jitter (RAP, %):", float(_jitter_rap))
print("Shimmer (local, %):", float(_shimmer))
print("Shimmer (APQ5, %):", float(_shimmer_apq5))
print("Vocal Fry Index:", float(_vocal_fry_index))
print("Voice Onset Time:", _voice_onset_time)
print("Vocal Tract Length (cm):", _vocal_tract_length)
print("HNR (ACF, dB):", float(_hnr_acf))
print("RMS Functionals [mean,std,min,max,skew,kurt]:", rms_functionals)
print("Estimated Tempo (BPM):", estimated_tempo)
print("Beat Times (seconds):", beat_times[:5])
