
import torch
import audiofeat

# Create a dummy audio signal
sample_rate = 22050
duration = 5
audio_data = torch.randn(sample_rate * duration)

# Compute features
rms = audiofeat.rms(audio_data, frame_length=2048, hop_length=512)
_zcr = audiofeat.zero_crossing_rate(audio_data)
_spectral_centroid = audiofeat.spectral_centroid(audio_data)
_spectral_rolloff = audiofeat.spectral_rolloff(audio_data)
_spectral_flux = audiofeat.spectral_flux(audio_data)
_spectral_flatness = audiofeat.spectral_flatness(audio_data)
_spectral_entropy = audiofeat.spectral_entropy(audio_data, n_fft=2048)
_spectral_skewness, _ = audiofeat.spectral_skewness(audio_data, n_fft=2048)
_low_high_energy_ratio = audiofeat.low_high_energy_ratio(audio_data, fs=sample_rate)
_amplitude_modulation_depth = audiofeat.amplitude_modulation_depth(audio_data, window=512)
_breath_group_duration = audiofeat.breath_group_duration(audio_data, fs=sample_rate)
_speech_rate = audiofeat.speech_rate(audio_data, fs=sample_rate)

# Pitch features
f0_autocorr = audiofeat.fundamental_frequency_autocorr(audio_data, fs=sample_rate, frame_length=2048, hop_length=512)
f0_yin = audiofeat.fundamental_frequency_yin(audio_data, fs=sample_rate, frame_length=2048, hop_length=512)
semitone_sd = audiofeat.semitone_sd(f0_autocorr)

# Voice features
# jitter = audiofeat.jitter(torch.randn(10))
# shimmer = audiofeat.shimmer(torch.randn(10))
# subharmonic_to_harmonic_ratio = audiofeat.subharmonic_to_harmonic_ratio(torch.randn(10), f0_bin=5, num_harmonics=3)
# normalized_amplitude_quotient = audiofeat.normalized_amplitude_quotient(torch.randn(1), torch.randn(1), torch.randn(1))
# closed_quotient = audiofeat.closed_quotient(torch.randn(1), torch.randn(1), torch.randn(1))
# glottal_closure_time = audiofeat.glottal_closure_time(torch.randn(10), torch.randn(10), torch.randn(10))
# soft_phonation_index = audiofeat.soft_phonation_index(torch.randn(1), torch.randn(1))
# speed_quotient = audiofeat.speed_quotient(torch.randn(10), torch.randn(10))
# vocal_fry_index = audiofeat.vocal_fry_index(torch.randn(10))
# voice_onset_time = audiofeat.voice_onset_time(audio_data, fs=sample_rate, frame_length=2048, hop_length=512)
# glottal_to_noise_excitation = audiofeat.glottal_to_noise_excitation(torch.randn(6, 10))
# vocal_tract_length = audiofeat.vocal_tract_length(1000, 2000)
# maximum_flow_declination_rate = audiofeat.maximum_flow_declination_rate(torch.randn(10), fs=sample_rate)
# nasality_index = audiofeat.nasality_index(torch.randn(100), torch.randn(100), fs=sample_rate)

# Spectral features (additional)
harmonic_richness_factor = audiofeat.harmonic_richness_factor(torch.randn(10))
inharmonicity_index = audiofeat.inharmonicity_index(torch.randn(10), f0=100)
phase_coherence = audiofeat.phase_coherence(torch.randn(10))
formant_frequencies = audiofeat.formant_frequencies(audio_data, fs=sample_rate, order=10)
formant_bandwidths = audiofeat.formant_bandwidths(torch.randn(10), fs=sample_rate)
formant_dispersion = audiofeat.formant_dispersion(torch.randn(10))
sibilant_spectral_peak_frequency = audiofeat.sibilant_spectral_peak_frequency(audio_data, fs=sample_rate)


# Print the first 5 values of each feature
print("RMS:", rms[:5])
print("Zero-Crossing Rate:", _zcr[:5])
print("Spectral Centroid:", _spectral_centroid[:5])
print("Spectral Rolloff:", _spectral_rolloff[:5])
print("Spectral Flux:", _spectral_flux[:5])
print("Spectral Flatness:", _spectral_flatness[:5])
print("Spectral Entropy:", _spectral_entropy)
print("Spectral Skewness:", _spectral_skewness)
print("Low-High Energy Ratio:", _low_high_energy_ratio)
print("Amplitude Modulation Depth:", _amplitude_modulation_depth)
print("Breath Group Duration:", _breath_group_duration)
print("Speech Rate:", _speech_rate)
print("F0 Autocorrelation:", f0_autocorr[:5])
print("F0 YIN:", f0_yin[:5])
print("Semitone SD:", semitone_sd)
print("Harmonic Richness Factor:", harmonic_richness_factor)
print("Inharmonicity Index:", inharmonicity_index)
print("Phase Coherence:", phase_coherence)
print("Formant Frequencies:", formant_frequencies)
print("Formant Bandwidths:", formant_bandwidths)
print("Formant Dispersion:", formant_dispersion)
print("Sibilant Spectral Peak Frequency:", sibilant_spectral_peak_frequency)
