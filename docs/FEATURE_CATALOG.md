| Component | Function | Signature | Description |
|---|---|---|---|
| `audiofeat.cepstral` | `delta` | `(x: torch.Tensor, width: int = 9)` | Computes the first-order derivative (delta) of a feature contour |
| `audiofeat.cepstral` | `delta_delta` | `(x: torch.Tensor, width: int = 9)` | Computes the second-order derivative (delta-delta) of a feature contour |
| `audiofeat.cepstral` | `gtcc` | `(audio_data: torch.Tensor, sample_rate: int, n_gtcc: int = 20, n_fft: int = 2048, hop_length: int = 512, n_bands: int = 128)` | Computes the Gammatone Cepstral Coefficients (GTCCs) of an audio signal |
| `audiofeat.cepstral` | `lpcc` | `(audio_data: torch.Tensor, sample_rate: int, n_lpcc: int = 12, n_fft: int = 2048, hop_length: int = 512, lpc_order: int = 12)` | Computes the Linear Predictive Cepstral Coefficients (LPCCs) of an audio signal |
| `audiofeat.io` | `extract_core_features` | `(waveform: 'torch.Tensor', sample_rate: 'int', *, frame_length: 'int' = 2048, hop_length: 'int' = 512) -> 'dict[str, float]'` | Extract a compact, production-friendly feature summary |
| `audiofeat.io` | `extract_features_for_directory` | `(input_dir: 'str | Path', *, sample_rate: 'int' = 22050, frame_length: 'int' = 2048, hop_length: 'int' = 512, skip_errors: 'bool' = True, errors: 'list[str] | None' = None) -> 'list[dict[str, float | str]]'` | No description available. |
| `audiofeat.io` | `extract_features_from_file` | `(audio_path: 'str | Path', *, sample_rate: 'int' = 22050, frame_length: 'int' = 2048, hop_length: 'int' = 512) -> 'dict[str, float | str]'` | No description available. |
| `audiofeat.io` | `iter_audio_files` | `(input_dir: 'str | Path', *, extensions: 'Iterable[str]' = {'.mp3', '.wav', '.ogg', '.flac', '.m4a'}) -> 'list[Path]'` | No description available. |
| `audiofeat.io` | `load_audio` | `(audio_path: 'str | Path', *, target_sample_rate: 'int | None' = 22050, mono: 'bool' = True) -> 'tuple[torch.Tensor, int]'` | Load audio from disk with optional mono conversion and resampling |
| `audiofeat.io` | `resample_if_needed` | `(waveform: 'torch.Tensor', sample_rate: 'int', target_sample_rate: 'int') -> 'torch.Tensor'` | No description available. |
| `audiofeat.io` | `summarize_matrix` | `(prefix: 'str', matrix: 'torch.Tensor') -> 'dict[str, float]'` | No description available. |
| `audiofeat.io` | `summarize_series` | `(prefix: 'str', series: 'torch.Tensor') -> 'dict[str, float]'` | No description available. |
| `audiofeat.io` | `to_mono` | `(waveform: 'torch.Tensor') -> 'torch.Tensor'` | Convert a loaded waveform to mono and return shape `(num_samples,)` |
| `audiofeat.io` | `write_feature_rows_to_csv` | `(rows: 'list[dict[str, float | str]]', output_csv: 'str | Path') -> 'Path'` | No description available. |
| `audiofeat.pitch` | `predict_crepe` | `(audio: torch.Tensor, sample_rate: int, output_activation: bool = False, model_capacity: str = 'full') -> torch.Tensor` | Estimate pitch using CREPE (Convolutional REpresentation for Pitch Estimation) |
| `audiofeat.pitch` | `fundamental_frequency_autocorr` | `(x: torch.Tensor, fs: int, frame_length: int, hop_length: int, fmin: int = 50, fmax: int = 600)` | Estimate F0 via autocorrelation per frame |
| `audiofeat.pitch` | `fundamental_frequency_yin` | `(x: torch.Tensor, fs: int, frame_length: int, hop_length: int, fmin: int = 50, fmax: int = 600, threshold: float = 0.1)` | Estimate F0 per frame using the YIN algorithm |
| `audiofeat.pitch` | `fundamental_frequency_pyin` | `(x: 'torch.Tensor', fs: 'int', frame_length: 'int' = 2048, hop_length: 'int' = 512, fmin: 'float' = 50.0, fmax: 'float' = 600.0, fill_unvoiced: 'float' = 0.0) -> 'torch.Tensor'` | Estimate F0 with probabilistic YIN (pYIN) via librosa |
| `audiofeat.pitch` | `semitone_sd` | `(f0: torch.Tensor)` | Standard deviation of F0 in semitones |
| `audiofeat.pitch` | `pitch_strength` | `(x: torch.Tensor, fs: int, frame_length: int, hop_length: int, fmin: int = 50, fmax: int = 600)` | Computes the pitch strength using the autocorrelation method |
| `audiofeat.spectral` | `spectral_bandwidth` | `(waveform: torch.Tensor, sample_rate: int, n_fft: int = 2048, hop_length: int = 512) -> torch.Tensor` | Computes the spectral bandwidth (or spread) of an audio waveform |
| `audiofeat.spectral` | `spectral_centroid` | `(audio_data: torch.Tensor, frame_length: int = 2048, hop_length: int = 512, sample_rate: int = 22050)` | Computes the spectral centroid of an audio signal |
| `audiofeat.spectral` | `chroma` | `(audio_data: torch.Tensor, sample_rate: int, n_fft: int = 2048, hop_length: int = 512, n_chroma: int = 12)` | Computes the Chroma features of an audio signal |
| `audiofeat.spectral` | `spectral_contrast` | `(x: torch.Tensor, fs: int, n_fft: int = 2048, n_bands: int = 6)` | Computes the spectral contrast of an audio signal |
| `audiofeat.spectral` | `cqt` | `(waveform: torch.Tensor, sample_rate: int, hop_length: int = 512, f_min: float = 32.7, n_bins: int = 84, bins_per_octave: int = 12) -> torch.Tensor` | Computes the Constant-Q Transform (CQT) of an audio waveform |
| `audiofeat.spectral` | `spectral_crest_factor` | `(x: torch.Tensor, n_fft: int)` | Computes the spectral crest factor of an audio signal |
| `audiofeat.spectral` | `spectral_deviation` | `(x: torch.Tensor, n_fft: int)` | Quantifies the "jaggedness" of the local spectrum |
| `audiofeat.spectral` | `low_high_energy_ratio` | `(x: torch.Tensor, fs: int, n_fft: int = 1024)` | Ratio of energy below 1 kHz to that above 3 kHz |
| `audiofeat.spectral` | `spectral_entropy` | `(x: torch.Tensor, n_fft: int)` | Spectral entropy of a frame |
| `audiofeat.spectral` | `spectral_flatness` | `(audio_data: torch.Tensor, frame_length=2048, hop_length=512)` | Computes the spectral flatness of an audio signal |
| `audiofeat.spectral` | `spectral_flux` | `(audio_data: torch.Tensor, frame_length=2048, hop_length=512)` | Computes the spectral flux of an audio signal |
| `audiofeat.spectral` | `formant_bandwidths` | `(a: 'torch.Tensor', fs: 'int')` | Formant bandwidths from LPC polynomial roots |
| `audiofeat.spectral` | `formant_contours` | `(x: 'torch.Tensor', fs: 'int', order: 'int | None' = 10, num_formants: 'int' = 5, max_formant: 'float' = 5500.0, frame_length_ms: 'float' = 25.0, hop_length_ms: 'float | None' = None, pre_emphasis: 'float' = 0.97, max_bandwidth: 'float' = 700.0, method: 'str' = 'burg') -> 'tuple[torch.Tensor, torch.Tensor]'` | Extract time-varying formant contours |
| `audiofeat.spectral` | `formant_dispersion` | `(formants: 'torch.Tensor')` | Average spacing between consecutive finite formants |
| `audiofeat.spectral` | `formant_frequencies` | `(x: 'torch.Tensor', fs: 'int', order: 'int | None' = 10, num_formants: 'int' = 5, max_formant: 'float' = 5500.0, frame_length_ms: 'float' = 25.0, hop_length_ms: 'float | None' = None, pre_emphasis: 'float' = 0.97, max_bandwidth: 'float' = 700.0, method: 'str' = 'burg')` | Estimate median formant frequencies [F1, F2, ..., Fn] |
| `audiofeat.spectral` | `gfcc` | `(waveform: torch.Tensor, sample_rate: int, n_gfcc: int = 40, n_fft: int = 2048, hop_length: int = 512, n_bands: int = 128) -> torch.Tensor` | Computes Gammatone Frequency Cepstral Coefficients (GFCCs) of an audio waveform |
| `audiofeat.spectral` | `harmonic_richness_factor` | `(magnitudes: torch.Tensor)` | Harmonic richness factor given harmonic magnitudes starting at F0 |
| `audiofeat.spectral` | `inharmonicity_index` | `(peaks: torch.Tensor, f0: float)` | Inharmonicity from peak frequencies and fundamental |
| `audiofeat.spectral` | `harmonic_to_noise_ratio` | `(harmonic_energy: torch.Tensor, noise_energy: torch.Tensor)` | Computes the Harmonic-to-Noise Ratio (HNR) |
| `audiofeat.spectral` | `hps` | `(waveform: torch.Tensor, sample_rate: int, n_fft: int = 2048, hop_length: int = 512, margin_h: float = 3.0, margin_p: float = 3.0) -> tuple[torch.Tensor, torch.Tensor]` | Performs Harmonic-Percussive Separation (HPS) on an audio waveform |
| `audiofeat.spectral` | `spectral_irregularity` | `(x: torch.Tensor, n_fft: int = 2048) -> torch.Tensor` | Compute Jensen's *spectral irregularity* of a signal |
| `audiofeat.spectral` | `key_detect` | `(waveform: torch.Tensor, sample_rate: int, n_fft: int = 4096, hop_length: int = 2048, n_chroma: int = 12) -> str` | Detects the musical key of an audio waveform |
| `audiofeat.spectral` | `log_mel_spectrogram` | `(waveform: torch.Tensor, sample_rate: int, n_fft: int = 2048, hop_length: int = 512, n_mels: int = 128, f_min: float = 0.0, f_max: float = None) -> torch.Tensor` | Computes the log-Mel spectrogram of an audio waveform using torchaudio |
| `audiofeat.spectral` | `lpc_coefficients` | `(audio_frame: torch.Tensor, order: int)` | Computes Linear Prediction Coefficients (LPC) for a single audio frame |
| `audiofeat.spectral` | `lsp_coefficients` | `(lpc_coeffs: torch.Tensor)` | Converts Linear Prediction Coefficients (LPC) to Line Spectral Pairs (LSP) |
| `audiofeat.spectral` | `mfcc` | `(audio_data: torch.Tensor, sample_rate: int, n_mfcc: int = 40, n_fft: int = 2048, hop_length: int = 512, n_mels: int = 128)` | Computes the Mel-Frequency Cepstral Coefficients (MFCCs) of an audio signal |
| `audiofeat.spectral` | `spectral_skewness` | `(x: torch.Tensor, n_fft: int)` | No description available. |
| `audiofeat.spectral` | `spectral_spread` | `(x: torch.Tensor, n_fft: int, sample_rate: int)` | Computes the spectral spread (bandwidth) of an audio signal |
| `audiofeat.spectral` | `phase_coherence` | `(phases: torch.Tensor)` | Compute phase coherence from instantaneous phase |
| `audiofeat.spectral` | `spectral_rolloff` | `(audio_data: torch.Tensor, frame_length: int = 2048, hop_length: int = 512, rolloff_percent: float = 0.85, sample_rate: int = 22050)` | Computes the spectral rolloff of an audio signal |
| `audiofeat.spectral` | `spectral_roughness` | `(x: torch.Tensor, sample_rate: int = 22050, n_fft: int = 2048, top_db: float = 60.0) -> torch.Tensor` | Compute the *spectral roughness* of an audio signal |
| `audiofeat.spectral` | `spectral_sharpness` | `(x: torch.Tensor, sample_rate: int = 22050, n_fft: int = 2048, power: float = 2.0) -> torch.Tensor` | Compute Zwicker *sharpness* (in acum) of an audio signal |
| `audiofeat.spectral` | `sibilant_spectral_peak_frequency` | `(x: torch.Tensor, fs: int, n_fft: int = 1024)` | Peak frequency of sibilant energy between 3 and 12 kHz |
| `audiofeat.spectral` | `spectral_slope` | `(x: torch.Tensor, n_fft: int)` | Computes the spectral slope of an audio signal |
| `audiofeat.spectral` | `cqt_spectrogram` | `(audio_data: torch.Tensor, sample_rate: int, hop_length: int = 512, fmin: float = 32.7, n_bins: int = 84, bins_per_octave: int = 12)` | Computes the Constant-Q Transform (CQT) spectrogram of an audio signal |
| `audiofeat.spectral` | `linear_spectrogram` | `(audio_data: torch.Tensor, n_fft: int = 2048, hop_length: int = 512)` | Computes the linear spectrogram (STFT) of an audio signal |
| `audiofeat.spectral` | `mel_spectrogram` | `(audio_data: torch.Tensor, sample_rate: int, n_fft: int = 2048, hop_length: int = 512, n_mels: int = 128)` | Computes the Mel spectrogram of an audio signal |
| `audiofeat.spectral` | `spectral_tonality` | `(x: torch.Tensor, n_fft: int = 2048, top_db: float = 60.0) -> torch.Tensor` | Compute a simple *tonality coefficient* (0..1) |
| `audiofeat.spectral` | `tonnetz` | `(chroma_features: torch.Tensor)` | Computes the Tonnetz (Tonal Centroid Features) from Chroma features |
| `audiofeat.standards` | `available_opensmile_feature_levels` | `() -> 'list[str]'` | Return available standard openSMILE feature levels |
| `audiofeat.standards` | `available_opensmile_feature_sets` | `() -> 'list[str]'` | Return available standard openSMILE feature sets |
| `audiofeat.standards` | `extract_opensmile_features` | `(audio: 'str | Path | torch.Tensor', *, sample_rate: 'int | None' = None, feature_set: 'OpenSmileFeatureSet' = 'eGeMAPSv02', feature_level: 'OpenSmileFeatureLevel' = 'Functionals', flatten: 'bool' = True)` | Extract standardized openSMILE descriptors (e.g., eGeMAPSv02 or ComParE_2016) |
| `audiofeat.stats` | `compute_functionals` | `(feature_series: torch.Tensor)` | Computes a set of statistical functionals (mean, std, min, max, skewness, kurtosis) |
| `audiofeat.temporal` | `amplitude_modulation_depth` | `(env: torch.Tensor, window: int)` | Amplitude modulation depth over a sliding window |
| `audiofeat.temporal` | `log_attack_time` | `(audio_data: torch.Tensor, sample_rate: int, threshold: float = 0.01)` | Computes the log attack time of an audio signal |
| `audiofeat.temporal` | `beat_track` | `(waveform: torch.Tensor, sample_rate: int, n_fft: int = 2048, hop_length: int = 512, tempo_min: float = 60.0, tempo_max: float = 240.0) -> tuple[torch.Tensor, torch.Tensor]` | Performs beat tracking on an audio waveform to estimate tempo and beat times |
| `audiofeat.temporal` | `temporal_centroid` | `(waveform: torch.Tensor, sample_rate: int) -> torch.Tensor` | Computes the temporal centroid of an audio waveform |
| `audiofeat.temporal` | `decay_time` | `(x: torch.Tensor, sample_rate: int, threshold_db: float = -20.0) -> torch.Tensor` | Compute *decay time* of an audio signal |
| `audiofeat.temporal` | `energy` | `(signal: torch.Tensor, sample_rate: int, window_size: float = 0.05, hop_size: float = 0.025)` | Calculates the short-term energy of an audio signal |
| `audiofeat.temporal` | `entropy_of_energy` | `(audio_data: torch.Tensor, frame_length: int, hop_length: int, n_sub_frames: int = 10)` | Computes the entropy of energy of an audio signal |
| `audiofeat.temporal` | `loudness` | `(waveform: torch.Tensor, sample_rate: int) -> torch.Tensor` | Computes the perceived loudness of an audio waveform |
| `audiofeat.temporal` | `onset_detect` | `(waveform: torch.Tensor, sample_rate: int, n_fft: int = 2048, hop_length: int = 512, backtrack: bool = True) -> torch.Tensor` | Computes an onset detection function (ODF) and optionally backtracks to find precise onset times |
| `audiofeat.temporal` | `breath_group_duration` | `(env: torch.Tensor, fs: int)` | Estimate breath group durations from envelope |
| `audiofeat.temporal` | `speech_rate` | `(x: torch.Tensor, fs: int, threshold_ratio: float = 0.3, min_gap: float = 0.1)` | Estimate speech rate in syllables per second |
| `audiofeat.temporal` | `temporal_centroid` | `(audio_data: torch.Tensor, frame_length: int, hop_length: int)` | Computes the temporal centroid of an audio signal |
| `audiofeat.temporal` | `beat_track` | `(audio_data: torch.Tensor, sample_rate: int, n_fft: int = 2048, hop_length: int = 512)` | Performs simple beat tracking on an audio signal |
| `audiofeat.temporal` | `tempo` | `(audio_data: torch.Tensor, sample_rate: int, n_fft: int = 2048, hop_length: int = 512)` | Estimates the tempo (BPM) of an audio signal |
| `audiofeat.temporal` | `frame_signal` | `(x: torch.Tensor, frame_length: int, hop_length: int)` | Frame a 1D signal into overlapping frames |
| `audiofeat.temporal` | `hann_window` | `(L: int)` | Return an L-point Hann window |
| `audiofeat.temporal` | `rms` | `(x: torch.Tensor, frame_length: int, hop_length: int)` | Root-mean-square amplitude per frame |
| `audiofeat.temporal` | `short_time_energy` | `(x: torch.Tensor, frame_length: int, hop_length: int)` | Computes the short-time energy of an audio signal |
| `audiofeat.temporal` | `teager_energy_operator` | `(x: torch.Tensor) -> torch.Tensor` | Compute the average *Teager Energy* of a signal |
| `audiofeat.temporal` | `tristimulus` | `(waveform: torch.Tensor, sample_rate: int, n_fft: int = 2048, hop_length: int = 512) -> torch.Tensor` | Computes the tristimulus of an audio waveform |
| `audiofeat.temporal` | `zero_crossing_count` | `(audio_data: torch.Tensor, frame_length=2048, hop_length=512)` | Computes the number of zero-crossings in each frame of an audio signal |
| `audiofeat.temporal` | `zero_crossing_rate` | `(audio_data: torch.Tensor, frame_length=2048, hop_length=512)` | Computes the normalized zero-crossing rate of an audio signal |
| `audiofeat.validation` | `apply_speaker_profile` | `(*, speaker_profile: 'str' = 'neutral', pitch_floor: 'float | None' = None, pitch_ceiling: 'float | None' = None, max_formant: 'float | None' = None) -> 'dict[str, float]'` | No description available. |
| `audiofeat.validation` | `build_praat_comparison_report` | `(waveform: 'torch.Tensor', sample_rate: 'int', praat_reference: 'Mapping[str, Mapping[str, float]]', *, frame_length: 'int', hop_length: 'int', speaker_profile: 'str' = 'neutral', pitch_floor: 'float | None' = None, pitch_ceiling: 'float | None' = None, pitch_method: 'str' = 'autocorr', yin_threshold: 'float' = 0.1, formant_order: 'int | None' = None, num_formants: 'int' = 5, max_formant: 'float | None' = None, formant_method: 'str' = 'burg', formant_window_length_sec: 'float' = 0.025, formant_time_step_sec: 'float' = 0.01, pre_emphasis_from_hz: 'float' = 50.0) -> 'dict'` | No description available. |
| `audiofeat.validation` | `compare_audio_to_praat_reference` | `(audio_path: 'str | Path', praat_reference: 'Mapping[str, Mapping[str, float]]', *, sample_rate: 'int' = 22050, frame_length: 'int | None' = None, hop_length: 'int | None' = None, speaker_profile: 'str' = 'neutral', pitch_floor: 'float | None' = None, pitch_ceiling: 'float | None' = None, pitch_method: 'str' = 'autocorr', yin_threshold: 'float' = 0.1, formant_order: 'int | None' = None, num_formants: 'int' = 5, max_formant: 'float | None' = None, formant_method: 'str' = 'burg', time_step_sec: 'float' = 0.01, formant_window_length_sec: 'float' = 0.025, pre_emphasis_from_hz: 'float' = 50.0) -> 'dict'` | No description available. |
| `audiofeat.validation` | `compute_audiofeat_reference` | `(waveform: 'torch.Tensor', sample_rate: 'int', *, frame_length: 'int', hop_length: 'int', speaker_profile: 'str' = 'neutral', pitch_floor: 'float | None' = None, pitch_ceiling: 'float | None' = None, pitch_method: 'str' = 'autocorr', yin_threshold: 'float' = 0.1, formant_order: 'int | None' = None, num_formants: 'int' = 5, max_formant: 'float | None' = None, formant_method: 'str' = 'burg', formant_window_length_sec: 'float' = 0.025, formant_time_step_sec: 'float' = 0.01, pre_emphasis_from_hz: 'float' = 50.0) -> 'dict[str, dict[str, float]]'` | No description available. |
| `audiofeat.validation` | `evaluate_praat_report` | `(report: 'Mapping', tolerances: 'Mapping[str, float] | None' = None) -> 'dict[str, object]'` | No description available. |
| `audiofeat.validation` | `extract_praat_reference` | `(audio_path: 'str | Path', *, speaker_profile: 'str' = 'neutral', pitch_floor: 'float | None' = None, pitch_ceiling: 'float | None' = None, num_formants: 'int' = 5, max_formant: 'float | None' = None, time_step: 'float' = 0.01, formant_window_length: 'float' = 0.025, pre_emphasis_from_hz: 'float' = 50.0) -> 'dict[str, dict[str, float]]'` | Extract Praat reference values directly using parselmouth |
| `audiofeat.validation` | `load_praat_reference` | `(reference_json: 'str | Path') -> 'dict'` | No description available. |
| `audiofeat.validation` | `relative_error` | `(value: 'float', reference: 'float') -> 'float'` | No description available. |
| `audiofeat.validation` | `save_json` | `(data: 'Mapping', output_path: 'str | Path') -> 'Path'` | No description available. |
| `audiofeat.validation` | `run_gold_standard_scorecard` | `(*, sample_rate: 'int' = 22050, frame_length: 'int' = 1024, hop_length: 'int' = 256, praat_audio_path: 'str | Path | None' = None, include_optional: 'bool' = True) -> 'dict[str, Any]'` | Run reproducible quality checks and produce a score normalized to 100 |
| `audiofeat.voice` | `alpha_ratio` | `(x: torch.Tensor, fs: int, n_fft: int = 2048)` | Computes the Alpha Ratio: ratio of energy in 50-1000 Hz to 1000-5000 Hz |
| `audiofeat.voice` | `cepstral_peak_prominence` | `(waveform: torch.Tensor, sample_rate: int, f0_min: float = 60.0, f0_max: float = 333.3, frame_length_ms: float = 40.0, hop_length_ms: float = 10.0) -> torch.Tensor` | Compute Cepstral Peak Prominence (CPP) for voice quality assessment |
| `audiofeat.voice` | `delta_cpp` | `(cpp: torch.Tensor)` | Frame-wise difference of cepstral peak prominence |
| `audiofeat.voice` | `glottal_to_noise_excitation` | `(spec: torch.Tensor)` | Approximate GNE using band cross-correlations |
| `audiofeat.voice` | `maximum_flow_declination_rate` | `(flow: torch.Tensor, fs: int)` | Approximate MFDR from differentiated glottal flow |
| `audiofeat.voice` | `hammarberg_index` | `(x: torch.Tensor, fs: int, n_fft: int = 2048)` | Computes the Hammarberg Index: ratio of highest energy peak in 0-2 kHz to 2-5 kHz |
| `audiofeat.voice` | `harmonic_differences` | `(magnitudes: torch.Tensor, f0_hz: float, fs: int, h_indices: list = None)` | Computes harmonic differences (e.g., H1-H2, H1-A3) |
| `audiofeat.voice` | `jitter_ddp` | `(periods: torch.Tensor) -> torch.Tensor` | Compute Jitter (DDP): Difference of Differences of Periods |
| `audiofeat.voice` | `jitter_local` | `(periods: torch.Tensor) -> torch.Tensor` | Compute Jitter (local): Average absolute difference between consecutive periods, |
| `audiofeat.voice` | `jitter_ppq5` | `(periods: torch.Tensor) -> torch.Tensor` | Compute Jitter (PPQ5): Five-point Period Perturbation Quotient |
| `audiofeat.voice` | `nasality_index` | `(nasal: torch.Tensor, oral: torch.Tensor, fs: int, n_fft: int = 1024)` | Compute nasality index from nasal and oral microphone signals |
| `audiofeat.voice` | `voice_onset_time` | `(x: torch.Tensor, fs: int, frame_length: int, hop_length: int)` | Simplified voice onset time estimation |
| `audiofeat.voice` | `closed_quotient` | `(open_time: torch.Tensor, close_time: torch.Tensor, period: torch.Tensor)` | Closed quotient from EGG timings per cycle |
| `audiofeat.voice` | `glottal_closure_time` | `(open_times: torch.Tensor, close_times: torch.Tensor, periods: torch.Tensor)` | Average relative glottal closure time |
| `audiofeat.voice` | `jitter` | `(periods: torch.Tensor)` | Cycle-to-cycle F0 variation (local jitter) |
| `audiofeat.voice` | `normalized_amplitude_quotient` | `(peak_flow: torch.Tensor, mfdr: torch.Tensor, period: torch.Tensor)` | NAQ computed from peak glottal flow, MFDR and period |
| `audiofeat.voice` | `shimmer` | `(amplitudes: torch.Tensor)` | Cycle-to-cycle amplitude variation (local shimmer) |
| `audiofeat.voice` | `soft_phonation_index` | `(low_band_energy: torch.Tensor, high_band_energy: torch.Tensor)` | Soft phonation index from low/high band energies |
| `audiofeat.voice` | `speed_quotient` | `(open_times: torch.Tensor, close_times: torch.Tensor)` | Speed quotient from glottal flow opening and closing times |
| `audiofeat.voice` | `subharmonic_to_harmonic_ratio` | `(mag: torch.Tensor, f0_bin: int, num_harmonics: int)` | Compute SHR from magnitude spectrum |
| `audiofeat.voice` | `vocal_fry_index` | `(f0: torch.Tensor)` | Ratio of fry frames to voiced frames based on F0 and period variation |
| `audiofeat.voice` | `shimmer_apq3` | `(amplitudes: torch.Tensor) -> torch.Tensor` | Compute Shimmer (APQ3): Amplitude Perturbation Quotient (3-point) |
| `audiofeat.voice` | `shimmer_dda` | `(amplitudes: torch.Tensor) -> torch.Tensor` | Compute Shimmer (DDA): Difference of Differences of Amplitudes |
| `audiofeat.voice` | `shimmer_local` | `(amplitudes: torch.Tensor) -> torch.Tensor` | Compute Shimmer (local): Average absolute difference between consecutive amplitudes, |
| `audiofeat.voice` | `shimmer_local_db` | `(amplitudes: torch.Tensor) -> torch.Tensor` | Compute Shimmer (local, dB): Average absolute difference in log-amplitudes |
| `audiofeat.voice` | `vocal_tract_length` | `(F1: float, F2: float, c: float = 34400)` | Estimate vocal tract length from first two formants |