import torch
import numpy as np


def frame_signal(x: torch.Tensor, frame_length: int, hop_length: int):
    """Frame a 1D signal into overlapping frames."""
    num_frames = 1 + (x.numel() - frame_length) // hop_length
    strides = (x.stride(0) * hop_length, x.stride(0))
    shape = (num_frames, frame_length)
    return x.as_strided(shape, strides)


def hann_window(L: int):
    """Return an L-point Hann window."""
    n = torch.arange(L, dtype=torch.float32)
    return 0.5 * (1 - torch.cos(2 * np.pi * n / (L - 1)))


def rms(x: torch.Tensor, frame_length: int, hop_length: int):
    """Root-mean-square amplitude per frame."""
    frames = frame_signal(x, frame_length, hop_length)
    w = hann_window(frame_length).to(x.device)
    win_frames = frames * w
    return torch.sqrt(torch.mean(win_frames ** 2, dim=1))


def fundamental_frequency_autocorr(x: torch.Tensor, fs: int, frame_length: int, hop_length: int, fmin=50, fmax=600):
    """Estimate F0 via autocorrelation per frame."""
    frames = frame_signal(x, frame_length, hop_length)
    w = hann_window(frame_length).to(x.device)
    win = frames * w
    autocorr = torch.fft.irfft(torch.fft.rfft(win, n=2*frame_length), n=2*frame_length)
    min_lag = int(fs / fmax)
    max_lag = int(fs / fmin)
    ac_segment = autocorr[:, min_lag:max_lag]
    lag = ac_segment.argmax(dim=1) + min_lag
    return fs / lag.float()


def spectral_entropy(x: torch.Tensor, n_fft: int):
    """Spectral entropy of a frame."""
    X = torch.fft.rfft(x * hann_window(x.numel()).to(x.device), n=n_fft)
    P = (X.abs() ** 2)
    P = P / P.sum()
    return -(P * torch.log2(P + 1e-12)).sum()


def spectral_rolloff(x: torch.Tensor, fs: int, roll_percent=0.95, n_fft=1024):
    """Frequency below which `roll_percent` of spectral energy is contained."""
    X = torch.fft.rfft(x * hann_window(x.numel()).to(x.device), n=n_fft)
    power = X.abs() ** 2
    cumulative = torch.cumsum(power, dim=0)
    threshold = roll_percent * cumulative[-1]
    idx = torch.searchsorted(cumulative, threshold)
    return fs * idx / n_fft


def spectral_flux(prev_mag: torch.Tensor, cur_mag: torch.Tensor):
    """Spectral flux between two magnitude spectra."""
    return torch.sum((cur_mag - prev_mag).pow(2))


def pitch_strength(x: torch.Tensor, fs: int, frame_length: int, hop_length: int):
    """Normalized autocorrelation peak strength as a measure of pitch clarity."""
    frames = frame_signal(x, frame_length, hop_length)
    w = hann_window(frame_length).to(x.device)
    win = frames * w
    autocorr = torch.fft.irfft(torch.fft.rfft(win, n=2*frame_length), n=2*frame_length)
    min_lag = int(fs / 600)
    max_lag = int(fs / 50)
    ac_segment = autocorr[:, min_lag:max_lag]
    values, _ = ac_segment.max(dim=1)
    energy = (win ** 2).sum(dim=1)
    return values / (energy + 1e-8)


def jitter(periods: torch.Tensor):
    """Cycle-to-cycle F0 variation (local jitter)."""
    diffs = torch.abs(periods[:-1] - periods[1:])
    return diffs.mean() / periods.mean()


def shimmer(amplitudes: torch.Tensor):
    """Cycle-to-cycle amplitude variation (local shimmer)."""
    diffs = torch.abs(amplitudes[:-1] - amplitudes[1:])
    return diffs.mean() / amplitudes.mean()


def spectral_skewness(x: torch.Tensor, n_fft: int):
    X = torch.fft.rfft(x * hann_window(x.numel()).to(x.device), n=n_fft)
    P = X.abs() ** 2
    freqs = torch.linspace(0, n_fft // 2, P.numel(), device=x.device)
    mean = torch.sum(freqs * P) / torch.sum(P)
    var = torch.sum((freqs - mean) ** 2 * P) / torch.sum(P)
    skew = torch.sum((freqs - mean) ** 3 * P) / (torch.sum(P) * var.sqrt() ** 3)
    kurt = torch.sum((freqs - mean) ** 4 * P) / (torch.sum(P) * var ** 2) - 3
    return skew, kurt


def low_high_energy_ratio(x: torch.Tensor, fs: int, n_fft: int = 1024):
    """Ratio of energy below 1 kHz to that above 3 kHz."""
    X = torch.fft.rfft(x * hann_window(x.numel()).to(x.device), n=n_fft)
    P = X.abs() ** 2
    freqs = torch.linspace(0, fs / 2, P.numel(), device=x.device)
    low = P[freqs < 1000].sum()
    high = P[freqs > 3000].sum()
    return 10 * torch.log10(low / (high + 1e-8))


def amplitude_modulation_depth(env: torch.Tensor, window: int):
    """Amplitude modulation depth over a sliding window."""
    if env.numel() < window:
        return torch.tensor(0.0, device=env.device)
    frames = frame_signal(env, window, window)
    max_e = frames.max(dim=1).values
    min_e = frames.min(dim=1).values
    return ((max_e - min_e) / (max_e + min_e + 1e-8)).mean()


def spectral_flux_frames(x: torch.Tensor, n_fft: int, hop: int):
    """Frame-wise spectral flux for a signal."""
    frames = frame_signal(x, n_fft, hop)
    w = hann_window(n_fft).to(x.device)
    specs = torch.fft.rfft(frames * w, n=n_fft)
    mags = specs.abs()
    flux = torch.zeros(mags.size(0)-1, device=x.device)
    for i in range(1, mags.size(0)):
        flux[i-1] = spectral_flux(mags[i-1], mags[i])
    return flux


def breath_group_duration(env: torch.Tensor, fs: int):
    """Estimate breath group durations from envelope."""
    threshold = env.mean() * 0.25
    below = env < threshold
    indices = torch.nonzero(below).squeeze()
    if indices.numel() == 0:
        return torch.tensor([])
    diffs = indices[1:] - indices[:-1]
    starts = indices[:-1][diffs > int(0.25 * fs)]
    if starts.numel() < 2:
        return torch.tensor([])
    durations = (starts[1:] - starts[:-1]).float() / fs
    return durations


def delta_cpp(cpp: torch.Tensor):
    """Frame-wise difference of cepstral peak prominence."""
    return cpp[1:] - cpp[:-1]


def normalized_amplitude_quotient(peak_flow: torch.Tensor, mfdr: torch.Tensor, period: torch.Tensor):
    """NAQ computed from peak glottal flow, MFDR and period."""
    return peak_flow / (mfdr * period)


def closed_quotient(open_time: torch.Tensor, close_time: torch.Tensor, period: torch.Tensor):
    """Closed quotient from EGG timings per cycle."""
    return (close_time - open_time) / period


def glottal_closure_time(open_times: torch.Tensor, close_times: torch.Tensor, periods: torch.Tensor):
    """Average relative glottal closure time."""
    return ((close_times - open_times) / periods).mean()


def harmonic_richness_factor(magnitudes: torch.Tensor):
    """Harmonic richness factor given harmonic magnitudes starting at F0."""
    if magnitudes.numel() < 2:
        return torch.tensor(0.0, device=magnitudes.device)
    numerator = magnitudes[1:].pow(2).sum()
    denominator = magnitudes[0].pow(2)
    return 10 * torch.log10(numerator / (denominator + 1e-8))


def inharmonicity_index(peaks: torch.Tensor, f0: float):
    """Inharmonicity from peak frequencies and fundamental."""
    k = torch.arange(1, peaks.numel() + 1, device=peaks.device)
    return torch.mean(torch.abs(peaks / (k * f0) - 1))


def phase_coherence(phases: torch.Tensor):
    """Compute phase coherence from instantaneous phase."""
    return torch.abs(torch.mean(torch.exp(1j * phases)))


def soft_phonation_index(low_band_energy: torch.Tensor, high_band_energy: torch.Tensor):
    """Soft phonation index from low/high band energies."""
    return 10 * torch.log10(high_band_energy / (low_band_energy + 1e-8))


def semitone_sd(f0: torch.Tensor):
    """Standard deviation of F0 in semitones."""
    mean_f0 = f0[f0 > 0].mean()
    if torch.isnan(mean_f0):
        return torch.tensor(0.0, device=f0.device)
    semitones = 12 * torch.log2(f0[f0 > 0] / mean_f0)
    return semitones.std()


def voice_onset_time(x: torch.Tensor, fs: int, frame_length: int, hop_length: int):
    """Simplified voice onset time estimation."""
    frames = frame_signal(x, frame_length, hop_length)
    energy = (frames ** 2).sum(dim=1)
    burst = (energy > energy.max() * 0.1).nonzero(as_tuple=False)
    if burst.numel() == 0:
        return 0.0
    nb = burst[0,0]
    autocorr = torch.fft.irfft(torch.fft.rfft(frames, n=2*frame_length), n=2*frame_length)
    nv = None
    for i in range(nb, frames.size(0)):
        ac = autocorr[i]
        r = ac[int(0.002*fs):int(0.015*fs)].max() / ac[0]
        if r > 0.3:
            nv = i
            break
    if nv is None:
        return 0.0
    return (nv - nb) * hop_length / fs


def subharmonic_to_harmonic_ratio(mag: torch.Tensor, f0_bin: int, num_harmonics: int):
    """Compute SHR from magnitude spectrum."""
    harmonic_indices = torch.arange(1, num_harmonics + 1, device=mag.device) * f0_bin
    subharmonic_indices = harmonic_indices + f0_bin // 2
    harmonic_power = (mag[harmonic_indices] ** 2).sum()
    sub_power = (mag[subharmonic_indices] ** 2).sum()
    return 10 * torch.log10(sub_power / (harmonic_power + 1e-8))


def formant_frequencies(x: torch.Tensor, fs: int, order: int):
    """Estimate formant frequencies using LPC."""
    x = x - 0.97 * torch.nn.functional.pad(x[:-1], (1,0))
    autocorr = torch.fft.irfft(torch.fft.rfft(x, n=2*order), n=2*order)
    R = torch.linalg.toeplitz(autocorr[:order])
    r = autocorr[1:order+1]
    coeffs = torch.linalg.solve(R, r)
    a = torch.cat([torch.ones(1, device=x.device), -coeffs])
    roots = torch.roots(a)
    roots = roots[(roots.imag >= 0)]
    angles = torch.angle(roots)
    formants = angles * fs / (2 * np.pi)
    formants = formants.sort().values
    return formants


def formant_bandwidths(a: torch.Tensor, fs: int):
    """Formant bandwidths from LPC polynomial roots."""
    roots = torch.roots(a)
    roots = roots[(roots.imag >= 0)]
    freqs = torch.angle(roots) * fs / (2 * np.pi)
    bandwidths = -2 * fs * torch.log(torch.abs(roots)) / (2 * np.pi)
    order = freqs.argsort()
    return bandwidths[order]


def glottal_to_noise_excitation(spec: torch.Tensor):
    """Approximate GNE using band cross-correlations."""
    bands = spec.view(6, -1)
    corr = torch.nn.functional.conv1d(bands.unsqueeze(1), bands.unsqueeze(1), padding=0).max(dim=-1).values
    g = corr.max()
    return 10 * torch.log10(g / (1 - g + 1e-8))


def formant_dispersion(formants: torch.Tensor):
    """Average spacing between first five formants."""
    if formants.numel() < 5:
        return torch.tensor(0.0, device=formants.device)
    d = 0
    for i in range(4):
        d += formants[i+1] - formants[i]
    return d / 4


def vocal_tract_length(F1: float, F2: float, c: float = 34400):
    """Estimate vocal tract length from first two formants."""
    return c / 4 * (1 / F1 + 1 / (F2 - F1))


def maximum_flow_declination_rate(flow: torch.Tensor, fs: int):
    """Approximate MFDR from differentiated glottal flow."""
    dflow = torch.diff(flow) * fs
    return dflow.max()


def speech_rate(x: torch.Tensor, fs: int, threshold_ratio: float = 0.3, min_gap: float = 0.1):
    """Estimate speech rate in syllables per second."""
    env = torch.abs(x)
    win_len = max(1, int(0.02 * fs))
    kernel = torch.ones(win_len, device=x.device) / win_len
    env = torch.nn.functional.conv1d(env.view(1,1,-1), kernel.view(1,1,-1), padding=win_len//2).squeeze()
    threshold = env.mean() * threshold_ratio
    peaks = (env[1:-1] > env[:-2]) & (env[1:-1] > env[2:]) & (env[1:-1] > threshold)
    indices = torch.nonzero(peaks).squeeze() + 1
    if indices.numel() == 0:
        return torch.tensor(0.0, device=x.device)
    keep = torch.cat([torch.tensor([True], device=x.device), (indices[1:] - indices[:-1]) > int(min_gap * fs)])
    syllables = indices[keep]
    return syllables.numel() / (x.numel() / fs)


def nasality_index(nasal: torch.Tensor, oral: torch.Tensor, fs: int, n_fft: int = 1024):
    """Compute nasality index from nasal and oral microphone signals."""
    N = torch.fft.rfft(nasal * hann_window(nasal.numel()).to(nasal.device), n=n_fft)
    O = torch.fft.rfft(oral * hann_window(oral.numel()).to(oral.device), n=n_fft)
    freqs = torch.linspace(0, fs / 2, N.numel(), device=nasal.device)
    mask = (freqs >= 300) & (freqs <= 800)
    n_power = (N.abs() ** 2)[mask].sum()
    o_power = (O.abs() ** 2)[mask].sum()
    return 10 * torch.log10((n_power + 1e-8) / (o_power + 1e-8))


def speed_quotient(open_times: torch.Tensor, close_times: torch.Tensor):
    """Speed quotient from glottal flow opening and closing times."""
    return (open_times.mean() / (close_times.mean() + 1e-8))


def vocal_fry_index(f0: torch.Tensor):
    """Ratio of fry frames to voiced frames based on F0 and period variation."""
    voiced = f0 > 0
    if voiced.sum() < 2:
        return torch.tensor(0.0, device=f0.device)
    periods = torch.where(voiced, 1.0 / (f0 + 1e-8), 0.0)
    diffs = torch.abs(periods[1:] - periods[:-1]) / (periods[:-1] + 1e-8)
    fry = (f0[:-1] < 70) & (diffs > 0.2)
    voiced_frames = voiced[:-1]
    return fry.sum().float() / (voiced_frames.sum().float() + 1e-8)


def sibilant_spectral_peak_frequency(x: torch.Tensor, fs: int, n_fft: int = 1024):
    """Peak frequency of sibilant energy between 3 and 12 kHz."""
    X = torch.fft.rfft(x * hann_window(x.numel()).to(x.device), n=n_fft)
    P = X.abs() ** 2
    freqs = torch.linspace(0, fs / 2, P.numel(), device=x.device)
    mask = (freqs >= 3000) & (freqs <= 12000)
    if mask.sum() == 0:
        return torch.tensor(0.0, device=x.device)
    peak_idx = P[mask].argmax()
    return freqs[mask][peak_idx]

