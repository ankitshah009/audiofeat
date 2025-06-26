import torch
from ..temporal.rms import frame_signal, hann_window

def jitter(periods: torch.Tensor):
    """Cycle-to-cycle F0 variation (local jitter)."""
    diffs = torch.abs(periods[:-1] - periods[1:])
    return diffs.mean() / periods.mean()

def shimmer(amplitudes: torch.Tensor):
    """Cycle-to-cycle amplitude variation (local shimmer)."""
    diffs = torch.abs(amplitudes[:-1] - amplitudes[1:])
    return diffs.mean() / amplitudes.mean()

def subharmonic_to_harmonic_ratio(mag: torch.Tensor, f0_bin: int, num_harmonics: int):
    """Compute SHR from magnitude spectrum."""
    harmonic_indices = torch.arange(1, num_harmonics + 1, device=mag.device) * f0_bin
    subharmonic_indices = harmonic_indices + f0_bin // 2
    harmonic_power = (mag[harmonic_indices] ** 2).sum()
    sub_power = (mag[subharmonic_indices] ** 2).sum()
    return 10 * torch.log10(sub_power / (harmonic_power + 1e-8))

def normalized_amplitude_quotient(peak_flow: torch.Tensor, mfdr: torch.Tensor, period: torch.Tensor):
    """NAQ computed from peak glottal flow, MFDR and period."""
    return peak_flow / (mfdr * period)

def closed_quotient(open_time: torch.Tensor, close_time: torch.Tensor, period: torch.Tensor):
    """Closed quotient from EGG timings per cycle."""
    return (close_time - open_time) / period

def glottal_closure_time(open_times: torch.Tensor, close_times: torch.Tensor, periods: torch.Tensor):
    """Average relative glottal closure time."""
    return ((close_times - open_times) / periods).mean()

def soft_phonation_index(low_band_energy: torch.Tensor, high_band_energy: torch.Tensor):
    """Soft phonation index from low/high band energies."""
    return 10 * torch.log10(high_band_energy / (low_band_energy + 1e-8))

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