
import torch
import pytest
from audiofeat.cepstral.lpcc import lpcc
from audiofeat.cepstral.gtcc import gtcc
from audiofeat.cepstral.deltas import delta, delta_delta
from audiofeat.spectral.lpc import lpc_coefficients
from audiofeat.spectral.lsp import lsp_coefficients

def test_lpc_coefficients():
    audio_frame = torch.randn(2048)
    order = 10
    result = lpc_coefficients(audio_frame, order)
    assert isinstance(result, torch.Tensor)
    assert result.shape[0] == order

def test_lsp_coefficients():
    lpc_coeffs = torch.randn(10) # Dummy LPC coefficients
    result = lsp_coefficients(lpc_coeffs)
    assert isinstance(result, torch.Tensor)
    # The number of LSP coefficients should be equal to the LPC order
    assert result.shape[0] == lpc_coeffs.shape[0]

def test_lpcc():
    audio_data = torch.randn(22050 * 5)
    result = lpcc(audio_data, sample_rate=22050)
    assert isinstance(result, torch.Tensor)
    assert result.shape[0] > 0
    assert result.shape[1] > 0

def test_gtcc():
    audio_data = torch.randn(22050 * 5)
    result = gtcc(audio_data, sample_rate=22050)
    assert isinstance(result, torch.Tensor)
    assert result.shape[0] > 0
    assert result.shape[1] > 0

def test_delta():
    features = torch.randn(10, 5) # time_steps, features
    result = delta(features)
    assert isinstance(result, torch.Tensor)
    assert result.shape == features.shape

def test_delta_delta():
    features = torch.randn(10, 5) # time_steps, features
    result = delta_delta(features)
    assert isinstance(result, torch.Tensor)
    assert result.shape == features.shape
