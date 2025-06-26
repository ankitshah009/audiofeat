
import torch
import pytest
from audiofeat.stats.functionals import compute_functionals

def test_compute_functionals():
    # Test with a simple 2D tensor
    feature_series = torch.tensor([
        [1.0, 2.0],
        [2.0, 3.0],
        [3.0, 4.0],
        [4.0, 5.0]
    ])
    
    result = compute_functionals(feature_series)
    assert isinstance(result, torch.Tensor)
    # Expected output: 2 features * 6 functionals = 12 elements
    assert result.shape[0] == feature_series.shape[1] * 6

    # Test with a single feature
    feature_series_single = torch.tensor([
        [1.0],
        [2.0],
        [3.0],
        [4.0]
    ])
    result_single = compute_functionals(feature_series_single)
    assert isinstance(result_single, torch.Tensor)
    assert result_single.shape[0] == 1 * 6

    # Test with random data
    random_data = torch.randn(100, 5)
    result_random = compute_functionals(random_data)
    assert isinstance(result_random, torch.Tensor)
    assert result_random.shape[0] == 5 * 6

    # Test with zero standard deviation (should handle division by zero)
    zero_std_data = torch.tensor([
        [1.0, 1.0],
        [1.0, 1.0],
        [1.0, 1.0]
    ])
    result_zero_std = compute_functionals(zero_std_data)
    assert isinstance(result_zero_std, torch.Tensor)
    assert result_zero_std.shape[0] == 2 * 6
    # Skewness and Kurtosis should be close to 0 or NaN if std is 0, but handled by epsilon
    assert torch.allclose(result_zero_std[4], torch.tensor(1.0)) # Min of first feature
    assert torch.allclose(result_zero_std[5], torch.tensor(1.0)) # Min of second feature
    assert torch.allclose(result_zero_std[6], torch.tensor(1.0)) # Max of first feature
    assert torch.allclose(result_zero_std[7], torch.tensor(1.0)) # Max of second feature
    assert torch.allclose(result_zero_std[2], torch.tensor(0.0)) # Std of first feature
    assert torch.allclose(result_zero_std[3], torch.tensor(0.0)) # Std of second feature

    # Test with empty tensor
    empty_data = torch.empty(0, 5)
    result_empty = compute_functionals(empty_data)
    assert isinstance(result_empty, torch.Tensor)
    assert result_empty.shape[0] == 5 * 6
    assert torch.all(torch.isnan(result_empty))

    # Test with incorrect dimensions
    with pytest.raises(ValueError):
        compute_functionals(torch.randn(10))
