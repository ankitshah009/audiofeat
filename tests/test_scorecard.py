import pytest

from audiofeat.validation.scorecard import run_gold_standard_scorecard


def test_gold_standard_scorecard_runs_without_optional_dependencies():
    report = run_gold_standard_scorecard(include_optional=False)
    assert isinstance(report, dict)
    assert 0.0 <= float(report["score"]) <= 100.0
    assert report["summary"]["available_points"] > 0
    assert isinstance(report["checks"], list)
    # With optional checks disabled, all required checks should pass.
    assert report["passed"] is True


def test_gold_standard_scorecard_validates_numeric_args():
    with pytest.raises(ValueError):
        run_gold_standard_scorecard(sample_rate=0, include_optional=False)
    with pytest.raises(ValueError):
        run_gold_standard_scorecard(frame_length=0, include_optional=False)
    with pytest.raises(ValueError):
        run_gold_standard_scorecard(hop_length=0, include_optional=False)
