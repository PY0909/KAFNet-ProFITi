import pytest
import torch

from kaf_profiti.industrial.risk import threshold_risk_score


def test_threshold_risk_score_returns_sample_crossing_probability():
    samples = torch.tensor(
        [
            [
                [[0.0, 0.2], [0.1, -0.3]],
                [[1.5, 0.2], [0.1, -0.3]],
                [[0.0, 0.2], [0.1, 1.2]],
            ]
        ]
    )
    lower_limits = torch.tensor([-1.0, -1.0])
    upper_limits = torch.tensor([1.0, 1.0])

    risk = threshold_risk_score(samples, upper_limits, lower_limits)

    assert risk.shape == (1,)
    assert risk.item() == pytest.approx(2.0 / 3.0)


def test_threshold_risk_score_returns_zero_when_no_sample_crosses_limits():
    samples = torch.zeros(2, 4, 3, 5)
    lower_limits = torch.full((5,), -1.0)
    upper_limits = torch.full((5,), 1.0)

    risk = threshold_risk_score(samples, upper_limits, lower_limits)

    assert torch.equal(risk, torch.zeros(2))


def test_threshold_risk_score_rejects_mismatched_sensor_limits():
    samples = torch.zeros(1, 2, 3, 4)
    lower_limits = torch.full((3,), -1.0)
    upper_limits = torch.full((3,), 1.0)

    with pytest.raises(ValueError, match="threshold dimension"):
        threshold_risk_score(samples, upper_limits, lower_limits)
