import pytest
import torch

from kaf_profiti.industrial.risk import threshold_exceedance_risk_score, threshold_risk_score
from kaf_profiti.experiments.metrics import risk_score_diagnostics


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


def test_threshold_exceedance_risk_score_uses_crossing_fraction_not_any_crossing():
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

    risk = threshold_exceedance_risk_score(samples, upper_limits, lower_limits)

    assert risk.shape == (1,)
    assert risk.item() == pytest.approx(2.0 / 12.0)


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


def test_risk_score_diagnostics_reports_label_and_score_distribution():
    labels = torch.tensor([0.0, 1.0, 1.0])
    scores = torch.tensor([0.2, 0.4, 0.8])

    diagnostics = risk_score_diagnostics(labels.numpy(), scores.numpy())

    assert diagnostics["label_positive_rate"] == pytest.approx(2.0 / 3.0)
    assert diagnostics["label_unique_count"] == 2
    assert diagnostics["risk_score_min"] == pytest.approx(0.2)
    assert diagnostics["risk_score_max"] == pytest.approx(0.8)
    assert diagnostics["risk_score_std"] > 0
    assert diagnostics["risk_score_is_constant"] is False
