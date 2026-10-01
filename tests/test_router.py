import pytest
from fakes import make_config

from gearbox.difficulty import DifficultyEstimate
from gearbox.router import base_tier_for, decide


@pytest.mark.parametrize("score, tier", [(0.0, 0), (0.33, 0), (0.34, 1), (0.66, 1), (0.67, 2), (1.0, 2)])
def test_base_tier_bands(score, tier):
    assert base_tier_for(score, 3) == tier


def test_constant_leverage_shifts_up_and_clamps():
    config = make_config(leverage=1)
    assert decide(config, DifficultyEstimate(0.1, 0.9)).tier == 1
    assert decide(config, DifficultyEstimate(0.9, 0.9)).tier == 2  # already top gear


def test_risk_scaled_leverage():
    config = make_config(leverage=1, risk_scaled_leverage=True)
    sure_easy = DifficultyEstimate(0.1, 0.9)
    unsure_easy = DifficultyEstimate(0.1, 0.3)
    assert decide(config, sure_easy, risk="low").leverage == 0
    assert decide(config, sure_easy, risk="high").leverage == 2
    assert decide(config, unsure_easy, risk="medium").leverage == 2  # +1 for low confidence
    assert decide(config, sure_easy, risk=None).leverage == 1  # falls back to constant


def test_explicit_leverage_and_bounds_win():
    config = make_config(leverage=2, risk_scaled_leverage=True)
    est = DifficultyEstimate(0.1, 0.9)
    assert decide(config, est, risk="high", leverage=0).tier == 0
    assert decide(config, est, leverage=0, min_tier=1).tier == 1
    assert decide(config, est, max_tier=1).tier == 1


def test_invalid_risk_rejected():
    with pytest.raises(ValueError):
        decide(make_config(risk_scaled_leverage=True), DifficultyEstimate(0.5, 0.5), risk="extreme")
    with pytest.raises(ValueError):  # also when risk-scaling is off
        decide(make_config(risk_scaled_leverage=False), DifficultyEstimate(0.5, 0.5), risk="extreme")
