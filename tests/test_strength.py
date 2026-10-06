"""Check strength (gearbox/verify/strength.py) and check-aware leverage (gearbox/router.py)."""

import asyncio
import math

import pytest
from fakes import FakeProvider, make_config

from gearbox.delegate.runtime import DelegationRuntime
from gearbox.difficulty import DifficultyEstimate
from gearbox.router import decide
from gearbox.verify import check_strength, count_cases


@pytest.mark.parametrize("code, cases, strength", [
    ("assert f(1) == 2", 1, "weak"),
    ("assert f(1) == 2\nassert f(2) == 3\nassert f(3) == 4", 3, "weak"),
    ("assert a\nassert b\nassert c\nassert d", 4, "medium"),
    ("for x in [1, 2, 3, 4, 5]:\n    assert f(x)", 5, "medium"),
    ("for x in range(20):\n    assert f(x) == x", 20, "strong"),
    ("for x in DATA:\n    assert f(x)", math.inf, "strong"),            # data-driven: uncountable
    ("for x in DATA:\n    if f(x) != 1:\n        raise AssertionError(x)", math.inf, "strong"),  # EvalPlus style
    ("def test_a():\n    assert f(1)\n\ndef test_b():\n    assert f(2)", 2, "weak"),
    ("assert all(f(x) for x in range(5))", 1, "weak"),               # one assert, however many inside
    ("x = 1", 0, "weak"),
    ("not python (", 0, "weak"),
])
def test_check_strength(code, cases, strength):
    assert count_cases(code) == cases
    assert check_strength(code) == strength


def test_check_leverage_from_strength_plus_risk():
    config = make_config(5, leverage=1, risk_scaled_leverage=True)
    easy = DifficultyEstimate(0.0, 0.9)
    assert decide(config, easy, check_strength="weak").tier == 2
    assert decide(config, easy, check_strength="medium").tier == 1
    assert decide(config, easy, check_strength="strong").tier == 0     # start cheap; escalation does the rest
    assert decide(config, easy, risk="high", check_strength="weak").tier == 4
    assert decide(config, easy, check_strength=None).tier == 1          # no check: the constant leverage
    assert decide(config, easy, leverage=0, check_strength="weak").tier == 0  # an explicit leverage wins
    assert "weak check: leverage +2" in decide(config, easy, check_strength="weak").rationale


def test_check_leverage_can_be_turned_off_or_changed():
    easy = DifficultyEstimate(0.0, 0.9)
    assert decide(make_config(5, leverage=1, check_leverage={}), easy, check_strength="weak").tier == 1
    assert decide(make_config(5, check_leverage={"weak": 3}), easy, check_strength="weak").tier == 3
    with pytest.raises(ValueError):
        make_config(check_leverage={"flimsy": 1})
    with pytest.raises(ValueError):
        make_config(check_leverage={"weak": -1})


def test_runtime_routes_by_the_checks_strength():
    async def tiers():
        rt = DelegationRuntime(make_config(5, leverage=0, code_checks=True, max_escalations=0),
                               provider=FakeProvider({f"t{i}": "def f():\n    return 1" for i in range(5)}))
        weak = await rt.run("Write f().", check="assert f() == 1")
        strong = await rt.run("Write f().", check="for _ in range(12):\n    assert f() == 1")
        return weak.decision, strong.decision

    weak, strong = asyncio.run(tiers())
    assert weak.tier - weak.base_tier == 2
    assert strong.tier == strong.base_tier
