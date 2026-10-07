"""Learned routing (gearbox/difficulty/learned.py) and the start floor."""

import asyncio
import json

import pytest
from fakes import FakeProvider, make_config

from gearbox.cost.ledger import Ledger
from gearbox.difficulty import DifficultyEstimate, LearnedEstimator, RouterModel
from gearbox.difficulty.learned import parse_rating
from gearbox.router import Router, decide

MODEL = {  # 4 known tiers (t0..t3) of 5; t4 unknown to the router file; judge is t1
    "judge_model": "fake/t1", "rating_mean": 5.0, "rating_sd": 2.0, "intercept": 0.0, "slope": 1.0,
    "abilities": {"fake/t0": -2.0, "fake/t1": 0.0, "fake/t2": 1.5, "fake/t3": 3.0},
}


def test_pass_chance_falls_with_rating_and_rises_with_ability():
    m = RouterModel(MODEL)
    assert m.pass_prob("fake/t2", 2) > m.pass_prob("fake/t2", 8)
    assert m.pass_prob("fake/t3", 5) > m.pass_prob("fake/t2", 5)
    assert m.pass_prob("fake/unknown", 5) is None
    assert parse_rating("I'd say 7.") == 7 and parse_rating("10") == 10 and parse_rating("hard") is None


def test_estimator_asks_the_judge_and_predicts_every_tier():
    config = make_config(5)
    ledger = Ledger(config.host)
    est = asyncio.run(LearnedEstimator(config, FakeProvider({"t1": "7"}), RouterModel(MODEL), ledger).estimate("task"))
    assert est.rationale.startswith("judge t1 rated 7/10")
    assert len(est.pass_prob) == 5 and est.pass_prob[4] is None
    assert [(r.tier, r.role) for r in ledger.records] == [("t1", "judge")]  # the judge's tokens are charged


def test_estimator_falls_back_when_the_judge_is_unusable():
    config = make_config(5)
    est = asyncio.run(LearnedEstimator(config, FakeProvider({"t1": "no idea"}), RouterModel(MODEL)).estimate("task"))
    assert est.pass_prob is None and "unparseable" in est.rationale
    with pytest.raises(ValueError):  # the judge's model must be one of the tiers
        LearnedEstimator(make_config(1), FakeProvider(), RouterModel(MODEL))


def test_learned_routing_picks_the_cheapest_tier_over_the_bar():
    config = make_config(5, learned_tau={"none": 0.9, "weak": 0.9, "medium": 0.8, "strong": 0.0})
    est = DifficultyEstimate(0.5, 1.0, "judge", pass_prob=(0.3, 0.6, 0.85, 0.95, None))
    assert decide(config, est).tier == 3                            # no check: needs 0.9
    assert decide(config, est, check_strength="medium").tier == 2   # 0.8
    assert decide(config, est, check_strength="strong").tier == 0   # start cheap, escalate on failure
    assert decide(config, est, risk="medium").tier == 4             # risk still adds a tier
    assert "t3 passes with p=0.95" in decide(config, est).rationale
    hopeless = DifficultyEstimate(0.9, 1.0, "judge", pass_prob=(0.1, 0.2, 0.3, 0.4, None))
    assert decide(config, hopeless).tier == 4                       # nothing clears the bar: the top
    with pytest.raises(ValueError):
        decide(config, est, risk="extreme")


def test_start_floor_applies_to_every_estimator():
    config = make_config(5, start_floor="t2", leverage=0)
    assert decide(config, DifficultyEstimate(0.0, 0.9)).tier == 2
    learned = DifficultyEstimate(0.5, 1.0, "judge", pass_prob=(0.99, 0.99, 0.99, 0.99, None))
    assert decide(config, learned, check_strength="strong").tier == 2
    with pytest.raises(ValueError):
        make_config(5, start_floor="t9")


def test_learned_config_loads_the_router_file(tmp_path):
    path = tmp_path / "router.json"
    path.write_text(json.dumps(MODEL))
    config = make_config(5, difficulty="learned", router_model=str(path))
    router = Router.from_config(config, FakeProvider({"t1": "2"}), Ledger(config.host))
    decision = asyncio.run(router.route("Write f()."))
    assert decision.tier_name in {"t2", "t3"}
    with pytest.raises(ValueError):
        make_config(difficulty="learned")  # no router_model


def test_offset_makes_every_task_harder_or_easier():
    harder = RouterModel({**MODEL, "offset": 1.0})
    assert harder.pass_prob("fake/t2", 5) < RouterModel(MODEL).pass_prob("fake/t2", 5)
    assert RouterModel(MODEL).offset == 0.0  # files without an offset still load


def test_fit_offset_recovers_a_shift():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bench"))
    import numpy as np
    from router_train import fit_offset, sigmoid

    rng = np.random.default_rng(0)
    a, b = np.array([-1.0, 0.0, 1.0, 2.0]), rng.normal(size=400)
    Y = (rng.random((400, 4)) < sigmoid(a[None] - b[:, None] - 0.8)).astype(float)  # tasks 0.8 harder
    assert abs(fit_offset(a, b, Y) - 0.8) < 0.15
