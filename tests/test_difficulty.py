import asyncio

from fakes import FakeProvider, make_config

from gearbox.cost.ledger import Ledger
from gearbox.difficulty import HeuristicEstimator, JudgeEstimator
from gearbox.difficulty.judge import parse_judgement


def test_heuristic_orders_easy_below_hard():
    h = HeuristicEstimator()
    easy = h.estimate_sync("Fix the typo in this docstring")
    hard = h.estimate_sync(
        "Design a distributed consensus algorithm, prove its safety invariant, and analyse its "
        "complexity under concurrent node failures and network partitions."
    )
    assert 0.0 <= easy.score < hard.score <= 1.0
    assert 0.0 < easy.confidence <= 1.0


def test_parse_judgement():
    est = parse_judgement('Sure! {"difficulty": 4, "confidence": 0.8, "reason": "multi-step"}')
    assert est.score == 0.75 and est.confidence == 0.8
    assert parse_judgement("no json here") is None
    assert parse_judgement('{"difficulty": "hard"}') is None
    assert parse_judgement('{"difficulty": 9}').score == 1.0  # clamped


def test_judge_charges_ledger_and_falls_back_on_garbage():
    config = make_config()
    ledger = Ledger(config.host)
    judge = JudgeEstimator(FakeProvider({"t0": "I think it's medium"}), config.tiers[0], ledger)
    est = asyncio.run(judge.estimate("Rename a variable"))
    assert "heuristic" in est.rationale
    assert [r.role for r in ledger.records] == ["judge"]


def test_judge_survives_provider_error():
    config = make_config()
    ledger = Ledger(config.host)
    judge = JudgeEstimator(FakeProvider({"t0": ConnectionError("down")}), config.tiers[0], ledger)
    est = asyncio.run(judge.estimate("Rename a variable"))
    assert "judge error" in est.rationale
    assert ledger.records == ()
