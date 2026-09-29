import pytest
from fakes import make_config

from gearbox.config import Pricing
from gearbox.cost.breakeven import break_even
from gearbox.cost.ledger import Ledger
from gearbox.providers import Completion

HOST = Pricing(input=10.0, output=50.0)
WORKER = Pricing(input=1.0, output=4.0)


def test_break_even_favours_long_output_short_brief():
    be = break_even(HOST, WORKER, brief_tokens=40, output_tokens=400, referenced_context_tokens=1000, p_fail=0.1)
    # direct 400*50 = 0.02; handoff 400*10 + 40*50 = 0.006; worker 1040*1 + 400*4 = 0.00264; retry 0.1*0.02
    assert be.host_direct_usd == pytest.approx(0.02)
    assert be.delegate_expected_usd == pytest.approx(0.006 + 0.00264 + 0.002)
    assert be.savings_usd == pytest.approx(0.00936)
    assert be.delegate


def test_break_even_rejects_long_brief_short_output():
    be = break_even(HOST, WORKER, brief_tokens=400, output_tokens=50)
    assert be.savings_usd < 0
    assert not be.delegate


def test_break_even_validates_p_fail():
    with pytest.raises(ValueError):
        break_even(HOST, WORKER, 10, 10, p_fail=1.5)


def test_ledger_savings_match_hand_computation(tmp_path):
    config = make_config()  # host 10/50, t0 1/4
    ledger = Ledger(config.host, path=tmp_path / "ledger.jsonl")
    ledger.record(config.tiers[0], Completion("ok", 100, 400), role="delegate", task_id="a", brief_tokens=40)
    s = ledger.summary()
    # worker (100*1 + 400*4) = 0.0017; direct 0.02; handoff 0.006
    assert s["total_cost_usd"] == pytest.approx(0.0017)
    assert s["estimated_host_savings_usd"] == pytest.approx(0.02 - 0.006 - 0.0017)
    assert s["delegated_tasks_succeeded"] == 1
    assert len((tmp_path / "ledger.jsonl").read_text().splitlines()) == 1


def test_ledger_counts_failed_task_as_pure_overhead():
    config = make_config()
    ledger = Ledger(config.host)
    ledger.record(config.tiers[0], Completion("UNSURE: x", 100, 400), role="delegate", task_id="a", ok=False, brief_tokens=40)
    s = ledger.summary()
    assert s["estimated_host_savings_usd"] == pytest.approx(-0.006 - 0.0017)
    assert s["delegated_tasks_succeeded"] == 0
