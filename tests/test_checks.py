import asyncio
import os

import pytest
from fakes import FakeProvider, make_config

from gearbox.delegate.runtime import DelegationRuntime, TaskState
from gearbox.verify.checks import extract_code, run_check

GOOD = "```python\ndef add(a, b):\n    return a + b\n```"
BUGGY = "```python\ndef add(a, b):\n    return a - b\n```"
CHECK = "assert add(2, 3) == 5\nassert add(-1, 1) == 0"


def check(result: str, code: str, timeout_s: float = 10.0):
    return asyncio.run(run_check(result, code, timeout_s))


def test_extract_code_prefers_python_fences():
    reply = "Here you go:\n```python\nx = 1\n```\n```bash\nrm -rf /\n```\n```\ny = 2\n```"
    assert extract_code(reply) == "x = 1\n\ny = 2"
    assert extract_code("x = 1") == "x = 1"


def test_extract_code_handles_unbalanced_fences():
    # ministral-3:8b on the A5000: code, then a closing fence with no opening one
    assert extract_code("def f():\n    return 1\n```") == "def f():\n    return 1"
    assert extract_code("```python\ndef f():\n    return 1") == "def f():\n    return 1"
    assert extract_code("~~~\nx = 1\n") == "x = 1"
    # a fence inside a string literal is not a fence line, so it stays
    assert extract_code('s = "```"') == 's = "```"'
    assert check("def add(a, b):\n    return a + b\n```", CHECK).passed
    from gearbox.verify import has_unbalanced_fences
    assert has_unbalanced_fences("def f(): pass\n```")
    assert not has_unbalanced_fences("```python\ndef f(): pass\n```")
    assert not has_unbalanced_fences("def f(): pass")


def test_passing_and_failing_checks():
    assert check(GOOD, CHECK).passed
    failed = check(BUGGY, CHECK)
    assert not failed.passed
    assert "assert add(2, 3) == 5" in failed.output  # the failing line is shown


def test_failed_assert_reports_actual_values():
    failed = check(BUGGY, CHECK)
    assert failed.output.splitlines()[-1] == "Failed: add(2, 3) returned -1, expected 5"
    computed = check(BUGGY, "assert add(2, 3) == add(5, 0)")
    assert computed.output.splitlines()[-1] == "Failed: add(2, 3) returned -1, expected add(5, 0) = 5"
    falsy = check("```python\ndef ok():\n    return 0\n```", "assert ok()")
    assert falsy.output.splitlines()[-1] == "Failed: ok() returned 0, expected a truthy value"
    negated = check("```python\ndef ok():\n    return 1\n```", "assert not ok()")
    assert negated.output.splitlines()[-1] == "Failed: ok() returned 1, expected a falsy value"
    # an assert with its own message keeps that message instead
    custom = check(BUGGY, "assert add(2, 3) == 5, 'add is broken'")
    assert "add is broken" in custom.output and "Failed:" not in custom.output


def test_test_functions_are_called():
    assert check(GOOD, "def test_add():\n    assert add(1, 1) == 2").passed
    assert not check(GOOD, "def test_add():\n    assert add(1, 1) == 3").passed


def test_result_text_available_for_non_code_checks():
    assert check('{"ok": true}', "import json\nassert json.loads(RESULT)['ok'] is True").passed


def test_unloadable_answer_is_reported():
    failed = check("Sorry, I can't do that.", CHECK)
    assert not failed.passed
    assert "did not compile" in failed.output


def test_crash_after_definitions_keeps_the_definitions():
    # Seen on the A5000: a demo call after the function crashed the load, but the function
    # was defined and returned a wrong value, which is a logic failure, not a missing function.
    answer = "```python\ndef add(a, b):\n    return a - b\n\nprint(add(int(input()), 2))\n```"
    failed = check(answer, CHECK)
    assert "raised an error while loading" in failed.output
    assert failed.output.splitlines()[-1] == "Failed: add(2, 3) returned -1, expected 5"


def test_timeout_kills_runaway_code():
    result = check("while True:\n    pass", "assert True", timeout_s=2)
    assert not result.passed and "timed out" in result.output


def test_signal_death_is_reported():
    result = check("x = 1", "import os, signal\nos.kill(os.getpid(), signal.SIGKILL)")
    assert not result.passed and "SIGKILL" in result.output


def test_timeout_race_with_already_dead_process(monkeypatch):
    """The process can die (e.g. by CPU limit) just as the wall-clock timeout fires;
    killing it then must not crash the check (seen on Linux/WSL)."""
    import gearbox.verify.checks as checks_module

    async def finish_then_time_out(awaitable, timeout):
        await awaitable  # let the process exit and be reaped
        raise asyncio.TimeoutError

    monkeypatch.setattr(checks_module.asyncio, "wait_for", finish_then_time_out)
    result = check(GOOD, CHECK)
    assert not result.passed and "timed out" in result.output


def test_environment_is_not_inherited(monkeypatch):
    monkeypatch.setenv("GEARBOX_TEST_SECRET", "s3cret")
    assert os.environ["GEARBOX_TEST_SECRET"] == "s3cret"
    assert check(GOOD, "import os\nassert 'GEARBOX_TEST_SECRET' not in os.environ").passed


def test_main_guard_in_answer_does_not_run():
    answer = "```python\ndef add(a, b):\n    return a + b\nif __name__ == '__main__':\n    input()\n```"
    assert check(answer, CHECK, timeout_s=5).passed


# --- the verify-then-accept ladder in the runtime ---

def repairing_worker(messages):
    """Buggy on a fresh brief; fixed only when shown why the last answer failed."""
    return GOOD if "failed the check" in messages[-1]["content"] else BUGGY


def run_task(provider, check_code=CHECK, **overrides):
    async def scenario():
        rt = DelegationRuntime(make_config(code_checks=True, **overrides), provider=provider)
        dt = await rt.run("Write add(a, b)", tier="t0", check=check_code)
        return dt, rt

    return asyncio.run(scenario())


def test_failed_check_escalates_with_repair_feedback():
    provider = FakeProvider({"t0": BUGGY, "t1": repairing_worker})
    dt, rt = run_task(provider, max_escalations=1)
    assert dt.state is TaskState.DONE
    assert dt.view()["verified"] is True
    assert [(a.tier, a.outcome) for a in dt.attempts] == [("t0", "check_failed"), ("t1", "ok")]
    assert dt.attempts[0].check["passed"] is False and dt.attempts[1].check["passed"] is True
    assert dt.result == GOOD


def test_failed_check_without_escalation_fails_and_keeps_answer():
    dt, rt = run_task(FakeProvider({"t0": BUGGY}), max_escalations=0)
    assert dt.state is TaskState.FAILED
    assert dt.view()["verified"] is False
    assert dt.result == BUGGY and "check failed on t0" in dt.error


def test_ledger_reports_false_done_rate_per_tier():
    provider = FakeProvider({"t0": BUGGY, "t1": repairing_worker})
    _, rt = run_task(provider, max_escalations=1)
    summary = rt.ledger.summary()
    assert summary["false_done_rate"] == {"t0": 1.0, "t1": 0.0}
    assert summary["by_tier"]["t0"]["check_failures"] == 1


def test_no_check_means_unverified_not_failed():
    async def scenario():
        rt = DelegationRuntime(make_config(code_checks=True), provider=FakeProvider())
        return await rt.run("Summarise", tier="t0")

    dt = asyncio.run(scenario())
    assert dt.state is TaskState.DONE and dt.view()["verified"] is None


def test_checks_are_opt_in():
    async def scenario():
        rt = DelegationRuntime(make_config(), provider=FakeProvider())
        with pytest.raises(ValueError, match="code_checks"):
            rt.delegate("Write add", check=CHECK)

    asyncio.run(scenario())


def test_worker_sees_the_check_as_its_spec():
    seen = []
    provider = FakeProvider({"t0": lambda messages: seen.append(messages[-1]["content"]) or GOOD})
    run_task(provider)
    assert "must pass this check" in seen[0] and "assert add(2, 3) == 5" in seen[0]


def test_hidden_check_is_run_but_not_shown():
    seen = []
    provider = FakeProvider({"t0": lambda messages: seen.append(messages[-1]["content"]) or GOOD})

    async def scenario():
        rt = DelegationRuntime(make_config(code_checks=True), provider=provider)
        return await rt.run("Write add(a, b)", tier="t0", check=CHECK, show_check=False)

    dt = asyncio.run(scenario())
    assert "assert add" not in seen[0]  # the worker never saw the tests
    assert dt.view()["verified"] is True  # but they still ran


def test_worker_functions_named_test_are_not_run_as_tests():
    # MBPP's Mbpp/19 asks for a function named test_duplicate(arraynums)
    answer = "def test_duplicate(nums):\n    return len(nums) != len(set(nums))"
    assert check(answer, "assert test_duplicate([1, 1])\nassert not test_duplicate([1, 2])").passed


def test_cancelling_a_check_kills_its_process():
    import os
    import signal
    import time

    async def scenario(tmp):
        marker = tmp / "pid"
        task = asyncio.ensure_future(run_check(
            "x = 1", f"import os, time\nopen({str(marker)!r}, 'w').write(str(os.getpid()))\ntime.sleep(60)", timeout_s=60))
        for _ in range(100):
            await asyncio.sleep(0.05)
            if marker.exists() and marker.read_text():
                break
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        return int(marker.read_text())

    import tempfile
    from pathlib import Path as _P
    with tempfile.TemporaryDirectory() as tmp:
        pid = asyncio.run(scenario(_P(tmp)))
    time.sleep(0.5)
    try:
        os.kill(pid, 0)
        alive = True
    except ProcessLookupError:
        alive = False
    if alive:
        os.kill(pid, signal.SIGKILL)
    assert not alive, "the cancelled check kept running"
