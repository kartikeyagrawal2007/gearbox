"""bench/false_done.py: telling format failures from logic failures."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bench"))
from false_done import failure_kind  # noqa: E402

TARGETS = ("slugify",)


def test_missing_requested_function_is_format():
    assert failure_kind("Traceback ...\nNameError: name 'slugify' is not defined", TARGETS) == "format"
    out = "The worker's code did not compile:\nSyntaxError\nNameError: name 'slugify' is not defined"
    assert failure_kind(out, TARGETS) == "format"


def test_wrong_value_is_logic_even_if_loading_crashed():
    # granite4.2:8b and qwen3.5:2b on the A5000 were mislabelled format by the old rule
    out = ("The worker's code raised an error while loading (definitions before the error are still used):\n"
           "EOFError\nTraceback ...\nFailed: slugify('Héllo') returned 'helloworld', expected 'hello'")
    assert failure_kind(out, TARGETS) == "logic"


def test_undefined_helper_is_logic():
    assert failure_kind("Traceback ...\nNameError: name 'helper' is not defined", TARGETS) == "logic"


# --- HumanEval+ checks (bench/tasksets.py) ---

import asyncio  # noqa: E402
import gzip  # noqa: E402
import json  # noqa: E402

import pytest  # noqa: E402

from gearbox.verify.checks import run_check  # noqa: E402
from tasksets import SMOKE, evalplus_check, humaneval_check, load_tasks  # noqa: E402

PROBLEM = {  # shaped like an EvalPlus record
    "task_id": "Demo/0", "entry_point": "truncate_number", "atol": 0,
    "prompt": "def truncate_number(number: float) -> float:\n    \"\"\"Return the decimal part.\"\"\"\n",
    "canonical_solution": "    return number - int(number)\n",
    "base_input": [[3.5], [1.25]], "plus_input": [[123.456]],
}


def run(answer: str, problem=PROBLEM):
    return asyncio.run(run_check(answer, humaneval_check(problem), timeout_s=30))


def test_humaneval_check_accepts_a_correct_answer():
    assert run("def truncate_number(number):\n    return number % 1.0").passed  # floats within tolerance


def test_humaneval_check_reports_the_failing_input():
    r = run("def truncate_number(number):\n    return 0.5")
    assert not r.passed
    assert "truncate_number(1.25) returned 0.5, expected 0.25 (input 2 of 3)" in r.output


def test_humaneval_check_reports_exceptions_with_the_input():
    r = run("def truncate_number(number):\n    raise ValueError('nope')")
    assert "truncate_number(3.5) raised ValueError: nope" in r.output


def test_humaneval_missing_function_is_a_format_failure():
    r = run("def something_else():\n    pass")
    assert failure_kind(r.output, ("truncate_number",)) == "format"


def test_humaneval_inputs_are_copied():
    problem = dict(PROBLEM, entry_point="first", prompt="def first(xs):\n", canonical_solution="    return xs[0]\n",
                   base_input=[[[1, 2]], [[3, 4]]], plus_input=[])
    assert run("def first(xs):\n    return xs.pop(0)", problem).passed  # mutating its input can't break the next call


def test_find_zero_accepts_any_root():
    problem = {"task_id": "Demo/32", "entry_point": "find_zero", "atol": 1e-6,
               "prompt": "def poly(xs, x):\n    return sum(c * x ** i for i, c in enumerate(xs))\n\ndef find_zero(xs):\n",
               "canonical_solution": "    return 1.0\n",  # root of x^2 - 1 (the other root is -1)
               "base_input": [[[-1, 0, 1]]], "plus_input": []}
    assert run("def find_zero(xs):\n    return -1.0", problem).passed


DATA = Path(__file__).resolve().parent.parent / "bench/data"


@pytest.mark.skipif(not (DATA / "MbppPlus.jsonl.gz").exists(), reason="EvalPlus data not downloaded")
def test_every_mbpp_plus_reference_passes_its_own_check():
    # Catches grader drift from EvalPlus: input conversion, special oracles, data quirks.
    problems = [json.loads(line) for line in gzip.open(DATA / "MbppPlus.jsonl.gz", "rt")]

    async def all_checks():
        sem = asyncio.Semaphore(4)

        async def one(p):
            async with sem:  # Mbpp/599 alone needs ~18 s; leave headroom for a busy machine
                return await run_check("```python\n" + p["canonical_solution"] + "\n```",
                                       evalplus_check(p, "mbpp"), timeout_s=120)

        return await asyncio.gather(*(one(p) for p in problems))

    results = asyncio.run(all_checks())
    assert [p["task_id"] for p, r in zip(problems, results) if not r.passed] == []
    assert len(load_tasks("mbpp+")) == 378


@pytest.mark.skipif(not (DATA / "HumanEvalPlus-Mini.jsonl.gz").exists(), reason="EvalPlus data not downloaded")
def test_every_reference_solution_passes_its_own_mini_check():
    path = Path(__file__).resolve().parent.parent / "bench/data/HumanEvalPlus-Mini.jsonl.gz"
    problems = [json.loads(line) for line in gzip.open(path, "rt")]

    async def all_checks():
        return await asyncio.gather(*(run_check("```python\n" + p["prompt"] + p["canonical_solution"] + "\n```",
                                                humaneval_check(p), timeout_s=60) for p in problems))

    results = asyncio.run(all_checks())
    assert [p["task_id"] for p, r in zip(problems, results) if not r.passed] == []
    assert len(load_tasks("humaneval+mini")) == 164 and len(load_tasks("smoke")) == len(SMOKE) == 8


def test_mbpp_not_none_tasks_compare_against_is_not_none():
    # Mbpp/737-style: the output (a re.Match or None) is only checked for "is not None"
    problem = {"task_id": "Mbpp/737", "entry_point": "check_str", "atol": 0,
               "prompt": "", "canonical_solution": "import re\ndef check_str(s):\n    return re.search('^[aeiou]', s)\n",
               "base_input": [["annie"], ["dawood"]], "plus_input": {}}
    answer = "import re\ndef check_str(s):\n    return re.match(r'[aeiouAEIOU]', s)"
    assert asyncio.run(run_check(answer, evalplus_check(problem, "mbpp"), timeout_s=30)).passed


def test_results_table_for_terminals(tmp_path):
    from results import render

    assert "No results" in render(tmp_path)
    (tmp_path / "he.json").write_text(json.dumps([
        {"tier": "qwen3.5-0.8b", "hatch": "on", "task_set": "humaneval+", "tasks": 164, "passed": 80, "unsure": 0,
         "false_done_rate": 0.512, "logic_false_done_rate": 0.512, "format_failures": 0, "fences_repaired": 2, "rows": []},
    ]))
    (tmp_path / "old.json").write_text(json.dumps([
        {"tier": "q0.5b", "tasks": 8, "passed": 1, "false_done_rate": 0.5, "rows": [{"outcome": "unsure"}]},
    ]))
    out = render(tmp_path)
    assert "== humaneval+" in out and "== smoke" in out
    assert "qwen3.5-0.8b" in out and "80/164 (49%)" in out and " 51%" in out
    assert "q0.5b" in out  # older files without the newer fields still render


def test_results_follow_the_size_ladder(tmp_path):
    from results import render

    rows = [{"tier": t, "hatch": "on", "task_set": "humaneval+", "tasks": 1, "passed": 1, "unsure": 0,
             "false_done_rate": 0, "rows": []} for t in ("qwen3.5-27b", "qwen3.5-2b", "qwen3.5-0.8b")]
    (tmp_path / "r.json").write_text(json.dumps(rows))
    out = render(tmp_path)
    assert out.index("qwen3.5-0.8b") < out.index("qwen3.5-2b") < out.index("qwen3.5-27b")


def test_plots_render_from_results(tmp_path):
    pytest.importorskip("matplotlib")
    from plots import params_b, size_ladder, vendor, vendors

    assert params_b("qwen3.5-0.8b") == 0.8 and params_b("granite4.2-8b") == 8.0
    assert vendor("ministral3-3b") == "Ministral 3 (Mistral)" and vendor("qwen2.5-coder-0.5b") is None
    rows = [{"tier": t, "hatch": "on", "task_set": "humaneval+", "tasks": 10, "passed": p, "unsure": 0,
             "logic_false_done_rate": (10 - p) / 10}
            for t, p in (("qwen3.5-0.8b", 2), ("qwen3.5-4b", 7), ("granite4.2-3b", 7), ("gemma3-12b", 8))]
    made = size_ladder(rows, "humaneval+", tmp_path) + vendors(rows, "humaneval+", tmp_path)
    assert all(p.exists() and p.stat().st_size > 1000 for p in made)


# --- Async delegation experiment (bench/async_bench.py) ---

import dataclasses as _dc  # noqa: E402

from async_bench import MODES, Episode, RemoteHost, modes_for, run_episode, summarize  # noqa: E402
from fakes import FakeProvider, make_config  # noqa: E402

from tasksets import Task  # noqa: E402


def test_async_episode_overlaps_host_work_with_workers():
    config = _dc.replace(make_config(2), code_checks=True, max_concurrent=4, max_escalations=0)
    provider = FakeProvider({"t0": "def f():\n    return 1", "t1": "def f():\n    return 1"}, delay=0.3)
    subtask = Task("p", "Write f() returning 1.", "assert f() == 1", ("f",), show_check=False)
    episode = Episode((subtask, subtask), ("step one", "step two"))

    async def all_modes():
        return {m: await run_episode(m, episode, config, provider, config.tiers[1], "t0", 64) for m in MODES}

    runs = asyncio.run(all_modes())
    assert all(r["outcomes"] == ["ok", "ok"] for r in runs.values())
    assert runs["blocking"]["host_blocked_s"] > 0.3          # waited for each worker in turn
    assert runs["async"]["wall_s"] < 0.8 * runs["blocking"]["wall_s"]  # host steps overlapped the workers
    assert runs["host_only"]["worker_busy_s"] == 0           # nothing was delegated
    records = [{**r, "host": "t1", "worker": "t0", "k": 2, "host_tokens": 64, "rep": 0, "placement": {}}
               for r in runs.values()]
    assert "t0" in summarize(records)


def test_remote_host_is_emulated_by_time_and_skips_host_only():
    config = _dc.replace(make_config(1), code_checks=True, max_concurrent=4, max_escalations=0)
    provider = FakeProvider({"t0": "def f():\n    return 1"}, delay=0.3)
    subtask = Task("p", "Write f() returning 1.", "assert f() == 1", ("f",), show_check=False)
    episode = Episode((subtask, subtask), ("step one", "step two"))
    host = RemoteHost(ttft_s=0.1, tokens_per_s=1000)  # 0.1 s + 200 tokens / 1000 = 0.3 s per step
    assert "host_only" not in modes_for(host)

    async def all_modes():
        return {m: await run_episode(m, episode, config, provider, host, "t0", 200) for m in modes_for(host)}

    runs = asyncio.run(all_modes())
    assert provider.calls.count("t0") == 2 * len(runs)       # only workers called a model
    assert abs(runs["parallel"]["host_work_s"] - 0.6) < 0.1  # two emulated steps of 0.3 s
    assert runs["async"]["wall_s"] < 0.8 * runs["blocking"]["wall_s"]
