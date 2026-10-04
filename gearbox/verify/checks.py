"""Executable acceptance checks: run the host's test code against a worker's output.

The worker's code (fenced blocks, or the whole reply) is loaded first, then the
check runs in the same namespace: plain asserts, or `test_*` functions that are
called one by one. `RESULT` holds the raw reply for non-code checks, e.g.
`assert json.loads(RESULT)["ok"]`.

This executes model-written code. Isolation is best effort, NOT a security
boundary: a fresh temp dir, an empty environment (no API keys), Python isolated
mode, stdin closed, CPU/file-size limits, a wall-clock timeout, and no network
where the OS allows it (sandbox-exec on macOS, `unshare -rn` on Linux).
Run Gearbox in a VM or container if that is not enough.
"""

from __future__ import annotations

import asyncio
import functools
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

PASS_MARKER = "GEARBOX_CHECK_PASSED"
MAX_OUTPUT_CHARS = 2000
_FENCE_RE = re.compile(r"```[ \t]*([\w+-]*)[^\n]*\n(.*?)```", re.DOTALL)
_FENCE_LINE_RE = re.compile(r"^[ \t]*(?:```|~~~)[\w+-]*[ \t]*$", re.MULTILINE)
_MACOS_NO_NETWORK = "(version 1)(allow default)(deny network*)"

RUNNER = f'''
import ast, resource, sys, traceback

def _limit(kind, value):
    try:
        resource.setrlimit(kind, (value, value))
    except (ValueError, OSError):
        pass

_limit(resource.RLIMIT_CPU, int(sys.argv[1]))
_limit(resource.RLIMIT_FSIZE, 10 * 2**20)

ns = {{"__name__": "__gearbox_check__", "RESULT": open("result.txt", encoding="utf-8").read()}}
def explain(exc):
    """For a bare failed `assert a == b` / `assert f(x)`, re-evaluate the parts so the
    message says what the answer actually returned (plain asserts don't)."""
    frame = next((f for f in reversed(traceback.extract_tb(exc.__traceback__)) if f.filename == "check.py"), None)
    if str(exc) or frame is None or not frame.line:
        return None
    try:
        test = ast.parse(frame.line.strip()).body[0].test
        value = lambda node: eval(compile(ast.Expression(node), "check.py", "eval"), ns)
        if isinstance(test, ast.Compare) and len(test.ops) == 1:
            right = test.comparators[0]
            expected = repr(value(right)) if isinstance(right, ast.Constant) else f"{{ast.unparse(right)}} = {{value(right)!r}}"
            return f"{{ast.unparse(test.left)}} returned {{value(test.left)!r}}, expected {{expected}}"
        if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
            return f"{{ast.unparse(test.operand)}} returned {{value(test.operand)!r}}, expected a falsy value"
        return f"{{ast.unparse(test)}} returned {{value(test)!r}}, expected a truthy value"
    except BaseException:
        return None

load_error = None
try:
    solution = compile(open("solution.py", encoding="utf-8").read(), "solution.py", "exec")
except SyntaxError:
    load_error = "The worker's code did not compile:\\n" + traceback.format_exc(limit=0)
else:
    try:
        exec(solution, ns)
    except BaseException:
        # e.g. a demo call after the definitions; what was defined before it still counts
        load_error = ("The worker's code raised an error while loading (definitions before the "
                      "error are still used):\\n" + traceback.format_exc(limit=2))

try:
    before = dict(ns)
    exec(compile(open("check.py", encoding="utf-8").read(), "check.py", "exec"), ns)
    # Only test_ functions the check itself defined: the worker's code may define its own
    # (e.g. MBPP's Mbpp/19 asks for a function named test_duplicate).
    tests = [(name, fn) for name, fn in list(ns.items())
             if name.startswith("test_") and callable(fn) and before.get(name) is not fn]
    for name, fn in tests:
        fn()
except BaseException as exc:
    if load_error:
        sys.stderr.write(load_error + "\\n")
    traceback.print_exc(limit=-3)
    detail = explain(exc) if isinstance(exc, AssertionError) else None
    if detail:
        sys.stderr.write("Failed: " + detail + "\\n")
    sys.exit(1)
print("{PASS_MARKER} tests=%d" % len(tests))
'''


@dataclass(frozen=True)
class CheckResult:
    passed: bool
    output: str  # tail of the check's stdout/stderr, for the next tier and the host
    duration_s: float
    isolation: str

    def as_dict(self) -> dict:
        return {
            "passed": self.passed,
            "output": self.output,
            "duration_s": round(self.duration_s, 3),
            "isolation": self.isolation,
        }


def extract_code(text: str) -> str:
    """Python from fenced blocks if there are any (other languages skipped), else the whole
    reply minus stray fence lines. Some models leave fences unbalanced: ministral-3:8b
    writes the code with a closing ``` but no opening one."""
    blocks = _FENCE_RE.findall(text)
    if not blocks:
        return _FENCE_LINE_RE.sub("", text).strip()
    python = [body for lang, body in blocks if lang.lower() in ("", "py", "python", "python3")]
    return "\n\n".join(b.strip() for b in python)


def has_unbalanced_fences(text: str) -> bool:
    """True when extract_code had to drop stray fence lines (a formatting slip worth counting)."""
    return not _FENCE_RE.findall(text) and bool(_FENCE_LINE_RE.search(text))


@functools.lru_cache(maxsize=1)
def isolation_mode() -> str:
    """Best network isolation available on this machine, probed once."""
    if sys.platform == "darwin" and shutil.which("sandbox-exec"):
        return "sandbox-exec (no network)"
    if sys.platform.startswith("linux") and shutil.which("unshare"):
        probe = subprocess.run(["unshare", "-rn", "true"], capture_output=True)
        if probe.returncode == 0:
            return "unshare (no network)"
    return "subprocess (network allowed)"


def _command(workdir: Path, cpu_s: int) -> list[str]:
    python = [sys.executable, "-I", str(workdir / "runner.py"), str(cpu_s)]
    mode = isolation_mode()
    if mode.startswith("sandbox-exec"):
        return ["sandbox-exec", "-p", _MACOS_NO_NETWORK, *python]
    if mode.startswith("unshare"):
        return ["unshare", "-rn", *python]
    return python


def _tail(text: str) -> str:
    text = text.strip()
    return text if len(text) <= MAX_OUTPUT_CHARS else "…" + text[-MAX_OUTPUT_CHARS:]


async def run_check(result_text: str, check_code: str, timeout_s: float = 20.0) -> CheckResult:
    start = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="gearbox-check-") as tmp:
        workdir = Path(tmp)
        (workdir / "runner.py").write_text(RUNNER, encoding="utf-8")
        (workdir / "result.txt").write_text(result_text, encoding="utf-8")
        (workdir / "solution.py").write_text(extract_code(result_text), encoding="utf-8")
        (workdir / "check.py").write_text(check_code, encoding="utf-8")
        proc = await asyncio.create_subprocess_exec(
            # The CPU limit is a backstop set past the wall-clock timeout, so the timeout acts
            # first. With equal limits, Linux kills a busy loop by CPU limit at the same moment.
            *_command(workdir, cpu_s=int(timeout_s) + 5),
            cwd=workdir,
            env={"PATH": "/usr/bin:/bin", "HOME": tmp, "PYTHONDONTWRITEBYTECODE": "1"},
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        try:
            out, _ = await asyncio.wait_for(proc.communicate(), timeout_s)
        except BaseException as e:
            # Timeout, or the caller was cancelled: never leave the check process running.
            try:
                proc.kill()
            except ProcessLookupError:  # it already exited (e.g. by its CPU limit)
                pass
            if not isinstance(e, asyncio.TimeoutError):
                raise
            await proc.wait()
            return CheckResult(False, f"check timed out after {timeout_s}s", time.perf_counter() - start, isolation_mode())
    text = out.decode("utf-8", errors="replace")
    if proc.returncode is not None and proc.returncode < 0:
        name = signal.Signals(-proc.returncode).name
        reason = "exceeded its CPU time limit" if name in ("SIGXCPU", "SIGKILL") else "was killed"
        text += f"\ncheck process {reason} ({name})"
    passed = proc.returncode == 0 and PASS_MARKER in text
    return CheckResult(passed, _tail(text), time.perf_counter() - start, isolation_mode())
