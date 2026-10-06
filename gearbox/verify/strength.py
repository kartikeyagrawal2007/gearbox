"""How strong is a check? Counts its test cases without running it.

A check's strength decides where a checked subtask starts on the ladder (see
gearbox/router.py): a weak check lets more wrong answers through, so it needs more
leverage. On HumanEval+ and MBPP+ (bench/weak_checks.py), a cascade with a 1-3 test check
needed to start two tiers up to stay accurate; with the full hidden suite, starting at the
cheapest tier was best.

Counting rules, on the check's syntax tree:
  - each `assert` (or `raise AssertionError(...)`) is one case
  - an `assert` in a loop over a literal list/tuple/set, or over range(n), counts per item
  - an `assert` in any other loop or comprehension (e.g. over loaded test data) can't be
    counted, so the check is treated as strong
"""

from __future__ import annotations

import ast
import math

WEAK_MAX = 3    # 1-3 cases: weak
MEDIUM_MAX = 9  # 4-9 cases: medium; 10 or more (or uncountable loops): strong


def _loop_items(loop: ast.For) -> float:
    it = loop.iter
    if isinstance(it, (ast.List, ast.Tuple, ast.Set)):
        return len(it.elts)
    if (isinstance(it, ast.Call) and isinstance(it.func, ast.Name) and it.func.id == "range"
            and all(isinstance(a, ast.Constant) and isinstance(a.value, int) for a in it.args)):
        return len(range(*(a.value for a in it.args)))
    return math.inf


def _is_case(node: ast.AST) -> bool:
    """`assert ...`, or `raise AssertionError(...)` (how data-driven checks like EvalPlus's fail)."""
    if isinstance(node, ast.Assert):
        return True
    if isinstance(node, ast.Raise) and node.exc is not None:
        exc = node.exc.func if isinstance(node.exc, ast.Call) else node.exc
        return isinstance(exc, ast.Name) and exc.id == "AssertionError"
    return False


def _count(node: ast.AST) -> float:
    if _is_case(node):
        return 1
    if isinstance(node, (ast.For, ast.AsyncFor)):
        body = sum(_count(n) for n in node.body)
        return (body * _loop_items(node) if body else 0) + sum(_count(n) for n in node.orelse)
    inner = sum(_count(n) for n in ast.iter_child_nodes(node))
    if isinstance(node, (ast.While, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)) and inner:
        return math.inf
    return inner


def count_cases(check_code: str) -> float:
    """Number of test cases in the check (math.inf if a loop makes it uncountable)."""
    try:
        return _count(ast.parse(check_code))
    except SyntaxError:
        return 0


def check_strength(check_code: str) -> str:
    """'weak', 'medium' or 'strong' (an unparseable or assert-free check is weak)."""
    n = count_cases(check_code)
    return "weak" if n <= WEAK_MAX else "medium" if n <= MEDIUM_MAX else "strong"
