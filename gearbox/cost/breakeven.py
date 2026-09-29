"""Should the host delegate a subtask, or do it itself?

Cost model (all in USD, prices per million tokens):

  host does it itself : host writes the output.
                        The context is assumed to be in its window already,
                        which makes "direct" cheap and keeps savings claims conservative.
  host delegates      : host writes the brief (host OUTPUT tokens)
                      + host reads the result back (host INPUT tokens)
                      + worker reads brief + referenced context and writes the output
                      + p_fail * (host redoes it directly)

The brief is billed at the host's output price, usually its most expensive
rate. Delegation therefore pays when the output is much longer than the brief,
or when the worker reads context by reference (files, URLs) instead of the host
pasting it into the brief.
"""

from __future__ import annotations

from dataclasses import dataclass

from gearbox.config import Pricing


@dataclass(frozen=True)
class BreakEven:
    host_direct_usd: float
    delegate_expected_usd: float
    savings_usd: float
    delegate: bool

    def as_dict(self) -> dict:
        return {
            "host_direct_usd": round(self.host_direct_usd, 6),
            "delegate_expected_usd": round(self.delegate_expected_usd, 6),
            "savings_usd": round(self.savings_usd, 6),
            "delegate": self.delegate,
        }


def break_even(
    host: Pricing,
    worker: Pricing,
    brief_tokens: int,
    output_tokens: int,
    referenced_context_tokens: int = 0,
    p_fail: float = 0.1,
    min_savings_usd: float = 0.0,
) -> BreakEven:
    if not 0.0 <= p_fail <= 1.0:
        raise ValueError("p_fail must be in [0, 1]")
    host_direct = host.cost(input_tokens=0, output_tokens=output_tokens)
    handoff = host.cost(input_tokens=output_tokens, output_tokens=brief_tokens)
    worker_cost = worker.cost(input_tokens=brief_tokens + referenced_context_tokens, output_tokens=output_tokens)
    expected = handoff + worker_cost + p_fail * host_direct
    savings = host_direct - expected
    return BreakEven(host_direct, expected, savings, delegate=savings > min_savings_usd)
