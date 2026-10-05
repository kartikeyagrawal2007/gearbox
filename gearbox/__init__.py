"""Gearbox: automatic transmission for LLM agents.

A host model (the expensive agent) hands subtasks to cheaper models and keeps working.
Where things live:
  config.py       the tier ladder (cheapest -> strongest model) and settings, from YAML
  router.py       picks a tier from a difficulty estimate plus "leverage" (a safety margin)
  difficulty/     difficulty estimators: a free heuristic and a cheap-model judge
  delegate/       running a delegation in the background: routing, retries, escalation, checks
  verify/         executable checks that decide whether an answer really works
  providers.py    talking to models (LiteLLM: Ollama, Anthropic, OpenAI, Gemini, ...)
  cost/           ledger of tokens and money, break-even rule, GPU energy meter
  race.py         blocking vs async delegation on the same workload, with timelines
  runs.py         reading benchmark result files
  integrations/   the MCP server, for agents like Claude Code and Antigravity
  ui/             the local web dashboard
  cli.py          the `gearbox` command
Benchmarks and their task sets live outside the package, in bench/. See EXPLANATION.md.
"""

from gearbox.config import GearboxConfig, Pricing, Tier, load_config
from gearbox.cost.ledger import Ledger
from gearbox.delegate.runtime import DelegationRuntime, TaskState
from gearbox.router import Router, RoutingDecision

__all__ = [
    "DelegationRuntime",
    "GearboxConfig",
    "Ledger",
    "Pricing",
    "Router",
    "RoutingDecision",
    "TaskState",
    "Tier",
    "load_config",
]

__version__ = "0.1.0"
