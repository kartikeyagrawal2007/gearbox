"""Delegation: hand a subtask to a cheaper model. runtime.py runs it in the background
(routing, retries, escalation, checks); brief.py writes the short instructions the worker sees."""

from gearbox.delegate.runtime import DelegatedTask, DelegationRuntime, TaskState

__all__ = ["DelegatedTask", "DelegationRuntime", "TaskState"]
