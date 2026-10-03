"""The compact brief a worker receives instead of the host's full history."""

from __future__ import annotations

UNSURE_PREFIX = "UNSURE:"

# Small models over-use an escape hatch: with "if you cannot complete it reliably, reply
# UNSURE", qwen2.5-coder:1.5b answered UNSURE (restating the task) on trivial coding
# subtasks it solves fine without the hatch. Steer hard toward doing the work.
WORKER_SYSTEM = (
    "You are a focused worker model. Do the subtask and reply with the result only, no preamble. "
    "Almost every subtask can be done: just do it. Only if it is impossible because information "
    f"is missing, reply with one line: {UNSURE_PREFIX} <the missing information>."
)


def build_messages(task: str, context: str = "", acceptance: str = "", check: str = "") -> list[dict[str, str]]:
    parts = [f"## Subtask\n{task.strip()}"]
    if context.strip():
        parts.append(f"## Context\n{context.strip()}")
    if acceptance.strip():
        parts.append(f"## Done when\n{acceptance.strip()}")
    if check.strip():
        parts.append(f"## Your answer must pass this check\n```python\n{check.strip()}\n```")
    return [
        {"role": "system", "content": WORKER_SYSTEM},
        {"role": "user", "content": "\n\n".join(parts)},
    ]


def is_unsure(text: str) -> bool:
    return text.lstrip().upper().startswith(UNSURE_PREFIX)


def estimate_tokens(text: str) -> int:
    """Rough token count (~4 chars/token); only used for savings estimates."""
    return (len(text) + 3) // 4


def repair_messages(messages: list[dict[str, str]], failed_answer: str, check_output: str) -> list[dict[str, str]]:
    """Messages for the next tier after a failed check: the previous answer plus why it failed."""
    return [
        *messages,
        {"role": "assistant", "content": failed_answer},
        {"role": "user", "content": (
            f"That answer failed the check:\n```\n{check_output}\n```\n"
            "Fix it. Reply with the complete corrected result only."
        )},
    ]
