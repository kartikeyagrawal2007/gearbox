"""The compact brief a worker receives instead of the host's full history."""

from __future__ import annotations

UNSURE_PREFIX = "UNSURE:"

WORKER_SYSTEM = (
    "You are a focused worker model. Complete only the subtask you are given, using only "
    "the provided context. Reply with the result itself: no preamble, no restating the task. "
    f"If you cannot complete it reliably, reply with a first line starting with {UNSURE_PREFIX} "
    "followed by what is missing."
)


def build_messages(task: str, context: str = "", acceptance: str = "") -> list[dict[str, str]]:
    parts = [f"## Subtask\n{task.strip()}"]
    if context.strip():
        parts.append(f"## Context\n{context.strip()}")
    if acceptance.strip():
        parts.append(f"## Done when\n{acceptance.strip()}")
    return [
        {"role": "system", "content": WORKER_SYSTEM},
        {"role": "user", "content": "\n\n".join(parts)},
    ]


def is_unsure(text: str) -> bool:
    return text.lstrip().upper().startswith(UNSURE_PREFIX)


def estimate_tokens(text: str) -> int:
    """Rough token count (~4 chars/token); only used for savings estimates."""
    return (len(text) + 3) // 4
