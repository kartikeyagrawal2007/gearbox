"""The MCP server's stdout carries JSON-RPC; LiteLLM must never write to it."""

import subprocess
import sys

PROBE = """
import sys
import gearbox.integrations.mcp_server as server   # builds MCPServer: root logger -> INFO
if sys.argv[1] == "protected":
    server.protect_stdout()
import litellm
from litellm._logging import verbose_logger
verbose_logger.info("PROBE-INFO")
"""


def run_probe(mode: str) -> str:
    done = subprocess.run([sys.executable, "-c", PROBE, mode], capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    return done.stdout


def test_litellm_logs_stay_off_stdout_when_protected():
    assert "PROBE-INFO" not in run_probe("protected")


def test_premise_litellm_logs_to_stdout_unprotected():
    # If this starts failing, LiteLLM stopped logging to stdout and protect_stdout()
    # may no longer be needed.
    assert "PROBE-INFO" in run_probe("unprotected")


def test_litellm_skips_remote_price_list_download():
    code = "import os, gearbox.providers; print(os.environ.get('LITELLM_LOCAL_MODEL_COST_MAP'))"
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60,
                          env={k: v for k, v in __import__("os").environ.items() if k != "LITELLM_LOCAL_MODEL_COST_MAP"})
    assert done.stdout.strip() == "True"
