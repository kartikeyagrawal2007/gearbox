import asyncio

from fakes import FakeProvider, make_config

from gearbox.delegate.runtime import DelegationRuntime
from gearbox.integrations import mcp_server


def test_tools_registered():
    tools = asyncio.run(mcp_server.mcp.list_tools())
    assert {t.name for t in tools} == {"route", "delegate", "await_result", "status", "cancel", "ledger"}


def test_delegate_await_ledger_flow(monkeypatch):
    async def scenario():
        rt = DelegationRuntime(make_config(), provider=FakeProvider(delay=0.05))
        monkeypatch.setattr(mcp_server, "_runtime", rt)
        reply = await mcp_server.delegate("Write docstrings for utils.py", expected_output_tokens=800)
        listing = await mcp_server.status()
        result = await mcp_server.await_result(reply["task_id"], timeout_s=5)
        ledger = await mcp_server.ledger()
        return reply, listing, result, ledger

    reply, listing, result, ledger = asyncio.run(scenario())
    assert reply["state"] in ("routing", "running")
    assert reply["break_even"]["delegate"] is True
    assert len(listing["tasks"]) == 1
    assert result["state"] == "done"
    assert ledger["delegated_tasks_succeeded"] == 1


def test_delegate_with_check_reports_verified(monkeypatch):
    good = "```python\ndef double(x):\n    return 2 * x\n```"

    async def scenario():
        rt = DelegationRuntime(make_config(code_checks=True), provider=FakeProvider({"t0": good}))
        monkeypatch.setattr(mcp_server, "_runtime", rt)
        reply = await mcp_server.delegate("Write double(x)", tier="t0", check="assert double(4) == 8")
        return await mcp_server.await_result(reply["task_id"], timeout_s=10)

    view = asyncio.run(scenario())
    assert view["state"] == "done" and view["verified"] is True
