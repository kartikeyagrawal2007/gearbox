"""End to end over real stdio: MCP client -> gearbox-mcp -> LiteLLM -> a local
fake OpenAI-compatible endpoint, so no model or network is needed."""

import asyncio
import json
import logging
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

COMPLETION = {
    "id": "chatcmpl-test",
    "object": "chat.completion",
    "created": 0,
    "model": "fake",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "hello from fake"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15},
}


class FakeOpenAI(BaseHTTPRequestHandler):
    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        body = json.dumps(COMPLETION).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def test_delegate_round_trip_and_readable_errors(tmp_path, caplog):
    server = HTTPServer(("127.0.0.1", 0), FakeOpenAI)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    config = tmp_path / "gearbox.yaml"
    config.write_text(
        "tiers:\n"
        "  - name: t0\n"
        "    model: openai/fake\n"
        f"    api_base: http://127.0.0.1:{server.server_port}/v1\n"
        "    params: {api_key: test-key-not-real}\n"
    )
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "gearbox.integrations.mcp_server"],
        env={"GEARBOX_CONFIG": str(config)},
    )

    async def scenario():
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                reply = await session.call_tool("delegate", {"task": "say hi"})
                task_id = json.loads(reply.content[0].text)["task_id"]
                result = await session.call_tool("await_result", {"task_id": task_id, "timeout_s": 30})
                bad = await session.call_tool("delegate", {"task": "x", "tier": "nope"})
                return json.loads(result.content[0].text), bad

    try:
        with caplog.at_level(logging.ERROR, logger="mcp.client.stdio"):
            view, bad = asyncio.run(scenario())
    finally:
        server.shutdown()

    assert not [r for r in caplog.records if "Failed to parse" in r.getMessage()]
    assert view["state"] == "done"
    assert view["result"] == "hello from fake"
    assert view["attempts"][0]["input_tokens"] == 12  # usage read from the provider response
    # A caller mistake comes back with its reason, not a generic "Error executing tool".
    assert bad.is_error
    assert "unknown tier 'nope'" in bad.content[0].text
