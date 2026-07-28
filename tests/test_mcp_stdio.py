"""Real stdio protocol smoke test for the packaged MCP server."""

from __future__ import annotations

import sys

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


@pytest.mark.asyncio
async def test_stdio_server_lists_and_calls_tools(tmp_path) -> None:
    sample = tmp_path / "sample.txt"
    sample.write_text("MCP connection test", encoding="utf-8", newline="\n")

    child_env = {
        "ALLOWED_ROOT": str(tmp_path),
        "AUDIT_LOG_PATH": str(tmp_path / "audit.jsonl"),
        "MAX_READ_BYTES": "1024",
        "MAX_WRITE_BYTES": "1024",
        "MAX_TOOL_ITERATIONS": "3",
        "OLLAMA_HOST": "http://127.0.0.1:11434",
        "OLLAMA_MODEL": "test-only",
    }
    parameters = StdioServerParameters(
        command=sys.executable,
        args=["-I", "-m", "ollama_mcp_agent.pc_tools_server"],
        env=child_env,
    )

    async with stdio_client(parameters) as (read_stream, write_stream), ClientSession(
        read_stream, write_stream
    ) as session:
        await session.initialize()
        listed = await session.list_tools()
        assert {tool.name for tool in listed.tools} == {
            "list_files",
            "read_text_file",
            "write_text_file",
        }

        result = await session.call_tool(
            "read_text_file",
            arguments={"relative_path": "sample.txt"},
        )

    assert result.isError is False
    assert result.structuredContent is not None
    assert result.structuredContent["content"] == "MCP connection test"
    assert result.structuredContent["byte_count"] == len(b"MCP connection test")
