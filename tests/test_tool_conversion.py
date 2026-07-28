from __future__ import annotations

import asyncio
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from ollama_mcp_agent.mcp_bridge import (
    MCPBridge,
    MCPServerError,
    MCPToolDefinitionError,
    ToolArgumentError,
    UnknownToolError,
    convert_mcp_tool_to_ollama,
    convert_mcp_tools_to_ollama,
    format_mcp_result,
    stdio_mcp_bridge,
    validate_tool_arguments,
)

READ_SCHEMA = {
    "type": "object",
    "properties": {"relative_path": {"type": "string", "minLength": 1}},
    "required": ["relative_path"],
}


@dataclass
class SDKToolDouble:
    name: str
    description: str
    inputSchema: dict[str, object]


def test_convert_sdk_tool_to_ollama_preserves_public_fields() -> None:
    tool = SDKToolDouble("read_text_file", "Read UTF-8 text", READ_SCHEMA)

    converted = convert_mcp_tool_to_ollama(tool)

    assert converted == {
        "type": "function",
        "function": {
            "name": "read_text_file",
            "description": "Read UTF-8 text",
            "parameters": READ_SCHEMA,
        },
    }
    assert converted["function"]["parameters"] is not READ_SCHEMA


def test_convert_mapping_accepts_snake_case_schema() -> None:
    converted = convert_mcp_tool_to_ollama(
        {"name": "list_files", "description": None, "input_schema": READ_SCHEMA}
    )

    assert converted["function"]["description"] == ""
    assert converted["function"]["parameters"] == READ_SCHEMA


def test_convert_list_tools_result_and_reject_duplicates() -> None:
    result = SimpleNamespace(
        tools=[
            SDKToolDouble("read_text_file", "Read", READ_SCHEMA),
            SDKToolDouble("list_files", "List", READ_SCHEMA),
        ]
    )
    assert [item["function"]["name"] for item in convert_mcp_tools_to_ollama(result)] == [
        "read_text_file",
        "list_files",
    ]

    with pytest.raises(MCPToolDefinitionError, match="duplicate"):
        convert_mcp_tools_to_ollama([result.tools[0], result.tools[0]])


@pytest.mark.parametrize(
    "tool",
    [
        {},
        {"name": "", "inputSchema": READ_SCHEMA},
        {"name": "bad", "inputSchema": []},
        {"name": "bad", "inputSchema": {"type": "string"}},
        {"name": "bad", "inputSchema": {"type": "not-a-json-schema-type"}},
    ],
)
def test_invalid_tool_definitions_are_rejected(tool: object) -> None:
    with pytest.raises(MCPToolDefinitionError):
        convert_mcp_tool_to_ollama(tool)


def test_validate_arguments_accepts_mapping_and_json_object() -> None:
    original = {"relative_path": "notes.txt"}
    validated = validate_tool_arguments(original, READ_SCHEMA)
    assert validated == original
    assert validated is not original
    assert validate_tool_arguments('{"relative_path":"notes.txt"}', READ_SCHEMA) == original


@pytest.mark.parametrize(
    "arguments",
    [
        None,
        [],
        "not JSON",
        "[]",
        {},
        {"relative_path": 7},
        {"relative_path": "notes.txt", "surprise": True},
    ],
)
def test_invalid_or_unknown_arguments_are_rejected(arguments: object) -> None:
    with pytest.raises(ToolArgumentError):
        validate_tool_arguments(arguments, READ_SCHEMA)


def test_schema_must_explicitly_allow_unknown_arguments() -> None:
    permissive = {**READ_SCHEMA, "additionalProperties": True}
    assert validate_tool_arguments({"relative_path": "notes.txt", "extra": 1}, permissive) == {
        "relative_path": "notes.txt",
        "extra": 1,
    }


def test_validation_error_does_not_reflect_sensitive_value() -> None:
    with pytest.raises(ToolArgumentError) as caught:
        validate_tool_arguments({"relative_path": 123456789}, READ_SCHEMA)
    assert "123456789" not in str(caught.value)


class SessionDouble:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.call_error: Exception | None = None

    async def call_tool(self, name: str, arguments: dict[str, object]) -> object:
        self.calls.append((name, arguments))
        if self.call_error is not None:
            raise self.call_error
        return {"content": [{"type": "text", "text": "ok"}], "isError": False}


@pytest.mark.asyncio
async def test_bridge_checks_registry_and_arguments_before_calling_session() -> None:
    session = SessionDouble()
    bridge = MCPBridge(session, [SDKToolDouble("read_text_file", "Read", READ_SCHEMA)])

    await bridge.call_tool("read_text_file", {"relative_path": "notes.txt"})
    assert session.calls == [("read_text_file", {"relative_path": "notes.txt"})]

    with pytest.raises(UnknownToolError):
        await bridge.call_tool("run_command", {})
    with pytest.raises(ToolArgumentError):
        await bridge.call_tool("read_text_file", {"relative_path": "notes.txt", "extra": 1})
    assert len(session.calls) == 1


@pytest.mark.asyncio
async def test_bridge_wraps_session_failure() -> None:
    session = SessionDouble()
    session.call_error = ConnectionError("stdio closed")
    bridge = MCPBridge(session, [SDKToolDouble("read_text_file", "Read", READ_SCHEMA)])

    with pytest.raises(MCPServerError, match="could not execute"):
        await bridge.call_tool("read_text_file", {"relative_path": "notes.txt"})


@pytest.mark.asyncio
async def test_bridge_does_not_wrap_asyncio_cancellation() -> None:
    class CancelledSession(SessionDouble):
        async def call_tool(self, name: str, arguments: dict[str, object]) -> object:
            raise asyncio.CancelledError

    bridge = MCPBridge(
        CancelledSession(),
        [SDKToolDouble("read_text_file", "Read", READ_SCHEMA)],
    )

    with pytest.raises(asyncio.CancelledError):
        await bridge.call_tool("read_text_file", {"relative_path": "notes.txt"})


def test_format_mcp_result_serializes_sdk_style_result() -> None:
    result = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="こんにちは")],
        structuredContent={"characters": 5},
        isError=False,
    )

    rendered = format_mcp_result(result)

    assert '"is_error":false' in rendered
    assert '"structured_content":{"characters":5}' in rendered
    assert "こんにちは" in rendered


@pytest.mark.asyncio
async def test_stdio_bridge_lists_real_server_tools(tmp_path: Path) -> None:
    """Smoke-test the installed stable MCP SDK without touching the real workspace."""

    server_env = dict(os.environ)
    server_env.update(
        {
            "ALLOWED_ROOT": str(tmp_path),
            "AUDIT_LOG_PATH": str(tmp_path / "audit.jsonl"),
            "MAX_READ_BYTES": "1024",
            "MAX_WRITE_BYTES": "1024",
            "MAX_TOOL_ITERATIONS": "2",
        }
    )

    async with stdio_mcp_bridge(
        sys.executable,
        ["-m", "ollama_mcp_agent.pc_tools_server"],
        env=server_env,
    ) as bridge:
        assert set(bridge.tool_names) == {
            "list_files",
            "read_text_file",
            "write_text_file",
        }
        result = await bridge.call_tool("list_files", {"relative_path": "."})
        assert getattr(result, "isError", False) is False
