"""Exercise the SDK's schema boundary through the project's public stdio bridge."""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

import jsonschema.validators
import pytest

from ollama_mcp_agent.mcp_bridge import MCPServerError, stdio_mcp_bridge

SERVER = Path(__file__).parent / "fixtures" / "schema_server.py"


@pytest.mark.asyncio
@pytest.mark.parametrize("scheme", ["http", "https", "file"])
async def test_output_schema_cannot_fetch_external_references(monkeypatch, tmp_path, scheme):
    attempted = []

    def forbid_open(*args, **kwargs):
        attempted.append(args)
        raise AssertionError("Output schema attempted external I/O")

    monkeypatch.setattr(urllib.request, "urlopen", forbid_open)
    # jsonschema also keeps an imported alias on versions supported by this project.
    monkeypatch.setattr(jsonschema.validators, "urlopen", forbid_open, raising=False)
    reference = (
        (tmp_path / "nonexistent-schema.json").as_uri()
        if scheme == "file"
        else f"{scheme}://schema.invalid/output.json"
    )
    async with stdio_mcp_bridge(sys.executable, ["-I", str(SERVER), reference]) as bridge:
        with pytest.raises(MCPServerError) as error:
            await bridge.call_tool("sample", {})
        assert attempted == []
        assert isinstance(error.value.__cause__, RuntimeError)
        assert "Invalid schema for tool" in str(error.value.__cause__)
    assert attempted == []


@pytest.mark.asyncio
async def test_output_schema_can_resolve_in_document_reference():
    async with stdio_mcp_bridge(
        sys.executable, ["-I", str(SERVER), "#/$defs/value"]
    ) as bridge:
        result = await bridge.call_tool("sample", {})
    assert result.structuredContent == {"value": "synthetic"}
