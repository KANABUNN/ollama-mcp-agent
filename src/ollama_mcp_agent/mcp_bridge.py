"""MCP-to-Ollama tool conversion and a small, testable MCP client facade.

The module deliberately contains no policy for filesystem access.  That policy is
enforced by the MCP server.  Its job here is to make the tool boundary strict:
only tools advertised by the connected server can be called, and arguments must
match the advertised JSON Schema before anything crosses that boundary.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterable, Mapping, Sequence
from contextlib import asynccontextmanager
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError


class MCPBridgeError(RuntimeError):
    """Base error for a local MCP connection or protocol operation."""


class MCPServerError(MCPBridgeError):
    """The MCP process/session could not complete a protocol operation."""


class MCPToolDefinitionError(MCPBridgeError, ValueError):
    """An MCP server advertised an unusable or ambiguous tool definition."""


class UnknownToolError(MCPBridgeError, ValueError):
    """A requested tool was not in the server's advertised tool list."""


class ToolArgumentError(MCPBridgeError, ValueError):
    """Tool arguments were not a JSON object matching the input schema."""


@dataclass(frozen=True, slots=True)
class RegisteredTool:
    """Normalized metadata retained for one MCP tool."""

    name: str
    description: str
    input_schema: dict[str, Any]

    def as_ollama_tool(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": deepcopy(self.input_schema),
            },
        }


def _field(value: object, *names: str, default: Any = None) -> Any:
    """Read an SDK model field or its mapping equivalent."""

    if isinstance(value, Mapping):
        for name in names:
            if name in value:
                return value[name]
        return default
    for name in names:
        if hasattr(value, name):
            return getattr(value, name)
    return default


def normalize_mcp_tool(tool: object) -> RegisteredTool:
    """Normalize an MCP SDK Tool (or equivalent mapping).

    MCP's Python models expose ``inputSchema`` while some serializers and test
    doubles use ``input_schema``.  Both representations are intentionally
    supported without depending on a private SDK attribute.
    """

    name = _field(tool, "name")
    if not isinstance(name, str) or not name.strip():
        raise MCPToolDefinitionError("MCP tool name must be a non-empty string")

    description = _field(tool, "description", default="")
    if description is None:
        description = ""
    if not isinstance(description, str):
        raise MCPToolDefinitionError(f"Description for tool {name!r} must be a string")

    schema = _field(tool, "inputSchema", "input_schema")
    if schema is None:
        schema = {"type": "object", "properties": {}}
    if not isinstance(schema, Mapping):
        raise MCPToolDefinitionError(f"Input schema for tool {name!r} must be an object")

    normalized_schema = deepcopy(dict(schema))
    try:
        Draft202012Validator.check_schema(normalized_schema)
    except SchemaError as exc:
        raise MCPToolDefinitionError(f"Input schema for tool {name!r} is invalid") from exc

    schema_type = normalized_schema.get("type")
    if schema_type not in (None, "object") and not (
        isinstance(schema_type, list) and "object" in schema_type
    ):
        raise MCPToolDefinitionError(f"Input schema for tool {name!r} must accept an object")

    return RegisteredTool(name.strip(), description, normalized_schema)


def convert_mcp_tool_to_ollama(tool: object) -> dict[str, Any]:
    """Convert one MCP tool definition to Ollama's function-tool shape."""

    return normalize_mcp_tool(tool).as_ollama_tool()


def _extract_tool_sequence(tools: object) -> Iterable[object]:
    listed_tools = _field(tools, "tools", default=None)
    if listed_tools is not None:
        tools = listed_tools
    if isinstance(tools, (str, bytes, bytearray, Mapping)) or not isinstance(tools, Iterable):
        raise MCPToolDefinitionError("MCP tool list must be an iterable of tools")
    return tools


def convert_mcp_tools_to_ollama(tools: object) -> list[dict[str, Any]]:
    """Convert a list-tools result or iterable, rejecting duplicate names."""

    converted: list[dict[str, Any]] = []
    names: set[str] = set()
    for raw_tool in _extract_tool_sequence(tools):
        normalized = normalize_mcp_tool(raw_tool)
        if normalized.name in names:
            raise MCPToolDefinitionError(
                f"MCP server advertised duplicate tool name {normalized.name!r}"
            )
        names.add(normalized.name)
        converted.append(normalized.as_ollama_tool())
    return converted


def _parse_arguments(arguments: object) -> dict[str, Any]:
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError as exc:
            raise ToolArgumentError("Tool arguments are not valid JSON") from exc
    if not isinstance(arguments, Mapping):
        raise ToolArgumentError("Tool arguments must be a JSON object")
    # Copy to detach validation/execution from a mutable SDK response object.
    return deepcopy(dict(arguments))


def _strict_root_schema(schema: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the project's deny-by-default rule to unknown root arguments.

    JSON Schema normally permits additional properties when the keyword is
    absent.  That default is too permissive at an LLM tool boundary.  An MCP
    server can still explicitly opt in with ``additionalProperties: true`` or
    provide a schema for additional properties.
    """

    strict = deepcopy(dict(schema))
    if (
        strict.get("type") == "object" or "properties" in strict
    ) and "additionalProperties" not in strict:
        strict["additionalProperties"] = False
    return strict


def _validation_location(error: object) -> str:
    path = getattr(error, "absolute_path", ())
    return "$" + "".join(f"[{part}]" if isinstance(part, int) else f".{part}" for part in path)


def validate_tool_arguments(
    arguments: object,
    input_schema: Mapping[str, Any],
) -> dict[str, Any]:
    """Parse and validate one tool call without mutating the supplied values."""

    if not isinstance(input_schema, Mapping):
        raise MCPToolDefinitionError("MCP input schema must be an object")
    parsed = _parse_arguments(arguments)
    strict_schema = _strict_root_schema(input_schema)
    try:
        Draft202012Validator.check_schema(strict_schema)
    except SchemaError as exc:
        raise MCPToolDefinitionError("MCP input schema is invalid") from exc

    error = next(iter(Draft202012Validator(strict_schema).iter_errors(parsed)), None)
    if error is not None:
        # Do not include the rejected value: tool arguments can contain file
        # contents and should not be reflected into logs or exception strings.
        keyword = getattr(error, "validator", "schema")
        location = _validation_location(error)
        raise ToolArgumentError(
            f"Tool arguments do not match the input schema at {location} ({keyword})"
        )
    return parsed


def _jsonable(value: object) -> object:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_jsonable(item) for item in value]
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _jsonable(model_dump(mode="json", by_alias=True, exclude_none=True))
    return str(value)


def format_mcp_result(result: object) -> str:
    """Serialize an MCP call result into a deterministic Ollama tool message."""

    if isinstance(result, (str, int, float, bool)) or result is None:
        payload: object = result
    else:
        content = _field(result, "content", default=None)
        structured = _field(result, "structuredContent", "structured_content", default=None)
        is_error = bool(_field(result, "isError", "is_error", default=False))
        if content is None and structured is None:
            payload = _jsonable(result)
        else:
            payload_dict: dict[str, object] = {
                "is_error": is_error,
                "content": _jsonable(content) if content is not None else [],
            }
            if structured is not None:
                payload_dict["structured_content"] = _jsonable(structured)
            payload = payload_dict
    return json.dumps(_jsonable(payload), ensure_ascii=False, separators=(",", ":"))


class MCPBridge:
    """Validate and invoke only the tools advertised by one MCP session."""

    def __init__(self, session: object, tools: Iterable[object] | None = None) -> None:
        self._session = session
        self._tools: dict[str, RegisteredTool] = {}
        if tools is not None:
            self._register_tools(tools)

    def _register_tools(self, tools: object) -> None:
        registered: dict[str, RegisteredTool] = {}
        for raw_tool in _extract_tool_sequence(tools):
            tool = normalize_mcp_tool(raw_tool)
            if tool.name in registered:
                raise MCPToolDefinitionError(
                    f"MCP server advertised duplicate tool name {tool.name!r}"
                )
            registered[tool.name] = tool
        self._tools = registered

    async def refresh_tools(self) -> tuple[RegisteredTool, ...]:
        try:
            result = await self._session.list_tools()
            self._register_tools(result)
        except MCPToolDefinitionError:
            raise
        except Exception as exc:
            raise MCPServerError("Unable to obtain the MCP tool list") from exc
        return tuple(self._tools.values())

    @property
    def registered_tools(self) -> tuple[RegisteredTool, ...]:
        return tuple(self._tools.values())

    @property
    def tool_names(self) -> tuple[str, ...]:
        return tuple(self._tools)

    @property
    def ollama_tools(self) -> list[dict[str, Any]]:
        return [tool.as_ollama_tool() for tool in self._tools.values()]

    def schema_for(self, name: str) -> dict[str, Any]:
        try:
            return deepcopy(self._tools[name].input_schema)
        except KeyError as exc:
            raise UnknownToolError(f"Tool {name!r} is not advertised by the MCP server") from exc

    def validate_call(self, name: object, arguments: object) -> dict[str, Any]:
        if not isinstance(name, str) or not name:
            raise UnknownToolError("Tool call does not contain a valid tool name")
        return validate_tool_arguments(arguments, self.schema_for(name))

    async def call_tool(self, name: object, arguments: object) -> object:
        validated = self.validate_call(name, arguments)
        assert isinstance(name, str)  # narrowed by validate_call
        try:
            return await self._session.call_tool(name, validated)
        except Exception as exc:
            raise MCPServerError(f"MCP server could not execute tool {name!r}") from exc


@asynccontextmanager
async def stdio_mcp_bridge(
    command: str,
    args: Sequence[str],
    *,
    env: Mapping[str, str] | None = None,
) -> AsyncIterator[MCPBridge]:
    """Start and reliably close a stdio MCP server using the public SDK API."""

    # Imports remain local so conversion-only tests do not need to construct
    # transport resources or depend on MCP SDK import-time behavior.
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    parameters = StdioServerParameters(
        command=command,
        args=list(args),
        env=dict(env) if env is not None else None,
    )
    consumer_failed = False
    try:
        async with (
            stdio_client(parameters) as (read_stream, write_stream),
            ClientSession(read_stream, write_stream) as session,
        ):
            await session.initialize()
            bridge = MCPBridge(session)
            await bridge.refresh_tools()
            try:
                yield bridge
            except BaseException:
                # Exceptions from the caller's ``async with`` body belong to
                # the caller.  Do not misreport them as a server crash.
                consumer_failed = True
                raise
    except MCPBridgeError:
        raise
    except Exception as exc:
        if consumer_failed:
            raise
        raise MCPServerError("The local MCP server session ended unexpectedly") from exc


# A concise alias for callers that prefer a verb at the call site.
connect_stdio_mcp = stdio_mcp_bridge
