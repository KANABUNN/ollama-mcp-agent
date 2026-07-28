from __future__ import annotations

import json
from collections.abc import Mapping
from contextlib import asynccontextmanager
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import ollama_mcp_agent.agent as agent_module
import ollama_mcp_agent.config as config_module
from ollama_mcp_agent.agent import (
    AgentConfigurationError,
    MCPUnavailableError,
    OllamaMCPAgent,
    ToolIterationLimitError,
    _mcp_server_environment,
    build_system_prompt,
    create_ollama_client,
    interactive_cli,
    make_safe_exists_checker,
    run_configured_agent,
    validate_loopback_ollama_host,
)
from ollama_mcp_agent.config import Config
from ollama_mcp_agent.mcp_bridge import MCPBridge

TOOLS = [
    {
        "name": "read_text_file",
        "description": "Read UTF-8 text",
        "inputSchema": {
            "type": "object",
            "properties": {"relative_path": {"type": "string"}},
            "required": ["relative_path"],
        },
    },
    {
        "name": "write_text_file",
        "description": "Write UTF-8 text",
        "inputSchema": {
            "type": "object",
            "properties": {
                "relative_path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["relative_path", "content"],
        },
    },
]


def tool_response(name: str, arguments: object) -> dict[str, object]:
    return {
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": name, "arguments": arguments}}],
        }
    }


def multiple_tool_response(*calls: tuple[str, object]) -> dict[str, object]:
    return {
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"function": {"name": name, "arguments": arguments}}
                for name, arguments in calls
            ],
        }
    }


def text_response(content: str = "done") -> dict[str, object]:
    return {"message": {"role": "assistant", "content": content}}


class OllamaDouble:
    def __init__(self, responses: list[object]) -> None:
        self.responses = responses
        self.calls: list[dict[str, object]] = []

    async def chat(self, **kwargs: object) -> object:
        self.calls.append(deepcopy(kwargs))
        if not self.responses:
            raise AssertionError("Unexpected Ollama request")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class SessionDouble:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.error: Exception | None = None

    async def call_tool(self, name: str, arguments: dict[str, object]) -> object:
        self.calls.append((name, arguments))
        if self.error is not None:
            raise self.error
        return {"content": [{"type": "text", "text": "tool result"}], "isError": False}


class AuditDouble:
    def __init__(self) -> None:
        self.records: list[dict[str, object]] = []

    def log(self, **record: object) -> None:
        self.records.append(record)


def make_agent(
    responses: list[object],
    *,
    approval: object = None,
    max_tools: int = 10,
    session: SessionDouble | None = None,
    audit: AuditDouble | None = None,
    exists_checker: object = None,
    audit_warning: object = None,
) -> tuple[OllamaMCPAgent, OllamaDouble, SessionDouble, AuditDouble]:
    ollama = OllamaDouble(responses)
    session = session or SessionDouble()
    audit = audit or AuditDouble()
    bridge = MCPBridge(session, TOOLS)
    callback = approval if callable(approval) else None
    agent = OllamaMCPAgent(
        ollama_client=ollama,
        bridge=bridge,
        model="tool-model",
        max_tool_iterations=max_tools,
        approval_callback=callback,
        audit_logger=audit,
        write_target_exists=(
            exists_checker if callable(exists_checker) else lambda _path: True
        ),
        audit_warning_callback=(audit_warning if callable(audit_warning) else None),
        public_config={"ollama_model": "tool-model"},
    )
    return agent, ollama, session, audit


@pytest.mark.asyncio
async def test_normal_answer_finishes_without_tool_call() -> None:
    agent, ollama, session, _audit = make_agent([text_response("hello")])

    assert await agent.run_turn("hi") == "hello"
    assert session.calls == []
    assert ollama.calls[0]["tools"] == agent.bridge.ollama_tools


@pytest.mark.asyncio
async def test_read_tool_runs_once_and_result_is_returned_to_ollama() -> None:
    agent, ollama, session, audit = make_agent(
        [
            tool_response("read_text_file", {"relative_path": "notes.txt"}),
            text_response("summary"),
        ]
    )

    assert await agent.run_turn("summarize notes") == "summary"
    assert session.calls == [("read_text_file", {"relative_path": "notes.txt"})]
    second_messages = ollama.calls[1]["messages"]
    assert isinstance(second_messages, list)
    assert second_messages[-1]["role"] == "tool"
    assert second_messages[-1]["tool_name"] == "read_text_file"
    assert audit.records[-1]["success"] is True
    assert audit.records[-1]["approval_required"] is False


@pytest.mark.asyncio
async def test_approved_write_runs_and_is_audited() -> None:
    approval_calls: list[tuple[str, str, bool]] = []

    def approve(path: str, content: str, *, exists: bool) -> bool:
        approval_calls.append((path, content, exists))
        return True

    agent, _ollama, session, audit = make_agent(
        [
            tool_response(
                "write_text_file",
                {"relative_path": "output/result.md", "content": "# Result\n"},
            ),
            text_response(),
        ],
        approval=approve,
    )

    await agent.run_turn("write result")

    assert approval_calls == [("output/result.md", "# Result\n", True)]
    assert session.calls[0][0] == "write_text_file"
    assert audit.records[0]["approved"] is True
    assert audit.records[0]["approval_required"] is True


@pytest.mark.asyncio
async def test_denied_write_returns_denial_to_ollama_without_execution() -> None:
    def deny(_path: str, _content: str, *, exists: bool) -> bool:
        assert exists is True
        return False

    agent, ollama, session, audit = make_agent(
        [
            tool_response(
                "write_text_file",
                {"relative_path": "output/result.md", "content": "secret body"},
            ),
            text_response("not written"),
        ],
        approval=deny,
    )

    assert await agent.run_turn("write") == "not written"
    assert session.calls == []
    denial = ollama.calls[1]["messages"][-1]["content"]
    assert "ApprovalDenied" in denial
    assert audit.records[0]["approved"] is False
    assert audit.records[0]["success"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize("truthy_non_bool", ["yes", "no", 1, object()])
async def test_write_approval_requires_literal_true(truthy_non_bool: object) -> None:
    def ambiguous(_path: str, _content: str, *, exists: bool) -> object:
        assert exists is True
        return truthy_non_bool

    agent, _ollama, session, audit = make_agent(
        [
            tool_response(
                "write_text_file",
                {"relative_path": "output/result.md", "content": "body"},
            ),
            text_response("not written"),
        ],
        approval=ambiguous,
    )

    await agent.run_turn("write")

    assert session.calls == []
    assert audit.records[0]["error_type"] == "ApprovalDenied"


@pytest.mark.asyncio
async def test_unknown_tool_is_rejected_and_reported_to_ollama() -> None:
    agent, ollama, session, audit = make_agent(
        [tool_response("run_command", {"command": "whoami"}), text_response("cannot")]
    )

    assert await agent.run_turn("do it") == "cannot"
    assert session.calls == []
    error_content = ollama.calls[1]["messages"][-1]["content"]
    assert "UnknownToolError" in error_content
    assert audit.records[0]["error_type"] == "UnknownToolError"


@pytest.mark.asyncio
async def test_invalid_arguments_are_rejected_before_approval_or_execution() -> None:
    approvals = 0

    def approve(*_args: object, **_kwargs: object) -> bool:
        nonlocal approvals
        approvals += 1
        return True

    agent, ollama, session, audit = make_agent(
        [
            tool_response(
                "write_text_file",
                {"relative_path": "result.md", "content": "x", "approved": True},
            ),
            text_response("invalid"),
        ],
        approval=approve,
    )

    assert await agent.run_turn("write") == "invalid"
    assert approvals == 0
    assert session.calls == []
    assert "ToolArgumentError" in ollama.calls[1]["messages"][-1]["content"]
    assert audit.records[0]["approved"] is False


@pytest.mark.asyncio
async def test_malformed_tool_call_is_safely_represented_in_next_ollama_request() -> None:
    agent, ollama, session, _audit = make_agent(
        [tool_response("", ["not", "an", "object"]), text_response("invalid")]
    )

    assert await agent.run_turn("try malformed") == "invalid"
    assert session.calls == []
    assistant_call = ollama.calls[1]["messages"][-2]["tool_calls"][0]["function"]
    tool_result = ollama.calls[1]["messages"][-1]
    assert assistant_call == {"name": "invalid_tool_call", "arguments": {}}
    assert tool_result["tool_name"] == "invalid_tool_call"


@pytest.mark.asyncio
async def test_maximum_tool_calls_stops_repeated_requests() -> None:
    agent, _ollama, session, _audit = make_agent(
        [
            tool_response("read_text_file", {"relative_path": "one.txt"}),
            tool_response("read_text_file", {"relative_path": "two.txt"}),
        ],
        max_tools=1,
    )

    with pytest.raises(ToolIterationLimitError, match="Maximum"):
        await agent.run_turn("keep reading")
    assert session.calls == [("read_text_file", {"relative_path": "one.txt"})]


@pytest.mark.asyncio
async def test_mcp_server_failure_is_clear_and_audited() -> None:
    session = SessionDouble()
    session.error = ConnectionError("stdio closed unexpectedly")
    agent, _ollama, _session, audit = make_agent(
        [tool_response("read_text_file", {"relative_path": "notes.txt"})],
        session=session,
    )

    with pytest.raises(MCPUnavailableError, match="local MCP server"):
        await agent.run_turn("read")
    assert audit.records[0]["success"] is False
    assert audit.records[0]["error_type"] == "MCPServerError"


@pytest.mark.asyncio
async def test_mcp_batch_failure_audits_and_completes_remaining_calls() -> None:
    session = SessionDouble()
    session.error = ConnectionError("stdio closed unexpectedly")
    agent, _ollama, _session, audit = make_agent(
        [
            multiple_tool_response(
                ("read_text_file", {"relative_path": "one.txt"}),
                ("read_text_file", {"relative_path": "two.txt"}),
            )
        ],
        session=session,
    )

    with pytest.raises(MCPUnavailableError):
        await agent.run_turn("read both")

    assert session.calls == [("read_text_file", {"relative_path": "one.txt"})]
    assert [record["error_type"] for record in audit.records] == [
        "MCPServerError",
        "NotExecutedAfterMCPFailure",
    ]
    assert [message["role"] for message in agent.messages[-2:]] == ["tool", "tool"]


@pytest.mark.asyncio
async def test_write_is_denied_if_target_state_changes_during_approval() -> None:
    observed_states = iter([False, True])
    agent, ollama, session, audit = make_agent(
        [
            tool_response(
                "write_text_file",
                {"relative_path": "result.md", "content": "body"},
            ),
            text_response("not written"),
        ],
        approval=lambda *_args, **_kwargs: True,
        exists_checker=lambda _path: next(observed_states),
    )

    assert await agent.run_turn("write") == "not written"
    assert session.calls == []
    assert audit.records[0]["error_type"] == "ApprovalTargetChanged"
    assert "ApprovalTargetChanged" in ollama.calls[1]["messages"][-1]["content"]


@pytest.mark.asyncio
async def test_audit_write_failure_warns_without_retrying_tool() -> None:
    class FailingAudit:
        def __init__(self) -> None:
            self.calls = 0

        def log(self, **_record: object) -> object:
            self.calls += 1
            return SimpleNamespace(written=False)

    failing_audit = FailingAudit()
    warnings: list[str] = []
    agent, _ollama, session, _audit = make_agent(
        [
            tool_response("read_text_file", {"relative_path": "notes.txt"}),
            text_response("done"),
        ],
        audit=failing_audit,  # type: ignore[arg-type]
        audit_warning=warnings.append,
    )

    assert await agent.run_turn("read") == "done"
    assert len(session.calls) == 1
    assert failing_audit.calls == 1
    assert len(warnings) == 1


def test_agent_fails_closed_when_no_write_approval_callback() -> None:
    agent, _ollama, _session, _audit = make_agent([])
    assert agent.approval_callback is None


@pytest.mark.parametrize(
    "host",
    ["http://127.0.0.1:11434", "http://localhost:11434", "http://[::1]:11434"],
)
def test_loopback_ollama_hosts_are_allowed(host: str) -> None:
    validate_loopback_ollama_host(host)


@pytest.mark.parametrize(
    "host",
    [
        "http://example.com:11434",
        "http://192.168.1.10:11434",
        "ftp://127.0.0.1:11434",
        "not-a-url",
    ],
)
def test_external_ollama_hosts_are_rejected(host: str) -> None:
    with pytest.raises(AgentConfigurationError):
        validate_loopback_ollama_host(host)


def test_ollama_client_disables_redirects_and_environment_proxies(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_client(**kwargs: object) -> object:
        captured.update(kwargs)
        return object()

    monkeypatch.setattr("ollama.AsyncClient", fake_client)

    create_ollama_client("http://127.0.0.1:11434")

    assert captured == {
        "host": "http://127.0.0.1:11434",
        "follow_redirects": False,
        "trust_env": False,
    }


def test_mcp_child_environment_contains_only_explicit_runtime_settings(tmp_path: Path) -> None:
    config = SimpleNamespace(
        ollama_host="http://127.0.0.1:11434",
        ollama_model="tool-model",
        allowed_root=tmp_path,
        audit_log_path=tmp_path / "audit.jsonl",
        max_read_bytes=10,
        max_write_bytes=20,
        max_tool_iterations=3,
        secret_token="must-not-be-copied",
    )

    child_env = _mcp_server_environment(config)

    assert child_env["ALLOWED_ROOT"] == str(tmp_path)
    assert child_env["MAX_READ_BYTES"] == "10"
    assert "secret_token" not in child_env
    assert "must-not-be-copied" not in child_env.values()


@pytest.mark.asyncio
async def test_configured_agent_closes_ollama_and_mcp_contexts(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config = Config(
        ollama_model="tool-model",
        allowed_root=tmp_path,
        audit_log_path=tmp_path / "audit.jsonl",
    )
    state: dict[str, object] = {}

    class ClientContext:
        async def __aenter__(self) -> ClientContext:
            state["client_entered"] = True
            return self

        async def __aexit__(self, *_args: object) -> None:
            state["client_closed"] = True

    @asynccontextmanager
    async def fake_stdio(_command: str, _args: object, *, env: object = None):
        state["child_env"] = env
        try:
            yield MCPBridge(SessionDouble(), TOOLS)
        finally:
            state["mcp_closed"] = True

    async def fake_cli(_agent: object) -> None:
        state["cli_ran"] = True

    monkeypatch.setattr(config_module, "load_config", lambda: config)
    monkeypatch.setattr(agent_module, "create_ollama_client", lambda _host: ClientContext())
    monkeypatch.setattr(agent_module, "stdio_mcp_bridge", fake_stdio)
    monkeypatch.setattr(agent_module, "interactive_cli", fake_cli)

    await run_configured_agent()

    assert state["client_entered"] is True
    assert state["client_closed"] is True
    assert state["mcp_closed"] is True
    assert state["cli_ran"] is True
    assert state["child_env"] == _mcp_server_environment(config)


def test_safe_exists_checker_never_checks_outside_root(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "inside.txt").write_text("ok", encoding="utf-8")
    checker = make_safe_exists_checker(root)

    assert checker("inside.txt") is True
    assert checker("../outside.txt") is False
    assert checker(str((tmp_path / "outside.txt").resolve())) is False


@pytest.mark.asyncio
async def test_cli_commands_do_not_call_ollama_or_disclose_unknown_config() -> None:
    agent, ollama, _session, _audit = make_agent([])
    answers = iter(["/help", "/tools", "/config", "/exit"])
    output: list[str] = []

    await interactive_cli(agent, input_fn=lambda _prompt: next(answers), output_fn=output.append)

    assert ollama.calls == []
    assert any("/exit" in line for line in output)
    assert any("read_text_file" in line for line in output)
    assert output[-1] == "ollama_model: tool-model"


@pytest.mark.asyncio
async def test_cli_escapes_terminal_controls_in_model_output() -> None:
    agent, _ollama, _session, _audit = make_agent([text_response("safe\x1b[31m\x08text")])
    answers = iter(["answer", "/exit"])
    output: list[str] = []

    await interactive_cli(agent, input_fn=lambda _prompt: next(answers), output_fn=output.append)

    assert output == [r"assistant: safe\x1b[31m\x08text"]


def test_system_prompt_treats_file_contents_as_untrusted() -> None:
    agent, _ollama, _session, _audit = make_agent([])
    system_message: Mapping[str, Any] = agent.messages[0]
    assert "untrusted data" in str(system_message["content"])


def test_system_prompt_identifies_allowed_root_and_relative_translation(tmp_path: Path) -> None:
    prompt = build_system_prompt(tmp_path)

    assert json.dumps(str(tmp_path), ensure_ascii=False) in prompt
    assert "use '.' as the tool path" in prompt
