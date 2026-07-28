from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import patch
from uuid import UUID

from ollama_mcp_agent.audit import AuditLogger, build_audit_record, sanitize_arguments


def test_logger_creates_parent_and_writes_utf8_jsonl(tmp_path: Path) -> None:
    log_path = tmp_path / "nested" / "audit.jsonl"
    logger = AuditLogger(log_path)

    result = logger.log(
        tool_name="read_text_file",
        arguments={"relative_path": "資料/メモ.txt"},
        approval_required=False,
        approved=True,
        success=True,
        duration_ms=12,
    )

    assert result.written
    UUID(result.request_id)
    raw = log_path.read_text(encoding="utf-8")
    assert raw.endswith("\n")
    assert "資料/メモ.txt" in raw
    record = json.loads(raw)
    timestamp = datetime.fromisoformat(record["timestamp"])
    assert timestamp.utcoffset() is not None
    assert record == {
        "timestamp": record["timestamp"],
        "request_id": result.request_id,
        "tool_name": "read_text_file",
        "arguments": {"relative_path": "資料/メモ.txt"},
        "approval_required": False,
        "approved": True,
        "success": True,
        "duration_ms": 12,
        "error_type": None,
    }


def test_logger_appends_one_json_object_per_line(tmp_path: Path) -> None:
    logger = AuditLogger(tmp_path / "audit.jsonl")

    first = logger.log(
        tool_name="list_files",
        arguments={},
        approval_required=False,
        approved=True,
        success=True,
        duration_ms=1,
    )
    second = logger.log(
        tool_name="write_text_file",
        arguments={"relative_path": "new.txt", "content": "data"},
        approval_required=True,
        approved=False,
        success=False,
        duration_ms=2,
        error_type="ApprovalDenied",
    )

    records = [json.loads(line) for line in logger.path.read_text(encoding="utf-8").splitlines()]
    assert first.written and second.written
    assert len(records) == 2
    assert records[1]["approved"] is False
    assert records[1]["error_type"] == "ApprovalDenied"


def test_sensitive_arguments_and_body_are_not_stored(tmp_path: Path) -> None:
    secrets = {
        "content": "TOP SECRET FILE BODY",
        "password": "correct horse battery staple",
        "access_token": "token-value-123",
        "env": {"PRIVATE_KEY": "private-env-value"},
        "headers": {"Authorization": "Bearer credential-value"},
        "nested": {"api-key": "api-key-value"},
        "relative_path": "output/result.md",
    }
    logger = AuditLogger(tmp_path / "audit.jsonl")

    logger.log(
        tool_name="write_text_file",
        arguments=secrets,
        approval_required=True,
        approved=True,
        success=True,
        duration_ms=3.5,
    )

    raw = logger.path.read_text(encoding="utf-8")
    for plaintext in (
        "TOP SECRET FILE BODY",
        "correct horse battery staple",
        "token-value-123",
        "private-env-value",
        "credential-value",
        "api-key-value",
    ):
        assert plaintext not in raw

    arguments = json.loads(raw)["arguments"]
    assert arguments["content"]["redacted"] is True
    assert arguments["content"]["character_count"] == len("TOP SECRET FILE BODY")
    assert len(arguments["content"]["sha256"]) == 64
    assert arguments["password"] == "[REDACTED]"
    assert arguments["env"] == "[REDACTED]"
    assert arguments["relative_path"] == "output/result.md"


def test_inline_secrets_and_dotenv_values_are_redacted() -> None:
    sanitized = sanitize_arguments(
        {
            "query": "token=visible-no-more",
            "diagnostic": "PUBLIC=one\nPRIVATE_KEY=two",
        }
    )

    encoded = json.dumps(sanitized)
    assert "visible-no-more" not in encoded
    assert "one" not in encoded
    assert "two" not in encoded


def test_complete_bearer_credentials_are_redacted() -> None:
    sanitized = sanitize_arguments(
        {
            "relative_path": "Authorization: Bearer TOPSECRET",
            "note": "X-Auth: Bearer SECONDSECRET",
            "other": "prefix Bearer THIRDSECRET suffix",
        }
    )

    encoded = json.dumps(sanitized)
    assert "TOPSECRET" not in encoded
    assert "SECONDSECRET" not in encoded
    assert "THIRDSECRET" not in encoded
    assert encoded.count("[REDACTED]") == 3


def test_exception_message_is_never_saved(tmp_path: Path) -> None:
    logger = AuditLogger(tmp_path / "audit.jsonl")

    logger.log(
        tool_name="read_text_file",
        arguments={},
        approval_required=False,
        approved=True,
        success=False,
        duration_ms=1,
        error_type=RuntimeError("password=hunter2"),
    )

    raw = logger.path.read_text(encoding="utf-8")
    assert "hunter2" not in raw
    assert json.loads(raw)["error_type"] == "RuntimeError"


def test_write_failure_is_reported_after_one_attempt(tmp_path: Path) -> None:
    logger = AuditLogger(tmp_path / "audit.jsonl")

    with patch.object(Path, "open", side_effect=OSError("disk unavailable")) as mocked_open:
        result = logger.log(
            tool_name="list_files",
            arguments={},
            approval_required=False,
            approved=True,
            success=True,
            duration_ms=0,
        )

    assert not result.written
    assert result.error == "OSError"
    UUID(result.request_id)
    assert mocked_open.call_count == 1


def test_invalid_request_id_is_replaced_with_uuid() -> None:
    record = build_audit_record(
        tool_name="list_files",
        arguments={},
        approval_required=False,
        approved=True,
        success=True,
        duration_ms=-1,
        request_id="not-a-uuid",
    )

    UUID(record.request_id)
    assert record.duration_ms == 0
