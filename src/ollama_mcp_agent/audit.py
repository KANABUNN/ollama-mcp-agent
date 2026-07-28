"""Best-effort, privacy-preserving JSON Lines audit logging."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Any
from uuid import UUID, uuid4

_CONTENT_KEYS = {
    "body",
    "content",
    "file_body",
    "file_content",
    "message",
    "prompt",
    "text",
}
_SECRET_KEY_PARTS = (
    "api_key",
    "apikey",
    "authorization",
    "cookie",
    "credential",
    "passwd",
    "password",
    "secret",
    "token",
)
_ENV_KEYS = {
    "env",
    "environ",
    "environment",
    "environment_variables",
}
_INLINE_SECRET_RE = re.compile(
    r"(?i)\b([A-Za-z0-9_.-]*(?:password|passwd|secret|token|api[_-]?key|authorization|x-auth)"
    r"[A-Za-z0-9_.-]*)(\s*[:=]\s*)([^\r\n,;]*)"
)
_BEARER_TOKEN_RE = re.compile(r"(?i)\bbearer\s+[^\s,;]+")
_ENV_ASSIGNMENT_RE = re.compile(r"(?m)^([A-Za-z_][A-Za-z0-9_]*)(\s*=\s*)(.*)$")
_MAX_DEPTH = 12
_MAX_ITEMS = 1_000
_MAX_STRING_CHARS = 4_096


def new_request_id() -> str:
    """Return a new canonical UUID string for one tool request."""

    return str(uuid4())


def _canonical_request_id(value: str | UUID | None) -> str:
    if value is None:
        return new_request_id()
    try:
        return str(UUID(str(value)))
    except (ValueError, AttributeError, TypeError):
        # Audit records must always contain a UUID.  A bad caller-provided ID is
        # replaced instead of making logging itself disrupt the agent.
        return new_request_id()


def _key_is_content(key: str) -> bool:
    lowered = key.casefold().replace("-", "_")
    return lowered in _CONTENT_KEYS or lowered.endswith(("_body", "_content"))


def _key_is_secret(key: str) -> bool:
    lowered = key.casefold().replace("-", "_")
    return any(part in lowered for part in _SECRET_KEY_PARTS)


def _key_is_environment(key: str) -> bool:
    lowered = key.casefold().replace("-", "_")
    return (
        lowered in _ENV_KEYS
        or lowered.startswith("env_")
        or lowered.endswith("_env")
        or re.fullmatch(r"[A-Z_][A-Z0-9_]*", key) is not None
    )


def _content_summary(value: Any) -> dict[str, Any]:
    """Describe content without retaining its plaintext."""

    if isinstance(value, str):
        encoded = value.encode("utf-8")
        return {
            "redacted": True,
            "character_count": len(value),
            "sha256": hashlib.sha256(encoded).hexdigest(),
        }
    if isinstance(value, (bytes, bytearray, memoryview)):
        encoded = bytes(value)
        return {
            "redacted": True,
            "byte_count": len(encoded),
            "sha256": hashlib.sha256(encoded).hexdigest(),
        }
    return {"redacted": True, "value_type": type(value).__name__}


def _redact_inline_secrets(value: str) -> str:
    """Redact common key/value secrets and dotenv-style values in strings."""

    redacted = _INLINE_SECRET_RE.sub(r"\1\2[REDACTED]", value)
    redacted = _BEARER_TOKEN_RE.sub("Bearer [REDACTED]", redacted)
    redacted = _ENV_ASSIGNMENT_RE.sub(r"\1\2[REDACTED_ENV_VALUE]", redacted)
    if len(redacted) > _MAX_STRING_CHARS:
        omitted = len(redacted) - _MAX_STRING_CHARS
        redacted = f"{redacted[:_MAX_STRING_CHARS]}... [{omitted} characters omitted]"
    return redacted


def _sanitize_value(value: Any, *, key: str | None, depth: int, seen: set[int]) -> Any:
    if key is not None:
        if _key_is_secret(key) or _key_is_environment(key):
            return "[REDACTED]"
        if _key_is_content(key):
            return _content_summary(value)

    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else "[NON_FINITE_NUMBER]"
    if isinstance(value, str):
        return _redact_inline_secrets(value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {"redacted": True, "byte_count": len(value)}
    if isinstance(value, Path):
        return _redact_inline_secrets(str(value))
    if depth >= _MAX_DEPTH:
        return "[MAX_DEPTH_REACHED]"

    track_identity = isinstance(value, (Mapping, Sequence, set, frozenset))
    identity = id(value)
    if track_identity:
        if identity in seen:
            return "[CYCLE]"
        seen.add(identity)

    try:
        if isinstance(value, Mapping):
            sanitized: dict[str, Any] = {}
            for index, (item_key, item_value) in enumerate(value.items()):
                if index >= _MAX_ITEMS:
                    sanitized["[TRUNCATED]"] = f"{len(value) - _MAX_ITEMS} items omitted"
                    break
                safe_key = item_key if isinstance(item_key, str) else f"[NON_STRING_KEY_{index}]"
                sanitized[safe_key] = _sanitize_value(
                    item_value,
                    key=safe_key,
                    depth=depth + 1,
                    seen=seen,
                )
            return sanitized

        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            sanitized_items = [
                _sanitize_value(item, key=None, depth=depth + 1, seen=seen)
                for item in value[:_MAX_ITEMS]
            ]
            if len(value) > _MAX_ITEMS:
                sanitized_items.append(f"[TRUNCATED: {len(value) - _MAX_ITEMS} items omitted]")
            return sanitized_items

        if isinstance(value, (set, frozenset)):
            # Sorting the already-sanitized representation keeps records stable
            # without invoking potentially sensitive object reprs.
            items = list(value)
            sanitized_items = [
                _sanitize_value(item, key=None, depth=depth + 1, seen=seen)
                for item in items[:_MAX_ITEMS]
            ]
            return sorted(sanitized_items, key=lambda item: json.dumps(item, sort_keys=True))

        # Avoid repr(value): arbitrary objects may include credentials there.
        return f"[UNSUPPORTED_TYPE:{type(value).__name__}]"
    finally:
        if track_identity:
            seen.discard(identity)


def sanitize_arguments(arguments: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return a JSON-safe copy with bodies and credentials removed."""

    if arguments is None:
        return {}
    sanitized = _sanitize_value(arguments, key=None, depth=0, seen=set())
    return sanitized if isinstance(sanitized, dict) else {}


def _timestamp(value: datetime | None = None) -> str:
    moment = value or datetime.now(UTC)
    if moment.tzinfo is None or moment.utcoffset() is None:
        moment = moment.replace(tzinfo=UTC)
    return moment.isoformat(timespec="milliseconds")


def _duration_ms(value: int | float) -> int | float:
    if isinstance(value, bool):
        return 0
    try:
        duration = float(value)
    except (TypeError, ValueError):
        return 0
    if not math.isfinite(duration) or duration < 0:
        return 0
    return int(duration) if duration.is_integer() else round(duration, 3)


def _safe_error_type(value: str | BaseException | type[BaseException] | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, BaseException):
        return type(value).__name__
    if isinstance(value, type) and issubclass(value, BaseException):
        return value.__name__
    candidate = str(value)
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]{0,127}", candidate):
        return candidate
    return "Error"


@dataclass(frozen=True, slots=True)
class AuditRecord:
    timestamp: str
    request_id: str
    tool_name: str
    arguments: dict[str, Any]
    approval_required: bool
    approved: bool
    success: bool
    duration_ms: int | float
    error_type: str | None


@dataclass(frozen=True, slots=True)
class AuditWriteResult:
    """Result of the logger's single best-effort write attempt."""

    request_id: str
    written: bool
    error: str | None = None


def build_audit_record(
    *,
    tool_name: str,
    arguments: Mapping[str, Any] | None,
    approval_required: bool,
    approved: bool,
    success: bool,
    duration_ms: int | float,
    error_type: str | BaseException | type[BaseException] | None = None,
    request_id: str | UUID | None = None,
    timestamp: datetime | None = None,
) -> AuditRecord:
    """Create a normalized record suitable for serialization."""

    safe_tool_name = (
        tool_name if re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", tool_name) else "invalid_tool"
    )
    return AuditRecord(
        timestamp=_timestamp(timestamp),
        request_id=_canonical_request_id(request_id),
        tool_name=safe_tool_name,
        arguments=sanitize_arguments(arguments),
        approval_required=bool(approval_required),
        approved=bool(approved),
        success=bool(success),
        duration_ms=_duration_ms(duration_ms),
        error_type=_safe_error_type(error_type),
    )


class AuditLogger:
    """Append audit records to a UTF-8 JSONL file exactly once per call."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = Lock()

    def log(
        self,
        *,
        tool_name: str,
        arguments: Mapping[str, Any] | None,
        approval_required: bool,
        approved: bool,
        success: bool,
        duration_ms: int | float,
        error_type: str | BaseException | type[BaseException] | None = None,
        request_id: str | UUID | None = None,
        timestamp: datetime | None = None,
    ) -> AuditWriteResult:
        """Build and append one record.

        Directory creation and file append are attempted once.  Any filesystem
        or serialization failure is reported to the caller and is not retried.
        """

        record = build_audit_record(
            tool_name=tool_name,
            arguments=arguments,
            approval_required=approval_required,
            approved=approved,
            success=success,
            duration_ms=duration_ms,
            error_type=error_type,
            request_id=request_id,
            timestamp=timestamp,
        )
        try:
            line = json.dumps(
                asdict(record),
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            )
            with self._lock:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8", newline="\n") as stream:
                    stream.write(f"{line}\n")
        except (OSError, TypeError, ValueError) as exc:
            return AuditWriteResult(
                request_id=record.request_id,
                written=False,
                error=type(exc).__name__,
            )
        return AuditWriteResult(request_id=record.request_id, written=True)


def write_audit_record(path: str | Path, **record_fields: Any) -> AuditWriteResult:
    """Convenience wrapper for callers that do not retain an ``AuditLogger``."""

    return AuditLogger(path).log(**record_fields)


__all__ = [
    "AuditLogger",
    "AuditRecord",
    "AuditWriteResult",
    "build_audit_record",
    "new_request_id",
    "sanitize_arguments",
    "write_audit_record",
]
