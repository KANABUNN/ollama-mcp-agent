"""Human approval helpers for operations that modify files.

Approval is deliberately handled by the agent process rather than by the MCP
server.  The server therefore cannot accept an LLM-provided flag claiming that
an operation has already been approved.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Callable
from dataclasses import dataclass

DEFAULT_PREVIEW_CHARS = 500
WRITE_TOOL_NAME = "write_text_file"

InputFunction = Callable[[str], str]
OutputFunction = Callable[[str], object]


def is_affirmative(response: str | None) -> bool:
    """Return ``True`` only for an explicit ``y`` or ``yes`` response."""

    return isinstance(response, str) and response.strip().casefold() in {"y", "yes"}


def requires_approval(tool_name: str) -> bool:
    """Return whether an MCP tool requires interactive approval.

    Unknown tool names are rejected elsewhere by the bridge.  At present the
    only exposed mutating tool is ``write_text_file``.
    """

    return tool_name == WRITE_TOOL_NAME


def _terminal_safe(value: str, *, preserve_layout: bool = False) -> str:
    """Escape terminal controls; only bounded content previews keep layout."""

    rendered: list[str] = []
    for character in value:
        if preserve_layout and character in {"\n", "\t"}:
            rendered.append(character)
        elif unicodedata.category(character).startswith("C"):
            codepoint = ord(character)
            escape = f"\\x{codepoint:02x}" if codepoint <= 0xFF else f"\\u{codepoint:04x}"
            rendered.append(escape)
        else:
            rendered.append(character)
    return "".join(rendered)


def make_content_preview(content: str, *, max_chars: int = DEFAULT_PREVIEW_CHARS) -> str:
    """Build a bounded, terminal-safe preview of text to be written."""

    if max_chars < 1:
        raise ValueError("max_chars must be at least 1")

    if not content:
        return "(empty)"

    preview = _terminal_safe(content[:max_chars], preserve_layout=True)
    omitted = len(content) - max_chars
    if omitted > 0:
        preview += f"\n... ({omitted} more characters omitted)"
    return preview


@dataclass(frozen=True, slots=True)
class WriteApprovalRequest:
    """The details shown to a user before a file write is authorized."""

    relative_path: str
    content: str
    exists: bool
    tool_name: str = WRITE_TOOL_NAME

    def render(self, *, preview_chars: int = DEFAULT_PREVIEW_CHARS) -> str:
        """Render the complete, bounded confirmation notice."""

        operation = "overwrite existing file" if self.exists else "create new file"
        preview = make_content_preview(self.content, max_chars=preview_chars)
        return "\n".join(
            (
                "File write approval required",
                f"Tool: {_terminal_safe(self.tool_name)}",
                f"Path: {_terminal_safe(self.relative_path)}",
                f"Operation: {operation}",
                f"Characters: {len(self.content)}",
                "Preview:",
                "---",
                preview,
                "---",
            )
        )


def request_write_approval(
    relative_path: str,
    content: str,
    *,
    exists: bool,
    input_fn: InputFunction = input,
    output_fn: OutputFunction = print,
    preview_chars: int = DEFAULT_PREVIEW_CHARS,
) -> bool:
    """Ask once for permission to write a file, defaulting safely to denial.

    ``input_fn`` and ``output_fn`` are injected so callers can test this code
    without touching the real console.  Empty, negative, malformed, interrupted,
    or unavailable input is denied; the function never keeps prompting.
    """

    request = WriteApprovalRequest(
        relative_path=str(relative_path),
        content=content,
        exists=bool(exists),
    )
    output_fn(request.render(preview_chars=preview_chars))

    try:
        response = input_fn("Proceed with this write? [y/N]: ")
    except (EOFError, KeyboardInterrupt):
        output_fn("Write denied.")
        return False

    approved = is_affirmative(response)
    output_fn("Write approved." if approved else "Write denied.")
    return approved


__all__ = [
    "DEFAULT_PREVIEW_CHARS",
    "WRITE_TOOL_NAME",
    "WriteApprovalRequest",
    "is_affirmative",
    "make_content_preview",
    "request_write_approval",
    "requires_approval",
]
