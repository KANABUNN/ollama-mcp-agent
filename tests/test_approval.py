from __future__ import annotations

import pytest

from ollama_mcp_agent.approval import (
    WriteApprovalRequest,
    is_affirmative,
    make_content_preview,
    request_write_approval,
    requires_approval,
)


@pytest.mark.parametrize("response", ["y", "yes", "Y", "YES", " Yes "])
def test_only_explicit_affirmative_responses_are_accepted(response: str) -> None:
    assert is_affirmative(response)


@pytest.mark.parametrize("response", [None, "", " ", "n", "no", "true", "ok", "yes please"])
def test_empty_negative_and_invalid_responses_are_denied(response: str | None) -> None:
    assert not is_affirmative(response)


def test_write_prompt_shows_all_required_details_for_new_file() -> None:
    output: list[str] = []
    prompts: list[str] = []

    def answer(prompt: str) -> str:
        prompts.append(prompt)
        return "yes"

    approved = request_write_approval(
        r"output\result.md",
        "# Result\nhello",
        exists=False,
        input_fn=answer,
        output_fn=output.append,
    )

    assert approved
    notice = output[0]
    assert "write_text_file" in notice
    assert r"output\result.md" in notice
    assert "create new file" in notice
    assert "Characters: 14" in notice
    assert "# Result\nhello" in notice
    assert prompts == ["Proceed with this write? [y/N]: "]
    assert output[-1] == "Write approved."


def test_write_prompt_identifies_overwrite_and_defaults_to_denial() -> None:
    output: list[str] = []

    approved = request_write_approval(
        "existing.txt",
        "replacement",
        exists=True,
        input_fn=lambda _prompt: "",
        output_fn=output.append,
    )

    assert not approved
    assert "overwrite existing file" in output[0]
    assert output[-1] == "Write denied."


@pytest.mark.parametrize("exception", [EOFError(), KeyboardInterrupt()])
def test_unavailable_or_interrupted_input_is_denied(exception: BaseException) -> None:
    output: list[str] = []

    def fail(_prompt: str) -> str:
        raise exception

    assert not request_write_approval(
        "file.txt",
        "value",
        exists=False,
        input_fn=fail,
        output_fn=output.append,
    )
    assert output[-1] == "Write denied."


def test_preview_is_bounded_and_escapes_terminal_controls() -> None:
    preview = make_content_preview("ab\x1b[31mcdef", max_chars=5)

    assert preview.startswith(r"ab\x1b[3")
    assert "more characters omitted" in preview
    assert "\x1b" not in preview.replace(r"\x1b", "")


def test_preview_length_must_be_positive() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        make_content_preview("text", max_chars=0)


def test_request_object_renders_character_count() -> None:
    notice = WriteApprovalRequest("file.txt", "日本語", exists=False).render()

    assert "Characters: 3" in notice


def test_path_control_characters_cannot_spoof_approval_lines() -> None:
    notice = WriteApprovalRequest(
        "safe.txt\nOperation: harmless",
        "content",
        exists=False,
    ).render()

    assert "Path: safe.txt\\x0aOperation: harmless" in notice
    assert "\nOperation: harmless" not in notice


def test_only_write_tool_requires_approval() -> None:
    assert requires_approval("write_text_file")
    assert not requires_approval("read_text_file")
    assert not requires_approval("list_files")
