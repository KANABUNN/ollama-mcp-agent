from __future__ import annotations

from pathlib import Path

import pytest

from ollama_mcp_agent.pc_tools_server import (
    BinaryFileError,
    FileSizeLimitError,
    FileToolError,
    FileTools,
    InvalidUtf8Error,
    PathAccessError,
)


@pytest.fixture
def file_tools(tmp_path: Path) -> FileTools:
    return FileTools(tmp_path, max_read_bytes=64, max_write_bytes=64)


def test_list_files_returns_sorted_metadata(file_tools: FileTools, tmp_path: Path) -> None:
    (tmp_path / "zeta.txt").write_text("hello", encoding="utf-8")
    (tmp_path / "Alpha").mkdir()
    (tmp_path / ".hidden").write_text("x", encoding="utf-8")

    entries = file_tools.list_files()

    assert [entry["name"] for entry in entries] == [".hidden", "Alpha", "zeta.txt"]
    by_name = {entry["name"]: entry for entry in entries}
    assert by_name["Alpha"]["type"] == "directory"
    assert by_name["Alpha"]["size_bytes"] is None
    assert by_name["zeta.txt"]["type"] == "file"
    assert by_name["zeta.txt"]["size_bytes"] == 5
    assert "+00:00" in str(by_name["zeta.txt"]["modified_at"])


def test_list_files_is_not_recursive(file_tools: FileTools, tmp_path: Path) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "inside.txt").write_text("inside", encoding="utf-8")

    entries = file_tools.list_files()

    assert [entry["name"] for entry in entries] == ["nested"]


def test_list_files_rejects_a_file(file_tools: FileTools, tmp_path: Path) -> None:
    (tmp_path / "file.txt").write_text("text", encoding="utf-8")

    with pytest.raises(FileToolError, match="not a directory"):
        file_tools.list_files("file.txt")


def test_read_utf8_returns_content_and_counts(
    file_tools: FileTools,
    tmp_path: Path,
) -> None:
    content = "こんにちは\n"
    (tmp_path / "note.txt").write_text(content, encoding="utf-8", newline="")

    result = file_tools.read_text_file("note.txt")

    assert result == {
        "relative_path": "note.txt",
        "content": content,
        "character_count": len(content),
        "byte_count": len(content.encode("utf-8")),
    }


def test_read_exact_size_limit_is_allowed(tmp_path: Path) -> None:
    (tmp_path / "exact.txt").write_bytes(b"1234")
    tools = FileTools(tmp_path, max_read_bytes=4)

    assert tools.read_text_file("exact.txt")["content"] == "1234"


def test_read_over_size_limit_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "large.txt").write_bytes(b"12345")
    tools = FileTools(tmp_path, max_read_bytes=4)

    with pytest.raises(FileSizeLimitError, match="read size limit"):
        tools.read_text_file("large.txt")


def test_read_invalid_utf8_is_rejected(file_tools: FileTools, tmp_path: Path) -> None:
    (tmp_path / "invalid.txt").write_bytes(b"\xff\xfe")

    with pytest.raises(InvalidUtf8Error, match="valid UTF-8"):
        file_tools.read_text_file("invalid.txt")


def test_read_binary_marker_is_rejected(file_tools: FileTools, tmp_path: Path) -> None:
    (tmp_path / "binary.dat").write_bytes(b"text\x00binary")

    with pytest.raises(BinaryFileError, match="binary"):
        file_tools.read_text_file("binary.dat")


def test_read_missing_file_is_rejected(file_tools: FileTools) -> None:
    with pytest.raises(FileToolError, match="does not exist"):
        file_tools.read_text_file("missing.txt")


def test_read_directory_as_file_is_rejected(file_tools: FileTools, tmp_path: Path) -> None:
    (tmp_path / "directory").mkdir()

    with pytest.raises(FileToolError, match="not a regular file"):
        file_tools.read_text_file("directory")


def test_new_write_creates_parent_and_normalises_newlines(
    file_tools: FileTools,
    tmp_path: Path,
) -> None:
    result = file_tools.write_text_file("output/result.md", "a\r\nb\rc\n")

    assert (tmp_path / "output" / "result.md").read_bytes() == b"a\nb\nc\n"
    assert result == {
        "relative_path": "output/result.md",
        "byte_count": 6,
        "character_count": 6,
        "created": True,
        "overwritten": False,
    }
    assert not list((tmp_path / "output").glob(".ollama-mcp-*.tmp"))


def test_write_overwrites_existing_file(file_tools: FileTools, tmp_path: Path) -> None:
    target = tmp_path / "result.txt"
    target.write_text("old", encoding="utf-8")

    result = file_tools.write_text_file("result.txt", "new")

    assert target.read_text(encoding="utf-8") == "new"
    assert result["created"] is False
    assert result["overwritten"] is True


def test_write_exact_byte_limit_is_allowed(tmp_path: Path) -> None:
    tools = FileTools(tmp_path, max_write_bytes=4)

    result = tools.write_text_file("exact.txt", "1234")

    assert result["byte_count"] == 4


def test_write_limit_counts_utf8_bytes(tmp_path: Path) -> None:
    tools = FileTools(tmp_path, max_write_bytes=4)

    with pytest.raises(FileSizeLimitError, match="write size limit"):
        tools.write_text_file("large.txt", "あい")
    assert not (tmp_path / "large.txt").exists()


def test_write_rejects_directory_target(file_tools: FileTools, tmp_path: Path) -> None:
    (tmp_path / "directory").mkdir()

    with pytest.raises(FileToolError, match="not a regular file"):
        file_tools.write_text_file("directory", "content")


@pytest.mark.parametrize("operation", ["list", "read", "write"])
def test_file_operations_reject_escape(file_tools: FileTools, operation: str) -> None:
    with pytest.raises(PathAccessError):
        if operation == "list":
            file_tools.list_files("..")
        elif operation == "read":
            file_tools.read_text_file("../secret.txt")
        else:
            file_tools.write_text_file("../secret.txt", "secret")


def test_file_tools_reject_invalid_limits(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="max_read_bytes"):
        FileTools(tmp_path, max_read_bytes=0)
    with pytest.raises(ValueError, match="max_write_bytes"):
        FileTools(tmp_path, max_write_bytes=-1)


def test_file_tools_reject_replaced_allowed_root(tmp_path: Path) -> None:
    root = tmp_path / "allowed"
    root.mkdir()
    tools = FileTools(root)
    root.rename(tmp_path / "original-root")
    root.mkdir()

    with pytest.raises(PathAccessError, match="changed after startup"):
        tools.list_files(".")


def test_read_rejects_hard_link_alias(tmp_path: Path) -> None:
    root = tmp_path / "allowed"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("outside secret", encoding="utf-8")
    try:
        (root / "alias.txt").hardlink_to(outside)
    except OSError as exc:
        pytest.skip(f"hard links are not available in this environment: {exc}")

    with pytest.raises(PathAccessError, match="multiple hard links"):
        FileTools(root).read_text_file("alias.txt")
