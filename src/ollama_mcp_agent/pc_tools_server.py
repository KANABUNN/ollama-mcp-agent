"""Restricted text-file tools and their stdio MCP server adapter."""

from __future__ import annotations

import os
import re
import stat
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from .config import Config, ConfigError, load_config


class FileToolError(RuntimeError):
    """Base class for safe, user-facing file tool failures."""


class PathAccessError(FileToolError):
    """Raised when a requested path is invalid or leaves the allowed root."""


class FileSizeLimitError(FileToolError):
    """Raised when a read or write exceeds its configured byte limit."""


class BinaryFileError(FileToolError):
    """Raised when a file has a binary marker and is unsuitable for this tool."""


class InvalidUtf8Error(FileToolError):
    """Raised when file bytes are not valid UTF-8."""


_WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{number}" for number in range(1, 10)),
    *(f"LPT{number}" for number in range(1, 10)),
}
_PATH_SEPARATOR_PATTERN = re.compile(r"[\\/]+")
_REPARSE_POINT_ATTRIBUTE = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)


def _is_link_or_reparse_point(path_stat: os.stat_result) -> bool:
    """Recognize POSIX links and Windows junction/reparse-point aliases."""

    attributes = getattr(path_stat, "st_file_attributes", 0)
    return stat.S_ISLNK(path_stat.st_mode) or bool(
        _REPARSE_POINT_ATTRIBUTE and attributes & _REPARSE_POINT_ATTRIBUTE
    )


def _validated_relative_parts(relative_path: str) -> tuple[str, ...]:
    if not isinstance(relative_path, str):
        raise PathAccessError("The path must be a relative string.")
    if any(ord(character) < 32 or ord(character) == 127 for character in relative_path):
        raise PathAccessError("The path contains an invalid character.")

    windows_path = PureWindowsPath(relative_path)
    posix_path = PurePosixPath(relative_path)
    if (
        Path(relative_path).is_absolute()
        or posix_path.is_absolute()
        or bool(windows_path.drive)
        or bool(windows_path.root)
    ):
        raise PathAccessError("Absolute paths are not allowed.")

    parts = tuple(
        part
        for part in _PATH_SEPARATOR_PATTERN.split(relative_path)
        if part not in {"", "."}
    )
    if any(part == ".." for part in parts):
        raise PathAccessError("Parent path components are not allowed.")

    # The application targets Windows.  Reject alternate data streams, device
    # names, and names Windows silently normalises, even when tests run on POSIX.
    for part in parts:
        if ":" in part or part.endswith((" ", ".")):
            raise PathAccessError("The path contains an unsupported Windows name.")
        stem = part.split(".", 1)[0].upper()
        if stem in _WINDOWS_RESERVED_NAMES:
            raise PathAccessError("The path contains a reserved Windows name.")

    return parts


def resolve_safe_path(allowed_root: str | Path, relative_path: str) -> Path:
    """Resolve a relative path and prove that it remains under ``allowed_root``.

    Existing symbolic links and Windows junctions are resolved before the
    containment check.  The returned path is the resolved path, so later file
    operations do not intentionally follow the user-supplied alias again.
    """

    try:
        root = Path(allowed_root).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise PathAccessError("The allowed root is not available.") from exc
    if not root.is_dir():
        raise PathAccessError("The allowed root is not a directory.")

    parts = _validated_relative_parts(relative_path)
    try:
        target = root.joinpath(*parts).resolve(strict=False)
        target.relative_to(root)
    except (OSError, RuntimeError, ValueError) as exc:
        raise PathAccessError("The requested path is outside the allowed root.") from exc
    return target


def _display_path(relative_path: str) -> str:
    parts = _validated_relative_parts(relative_path)
    return "/".join(parts) if parts else "."


def _modified_at(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, tz=UTC).isoformat()


class FileTools:
    """Testable implementation of the three restricted MCP file tools."""

    def __init__(
        self,
        allowed_root: str | Path,
        *,
        max_read_bytes: int = 1_048_576,
        max_write_bytes: int = 1_048_576,
    ) -> None:
        if (
            isinstance(max_read_bytes, bool)
            or not isinstance(max_read_bytes, int)
            or max_read_bytes <= 0
        ):
            raise ValueError("max_read_bytes must be a positive integer.")
        if (
            isinstance(max_write_bytes, bool)
            or not isinstance(max_write_bytes, int)
            or max_write_bytes <= 0
        ):
            raise ValueError("max_write_bytes must be a positive integer.")

        configured_root = Path(allowed_root)
        try:
            configured_stat = configured_root.lstat()
        except OSError as exc:
            raise PathAccessError("The allowed root is not available.") from exc
        if _is_link_or_reparse_point(configured_stat):
            raise PathAccessError("The allowed root must not be a link or junction.")

        # Resolve once and retain the directory identity. If the configured
        # root is later renamed and replaced, operations fail closed instead
        # of silently trusting the replacement as a new security boundary.
        self.allowed_root = resolve_safe_path(configured_root, ".")
        root_stat = self.allowed_root.lstat()
        self._root_identity = (root_stat.st_dev, root_stat.st_ino)
        self.max_read_bytes = max_read_bytes
        self.max_write_bytes = max_write_bytes

    @classmethod
    def from_config(cls, config: Config) -> FileTools:
        return cls(
            config.allowed_root,
            max_read_bytes=config.max_read_bytes,
            max_write_bytes=config.max_write_bytes,
        )

    def _resolve(self, relative_path: str) -> Path:
        self._verify_root_identity()
        target = resolve_safe_path(self.allowed_root, relative_path)
        self._verify_root_identity()
        return target

    def _verify_root_identity(self) -> None:
        try:
            root_stat = self.allowed_root.lstat()
        except OSError as exc:
            raise PathAccessError("The allowed root is not available.") from exc
        identity = (root_stat.st_dev, root_stat.st_ino)
        if (
            identity != self._root_identity
            or not stat.S_ISDIR(root_stat.st_mode)
            or _is_link_or_reparse_point(root_stat)
        ):
            raise PathAccessError("The allowed root changed after startup.")

    def list_files(self, relative_path: str = ".") -> list[dict[str, object]]:
        """List one directory level, including hidden entries."""

        target = self._resolve(relative_path)
        if not target.exists():
            raise FileToolError("The requested directory does not exist.")
        if not target.is_dir():
            raise FileToolError("The requested path is not a directory.")

        entries: list[dict[str, object]] = []
        try:
            children = sorted(target.iterdir(), key=lambda item: item.name.casefold())
            for item in children:
                item_stat = item.lstat()
                mode = item_stat.st_mode
                if stat.S_ISDIR(mode):
                    item_type = "directory"
                    size_bytes: int | None = None
                elif stat.S_ISREG(mode):
                    item_type = "file"
                    size_bytes = item_stat.st_size
                elif stat.S_ISLNK(mode):
                    # Do not follow a link merely to produce a listing.  Reads
                    # through it still undergo the containment check.
                    item_type = "symlink"
                    size_bytes = None
                else:
                    item_type = "other"
                    size_bytes = None

                entries.append(
                    {
                        "name": item.name,
                        "type": item_type,
                        "size_bytes": size_bytes,
                        "modified_at": _modified_at(item_stat.st_mtime),
                    }
                )
        except OSError as exc:
            raise FileToolError("The requested directory could not be listed.") from exc
        return entries

    def read_text_file(self, relative_path: str) -> dict[str, object]:
        """Read a bounded, regular UTF-8 text file."""

        target = self._resolve(relative_path)
        if not target.exists():
            raise FileToolError("The requested file does not exist.")
        if not target.is_file():
            raise FileToolError("The requested path is not a regular file.")

        try:
            with target.open("rb") as file_handle:
                opened_stat = os.fstat(file_handle.fileno())
                if not stat.S_ISREG(opened_stat.st_mode):
                    raise FileToolError("The requested path is not a regular file.")
                if opened_stat.st_nlink > 1:
                    raise PathAccessError("Files with multiple hard links are not allowed.")
                if opened_stat.st_size > self.max_read_bytes:
                    raise FileSizeLimitError("The file exceeds the read size limit.")
                data = file_handle.read(self.max_read_bytes + 1)
        except FileSizeLimitError:
            raise
        except OSError as exc:
            raise FileToolError("The requested file could not be read.") from exc

        # The bounded read protects against a file growing after the stat call.
        if len(data) > self.max_read_bytes:
            raise FileSizeLimitError("The file exceeds the read size limit.")
        if b"\x00" in data:
            raise BinaryFileError("The requested file appears to be binary.")
        try:
            content = data.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise InvalidUtf8Error("The requested file is not valid UTF-8.") from exc

        return {
            "relative_path": _display_path(relative_path),
            "content": content,
            "character_count": len(content),
            "byte_count": len(data),
        }

    def write_text_file(self, relative_path: str, content: str) -> dict[str, object]:
        """Atomically create or overwrite a bounded UTF-8 text file."""

        if not isinstance(content, str):
            raise FileToolError("Content must be text.")
        normalised_content = content.replace("\r\n", "\n").replace("\r", "\n")
        try:
            encoded_content = normalised_content.encode("utf-8", errors="strict")
        except UnicodeEncodeError as exc:
            raise InvalidUtf8Error("Content cannot be encoded as UTF-8.") from exc
        if len(encoded_content) > self.max_write_bytes:
            raise FileSizeLimitError("The content exceeds the write size limit.")

        target = self._resolve(relative_path)
        target_existed = target.exists()
        if target_existed and not target.is_file():
            raise FileToolError("The requested path is not a regular file.")

        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            # Re-resolve after creating parents so a link cannot bypass the
            # containment check simply because its parent did not exist before.
            target = self._resolve(relative_path)
            target_existed = target.exists()
            if target_existed and not target.is_file():
                raise FileToolError("The requested path is not a regular file.")

            descriptor, temporary_name = tempfile.mkstemp(
                dir=target.parent,
                prefix=".ollama-mcp-",
                suffix=".tmp",
            )
            temporary_path = Path(temporary_name)
            try:
                with os.fdopen(descriptor, "wb") as file_handle:
                    file_handle.write(encoded_content)
                    file_handle.flush()
                    os.fsync(file_handle.fileno())
                os.replace(temporary_path, target)
            except BaseException:
                temporary_path.unlink(missing_ok=True)
                raise
        except FileToolError:
            raise
        except OSError as exc:
            raise FileToolError("The requested file could not be written.") from exc

        return {
            "relative_path": _display_path(relative_path),
            "byte_count": len(encoded_content),
            "character_count": len(normalised_content),
            "created": not target_existed,
            "overwritten": target_existed,
        }


def create_mcp_server(
    config: Config | None = None,
    *,
    file_tools: FileTools | None = None,
) -> Any:
    """Build a FastMCP adapter over :class:`FileTools`.

    The SDK import is deliberately lazy so pure file-tool tests do not need to
    import or start MCP machinery.
    """

    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:  # pragma: no cover - depends on installation
        raise RuntimeError("The MCP Python SDK is not installed.") from exc

    if config is not None and file_tools is not None:
        raise ValueError("Pass config or file_tools, not both.")
    tools = file_tools or FileTools.from_config(config or load_config())
    server = FastMCP("Local PC Tools")

    @server.tool()
    def list_files(relative_path: str = ".") -> list[dict[str, object]]:
        """List files and directories one level below an allowed relative path."""

        return tools.list_files(relative_path)

    @server.tool()
    def read_text_file(relative_path: str) -> dict[str, object]:
        """Read one bounded UTF-8 text file from an allowed relative path."""

        return tools.read_text_file(relative_path)

    @server.tool()
    def write_text_file(relative_path: str, content: str) -> dict[str, object]:
        """Create or overwrite one bounded UTF-8 file at an allowed relative path."""

        return tools.write_text_file(relative_path, content)

    return server


def main() -> int:
    """Run the MCP server over stdio without writing ordinary logs to stdout."""

    try:
        server = create_mcp_server()
        server.run(transport="stdio")
    except (ConfigError, FileToolError, RuntimeError) as exc:
        sys.stderr.write(f"MCP server could not start: {exc}\n")
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised as a process
    raise SystemExit(main())
