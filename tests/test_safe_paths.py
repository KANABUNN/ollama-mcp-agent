from __future__ import annotations

from pathlib import Path

import pytest

import ollama_mcp_agent.config as config_module
from ollama_mcp_agent.config import Config, ConfigError, load_config, public_probe_main
from ollama_mcp_agent.pc_tools_server import PathAccessError, resolve_safe_path


def test_plain_file_is_allowed(tmp_path: Path) -> None:
    assert resolve_safe_path(tmp_path, "file.txt") == tmp_path / "file.txt"


def test_nested_windows_style_path_is_allowed_on_any_os(tmp_path: Path) -> None:
    expected = tmp_path / "subdir" / "file.txt"

    assert resolve_safe_path(tmp_path, r"subdir\file.txt") == expected


@pytest.mark.parametrize(
    "relative_path",
    [
        "../outside.txt",
        r"..\outside.txt",
        "nested/../../outside.txt",
        r"nested\..\..\outside.txt",
    ],
)
def test_parent_components_are_rejected(
    tmp_path: Path,
    relative_path: str,
) -> None:
    with pytest.raises(PathAccessError, match="Parent path"):
        resolve_safe_path(tmp_path, relative_path)


@pytest.mark.parametrize(
    "absolute_path",
    [
        "/outside.txt",
        r"C:\outside.txt",
        r"C:outside.txt",
        r"\outside.txt",
        r"\\server\share\outside.txt",
    ],
)
def test_absolute_and_drive_paths_are_rejected_on_any_os(
    tmp_path: Path,
    absolute_path: str,
) -> None:
    with pytest.raises(PathAccessError, match="Absolute paths"):
        resolve_safe_path(tmp_path, absolute_path)


def test_native_absolute_path_is_rejected(tmp_path: Path) -> None:
    outside = tmp_path.parent / "allowed-root-similar" / "outside.txt"

    with pytest.raises(PathAccessError, match="Absolute paths"):
        resolve_safe_path(tmp_path, str(outside.resolve(strict=False)))


def test_similarly_named_sibling_cannot_pass_containment_check(tmp_path: Path) -> None:
    allowed_root = tmp_path / "allowed"
    allowed_root.mkdir()
    similar_sibling = tmp_path / "allowed-backup"
    similar_sibling.mkdir()

    with pytest.raises(PathAccessError):
        resolve_safe_path(allowed_root, str(similar_sibling / "secret.txt"))


def test_symlink_escape_is_rejected_when_supported(tmp_path: Path) -> None:
    allowed_root = tmp_path / "allowed"
    allowed_root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    link = allowed_root / "escape"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"symlinks are not available in this environment: {exc}")

    with pytest.raises(PathAccessError, match="outside the allowed root"):
        resolve_safe_path(allowed_root, "escape/secret.txt")


def test_symlink_that_stays_inside_root_is_allowed_when_supported(
    tmp_path: Path,
) -> None:
    real_directory = tmp_path / "real"
    real_directory.mkdir()
    link = tmp_path / "inside-link"
    try:
        link.symlink_to(real_directory, target_is_directory=True)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"symlinks are not available in this environment: {exc}")

    assert resolve_safe_path(tmp_path, "inside-link/file.txt") == (
        real_directory / "file.txt"
    )


@pytest.mark.parametrize("relative_path", ["NUL", "con.txt", "file.txt:stream"])
def test_unsafe_windows_names_are_rejected_everywhere(
    tmp_path: Path,
    relative_path: str,
) -> None:
    with pytest.raises(PathAccessError, match="Windows name"):
        resolve_safe_path(tmp_path, relative_path)


@pytest.mark.parametrize("relative_path", ["line\nbreak.txt", "tab\tname.txt", "escape\x1b.txt"])
def test_path_control_characters_are_rejected(tmp_path: Path, relative_path: str) -> None:
    with pytest.raises(PathAccessError, match="invalid character"):
        resolve_safe_path(tmp_path, relative_path)


def test_missing_allowed_root_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(PathAccessError, match="allowed root"):
        resolve_safe_path(tmp_path / "missing", "file.txt")


@pytest.mark.parametrize(
    "host",
    [
        "http://127.0.0.1:11434",
        "http://127.42.0.9:11434",
        "http://localhost:11434",
        "http://[::1]:11434",
    ],
)
def test_config_accepts_only_expected_loopback_forms(
    tmp_path: Path,
    host: str,
) -> None:
    config = Config(ollama_host=host, allowed_root=tmp_path)

    assert config.ollama_host == host


@pytest.mark.parametrize(
    "host",
    [
        "http://example.com:11434",
        "http://192.168.1.20:11434",
        "http://0.0.0.0:11434",
        "http://localhost.example:11434",
        "http://user:password@localhost:11434",
    ],
)
def test_config_rejects_non_loopback_or_credentialed_hosts(
    tmp_path: Path,
    host: str,
) -> None:
    with pytest.raises(ConfigError, match="loopback"):
        Config(ollama_host=host, allowed_root=tmp_path)


@pytest.mark.parametrize(
    "network_path",
    [
        r"\\server\share\workspace",
        r"//server/share/workspace",
        r"\\?\UNC\server\share\workspace",
        r"\\.\C:\workspace",
    ],
)
def test_config_rejects_unc_and_device_allowed_roots(
    tmp_path: Path,
    network_path: str,
) -> None:
    with pytest.raises(ConfigError, match=r"local|UNC"):
        load_config({"ALLOWED_ROOT": network_path}, project_root=tmp_path)


def test_load_config_exposes_all_runtime_settings(tmp_path: Path) -> None:
    config = load_config(
        {
            "OLLAMA_HOST": "http://localhost:11435",
            "OLLAMA_MODEL": "local-model",
            "ALLOWED_ROOT": str(tmp_path),
            "AUDIT_LOG_PATH": "custom/audit.jsonl",
            "MAX_READ_BYTES": "10",
            "MAX_WRITE_BYTES": "20",
            "MAX_TOOL_ITERATIONS": "3",
        },
        project_root=tmp_path,
    )

    assert config.ollama_host == "http://localhost:11435"
    assert config.ollama_model == "local-model"
    assert config.allowed_root == tmp_path.resolve()
    assert config.audit_log_path == (tmp_path / "custom" / "audit.jsonl").resolve()
    assert config.max_read_bytes == 10
    assert config.max_write_bytes == 20
    assert config.max_tool_iterations == 3


def test_load_config_rejects_relative_allowed_root(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="absolute path"):
        load_config({"ALLOWED_ROOT": "relative/root"}, project_root=tmp_path)


def test_audit_log_path_cannot_leave_project(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    allowed_root = tmp_path / "allowed"
    allowed_root.mkdir()

    with pytest.raises(ConfigError, match="AUDIT_LOG_PATH"):
        load_config(
            {
                "ALLOWED_ROOT": str(allowed_root),
                "AUDIT_LOG_PATH": str(tmp_path / "outside" / "audit.jsonl"),
            },
            project_root=project_root,
        )


@pytest.mark.parametrize("setting", ["ALLOWED_ROOT", "AUDIT_LOG_PATH"])
def test_configured_paths_reject_control_characters(tmp_path: Path, setting: str) -> None:
    values = {"ALLOWED_ROOT": str(tmp_path), setting: "bad\npath"}

    with pytest.raises(ConfigError, match="control character"):
        load_config(values, project_root=tmp_path)


def test_public_config_probe_never_prints_underlying_invalid_value(
    monkeypatch,
    capsys,
) -> None:
    sentinel = "LEAK_SENTINEL"

    def invalid_config() -> Config:
        raise ConfigError(f"invalid value: {sentinel}")

    monkeypatch.setattr(config_module, "load_config", invalid_config)

    assert public_probe_main() == 2
    output = capsys.readouterr()
    assert sentinel not in output.out
    assert sentinel not in output.err
    assert output.out == ""
    assert output.err == ""


@pytest.mark.parametrize("value", ["0", "-1", "not-a-number"])
def test_load_config_rejects_invalid_size_limits(tmp_path: Path, value: str) -> None:
    with pytest.raises(ConfigError, match="MAX_READ_BYTES"):
        load_config(
            {"ALLOWED_ROOT": str(tmp_path), "MAX_READ_BYTES": value},
            project_root=tmp_path,
        )
