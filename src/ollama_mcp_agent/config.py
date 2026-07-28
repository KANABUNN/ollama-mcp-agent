"""Application configuration loaded from environment variables.

The module intentionally has no third-party dependency.  This keeps the stdio
server importable in small test environments and makes the rules used for the
security-relevant size and path settings explicit.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from ipaddress import ip_address
from pathlib import Path, PureWindowsPath
from urllib.parse import urlsplit

DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"
DEFAULT_ALLOWED_ROOT = Path(r"C:\LLM-Workspace")
DEFAULT_AUDIT_LOG_PATH = Path("logs/audit.jsonl")
DEFAULT_MAX_READ_BYTES = 1_048_576
DEFAULT_MAX_WRITE_BYTES = 1_048_576
DEFAULT_MAX_TOOL_ITERATIONS = 10
_CONFIG_ENV_NAMES = {
    "OLLAMA_HOST",
    "OLLAMA_MODEL",
    "ALLOWED_ROOT",
    "AUDIT_LOG_PATH",
    "MAX_READ_BYTES",
    "MAX_WRITE_BYTES",
    "MAX_TOOL_ITERATIONS",
}


class ConfigError(ValueError):
    """Raised when application configuration is missing or invalid."""


@dataclass(frozen=True, slots=True)
class Config:
    """Validated runtime configuration.

    ``OLLAMA_MODEL`` is deliberately allowed to be empty here.  The MCP server
    does not need a model, while the interactive agent can give a focused error
    before its first Ollama request.
    """

    ollama_host: str = DEFAULT_OLLAMA_HOST
    ollama_model: str = ""
    allowed_root: Path = DEFAULT_ALLOWED_ROOT
    audit_log_path: Path = DEFAULT_AUDIT_LOG_PATH
    max_read_bytes: int = DEFAULT_MAX_READ_BYTES
    max_write_bytes: int = DEFAULT_MAX_WRITE_BYTES
    max_tool_iterations: int = DEFAULT_MAX_TOOL_ITERATIONS

    def __post_init__(self) -> None:
        object.__setattr__(self, "allowed_root", Path(self.allowed_root))
        object.__setattr__(self, "audit_log_path", Path(self.audit_log_path))

        parsed_host = urlsplit(self.ollama_host)
        hostname = parsed_host.hostname
        if (
            parsed_host.scheme not in {"http", "https"}
            or not hostname
            or parsed_host.username is not None
            or parsed_host.password is not None
            or not _is_loopback_hostname(hostname)
        ):
            raise ConfigError(
                "OLLAMA_HOST must be a loopback HTTP(S) URL without credentials."
            )
        try:
            _ = parsed_host.port
        except ValueError as exc:
            raise ConfigError("OLLAMA_HOST contains an invalid port.") from exc

        _require_positive_int("MAX_READ_BYTES", self.max_read_bytes)
        _require_positive_int("MAX_WRITE_BYTES", self.max_write_bytes)
        _require_positive_int("MAX_TOOL_ITERATIONS", self.max_tool_iterations)


# A compatibility-friendly name for callers that prefer Settings.
Settings = Config


def _is_loopback_hostname(hostname: str) -> bool:
    normalised = hostname.rstrip(".").casefold()
    if normalised == "localhost":
        return True
    try:
        return ip_address(normalised).is_loopback
    except ValueError:
        return False


def _require_positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigError(f"{name} must be a positive integer.")


def _parse_positive_int(name: str, value: str, default: int) -> int:
    raw_value = value.strip() if value else str(default)
    try:
        parsed = int(raw_value, 10)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a positive integer.") from exc
    _require_positive_int(name, parsed)
    return parsed


def _read_dotenv(path: Path) -> dict[str, str]:
    """Read the small dotenv subset used by this project.

    Environment variables are overlaid afterwards.  Values are not expanded,
    which avoids surprising interpolation of paths or secrets.
    """

    if not path.is_file():
        return {}

    values: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ConfigError("The .env file could not be read as UTF-8.") from exc

    for line_number, original_line in enumerate(lines, start=1):
        line = original_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            raise ConfigError(f"Invalid .env entry on line {line_number}.")
        name, value = line.split("=", 1)
        name = name.strip()
        if not name or not name.replace("_", "A").isalnum() or name[0].isdigit():
            raise ConfigError(f"Invalid .env key on line {line_number}.")
        if name not in _CONFIG_ENV_NAMES:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        values[name] = value
    return values


def _looks_absolute(path_text: str) -> bool:
    """Recognise both native and Windows absolute paths on every test OS."""

    native_path = Path(path_text)
    windows_path = PureWindowsPath(path_text)
    return native_path.is_absolute() or windows_path.is_absolute()


def _reject_network_allowed_root(path_text: str, *, setting_name: str) -> None:
    if setting_name != "ALLOWED_ROOT":
        return

    normalized = path_text.replace("/", "\\")
    if normalized.startswith("\\\\"):
        raise ConfigError("ALLOWED_ROOT must be a local, non-UNC path.")

    if os.name == "nt":
        # A drive letter may itself be mapped to SMB even without UNC syntax.
        # GetDriveTypeW checks the drive class without opening user files.
        import ctypes

        drive_root = PureWindowsPath(path_text).anchor
        drive_type = ctypes.windll.kernel32.GetDriveTypeW(str(drive_root))
        drive_remote = 4
        if drive_type == drive_remote:
            raise ConfigError("ALLOWED_ROOT must not use a network drive.")


def _configured_path(
    value: str,
    *,
    setting_name: str,
    project_root: Path,
    require_absolute: bool,
) -> Path:
    value = value.strip()
    if not value:
        raise ConfigError(f"{setting_name} must not be empty.")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise ConfigError(f"{setting_name} contains an invalid control character.")

    if require_absolute and not _looks_absolute(value):
        raise ConfigError(f"{setting_name} must be an absolute path.")
    _reject_network_allowed_root(value, setting_name=setting_name)

    path = Path(value)
    # A Windows absolute path cannot be resolved meaningfully on a non-Windows
    # host.  Preserve it for configuration inspection; FileTools will reject it
    # if someone tries to use it on the wrong operating system.
    if os.name != "nt" and PureWindowsPath(value).is_absolute():
        return path
    if not path.is_absolute():
        path = project_root / path
    return path.resolve(strict=False)


def load_config(
    env: Mapping[str, str] | None = None,
    *,
    dotenv_path: str | Path | None = None,
    project_root: str | Path | None = None,
) -> Config:
    """Load configuration, with process environment taking priority over .env.

    Passing ``env`` makes tests deterministic: that mapping is used instead of
    ``os.environ``.  A missing dotenv file is intentionally harmless.
    """

    root = (
        Path(project_root).resolve(strict=False)
        if project_root is not None
        else Path(__file__).resolve().parents[2]
    )
    env_file = Path(dotenv_path) if dotenv_path is not None else root / ".env"

    values = _read_dotenv(env_file)
    source_env = os.environ if env is None else env
    # Read only this application's allowlisted settings. Credentials and other
    # unrelated process environment values never enter the configuration map.
    for name in _CONFIG_ENV_NAMES:
        if name in source_env:
            values[name] = source_env[name]

    ollama_host = values.get("OLLAMA_HOST", DEFAULT_OLLAMA_HOST).strip()
    ollama_model = values.get("OLLAMA_MODEL", "").strip()
    allowed_root = _configured_path(
        values.get("ALLOWED_ROOT", str(DEFAULT_ALLOWED_ROOT)),
        setting_name="ALLOWED_ROOT",
        project_root=root,
        require_absolute=True,
    )
    audit_log_path = _configured_path(
        values.get("AUDIT_LOG_PATH", str(DEFAULT_AUDIT_LOG_PATH)),
        setting_name="AUDIT_LOG_PATH",
        project_root=root,
        require_absolute=False,
    )
    try:
        audit_log_path.relative_to(root)
    except ValueError as exc:
        raise ConfigError("AUDIT_LOG_PATH must stay within the project directory.") from exc

    return Config(
        ollama_host=ollama_host,
        ollama_model=ollama_model,
        allowed_root=allowed_root,
        audit_log_path=audit_log_path,
        max_read_bytes=_parse_positive_int(
            "MAX_READ_BYTES",
            values.get("MAX_READ_BYTES", ""),
            DEFAULT_MAX_READ_BYTES,
        ),
        max_write_bytes=_parse_positive_int(
            "MAX_WRITE_BYTES",
            values.get("MAX_WRITE_BYTES", ""),
            DEFAULT_MAX_WRITE_BYTES,
        ),
        max_tool_iterations=_parse_positive_int(
            "MAX_TOOL_ITERATIONS",
            values.get("MAX_TOOL_ITERATIONS", ""),
            DEFAULT_MAX_TOOL_ITERATIONS,
        ),
    )


def load_settings(
    env: Mapping[str, str] | None = None,
    *,
    dotenv_path: str | Path | None = None,
    project_root: str | Path | None = None,
) -> Config:
    """Alias for :func:`load_config`."""

    return load_config(
        env,
        dotenv_path=dotenv_path,
        project_root=project_root,
    )


def public_probe_main() -> int:
    """Print only non-secret fields required by the PowerShell launcher."""

    try:
        config = load_config()
    except ConfigError:
        return 2
    print(
        json.dumps(
            {
                "ollama_host": config.ollama_host,
                "model_configured": bool(config.ollama_model),
            },
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - PowerShell launcher path
    raise SystemExit(public_probe_main())
