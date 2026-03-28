"""Configuration file loading, validation, and migration for TokLog proxy."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict


# Built-in defaults for all config values
_DEFAULTS: Dict[str, Any] = {
    "proxy": {
        "port": 4007,
        "host": "127.0.0.1",
        "enabled": True,
        "budget_usd": None,  # daily spend limit in USD, None = no enforcement
    },
    "logging": {
        "retention_days": 30,
        "max_file_size_mb": 100,
        "enabled": True,
    },
    "defaults": {
        "tags": {},
    },
    "skip_processes": [],
    "features": {
        "anomaly_detection": True,
        "daily_digest": True,
        "auto_start": True,
    },
    "pricing_overrides": {},
}


def get_config_path() -> Path:
    """Return the path to the config file."""
    return Path.home() / ".toklog" / "config.json"


def load_config(
    config_path: Path | None = None,
    validate: bool = True,
) -> Dict[str, Any]:
    """Load config from file or return defaults.

    Args:
        config_path: Path to config.json. If None, uses ~/.toklog/config.json
        validate: If True, validate schema after loading

    Returns:
        Dict with merged config (file overrides defaults)

    Raises:
        ValueError: If config file is invalid JSON or fails schema validation
    """
    path = config_path or get_config_path()

    # Start with defaults
    config = copy.deepcopy(_DEFAULTS)

    # Merge from file if exists
    if path.exists():
        try:
            with open(path, encoding="utf-8") as f:
                file_config = json.load(f)
            config = _deep_merge(config, file_config)
        except json.JSONDecodeError as e:
            raise ValueError(f"Config file {path} is not valid JSON: {e}")

    # Validate schema
    if validate:
        validate_config(config)

    return config


def validate_config(cfg: Dict[str, Any]) -> None:
    """Validate config schema.

    Raises:
        ValueError: If config structure is invalid
    """
    # Check proxy settings
    proxy = cfg.get("proxy", {})
    if not isinstance(proxy.get("port"), int):
        raise ValueError("proxy.port must be an integer")
    port = proxy["port"]
    if not (1 <= port <= 65535):
        raise ValueError(f"proxy.port must be 1-65535, got {port}")

    # Check logging settings
    logging_cfg = cfg.get("logging", {})
    if not isinstance(logging_cfg.get("retention_days"), int):
        raise ValueError("logging.retention_days must be an integer")
    if logging_cfg.get("retention_days", 30) < 1:
        raise ValueError("logging.retention_days must be at least 1")

    # Check skip_processes is a list of strings
    skip = cfg.get("skip_processes", [])
    if not isinstance(skip, list):
        raise ValueError("skip_processes must be a list")
    if not all(isinstance(p, str) for p in skip):
        raise ValueError("skip_processes must contain only strings")

    # Check features are booleans
    features = cfg.get("features", {})
    if not all(isinstance(v, bool) for v in features.values()):
        raise ValueError("All feature flags must be boolean")


def create_config_interactive(
    port_override: int | None = None,
    config_path: Path | None = None,
) -> Dict[str, Any]:
    """Create config interactively (used by setup wizard).

    Args:
        port_override: If provided, skip asking for port
        config_path: Where to write config. If None, uses ~/.toklog/config.json

    Returns:
        The created config dict
    """
    import click

    config = copy.deepcopy(_DEFAULTS)
    path = config_path or get_config_path()

    # Ask for port if not overridden
    if port_override is not None:
        config["proxy"]["port"] = port_override
    else:
        port = click.prompt(
            "Proxy port",
            type=int,
            default=config["proxy"]["port"],
        )
        config["proxy"]["port"] = port

    # Ask about auto-start
    auto_start = click.confirm("Enable auto-start on reboot?", default=True)
    config["features"]["auto_start"] = auto_start

    # Ask about log retention
    retention = click.prompt(
        "Keep logs for (days)",
        type=int,
        default=config["logging"]["retention_days"],
    )
    config["logging"]["retention_days"] = retention

    # Ask about anomaly detection
    anomalies = click.confirm(
        "Enable anomaly detection (background monitoring)?",
        default=True,
    )
    config["features"]["anomaly_detection"] = anomalies

    # Ask about daily digest
    digest = click.confirm(
        "Enable daily digest email (future feature)?",
        default=True,
    )
    config["features"]["daily_digest"] = digest

    # Write config
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    return config


def migrate_from_skip_processes_file(
    skip_file_path: Path | None = None,
    config_path: Path | None = None,
) -> bool:
    """Migrate old skip_processes file to config.json.

    If config.json doesn't exist but skip_processes does:
    - Create config.json with migrated patterns
    - Rename skip_processes to skip_processes.backup

    Args:
        skip_file_path: Path to old skip_processes file. If None, uses ~/.toklog/skip_processes
        config_path: Where to write config. If None, uses ~/.toklog/config.json

    Returns:
        True if migration occurred, False if nothing to do
    """
    skip_file = skip_file_path or Path.home() / ".toklog" / "skip_processes"
    config_file = config_path or get_config_path()

    # Nothing to do if config already exists
    if config_file.exists():
        return False

    # Nothing to do if old file doesn't exist
    if not skip_file.exists():
        return False

    # Load skip patterns from old file
    patterns = [
        line.strip().lower()
        for line in skip_file.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]

    # Create config with migrated patterns
    config = copy.deepcopy(_DEFAULTS)
    config["skip_processes"] = patterns

    # Write config file
    config_file.parent.mkdir(parents=True, exist_ok=True)
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    # Backup old file
    skip_file.rename(skip_file.with_stem(skip_file.stem + ".backup"))

    return True


def _deep_merge(base: Dict, override: Dict) -> Dict:
    """Recursively merge override dict into base dict."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result
