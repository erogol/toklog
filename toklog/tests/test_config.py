"""Tests for toklog config module."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from toklog.config import (
    _DEFAULTS,
    get_config_path,
    load_config,
    migrate_from_skip_processes_file,
    validate_config,
)


class TestLoadConfig:
    """Tests for load_config function."""

    def test_load_config_with_defaults(self):
        """If no config file, return defaults."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            config = load_config(config_path=config_path, validate=False)
            assert config["proxy"]["port"] == 4007
            assert config["logging"]["retention_days"] == 30
            assert config["skip_processes"] == []

    def test_load_config_from_file(self):
        """Load values from file, merge with defaults."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            file_config = {"proxy": {"port": 5000}}
            config_path.write_text(json.dumps(file_config))

            config = load_config(config_path=config_path, validate=False)
            assert config["proxy"]["port"] == 5000
            # Other defaults should still be present
            assert config["logging"]["retention_days"] == 30

    def test_load_config_invalid_json(self):
        """Invalid JSON raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            config_path.write_text("{ invalid json")

            with pytest.raises(ValueError, match="not valid JSON"):
                load_config(config_path=config_path)

    def test_load_config_skip_validation(self):
        """With validate=False, don't check schema."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            # Bad port value, but validate=False
            bad_config = {"proxy": {"port": "not_an_int"}}
            config_path.write_text(json.dumps(bad_config))

            # Should not raise
            config = load_config(config_path=config_path, validate=False)
            assert config["proxy"]["port"] == "not_an_int"

    def test_load_config_with_validation(self):
        """With validate=True, check schema."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            bad_config = {"proxy": {"port": "not_an_int"}}
            config_path.write_text(json.dumps(bad_config))

            with pytest.raises(ValueError, match="must be an integer"):
                load_config(config_path=config_path, validate=True)

    def test_get_config_path(self):
        """get_config_path returns expected path."""
        path = get_config_path()
        assert path == Path.home() / ".toklog" / "config.json"


class TestValidateConfig:
    """Tests for validate_config function."""

    def test_valid_config(self):
        """Valid config passes validation."""
        # Should not raise
        validate_config(_DEFAULTS)

    def test_invalid_port_type(self):
        """Port must be integer."""
        config = json.loads(json.dumps(_DEFAULTS))
        config["proxy"]["port"] = "4007"
        with pytest.raises(ValueError, match="must be an integer"):
            validate_config(config)

    def test_invalid_port_range_low(self):
        """Port must be >= 1."""
        config = json.loads(json.dumps(_DEFAULTS))
        config["proxy"]["port"] = 0
        with pytest.raises(ValueError, match="1-65535"):
            validate_config(config)

    def test_invalid_port_range_high(self):
        """Port must be <= 65535."""
        config = json.loads(json.dumps(_DEFAULTS))
        config["proxy"]["port"] = 99999
        with pytest.raises(ValueError, match="1-65535"):
            validate_config(config)

    def test_invalid_retention_type(self):
        """retention_days must be integer."""
        config = json.loads(json.dumps(_DEFAULTS))
        config["logging"]["retention_days"] = "30"
        with pytest.raises(ValueError, match="must be an integer"):
            validate_config(config)

    def test_invalid_retention_value(self):
        """retention_days must be >= 1."""
        config = json.loads(json.dumps(_DEFAULTS))
        config["logging"]["retention_days"] = 0
        with pytest.raises(ValueError, match="at least 1"):
            validate_config(config)

    def test_invalid_skip_processes_type(self):
        """skip_processes must be list."""
        config = json.loads(json.dumps(_DEFAULTS))
        config["skip_processes"] = "grep"
        with pytest.raises(ValueError, match="must be a list"):
            validate_config(config)

    def test_invalid_skip_processes_items(self):
        """skip_processes items must be strings."""
        config = json.loads(json.dumps(_DEFAULTS))
        config["skip_processes"] = ["grep", 123]
        with pytest.raises(ValueError, match="only strings"):
            validate_config(config)

    def test_invalid_feature_flags(self):
        """Feature flags must be booleans."""
        config = json.loads(json.dumps(_DEFAULTS))
        config["features"]["anomaly_detection"] = "yes"
        with pytest.raises(ValueError, match="must be boolean"):
            validate_config(config)


class TestMigrateSkipProcesses:
    """Tests for migrate_from_skip_processes_file function."""

    def test_migrate_when_old_file_exists(self):
        """Migrate patterns from old skip_processes file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            skip_file = tmpdir / "skip_processes"
            config_file = tmpdir / "config.json"

            # Create old skip_processes file
            skip_file.write_text("grep\nsed\n# comment\ncat\n")

            result = migrate_from_skip_processes_file(
                skip_file_path=skip_file,
                config_path=config_file,
            )

            assert result is True
            assert config_file.exists()

            # Check config was created with patterns
            config = json.loads(config_file.read_text())
            assert "grep" in config["skip_processes"]
            assert "sed" in config["skip_processes"]
            assert "cat" in config["skip_processes"]
            assert "comment" not in config["skip_processes"]  # Comments ignored

    def test_migrate_creates_backup(self):
        """Old file renamed to .backup."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            skip_file = tmpdir / "skip_processes"
            config_file = tmpdir / "config.json"

            skip_file.write_text("grep\nsed\n")

            migrate_from_skip_processes_file(
                skip_file_path=skip_file,
                config_path=config_file,
            )

            assert not skip_file.exists()
            assert (tmpdir / "skip_processes.backup").exists()

    def test_migrate_skip_if_config_exists(self):
        """Don't migrate if config.json already exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            skip_file = tmpdir / "skip_processes"
            config_file = tmpdir / "config.json"

            skip_file.write_text("grep\n")
            config_file.write_text("{}")

            result = migrate_from_skip_processes_file(
                skip_file_path=skip_file,
                config_path=config_file,
            )

            assert result is False
            assert skip_file.exists()  # Not renamed

    def test_migrate_skip_if_old_file_missing(self):
        """Don't migrate if skip_processes file doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            skip_file = tmpdir / "skip_processes"
            config_file = tmpdir / "config.json"

            result = migrate_from_skip_processes_file(
                skip_file_path=skip_file,
                config_path=config_file,
            )

            assert result is False
            assert not config_file.exists()

    def test_migrate_handles_whitespace(self):
        """Migration trims whitespace and lowercases patterns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            skip_file = tmpdir / "skip_processes"
            config_file = tmpdir / "config.json"

            skip_file.write_text("  GREP  \n  SED  \n")

            migrate_from_skip_processes_file(
                skip_file_path=skip_file,
                config_path=config_file,
            )

            config = json.loads(config_file.read_text())
            assert "grep" in config["skip_processes"]
            assert "sed" in config["skip_processes"]
            assert "GREP" not in config["skip_processes"]  # Lowercased

    def test_migrate_skip_empty_lines(self):
        """Migration skips empty lines."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            skip_file = tmpdir / "skip_processes"
            config_file = tmpdir / "config.json"

            skip_file.write_text("grep\n\n\nsed\n  \n")

            migrate_from_skip_processes_file(
                skip_file_path=skip_file,
                config_path=config_file,
            )

            config = json.loads(config_file.read_text())
            assert len(config["skip_processes"]) == 2
            assert config["skip_processes"] == ["grep", "sed"]