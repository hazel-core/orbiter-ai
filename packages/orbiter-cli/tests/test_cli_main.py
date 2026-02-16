"""Tests for orbiter_cli.main — CLI entry point, arg parsing, config loading."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from orbiter_cli.main import (
    CLIError,
    app,
    find_config,
    load_config,
    resolve_config,
)

runner = CliRunner()


# ---------------------------------------------------------------------------
# CLIError
# ---------------------------------------------------------------------------


class TestCLIError:
    def test_is_exception(self) -> None:
        assert issubclass(CLIError, Exception)

    def test_message(self) -> None:
        err = CLIError("bad config")
        assert str(err) == "bad config"


# ---------------------------------------------------------------------------
# find_config
# ---------------------------------------------------------------------------


class TestFindConfig:
    def test_finds_orbiter_yaml(self, tmp_path: Path) -> None:
        (tmp_path / ".orbiter.yaml").write_text("agents: {}")
        result = find_config(tmp_path)
        assert result is not None
        assert result.name == ".orbiter.yaml"

    def test_finds_orbiter_config_yaml(self, tmp_path: Path) -> None:
        (tmp_path / "orbiter.config.yaml").write_text("agents: {}")
        result = find_config(tmp_path)
        assert result is not None
        assert result.name == "orbiter.config.yaml"

    def test_prefers_orbiter_yaml_over_config(self, tmp_path: Path) -> None:
        (tmp_path / ".orbiter.yaml").write_text("a: 1")
        (tmp_path / "orbiter.config.yaml").write_text("b: 2")
        result = find_config(tmp_path)
        assert result is not None
        assert result.name == ".orbiter.yaml"

    def test_returns_none_when_no_config(self, tmp_path: Path) -> None:
        assert find_config(tmp_path) is None

    def test_defaults_to_cwd(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".orbiter.yaml").write_text("x: 1")
        result = find_config()
        assert result is not None
        assert result.name == ".orbiter.yaml"


# ---------------------------------------------------------------------------
# load_config
# ---------------------------------------------------------------------------


class TestLoadConfig:
    def test_loads_valid_yaml(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "test.yaml"
        cfg_file.write_text("agents:\n  bot:\n    model: openai:gpt-4o\n")
        result = load_config(cfg_file)
        assert "agents" in result
        assert result["agents"]["bot"]["model"] == "openai:gpt-4o"

    def test_raises_on_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(CLIError, match="Config file not found"):
            load_config(tmp_path / "nonexistent.yaml")

    def test_raises_on_invalid_yaml(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "bad.yaml"
        cfg_file.write_text("- just a list")
        with pytest.raises(CLIError, match="Invalid config"):
            load_config(cfg_file)

    def test_variable_substitution(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TEST_MODEL", "openai:gpt-4o")
        cfg_file = tmp_path / "env.yaml"
        cfg_file.write_text("agents:\n  bot:\n    model: ${TEST_MODEL}\n")
        result = load_config(cfg_file)
        assert result["agents"]["bot"]["model"] == "openai:gpt-4o"

    def test_vars_section_substitution(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "vars.yaml"
        cfg_file.write_text("vars:\n  temp: 0.7\nagents:\n  bot:\n    temperature: ${vars.temp}\n")
        result = load_config(cfg_file)
        assert result["agents"]["bot"]["temperature"] == 0.7

    def test_accepts_string_path(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "str.yaml"
        cfg_file.write_text("key: value\n")
        result = load_config(str(cfg_file))
        assert result["key"] == "value"


# ---------------------------------------------------------------------------
# resolve_config
# ---------------------------------------------------------------------------


class TestResolveConfig:
    def test_explicit_path(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "explicit.yaml"
        cfg_file.write_text("agents: {}\n")
        result = resolve_config(str(cfg_file))
        assert result is not None
        assert "agents" in result

    def test_auto_discovery(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".orbiter.yaml").write_text("agents:\n  a:\n    model: test\n")
        result = resolve_config(None)
        assert result is not None
        assert "agents" in result

    def test_no_config_returns_none(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        assert resolve_config(None) is None

    def test_explicit_overrides_discovery(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".orbiter.yaml").write_text("source: auto\n")
        explicit = tmp_path / "custom.yaml"
        explicit.write_text("source: explicit\n")
        result = resolve_config(str(explicit))
        assert result is not None
        assert result["source"] == "explicit"


# ---------------------------------------------------------------------------
# CLI arg parsing — run command
# ---------------------------------------------------------------------------


class TestCLIRun:
    def test_no_args_shows_help(self) -> None:
        result = runner.invoke(app, [])
        # Typer returns exit code 0 or 2 depending on version for no_args_is_help
        assert result.exit_code in (0, 2)
        assert "Usage" in result.output

    def test_run_without_config_exits_1(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["run", "hello"])
        assert result.exit_code == 1
        assert "No config file found" in result.output

    def test_run_with_config(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        cfg = tmp_path / ".orbiter.yaml"
        cfg.write_text("agents:\n  bot:\n    model: test\n")
        result = runner.invoke(app, ["run", "hello"])
        assert result.exit_code == 0
        assert "Running with input:" in result.output

    def test_run_with_explicit_config(self, tmp_path: Path) -> None:
        cfg = tmp_path / "custom.yaml"
        cfg.write_text("agents:\n  bot:\n    model: test\n")
        result = runner.invoke(app, ["run", "--config", str(cfg), "hello"])
        assert result.exit_code == 0

    def test_run_with_model_flag(self, tmp_path: Path) -> None:
        cfg = tmp_path / "m.yaml"
        cfg.write_text("agents: {}\n")
        result = runner.invoke(app, ["run", "-c", str(cfg), "-m", "openai:gpt-4o", "test"])
        assert result.exit_code == 0

    def test_run_with_stream_flag(self, tmp_path: Path) -> None:
        cfg = tmp_path / "s.yaml"
        cfg.write_text("agents: {}\n")
        result = runner.invoke(app, ["run", "-c", str(cfg), "--stream", "test"])
        assert result.exit_code == 0

    def test_run_verbose(self, tmp_path: Path) -> None:
        cfg = tmp_path / "v.yaml"
        cfg.write_text("agents:\n  a:\n    model: x\n")
        result = runner.invoke(app, ["--verbose", "run", "-c", str(cfg), "test"])
        assert result.exit_code == 0
        assert "Loaded config" in result.output

    def test_run_invalid_config_path(self) -> None:
        result = runner.invoke(app, ["run", "-c", "/nonexistent/path.yaml", "hi"])
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# CLI arg parsing — help text
# ---------------------------------------------------------------------------


class TestCLIHelp:
    def test_help_flag(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "multi-agent" in result.output.lower() or "orbiter" in result.output.lower()

    def test_run_help(self) -> None:
        result = runner.invoke(app, ["run", "--help"])
        assert result.exit_code == 0
        assert "--config" in result.output
        assert "--model" in result.output
        assert "--stream" in result.output


# ---------------------------------------------------------------------------
# Config search order
# ---------------------------------------------------------------------------


class TestConfigPrecedence:
    def test_orbiter_yaml_found_first(self, tmp_path: Path) -> None:
        (tmp_path / ".orbiter.yaml").write_text("first: true\n")
        (tmp_path / "orbiter.config.yaml").write_text("second: true\n")
        result = find_config(tmp_path)
        assert result is not None
        config = load_config(result)
        assert config.get("first") is True

    def test_orbiter_config_yaml_fallback(self, tmp_path: Path) -> None:
        (tmp_path / "orbiter.config.yaml").write_text("fallback: true\n")
        result = find_config(tmp_path)
        assert result is not None
        config = load_config(result)
        assert config.get("fallback") is True

    def test_ignores_non_config_files(self, tmp_path: Path) -> None:
        (tmp_path / "random.yaml").write_text("ignored: true\n")
        assert find_config(tmp_path) is None


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_yaml_dict(self, tmp_path: Path) -> None:
        cfg = tmp_path / "empty.yaml"
        cfg.write_text("{}\n")
        result = load_config(cfg)
        assert result == {}

    def test_config_with_nested_vars(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OUTER", "hello")
        cfg = tmp_path / "nested.yaml"
        cfg.write_text("vars:\n  inner: world\ngreeting: ${OUTER} ${vars.inner}\n")
        result = load_config(cfg)
        assert result["greeting"] == "hello world"

    def test_find_config_with_path_object(self, tmp_path: Path) -> None:
        (tmp_path / ".orbiter.yaml").write_text("ok: true\n")
        result = find_config(Path(tmp_path))
        assert result is not None
