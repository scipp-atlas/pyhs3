"""pyhs3 configures logging like a library: only its own logger, and only on request."""

from __future__ import annotations

import json
import logging
import subprocess
import sys

import pytest
from rich.logging import RichHandler
from typer.testing import CliRunner

import pyhs3.cli
import pyhs3.logging
from pyhs3.cli import app


def _python(code: str) -> str:
    """Run ``code`` in a fresh interpreter and return its stdout."""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        timeout=300,
    )
    return result.stdout.strip()


class TestImportingPyhs3:
    def test_keeps_existing_loggers_enabled(self):
        out = _python(
            "import logging\n"
            "other = logging.getLogger('some.other.library')\n"
            "import pyhs3\n"
            "print(other.disabled)"
        )
        assert out == "False"

    def test_leaves_the_root_logger_alone(self):
        out = _python(
            "import logging\n"
            "import pyhs3\n"
            "root = logging.getLogger()\n"
            "print(len(root.handlers), logging.getLevelName(root.level))"
        )
        assert out == "0 WARNING"

    def test_pyhs3_logger_has_no_handler(self):
        """No handler, not even NullHandler, which would hide warnings."""
        out = _python(
            "import logging\n"
            "import pyhs3\n"
            "print([type(h).__name__ for h in logging.getLogger('pyhs3').handlers])"
        )
        assert out == "[]"

    def test_warnings_reach_stderr_once_without_configuration(self):
        """With no handlers configured, Python's last-resort handler prints warnings."""
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import logging, pyhs3\n"
                    "logging.getLogger('pyhs3.test').warning('careful')"
                ),
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=300,
        )
        assert result.stderr.count("careful") == 1


@pytest.fixture
def restore_pyhs3_logger():
    """Undo setup() on the pyhs3 logger after the test."""
    logger = logging.getLogger("pyhs3")
    handlers, level, propagate = list(logger.handlers), logger.level, logger.propagate
    yield logger
    logger.handlers[:] = handlers
    logger.setLevel(level)
    logger.propagate = propagate


class TestSetup:
    def test_configures_only_the_pyhs3_logger(self, restore_pyhs3_logger):
        other = logging.getLogger("some.other.library")
        root_handlers = list(logging.getLogger().handlers)

        pyhs3.logging.setup()

        assert any(isinstance(h, RichHandler) for h in restore_pyhs3_logger.handlers)
        assert logging.getLogger().handlers == root_handlers
        assert not other.disabled

    def test_is_idempotent(self, restore_pyhs3_logger):
        pyhs3.logging.setup()
        pyhs3.logging.setup()
        rich = [h for h in restore_pyhs3_logger.handlers if isinstance(h, RichHandler)]
        assert len(rich) == 1

    @pytest.mark.usefixtures("restore_pyhs3_logger")
    def test_records_still_reach_root_handlers(self, caplog):
        """Applications (and caplog) listening on the root logger still see pyhs3."""
        pyhs3.logging.setup()
        with caplog.at_level(logging.WARNING):
            logging.getLogger("pyhs3.test").warning("careful")
        assert "careful" in caplog.text

    def test_sets_the_requested_level(self, restore_pyhs3_logger):
        pyhs3.logging.setup(level=logging.DEBUG)
        assert restore_pyhs3_logger.level == logging.DEBUG


class TestCommandLine:
    def test_commands_configure_pyhs3_logging(self, tmp_path, monkeypatch):
        calls = []
        monkeypatch.setattr(pyhs3.cli, "setup", lambda **kw: calls.append(kw))
        path = tmp_path / "workspace.json"
        path.write_text(
            json.dumps({"metadata": {"hs3_version": "0.2"}}), encoding="utf-8"
        )

        result = CliRunner().invoke(app, ["validate", str(path)])

        assert result.exit_code == 0, result.output
        assert calls == [{}]
