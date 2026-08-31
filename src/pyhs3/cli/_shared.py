"""Shared I/O and formatting helpers for the pyhs3 command-line interface.

Centralizes the pieces every subcommand needs: reading a workspace spec from a
file path or stdin (so ``curl ... | pyhs3 validate -`` works), detecting whether
stdout is an interactive terminal (to choose table vs. JSON output), and parsing
``name=value`` parameter overrides.
"""

from __future__ import annotations

import json
import math
import os
import stat
import sys
from pathlib import Path
from typing import Any

import typer

from pyhs3.workspace import Workspace

#: Sentinel path value meaning "read from standard input".
STDIN_MARKER = "-"


def stdout_is_interactive() -> bool:
    """Return True iff stdout is a TTY (not a pipe or regular file).

    Uses ``fstat`` to distinguish FIFOs and regular files from character
    devices, falling back to :func:`sys.stdout.isatty` when ``fstat`` fails
    (e.g. test harnesses that replace ``sys.stdout`` with an in-memory buffer).
    """
    try:
        mode = os.fstat(sys.stdout.fileno()).st_mode
    except (OSError, ValueError, AttributeError):
        return sys.stdout.isatty()
    if stat.S_ISFIFO(mode) or stat.S_ISREG(mode):
        return False
    return sys.stdout.isatty()


def _stdin_is_tty() -> bool:
    """Return True iff stdin is an interactive terminal (nothing piped in)."""
    try:
        return sys.stdin.isatty()
    except (OSError, ValueError, AttributeError):
        return False


def read_spec(path: str | None) -> dict[str, Any]:
    """Read a workspace JSON spec from *path*, ``-``, or stdin.

    A ``path`` of ``None`` or ``"-"`` reads from stdin; when stdin is an
    interactive terminal (nothing piped) that is an error, since the command
    would otherwise block waiting for input that never comes.

    Args:
        path: File path to read, ``"-"``/``None`` for stdin.

    Returns:
        The parsed JSON object as a dict.

    Raises:
        typer.BadParameter: stdin was requested but is an interactive terminal,
            or the parsed JSON's root is not an object.
        json.JSONDecodeError: the input is not valid JSON.
    """
    if path is None or path == STDIN_MARKER:
        if _stdin_is_tty():
            msg = "no workspace path given and stdin is a terminal; pass a file path or pipe JSON in"
            raise typer.BadParameter(msg)
        text = sys.stdin.read()
    else:
        text = Path(path).read_text(encoding="utf-8")
    spec = json.loads(text)
    if not isinstance(spec, dict):
        msg = f"expected a JSON object at the workspace root, got {type(spec).__name__}"
        raise typer.BadParameter(msg)
    return spec


def load_workspace(path: str | None) -> Workspace:
    """Read and validate a workspace from *path* or stdin.

    Args:
        path: File path, ``"-"``/``None`` for stdin.

    Returns:
        The validated :class:`~pyhs3.workspace.Workspace`.

    Raises:
        json.JSONDecodeError: the input is not valid JSON.
        pydantic.ValidationError: a field fails schema validation.
        pyhs3.exceptions.WorkspaceValidationError: a foreign-key reference
            cannot be resolved.
    """
    spec = read_spec(path)
    return Workspace(**spec)


def display_name(path: str | None) -> str:
    """Human-readable source name for error messages."""
    if path is None or path == STDIN_MARKER:
        return "<stdin>"
    return path


def parse_param(value: str) -> tuple[str, float]:
    """Parse a ``name=value`` override into a ``(name, float)`` pair.

    Raises:
        typer.BadParameter: *value* is missing the ``=`` separator or the
            right-hand side is not a float.
    """
    name, sep, raw = value.partition("=")
    if not sep or not name:
        msg = f"expected 'name=value', got {value!r}"
        raise typer.BadParameter(msg)
    try:
        return name, float(raw)
    except ValueError as exc:
        msg = f"parameter {name!r} value {raw!r} is not a number"
        raise typer.BadParameter(msg) from exc


def finite_or_none(value: float) -> float | None:
    """Map non-finite floats to ``None`` so the result is valid JSON.

    ``json.dumps`` would otherwise emit ``Infinity``/``NaN``, which are not
    valid JSON and break downstream parsers.
    """
    return value if math.isfinite(value) else None
