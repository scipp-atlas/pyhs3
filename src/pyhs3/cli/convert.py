"""Conversion pyhs3 CLI commands: ``convert to-hs3`` and ``convert to-pyhf``.

Converts between pyhf workspaces (HistFactory JSON) and HS3 with
:func:`pyhs3.from_hifa` and :func:`pyhs3.to_hifa`. Each direction is its own
command, so neither guesses the input's format and each takes only the options
that direction uses.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer

from pyhs3.cli._shared import read_spec
from pyhs3.exceptions import HS3Exception
from pyhs3.hifa import from_hifa, hifa_modifier_settings, to_hifa
from pyhs3.workspace import Workspace

app = typer.Typer(
    name="convert",
    help="Convert between pyhf (HistFactory JSON) and HS3 workspaces.",
    no_args_is_help=True,
    rich_markup_mode="markdown",
)

_OUTPUT_HELP = "File to write the converted workspace to. Default: stdout."


def _write(payload: Any, output: Path | None) -> None:
    """Write ``payload`` as JSON to ``output``, or to stdout without one."""
    text = json.dumps(payload, indent=2)
    if output is None:
        typer.echo(text)
    else:
        output.write_text(text + "\n", encoding="utf-8")


def _parse_interpcode(value: str) -> tuple[str, str]:
    """Parse a ``TYPE=CODE`` interpolation setting.

    Raises:
        typer.BadParameter: *value* is not ``TYPE=CODE``.
    """
    modifier_type, sep, code = value.partition("=")
    if not sep or not modifier_type or not code:
        msg = f"expected 'TYPE=CODE' (e.g. normsys=code1), got {value!r}"
        raise typer.BadParameter(msg)
    return modifier_type, code


@app.command("to-hs3")
def to_hs3(
    workspace: Annotated[
        str | None,
        typer.Argument(
            metavar="INPUT",
            help="Path to a pyhf workspace JSON file. Use '-' or omit to read from stdin.",
        ),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", dir_okay=False, help=_OUTPUT_HELP),
    ] = None,
    interpcode: Annotated[
        list[str] | None,
        typer.Option(
            "--interpcode",
            metavar="TYPE=CODE",
            help=(
                "Interpolation code of a modifier type, as pyhf's "
                "modifier_settings give it, e.g. normsys=code1. Repeatable. "
                "Default: normsys=code4 and histosys=code4p."
            ),
        ),
    ] = None,
) -> None:
    """Convert a pyhf workspace (HistFactory JSON) to an HS3 workspace.

    The input must already have any patches applied. An input that is not a
    valid pyhf workspace, or that HS3 cannot express, is reported as an error.
    """
    settings: dict[str, dict[str, str]] = {}
    for item in interpcode or []:
        modifier_type, code = _parse_interpcode(item)
        settings[modifier_type] = {"interpcode": code}
    try:
        ws = from_hifa(read_spec(workspace), modifier_settings=settings)
    except (ValueError, HS3Exception) as exc:
        # Invalid JSON, a pyhf schema error and a model HS3 cannot express all
        # land here (json and pydantic errors are ValueErrors).
        raise typer.BadParameter(str(exc)) from exc
    _write(ws.model_dump(mode="json", exclude_none=True), output)


@app.command("to-pyhf")
def to_pyhf(
    workspace: Annotated[
        str | None,
        typer.Argument(
            metavar="INPUT",
            help="Path to an HS3 workspace JSON file. Use '-' or omit to read from stdin.",
        ),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", dir_okay=False, help=_OUTPUT_HELP),
    ] = None,
    likelihood: Annotated[
        str | None,
        typer.Option(
            "--likelihood",
            help="Name of the likelihood to convert. Required when the likelihoods pair the channels with different data.",
        ),
    ] = None,
    poi: Annotated[
        str | None,
        typer.Option(
            "--poi",
            help="Parameter of interest, for an analysis listing several or none.",
        ),
    ] = None,
    parameter_set: Annotated[
        str | None,
        typer.Option(
            "--parameter-set",
            help="Parameter point to take initial values and const flags from, instead of each analysis's init.",
        ),
    ] = None,
    modifier_settings: Annotated[
        Path | None,
        typer.Option(
            "--modifier-settings",
            dir_okay=False,
            help="File to write the modifier_settings to. Default: print them to stderr.",
        ),
    ] = None,
) -> None:
    """Convert an HS3 workspace of HistFactory channels to a pyhf workspace.

    pyhf's JSON holds no interpolation codes, so the command also gives the
    ``modifier_settings`` to build the pyhf model with,
    ``pyhf.Workspace(spec).model(modifier_settings=...)``: written to
    ``--modifier-settings``, or printed to stderr so stdout stays the
    workspace. Content pyhf cannot express is reported as an error.
    """
    try:
        ws = Workspace(**read_spec(workspace))
        spec = to_hifa(ws, likelihood=likelihood, poi=poi, parameter_set=parameter_set)
        settings = hifa_modifier_settings(ws, likelihood=likelihood)
    except (ValueError, HS3Exception) as exc:
        raise typer.BadParameter(str(exc)) from exc
    _write(spec, output)
    if modifier_settings is None:
        typer.echo(f"modifier_settings: {json.dumps(settings)}", err=True)
    else:
        _write(settings, modifier_settings)
