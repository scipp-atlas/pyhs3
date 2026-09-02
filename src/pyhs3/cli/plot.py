"""Plot pyhs3 CLI command: render a workspace's observed data as a figure.

Follows the same matplotlib idioms already documented in
``docs/visualization.rst``: a ``hist.Hist.plot()`` bar/step histogram for a 1D
:class:`~pyhs3.data.BinnedData` entry, ``pcolormesh`` for a 2D one, and a
binned histogram of raw entries (via :meth:`~pyhs3.data.UnbinnedData.to_hist`)
for :class:`~pyhs3.data.UnbinnedData`. Requires the ``plot`` optional extra
(``pip install 'pyhs3[plot]'``), since matplotlib is not a core dependency.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import typer

from pyhs3.cli._shared import load_workspace
from pyhs3.data import BinnedData, Data, Datum, UnbinnedData

if TYPE_CHECKING:
    from matplotlib.figure import Figure

#: Output formats matplotlib's ``savefig`` can write directly.
_SUPPORTED_FORMATS = ("png", "pdf", "svg")


def _require_matplotlib_pyplot() -> Any:
    """Import ``matplotlib.pyplot`` on the non-interactive Agg backend.

    The backend must be selected before ``pyplot`` is imported, matching how
    ``docs/conf.py`` configures the Sphinx ``.. plot::`` directive, so this
    works headlessly (no display) in CI and other server environments.

    Raises:
        typer.BadParameter: matplotlib is not installed.
    """
    try:
        import matplotlib as mpl  # noqa: PLC0415  # pylint: disable=import-outside-toplevel

        mpl.use("Agg")
        import matplotlib.pyplot as plt  # noqa: PLC0415  # pylint: disable=import-outside-toplevel
    except ImportError as exc:
        msg = (
            "pyhs3 plot requires matplotlib: install it with "
            "`pip install 'pyhs3[plot]'`"
        )
        raise typer.BadParameter(msg) from exc
    return plt


def _plot_binned(fig: Figure, data: BinnedData) -> None:
    """Draw *data* onto *fig*: bar/step histogram (1D) or heatmap (2D)."""
    ndim = len(data.axes)
    h = data.to_hist()
    if ndim == 1:
        ax = fig.add_subplot()
        h.plot(ax=ax, histtype="fill", alpha=0.7, label=data.name)
        ax.set_xlabel(data.axes[0].name)
        ax.set_ylabel("Events")
        ax.legend()
    elif ndim == 2:
        ax = fig.add_subplot()
        # h.axes.edges has shape (naxes, nbins+1, 1); .T aligns it with
        # pcolormesh's (x_edges, y_edges) signature. h.values() is (nx, ny),
        # transposed to (ny, nx) to match pcolormesh's (row=y, col=x) convention.
        mesh = ax.pcolormesh(*h.axes.edges.T, h.values().T, cmap="viridis")
        fig.colorbar(mesh, ax=ax, label="Events")
        ax.set_xlabel(data.axes[0].name)
        ax.set_ylabel(data.axes[1].name)
    else:
        msg = (
            f"plotting binned data with {ndim} axes is not yet supported "
            "(only 1D and 2D)"
        )
        raise typer.BadParameter(msg)
    ax.set_title(data.name)


def _plot_unbinned(fig: Figure, data: UnbinnedData) -> None:
    """Draw *data* onto *fig*: a binned histogram of its raw entries (1D only)."""
    ndim = len(data.axes)
    if ndim != 1:
        msg = f"plotting unbinned data with {ndim} axes is not yet supported (only 1D)"
        raise typer.BadParameter(msg)
    h = data.to_hist(nbins=50)
    ax = fig.add_subplot()
    h.plot(ax=ax, histtype="fill", alpha=0.6, label=data.name)
    ax.set_xlabel(data.axes[0].name)
    ax.set_ylabel("Entries")
    ax.legend()
    ax.set_title(data.name)


def plot(
    workspace: Annotated[
        str | None,
        typer.Argument(
            metavar="WORKSPACE",
            help="Path to an HS3 workspace JSON file. Use '-' or omit to read from stdin.",
        ),
    ] = None,
    *,
    data_name: Annotated[
        str,
        typer.Option(
            "--data-name",
            help="Name of the entry in the workspace's data list to plot.",
        ),
    ],
    outfile: Annotated[
        Path | None,
        typer.Option(
            "--outfile",
            help="Output file path. Defaults to '{data-name}.{fmt}' in the current directory.",
        ),
    ] = None,
    fmt: Annotated[
        str,
        typer.Option("--fmt", help="Output image format: 'png', 'pdf', or 'svg'."),
    ] = "png",
) -> None:
    """Render one named data entry from a workspace as a matplotlib figure.

    Requires the ``plot`` extra (``pip install 'pyhs3[plot]'``), which
    installs ``hist[plot]`` -- matplotlib **and** ``mplhep``, since plotting a
    ``hist.Hist`` needs both; matplotlib alone is not enough. If matplotlib
    itself is missing, this reports a clean error rather than a raw
    traceback; if matplotlib is present but ``mplhep`` specifically is
    missing (an unusual partial install), plotting can still raise a raw
    ``ModuleNotFoundError`` -- installing the ``plot`` extra as documented
    avoids this case entirely.

    Only 1D ``BinnedData``/``UnbinnedData`` and 2D ``BinnedData`` are
    supported; other shapes raise a clear error rather than a misleading plot.
    """
    if fmt not in _SUPPORTED_FORMATS:
        msg = (
            f"unsupported --fmt {fmt!r}; choose one of {', '.join(_SUPPORTED_FORMATS)}"
        )
        raise typer.BadParameter(msg)

    plt = _require_matplotlib_pyplot()

    ws = load_workspace(workspace)
    data_collection = ws.data if ws.data is not None else Data([])
    if data_name not in data_collection:
        names = ", ".join(datum.name for datum in data_collection) or "<none>"
        msg = f"no data named {data_name!r} in the workspace (available: {names})"
        raise typer.BadParameter(msg)

    datum: Datum = data_collection[data_name]
    destination = outfile if outfile is not None else Path(f"{data_name}.{fmt}")

    fig = plt.figure()
    try:
        if isinstance(datum, BinnedData):
            _plot_binned(fig, datum)
        elif isinstance(datum, UnbinnedData):
            _plot_unbinned(fig, datum)
        else:
            msg = f"plotting data of type {type(datum).__name__!r} is not yet supported"
            raise typer.BadParameter(msg)
        fig.savefig(destination, format=fmt)
    finally:
        plt.close(fig)

    typer.echo(str(destination))
