# ruff: noqa: T201
"""Plot pyhs3 vs quickFit NLL scan curves into a single multi-panel PDF.

Each workspace gets its own axis showing two ΔNLL curves -- one from pyhs3 and
one from quickFit -- with both curves shifted so their minimum is zero, so the
shape (delta NLL) can be compared directly regardless of any constant offset.

Consumed by ``eval_simple_muscan.py --plot-nlls``; the ``plot_nll_curves`` function
takes the result dicts returned by ``run_scan`` (each containing ``mus``,
``qf_nlls``, ``pyhs3_nlls`` and ``workspace``).
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import numpy as np
from matplotlib import pyplot as plt

_CELL_W, _CELL_H = 5, 4  # inches per panel


def _delta(nlls: list[float]) -> np.ndarray:
    """Shift an NLL curve so its minimum is zero."""
    arr = np.asarray(nlls, dtype=np.float64)
    return arr - np.nanmin(arr)


def plot_nll_curves(results: list[dict], output_pdf: Path) -> Path:
    """Render one ΔNLL panel per result into *output_pdf*.

    Each result needs ``mus``, ``qf_nlls``, ``pyhs3_nlls`` and ``workspace``.
    """
    n = len(results)
    ncols = 1 if n == 1 else 2 
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(ncols * _CELL_W, nrows * _CELL_H),
        constrained_layout=True,
        squeeze=False,
    )
    flat_axes = axes.flat

    for ax, result in zip(flat_axes, results, strict=False):
        mus = np.asarray(result["mus"], dtype=np.float64)
        qf_delta = _delta(result["qf_nlls"])
        pyhs3_delta = _delta(result["pyhs3_nlls"])
        ws_name = result["workspace"].name

        ax.plot(
            mus,
            qf_delta,
            color="#d73027",
            marker="o",
            markersize=3,
            linewidth=1.2,
            label="quickFit",
        )
        ax.plot(
            mus,
            pyhs3_delta,
            color="#4575b4",
            marker="s",
            markersize=3,
            linewidth=1.2,
            label="pyhs3",
        )

        ax.set_xlabel(r"$\mu_{sig}$")
        ax.set_ylabel(r"$\Delta(-\ln L)$")
        ax.set_title(ws_name)
        ax.grid(True, alpha=0.25)
        ax.legend()

    # hide any unused panels in the grid
    for ax in list(flat_axes)[n:]:
        ax.set_visible(False)

    fig.suptitle("pyhs3 vs quickFit ΔNLL (min shifted to zero)")
    output_pdf = Path(output_pdf)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf)
    plt.close(fig)
    print(f"Wrote NLL curve plot to {output_pdf}")
    return output_pdf
