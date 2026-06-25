#!/usr/bin/env python3
# ruff: noqa: T201
"""Plot pyhs3 scan NLL vs quickFit reference NLL on the same canvas.

Reads scan results from runs/latest/scan_summary.json (omitting the last two
points) and the reference NLL values from examples/minimization_dihiggs.py
(extracted with ast.parse so pyhs3/pytensor need not be imported).

Reference NLL values are multiplied by 2 before plotting, since the quickFit
values are -log(L) while pyhs3 scans 2*(-log(L)/N) or similar — multiply by 2
to bring them to the same scale for visual comparison.

Both series are shifted so their minimum is 0 (ΔL plot).

Usage
-----
    python plot_scan.py                              # runs/latest → nll_comparison_scan.pdf
    python plot_scan.py --scan PATH/scan_summary.json --output out.pdf
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

_DEFAULT_SCAN = Path("runs/latest/scan_summary.json")
_DEFAULT_SCRIPT = (
    Path(__file__).resolve().parent / "examples" / "minimization_dihiggs.py"
)


def extract_reference(script_path: Path) -> dict[str, list[float]]:
    """Parse _REFERENCE from minimization_dihiggs.py without importing it."""
    source = script_path.read_text()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "_REFERENCE":
                    return ast.literal_eval(node.value)
    msg = f"_REFERENCE not found in {script_path}"
    raise ValueError(msg)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--scan",
        type=Path,
        default=_DEFAULT_SCAN,
        help=f"Path to scan_summary.json (default: {_DEFAULT_SCAN})",
    )
    parser.add_argument(
        "--script",
        type=Path,
        default=_DEFAULT_SCRIPT,
        help=f"Path to minimization_dihiggs.py containing _REFERENCE "
        f"(default: {_DEFAULT_SCRIPT})",
    )
    parser.add_argument(
        "--omit-last",
        type=int,
        default=0,
        metavar="N",
        help="Omit the last N scan points (default: 0)",
    )
    parser.add_argument(
        "--ref-scale",
        type=float,
        default=2.0,
        help="Multiply reference NLL values by this factor (default: 2.0)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output PDF path (default: same directory as --scan, nll_comparison_scan.pdf)",
    )
    args = parser.parse_args()

    # --- load scan results ---
    scan_path = args.scan.resolve()
    if not scan_path.exists():
        msg = f"ERROR: scan file not found: {scan_path}"
        raise SystemExit(msg)

    with scan_path.open() as f:
        scan_data = json.load(f)

    if args.omit_last > 0:
        scan_data = scan_data[: -args.omit_last]
        print(f"Using {len(scan_data)} scan points (omitted last {args.omit_last}).")
    else:
        print(f"Using all {len(scan_data)} scan points.")

    scan_mu = np.array([pt["mu"] for pt in scan_data])
    scan_nll = np.array([pt["nll"] for pt in scan_data])

    # --- load reference ---
    ref = extract_reference(args.script)
    ref_mu = np.array(ref["mu_HH"])
    ref_nll = np.array(ref["nll"]) * args.ref_scale
    print(f"Reference: {len(ref_mu)} points, scaled by x{args.ref_scale}.")

    # --- shift both to minimum = 0 ---
    scan_nll_shifted = scan_nll - scan_nll.min()
    ref_nll_shifted = ref_nll - ref_nll.min()

    # --- plot ---
    plt.figure()
    plt.scatter(
        ref_mu, ref_nll_shifted, label=f"reference NLL (x{args.ref_scale})", marker="o"
    )
    plt.scatter(scan_mu, scan_nll_shifted, label="pyhs3 scan NLL", marker="x")
    plt.xlabel("mu_HH")
    plt.ylabel(r"$-2\,\Delta(\ln\Lambda)$")
    plt.legend()

    if args.output is None:
        out_path = scan_path.parent / "nll_comparison_scan.pdf"
    else:
        out_path = args.output.resolve()

    plt.savefig(out_path)
    print(f"\nPlot saved to {out_path}")


if __name__ == "__main__":
    main()
