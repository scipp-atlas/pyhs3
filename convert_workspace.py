#!/usr/bin/env python3
# ruff: noqa: T201
"""Strip dummy near-zero-weight events from combData_* entries in a pyhs3 workspace.

The original workspace file stores unbinned datasets using a workaround: real
events have weight 1.0 and a large number of dummy bin-centre events have weight
~1e-9.  This causes pyhs3 to use the normalized weighted likelihood

    Σᵢ wᵢ log p(xᵢ | θ) / Σᵢ wᵢ

instead of the standard unbinned likelihood

    Σᵢ log p(xᵢ | θ)

This script produces a clean copy of the JSON where every combData_* entry
contains only the real events and has no weights field, so pyhs3 uses the
standard unweighted code path — matching what ROOT/RooFit computes.

Usage
-----
    python convert_workspace.py                         # uses built-in defaults
    python convert_workspace.py INPUT.json OUTPUT.json
    python convert_workspace.py --threshold 0.5 INPUT.json OUTPUT.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_DEFAULT_INPUT = (
    Path(__file__).resolve().parent
    / "data"
    / "test_hs3_unbinned_pyhs3_validation_issue41.json"
)
_WEIGHT_THRESHOLD = (
    0.5  # separates real events (weight 1.0) from dummies (weight ~1e-9)
)


def convert(
    input_path: Path, output_path: Path, threshold: float = _WEIGHT_THRESHOLD
) -> None:
    print(f"Reading  : {input_path}")
    with input_path.open() as f:
        ws = json.load(f)

    n_converted = 0
    for datum in ws["data"]:
        if not (
            datum.get("name", "").startswith("combData_")
            and datum.get("type") == "unbinned"
            and "weights" in datum
        ):
            continue

        entries = datum["entries"]
        weights = datum["weights"]
        n_before = len(entries)

        real_entries = [
            e for e, w in zip(entries, weights, strict=False) if w > threshold
        ]
        n_after = len(real_entries)

        datum["entries"] = real_entries
        del datum["weights"]

        print(
            f"  {datum['name']}: {n_before} → {n_after} events "
            f"(removed {n_before - n_after} dummy events)"
        )
        n_converted += 1

    if n_converted == 0:
        print("  No combData_* entries with weights found — nothing to convert.")
    else:
        print(f"\nConverted {n_converted} dataset(s).")

    print(f"Writing  : {output_path}")
    with output_path.open("w") as f:
        json.dump(ws, f, indent=2)
    print("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=_DEFAULT_INPUT,
        help=f"Input workspace JSON (default: {_DEFAULT_INPUT.name})",
    )
    parser.add_argument(
        "output",
        nargs="?",
        type=Path,
        default=None,
        help="Output JSON path (default: data/<input_stem>_clean.json)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=_WEIGHT_THRESHOLD,
        help=f"Weight threshold separating real from dummy events "
        f"(default: {_WEIGHT_THRESHOLD})",
    )
    args = parser.parse_args()

    input_path: Path = args.input.resolve()
    if not input_path.exists():
        sys.exit(f"ERROR: input file not found: {input_path}")

    if args.output is None:
        output_path = input_path.parent / f"{input_path.stem}_clean{input_path.suffix}"
    else:
        output_path = args.output.resolve()

    if output_path == input_path:
        sys.exit("ERROR: output path is the same as input — choose a different name.")

    convert(input_path, output_path, threshold=args.threshold)


if __name__ == "__main__":
    main()
