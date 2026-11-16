"""Write one seeded Poisson toy of a workspace's main-measurement data into both its pyhf and HS3 files.

Both HWW files in ``aheld/`` carry blinded Asimov data, so a fit starting at the nominal values
is already at its minimum. The toy gives the minimiser real work while keeping the two formats on
identical data.

Usage: python make_toy.py PYHF.json HS3.json OUTDIR
Writes OUTDIR/hww_toy_pyhf.json and OUTDIR/hww_toy_hs3.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

SEED = 20261116
HS3_OBSERVED_PREFIX = "obsData_"


def main(pyhf_path: Path, hs3_path: Path, outdir: Path) -> None:
    rng = np.random.default_rng(SEED)
    pyhf_ws = json.loads(pyhf_path.read_text(encoding="utf-8"))
    hs3_ws = json.loads(hs3_path.read_text(encoding="utf-8"))

    toy: dict[str, list[float]] = {}
    for observation in pyhf_ws["observations"]:
        counts = rng.poisson(np.asarray(observation["data"])).astype(float).tolist()
        toy[observation["name"]] = counts
        observation["data"] = counts

    replaced = 0
    for datum in hs3_ws["data"]:
        if datum["name"].startswith(HS3_OBSERVED_PREFIX):
            datum["contents"] = toy[datum["name"].removeprefix(HS3_OBSERVED_PREFIX)]
            replaced += 1
    if replaced != len(toy):
        msg = f"replaced {replaced} HS3 observed datasets but the pyhf file has {len(toy)} channels"
        raise ValueError(msg)

    (outdir / "hww_toy_pyhf.json").write_text(json.dumps(pyhf_ws), encoding="utf-8")
    (outdir / "hww_toy_hs3.json").write_text(json.dumps(hs3_ws), encoding="utf-8")
    print(f"channels {replaced}, total events {sum(map(sum, toy.values()))}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
