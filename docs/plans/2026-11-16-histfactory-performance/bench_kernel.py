"""Compile the vectorised kernel in several PyTensor modes, check it against the JS reference values, and time it.

Usage: python bench_kernel.py MODEL_DUMP.json MODES PROD_MODE
    MODES      comma-separated PyTensor modes, e.g. FAST_RUN,NUMBA or JAX
    PROD_MODE  "prod" or "unrolled" (see hf_kernel.padded_prod; JAX needs "unrolled")
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import hf_kernel
import numpy as np
import pytensor
import pytensor.tensor as pt
from pytensor.graph.traversal import applys_between

REPEATS = 500


def main(dump: Path, modes: list[str], prod_mode: hf_kernel.ProdMode) -> None:
    m = hf_kernel.load(dump)
    theta, twice_nll = hf_kernel.build(m, prod_mode)
    grad = pt.grad(twice_nll, theta)
    print(
        json.dumps(
            {
                "prod_mode": prod_mode,
                "raw_apply_nodes": len(
                    list(applys_between([theta], [twice_nll, grad]))
                ),
            }
        )
    )
    for mode in modes:
        t = time.perf_counter()
        fn = pytensor.function([theta], [twice_nll, grad], mode=mode)
        compile_s = time.perf_counter() - t
        worst_value = 0.0
        worst_grad = 0.0
        for ref in m["refs"]:
            value, gradient = fn(np.asarray(ref["x"]))
            ref_grad = np.asarray(ref["g"])
            worst_value = max(worst_value, abs(float(value) - ref["f"]) / abs(ref["f"]))
            # relative error per component, with an absolute floor for components that are ~0
            worst_grad = max(
                worst_grad,
                float(np.max(np.abs(gradient - ref_grad) / (np.abs(ref_grad) + 1e-6))),
            )
        x = np.asarray(m["refs"][1]["x"])
        fn(x)
        t = time.perf_counter()
        for _ in range(REPEATS):
            fn(x)
        per_call_ms = (time.perf_counter() - t) / REPEATS * 1e3
        print(
            json.dumps(
                {
                    "mode": mode,
                    "compile_s": round(compile_s, 2),
                    "value_grad_ms": round(per_call_ms, 4),
                    "compiled_apply_nodes": len(fn.maker.fgraph.apply_nodes),
                    "max_rel_err_value_vs_js": worst_value,
                    "max_rel_err_grad_vs_js": worst_grad,
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2].split(","), sys.argv[3])
