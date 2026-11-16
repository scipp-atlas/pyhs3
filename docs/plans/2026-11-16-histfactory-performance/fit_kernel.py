"""End-to-end Python fit: vectorised PyTensor kernel, ported JS BFGS, and covariance two ways.

The covariance comes from central differences of the analytic gradient (what the JS does) and,
for comparison, from PyTensor's autodiff Hessian.

Usage: python fit_kernel.py MODEL_DUMP.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import bfgs
import hf_kernel
import numpy as np
import pytensor
import pytensor.tensor as pt


def main(dump: Path) -> None:
    m = hf_kernel.load(dump)
    out: dict[str, object] = {}

    t = time.perf_counter()
    theta, twice_nll = hf_kernel.build(m, "prod")
    value_and_grad = pytensor.function(
        [theta], [twice_nll, pt.grad(twice_nll, theta)], mode="FAST_RUN"
    )
    out["compile_s"] = round(time.perf_counter() - t, 2)

    def fg(x: bfgs.FloatArray) -> tuple[float, bfgs.FloatArray]:
        value, grad = value_and_grad(x)
        # nu <= 0 makes 2NLL nan or inf: report inf so the line search backtracks, as the JS does
        return (float(value) if np.isfinite(value) else np.inf), np.asarray(grad)

    lower = np.asarray(m["lower"])
    upper = np.asarray(m["upper"])
    free = np.flatnonzero(~np.asarray(m["fixed"], dtype=bool))
    prefit = np.asarray(m["prefitUnc"])
    width = np.where(prefit > 0, prefit, np.maximum((upper - lower) / 10, 1e-3))

    t = time.perf_counter()
    fit = bfgs.minimize(
        fg, np.asarray(m["init"]), lower, upper, free, 0.5 * width[free] ** 2, tol=1e-6
    )
    out |= {
        "bfgs_s": round(time.perf_counter() - t, 3),
        "bfgs_calls": fit.calls,
        "bfgs_iterations": fit.iterations,
        "converged": fit.converged,
        "twice_nll": round(fit.f, 6),
        "poi_hat": round(float(fit.x[m["poiIndex"]]), 5),
    }

    # covariance over the parameters not pinned at a bound, as the JS does
    x, g = fit.x, fit.g
    pinned = ((x <= lower) & (g > 0)) | ((x >= upper) & (g < 0))
    inner = np.array([p for p in free if not pinned[p]])
    poi_row = int(np.flatnonzero(inner == m["poiIndex"])[0])
    steps = 1e-4 * np.where(
        prefit[inner] > 0, prefit[inner], np.maximum(np.abs(x[inner]), 1)
    )
    t = time.perf_counter()
    hessian_fd = bfgs.fd_hessian(fg, x, inner, steps)
    covariance_fd = 2 * np.linalg.inv(hessian_fd)  # 2NLL: covariance = 2 * inv(Hessian)
    out |= {
        "hessian_fd_s": round(time.perf_counter() - t, 2),
        "hessian_fd_calls": 2 * len(inner),
        "poi_unc_fd": round(float(np.sqrt(covariance_fd[poi_row, poi_row])), 5),
    }
    print(json.dumps(out), flush=True)

    # Autodiff Hessian. pt.prod has no second derivative in PyTensor, so this graph uses the unrolled product.
    theta2, twice_nll2 = hf_kernel.build(m, "unrolled")
    t = time.perf_counter()
    hessian_fn = pytensor.function(
        [theta2], pytensor.gradient.hessian(twice_nll2, theta2), mode="FAST_RUN"
    )
    out["hessian_ad_compile_s"] = round(time.perf_counter() - t, 2)
    t = time.perf_counter()
    hessian_ad = hessian_fn(x)[np.ix_(inner, inner)]
    out["hessian_ad_eval_s"] = round(time.perf_counter() - t, 2)
    covariance_ad = 2 * np.linalg.inv(hessian_ad)
    out["poi_unc_ad"] = round(float(np.sqrt(covariance_ad[poi_row, poi_row])), 5)
    out["max_rel_diff_hessian_fd_vs_ad"] = float(
        np.max(np.abs(hessian_fd - hessian_ad)) / np.max(np.abs(hessian_ad))
    )
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
