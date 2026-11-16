"""Compare SciPy and iminuit optimisers on the identical histfactory_js likelihood.

Only the optimiser changes between runs: function, gradient, starting point, and bounds are the same.

Usage: python drive_js_optimizers.py ENGINE_DIR WORKSPACE.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import iminuit
import numpy as np
import scipy.optimize
from bfgs import penalize_undefined
from js_bridge import FloatArray, JSLikelihood


def report(
    likelihood: JSLikelihood,
    label: str,
    twice_nll: float,
    x: FloatArray,
    t0: float,
    **extra: object,
) -> None:
    print(
        json.dumps(
            {
                "label": label,
                "calls": likelihood.calls,
                "s": round(time.perf_counter() - t0, 2),
                "twiceNll": round(twice_nll, 6),
                "poiHat": round(float(x[likelihood.poi_index]), 5),
                **extra,
            }
        ),
        flush=True,
    )


def lbfgsb(
    likelihood: JSLikelihood, label: str, *, scaled: bool, maxcor: int = 10
) -> None:
    """L-BFGS-B in raw parameters, or in z = (x - x0) / sigma_prefit (the engine's diagonal preconditioning)."""
    scale = likelihood.prefit_width if scaled else np.ones(likelihood.n_pars)

    @penalize_undefined
    def objective(z: FloatArray) -> tuple[float, FloatArray]:
        twice_nll, grad = likelihood(likelihood.init + scale * z)
        return twice_nll, grad * scale

    bounds = list(
        zip(
            (likelihood.lower - likelihood.init) / scale,
            (likelihood.upper - likelihood.init) / scale,
            strict=True,
        )
    )
    likelihood.calls = 0
    t0 = time.perf_counter()
    result = scipy.optimize.minimize(
        objective,
        np.zeros(likelihood.n_pars),
        jac=True,
        method="L-BFGS-B",
        bounds=bounds,
        options={
            "maxcor": maxcor,
            "maxiter": 20000,
            "maxfun": 50000,
            "ftol": 1e-13,
            "gtol": 1e-7,
        },
    )
    report(
        likelihood,
        label,
        float(result.fun),
        likelihood.init + scale * result.x,
        t0,
        nit=int(result.nit),
        message=str(result.message),
    )


def migrad(
    likelihood: JSLikelihood, label: str, *, strategy: int, analytic_gradient: bool
) -> None:
    def value(x: FloatArray) -> float:
        return likelihood(x)[0]

    def gradient(x: FloatArray) -> FloatArray:
        return likelihood(x)[1]

    minuit = iminuit.Minuit(
        value, likelihood.init, grad=gradient if analytic_gradient else None
    )
    minuit.errordef = 1.0  # the objective is 2NLL
    minuit.strategy = strategy
    minuit.limits = list(zip(likelihood.lower, likelihood.upper, strict=True))
    minuit.errors = likelihood.prefit_width
    likelihood.calls = 0
    t0 = time.perf_counter()
    minuit.migrad(ncall=500000)
    report(
        likelihood,
        label,
        float(minuit.fval),
        np.asarray(minuit.values),
        t0,
        valid=minuit.valid,
        nfcn=minuit.nfcn,
        ngrad=minuit.ngrad,
    )


def main(engine_dir: Path, workspace: Path) -> None:
    likelihood = JSLikelihood(engine_dir, workspace)
    try:
        lbfgsb(likelihood, "L-BFGS-B raw, maxcor=10", scaled=False)
        for maxcor in (10, 30, 100, 300):
            lbfgsb(
                likelihood,
                f"L-BFGS-B sigma-scaled, maxcor={maxcor}",
                scaled=True,
                maxcor=maxcor,
            )
        migrad(
            likelihood,
            "MIGRAD strategy 0, analytic gradient",
            strategy=0,
            analytic_gradient=True,
        )
        migrad(
            likelihood,
            "MIGRAD strategy 1, analytic gradient",
            strategy=1,
            analytic_gradient=True,
        )
        migrad(
            likelihood,
            "MIGRAD strategy 0, numerical gradient",
            strategy=0,
            analytic_gradient=False,
        )
    finally:
        likelihood.close()


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
