"""NumPy port of the histfactory_js box-constrained dense BFGS (``js/minimize.js``), algorithmically identical.

On the HWW toy this port follows the same 84-iteration, 279-call path as the JS and lands on the same
2NLL, which validates the port.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
ValueAndGrad = Callable[[FloatArray], tuple[float, FloatArray]]

# SciPy's L-BFGS-B line search cannot recover from a non-finite objective; a large finite value with
# a zero gradient makes it backtrack out of the region where the expected yields are not positive.
UNDEFINED_PENALTY = 1e10

ARMIJO = 1e-4
MAX_BACKTRACKS = 40
MAX_RESETS = 4
STALL_LIMIT = 3


@dataclass
class Result:
    x: FloatArray
    f: float
    g: FloatArray
    iterations: int
    converged: bool
    calls: int
    edm: float


def minimize(
    fg: ValueAndGrad,
    x0: FloatArray,
    lower: FloatArray,
    upper: FloatArray,
    free: IntArray,
    diag: FloatArray,
    tol: float = 1e-6,
    max_iter: int | None = None,
) -> Result:
    """Minimise ``fg`` over ``x[free]`` within [lower, upper].

    ``fg(x)`` returns the objective and its (N,) gradient, or ``inf`` where the objective is undefined.
    ``diag`` is the (n,) starting inverse-Hessian diagonal for the n free parameters; the JS uses
    ``0.5 * sigma_prefit**2`` because the objective is 2NLL. The (n, n) inverse Hessian is dense.
    Convergence is an estimated distance to minimum ``edm = 0.5 * g^T C g`` below ``tol``.
    """
    n = len(free)
    max_iter = max_iter or 200 + 20 * n
    lo = lower[free]
    hi = upper[free]
    calls = 0

    def call(x: FloatArray) -> tuple[float, FloatArray]:
        nonlocal calls
        calls += 1
        return fg(x)

    x = np.array(x0, dtype=float)
    x[free] = np.clip(x[free], lo, hi)
    f, g = call(x)
    inverse_hessian = np.diag(diag).astype(float)  # (n, n)
    edm = np.inf
    converged = False
    resets = 0
    stalled = 0
    iteration = 0
    for iteration in range(max_iter):  # noqa: B007 (the final value is reported)
        xf = x[free]
        g_free = g[free]
        # parameters at a bound whose gradient pushes outwards do not move this iteration
        active = ((xf <= lo) & (g_free > 0)) | ((xf >= hi) & (g_free < 0))
        gf = np.where(active, 0.0, g_free)
        step = inverse_hessian @ gf  # (n,)
        if active.any():
            # project the active directions out: step <- step - C[:, A] inv(C[A, A]) step[A]
            a_idx = np.flatnonzero(active)
            # a singular C[A, A] keeps the unprojected step, as the JS does when the inverse fails
            with contextlib.suppress(np.linalg.LinAlgError):
                step -= inverse_hessian[:, a_idx] @ np.linalg.solve(
                    inverse_hessian[np.ix_(a_idx, a_idx)], step[a_idx]
                )
            step[a_idx] = 0.0
        slope = -gf @ step
        edm = -0.5 * slope
        if 0 <= edm < tol:
            converged = True
            break
        if not slope < 0:
            resets += 1
            if resets > MAX_RESETS:
                break
            inverse_hessian = np.diag(diag).astype(float)
            continue

        # backtracking line search along -step, projected onto the bounds
        t = 1.0
        accepted = False
        for _ in range(MAX_BACKTRACKS):
            x_new = x.copy()
            x_new[free] = np.clip(xf - t * step, lo, hi)
            predicted = g[free] @ (x_new[free] - xf)
            f_new, g_new = call(x_new)
            if np.isfinite(f_new) and f_new <= f + ARMIJO * predicted:
                accepted = True
                break
            quadratic = (
                (-slope * t * t) / (2 * (f_new - f - slope * t))
                if np.isfinite(f_new)
                else 0.0
            )
            t = min(max(quadratic, 0.1 * t), 0.5 * t)
        if not accepted:
            resets += 1
            if resets > MAX_RESETS:
                break
            inverse_hessian = np.diag(diag).astype(float)
            continue

        s = x_new[free] - xf
        y = g_new[free] - g[free]
        sy = s @ y
        # BFGS update of the inverse Hessian, skipped when the curvature condition fails
        if sy > 1e-10 * np.sqrt((s @ s) * (y @ y)):
            cy = inverse_hessian @ y
            factor = (sy + y @ cy) / (sy * sy)
            inverse_hessian += (
                factor * np.outer(s, s) - (np.outer(cy, s) + np.outer(s, cy)) / sy
            )
        stalled = stalled + 1 if f - f_new < 1e-13 * (abs(f) + 1) else 0
        x, g, f = x_new, g_new, f_new
        if stalled >= STALL_LIMIT:
            break
    return Result(
        x=x, f=f, g=g, iterations=iteration, converged=converged, calls=calls, edm=edm
    )


def penalize_undefined(fg: ValueAndGrad) -> ValueAndGrad:
    """Wrap ``fg`` for SciPy: a non-finite objective becomes ``UNDEFINED_PENALTY`` with a zero gradient."""

    def guarded(x: FloatArray) -> tuple[float, FloatArray]:
        value, grad = fg(x)
        if not np.isfinite(value):
            return UNDEFINED_PENALTY, np.zeros_like(grad)
        return value, grad

    return guarded


def fd_hessian(
    fg: ValueAndGrad, x: FloatArray, index: IntArray, steps: FloatArray
) -> FloatArray:
    """(k, k) Hessian over ``x[index]`` from central differences of the analytic gradient (2k calls).

    Row a is ``(g(x + h_a e_a) - g(x - h_a e_a)) / (2 h_a)`` restricted to ``index``; the result is symmetrised.
    """
    k = len(index)
    hessian = np.empty((k, k))
    for row, (p, h) in enumerate(zip(index, steps, strict=True)):
        shifted = x.copy()
        shifted[p] = x[p] + h
        g_plus = fg(shifted)[1]
        shifted[p] = x[p] - h
        g_minus = fg(shifted)[1]
        hessian[row] = (g_plus[index] - g_minus[index]) / (2 * h)
    return 0.5 * (hessian + hessian.T)
