# ruff: noqa: T201
"""Profile-likelihood minimization demo for the ATLAS diHiggs bbyy workspace.

Demonstrates how to:
1. Build a pyhs3 Model from an HS3 workspace
2. Compile the joint log-probability into a fast pytensor function
3. Profile over nuisance parameters at a fixed mu_HH using scipy

Install: pip install "pyhs3" scipy skhep-testdata
Run:     python examples/minimization_dihiggs.py
"""

from __future__ import annotations

import argparse
import datetime
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pytensor.tensor as pt
from matplotlib import pyplot as plt
from pytensor.compile.function import function
from pytensor.graph.replace import clone_replace
from pytensor.graph.traversal import explicit_graph_inputs
from scipy.optimize import OptimizeResult, minimize
from skhep_testdata import data_path as skhep_testdata_path

import pyhs3

_REFERENCE = {
    "mu_HH": [
        -0.5,
        -0.4,
        -0.3,
        -0.2,
        -0.1,
        0.0,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0,
        1.1,
        1.2,
        1.3,
        1.4,
        1.5,
        1.6,
        1.7,
        1.8,
        1.9,
        2.0,
        2.1,
        2.2,
        2.3,
        2.4,
        2.5,
    ],
    "nll": [
        2103.000271377221,
        2102.8686029168957,
        2102.7485815518967,
        2102.6412720912876,
        2102.5465216345087,
        2102.463841521541,
        2102.3925958322193,
        2102.3321177645485,
        2102.281745494838,
        2102.240869154073,
        2102.2090012317617,
        2102.1856874851815,
        2102.170428875709,
        2102.1627459894376,
        2102.162186508772,
        2102.16832136193,
        2102.180743919424,
        2102.1990690691637,
        2102.222931922541,
        2102.2519863755706,
        2102.28590316104,
        2102.3243680706505,
        2102.367080069453,
        2102.41374963789,
        2102.4640980726444,
        2102.517857855883,
        2102.574777457445,
        2102.634610797755,
        2102.6971149612623,
        2102.762164407559,
        2102.828822854034,
    ],
}

_REFERENCE = {
    "mu_HH": [
        -0.5,
        -0.4,
        -0.3,
        -0.2,
        -0.1,
        0.0,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0,
        1.1,
        1.2,
        1.3,
        1.4,
        1.5,
        1.6,
        1.7,
        1.8,
        1.9,
        2.0,
        2.1,
        2.2,
        2.3,
        2.4,
        2.5,
    ],
    "nll": [
        2116.5050528141624,
        2116.32024442325,
        2116.147536871291,
        2115.9898748809364,
        2115.8484135401295,
        2115.7234904744287,
        2115.614760462136,
        2115.5217651455596,
        2115.4428493502382,
        2115.3774140758096,
        2115.3244289405097,
        2115.282917807648,
        2115.2519751699165,
        2115.2307707411514,
        2115.2185475973124,
        2115.214617056965,
        2115.218352131681,
        2115.229180543526,
        2115.2465778591986,
        2115.270060985844,
        2115.299182173344,
        2115.333523579179,
        2115.372692531467,
        2115.4163176659454,
        2115.4640463133146,
        2115.5155437672224,
        2115.5704953916675,
        2115.6286127532935,
        2115.689167112463,
        2115.7526215376183,
        2115.8184195879385,
    ],
}

_WS_FILE = "test_hs3_unbinned_pyhs3_validation_issue41.json"
_WS_LOCAL = Path(__file__).resolve().parent.parent / "data" / _WS_FILE


def resolve_workspace(ws_arg: Path | None = None) -> tuple[Path, Path]:
    """Return (ws_path, cache_dir) for the current run.

    ws_path  — workspace JSON to load.
    cache_dir — directory for ws.pkl and log_prob_fn.pkl, named after the
                workspace stem so different JSON files never share a cache.
                Created on first call.

    Priority: explicit --workspace argument > local data/ copy > skhep-testdata.
    """
    if ws_arg is not None:
        ws_path = ws_arg.resolve()
    elif _WS_LOCAL.exists():
        ws_path = _WS_LOCAL
    else:
        print("  (local workspace not found; using skhep-testdata)")
        ws_path = Path(skhep_testdata_path(_WS_FILE))

    cache_dir = Path("cache") / ws_path.stem
    cache_dir.mkdir(parents=True, exist_ok=True)
    return ws_path, cache_dir


def build_model(ws_path: Path, cache_dir: Path) -> pyhs3.Model:
    """Load workspace and build (or load cached) model.

    Cache is read from / written to cache_dir/ws.pkl.
    """
    model_cache = cache_dir / "ws.pkl"
    print(f"Loading workspace from {ws_path} ...")

    if model_cache.exists():
        print(f"Loading cached model from {model_cache} ...")
        with model_cache.open("rb") as f:
            return pickle.load(f)

    ws = pyhs3.Workspace.load(ws_path)
    analysis = ws.analyses["CombinedPdf_combData"]

    # Collect parameter points from the workspace
    pset_names = [
        "default_values",
        "nominalGlobs",
        "nominalNuis",
        "POI_muhat",
    ]
    param_set = pyhs3.parameter_points.ParameterSet(
        name="collected",
        parameters=[
            pp for pset_name in pset_names for pp in ws.parameter_points[pset_name]
        ],
    )

    print("Building symbolic model (this takes ~1 min) ...")
    model = ws.model(analysis, parameter_set=param_set, progress=True)
    with model_cache.open("wb") as f:
        pickle.dump(model, f)
    print(f"  Model cached to {model_cache}")
    return model


def compile_log_prob(
    model: pyhs3.Model,
    cache_dir: Path,
    analytic_grad: bool = False,
    use_jax: bool = False,
):
    """Compile model.log_prob into a callable function.

    Attempts the JAX backend by default (use_jax=True), which:
    - Transpiles the PyTensor symbolic graph to JAX/XLA via pyhs3.jaxify()
    - Compiles in seconds (not minutes) using all available CPU cores via XLA
    - Provides automatic differentiation at no extra cost via jax.value_and_grad
    - Caches compiled XLA binaries in cache_dir/xla_cache/ for instant reuse

    Falls back to PyTensor compiled functions when use_jax=False or when JAX
    is not importable.

    Before compiling, every input that is not a genuine free parameter
    (mu_HH or a nuisance) is baked as a pt.constant, reducing the symbolic
    graph from ~28 700 leaves to ~170 free parameters.

    Cache strategy:
    - JAX path:     writes ("jax", input_names) sentinel to log_prob_fn.pkl;
                    XLA handles the binary cache in xla_cache/.
    - PyTensor path: pickles (log_prob_fn, grad_fn, input_names) to the same file.

    Returns (log_prob_fn, grad_fn, input_names) where:
      log_prob_fn  — callable(*args) → [scalar_nll]  (-2 * log_prob)
      grad_fn      — callable(*args) → [g0, g1, …]  (one gradient per input),
                     or None if gradient compilation failed.
      input_names  — ordered list of parameter name strings both callables
                     expect as positional arguments.
    """
    fn_cache = cache_dir / "log_prob_fn.pkl"

    if fn_cache.exists():
        print(f"Loading cached compiled function from {fn_cache} ...")
        with fn_cache.open("rb") as f:
            cached = pickle.load(f)
        if (
            isinstance(cached, tuple)
            and len(cached) == 2
            and cached[0] in ("jax", "jax_export")
        ):
            if use_jax:
                if cached[0] == "jax_export":
                    export_path = cache_dir / "jax_export.bin"
                    if export_path.exists():
                        try:
                            import jax  # noqa: PLC0415
                            import jax.export as _jax_export  # noqa: PLC0415
                            import jax.numpy as jnp  # noqa: PLC0415

                            t0 = time.perf_counter()
                            print(f"  Loading JAX export from {export_path} ...")
                            _loaded = _jax_export.deserialize(export_path.read_bytes())
                            jax_input_names = list(cached[1])
                            print(
                                f"  Loaded in {time.perf_counter() - t0:.1f}s "
                                f"({export_path.stat().st_size // 1024} KB, "
                                f"{len(jax_input_names)} inputs) — skipping jaxify."
                            )
                            # XLA cache so first Exported.call() reuses compiled binary.
                            _xc = cache_dir / "xla_cache"
                            _xc.mkdir(exist_ok=True)
                            jax.config.update("jax_compilation_cache_dir", str(_xc))

                            _ec: list = [None, None, None]

                            def _jax_eval_export(args_tuple: tuple) -> None:
                                if _ec[0] != args_tuple:
                                    _jargs = [jnp.float64(v) for v in args_tuple]
                                    _v, _g = _loaded.call(*_jargs)
                                    _ec[0] = args_tuple
                                    _ec[1] = [float(_v)]
                                    _ec[2] = [float(g) for g in _g]

                            def log_prob_fn(*args):
                                _jax_eval_export(
                                    tuple(float(np.asarray(a)) for a in args)
                                )
                                return _ec[1]

                            def grad_fn(*args):
                                _jax_eval_export(
                                    tuple(float(np.asarray(a)) for a in args)
                                )
                                return _ec[2]

                            return log_prob_fn, grad_fn, jax_input_names

                        except Exception as exc:
                            print(
                                f"  JAX export load failed ({exc!r}); re-jaxifying ..."
                            )
                            # fall through to clone_replace + jaxify
                    else:
                        print("  JAX export binary missing; re-jaxifying ...")
                        # fall through
                else:
                    print(
                        "  JAX sentinel found — re-jaxifying (XLA cache handles compilation) ..."
                    )
                    # fall through
            else:
                print(
                    "  JAX sentinel found but --no-jax specified; recompiling with PyTensor ..."
                )
                fn_cache.unlink()
        elif len(cached) == 3:
            log_prob_fn, grad_fn, input_names = cached
            if grad_fn is not None:
                print(f"  {len(input_names)} free inputs (with analytic gradient).")
                return log_prob_fn, grad_fn, input_names
            if not analytic_grad:
                # grad_fn is None and no analytic gradient was requested — valid cache.
                print(
                    f"  {len(input_names)} free inputs (numerical gradients, cached)."
                )
                return log_prob_fn, grad_fn, input_names
            # grad_fn was None but analytic_grad=True — previous run failed to
            # compile the gradient.  Recompile.
            print(
                "  Cached gradient is None but --analytic-grad requested; recompiling ..."
            )
            fn_cache.unlink()
        else:
            # Old 2-tuple cache format — missing gradient function; recompile.
            print("  Old cache format (no gradient); recompiling ...")
            fn_cache.unlink()

    nll_expr = -2.0 * model.log_prob

    # Parameters that stay symbolic: mu_HH + all nuisance parameters.
    free_param_names: set[str] = set(model.free_params.keys())

    # Walk the raw graph to find every named leaf.
    all_raw_inputs = [
        v for v in explicit_graph_inputs([nll_expr]) if v.name is not None
    ]

    data_np = model.data
    nominal_vals = model.free_params  # best-fit / nominal values for baking

    subs: dict = {}
    free_vars: dict[str, pt.TensorVariable] = {}

    for var in all_raw_inputs:
        if var.name in data_np:
            # Observable data: bake as a compile-time constant.
            subs[var] = pt.constant(
                np.asarray(data_np[var.name], dtype=np.float64), name=var.name
            )
        elif var.name in free_param_names:
            # Keep as a canonical free symbolic scalar (deduplicate copies).
            if var.name not in free_vars:
                free_vars[var.name] = pt.scalar(var.name)
            subs[var] = free_vars[var.name]
        else:
            # Any other leaf (global observables, per-bin yields, …):
            # bake at the nominal / best-fit value from the parameter set.
            subs[var] = pt.constant(
                np.float64(nominal_vals.get(var.name, 0.0)), name=var.name
            )

    nll_reduced = clone_replace(nll_expr, replace=subs)

    # free_vars are now the only symbolic leaves; collect in a stable order.
    free_inputs = [
        v for v in explicit_graph_inputs([nll_reduced]) if v.name is not None
    ]

    n_baked = len(all_raw_inputs) - len(free_inputs)
    print(
        f"  Baked {n_baked} fixed inputs as constants; "
        f"{len(free_inputs)} free inputs remain (mu_HH + nuisances)."
    )

    # nll_scalar: a true 0-d tensor required by pt.grad and pyhs3.jaxify.
    # nll_reduced may have shape (1,) — squeeze it.
    nll_scalar = nll_reduced[0] if nll_reduced.ndim > 0 else nll_reduced

    # -----------------------------------------------------------------------
    # JAX path — fast compilation via XLA, automatic differentiation included
    # -----------------------------------------------------------------------
    if use_jax:
        try:
            import jax  # noqa: PLC0415
            import jax.numpy as jnp  # noqa: PLC0415

            # Enable persistent XLA compilation cache so subsequent runs are
            # near-instant (XLA re-loads the compiled binary from disk).
            xla_cache = cache_dir / "xla_cache"
            xla_cache.mkdir(exist_ok=True)
            jax.config.update("jax_compilation_cache_dir", str(xla_cache))

            t_jaxify = time.perf_counter()
            print("  Transpiling to JAX via pyhs3.jaxify() ...")
            # Pass free_inputs explicitly to skip the graph-traversal inside jaxify.
            jg = pyhs3.jaxify(nll_scalar, inputs=free_inputs)
            jax_input_names = list(jg.input_names)
            print(
                f"  jaxify done in {time.perf_counter() - t_jaxify:.1f}s "
                f"({len(jax_input_names)} free inputs)."
            )

            # JIT-compile a function that returns (nll_value, gradients) in one
            # forward+backward pass — no separate gradient compilation needed.
            _val_and_grad_fn = jax.value_and_grad(
                lambda *a: jg.fn(*a)[0],
                argnums=tuple(range(len(jax_input_names))),
            )
            _jax_val_and_grad = jax.jit(_val_and_grad_fn)

            # First call triggers XLA compilation (fast on cache hit).
            t_warmup = time.perf_counter()
            print(
                "  Warming up JAX JIT (first call compiles or loads from XLA cache) ..."
            )
            _dummy = [jnp.float64(0.0)] * len(jax_input_names)
            _jax_val_and_grad(*_dummy)
            print(f"  JIT warm-up done in {time.perf_counter() - t_warmup:.1f}s.")

            # Shared evaluation cache: scipy calls nll(x) then immediately jac(x)
            # with the same x; cache the last (val, grads) to avoid re-evaluation.
            _eval_cache: list = [None, None, None]  # [args_key, [val], [grads]]

            def _jax_eval(args_tuple: tuple) -> None:
                if _eval_cache[0] != args_tuple:
                    jax_args = [jnp.float64(v) for v in args_tuple]
                    val, grads = _jax_val_and_grad(*jax_args)
                    _eval_cache[0] = args_tuple
                    _eval_cache[1] = [float(val)]
                    _eval_cache[2] = [float(g) for g in grads]

            def log_prob_fn(*args):
                _jax_eval(tuple(float(np.asarray(a)) for a in args))
                return _eval_cache[1]

            def grad_fn(*args):
                _jax_eval(tuple(float(np.asarray(a)) for a in args))
                return _eval_cache[2]

            # Attempt JAX export serialization so subsequent runs skip jaxify
            # entirely.  jax.export produces a StableHLO artifact that can be
            # deserialized in milliseconds without re-running jaxify().
            sentinel_key = "jax"
            try:
                import jax.export as _jax_export  # noqa: PLC0415

                t_exp = time.perf_counter()
                print("  Exporting compiled JAX function to disk ...")
                abstract_args = [jax.ShapeDtypeStruct((), jnp.float64)] * len(
                    jax_input_names
                )
                _exported = _jax_export.export(jax.jit(_val_and_grad_fn))(
                    *abstract_args
                )
                blob = _exported.serialize()
                export_path = cache_dir / "jax_export.bin"
                export_path.write_bytes(blob)
                print(
                    f"  Exported in {time.perf_counter() - t_exp:.1f}s "
                    f"({len(blob) // 1024} KB) → {export_path.name}"
                )
                sentinel_key = "jax_export"
            except Exception as exc:
                print(f"  JAX export failed ({exc!r}); will re-jaxify on next run.")

            # Write sentinel (lightweight — we never pickle the JAX callable itself).
            with fn_cache.open("wb") as f:
                pickle.dump((sentinel_key, jax_input_names), f)
            print(f"  Cache sentinel ({sentinel_key!r}) written to {fn_cache.name}")

            return log_prob_fn, grad_fn, jax_input_names

        except Exception as exc:
            print(f"  JAX path failed ({exc!r}); falling back to PyTensor ...")

    # -----------------------------------------------------------------------
    # PyTensor path — fallback or --no-jax
    # -----------------------------------------------------------------------
    log_prob_fn = function(
        inputs=free_inputs,
        outputs=nll_reduced,
        mode=model.mode,
        on_unused_input="ignore",
        name=model._likelihood.name,
        trust_input=True,
    )

    # Store only the name strings — symbolic variables aren't needed after
    # compilation, and names are all that profile_nll uses for ordering.
    input_names = [v.name for v in free_inputs]

    # Compile analytic gradient via PyTensor automatic differentiation.
    # This replaces 2xN_free numerical NLL evaluations per gradient step with
    # a single forward+backward pass, giving exact gradients even for
    # weakly-constrained nuisance parameters where finite differences are noisy.
    #
    # We deliberately use mode="FAST_COMPILE" for the gradient function.
    # model.mode (typically FAST_RUN) applies expensive graph-rewrite passes
    # before C-compilation; on the larger gradient graph those rewrites can
    # take an hour or more.  FAST_COMPILE skips those rewrites and executes
    # via Python/NumPy instead of C, which is ~10-50x slower per call —
    # but one gradient call still replaces 2xN_free=338 NLL evaluations,
    # so the per-optimizer-step time is substantially lower overall.
    grad_fn = None
    if not analytic_grad:
        print(
            "  Analytic gradient not requested; using numerical gradients (scipy 3-point)."
        )
    else:
        try:
            print("  Compiling analytic gradient function (FAST_COMPILE mode) ...")
            grad_outputs = pt.grad(
                nll_scalar,
                free_inputs,
                disconnected_inputs="zero",  # treat disconnected inputs as ∂/∂x = 0
            )
            grad_fn = function(
                inputs=free_inputs,
                outputs=grad_outputs,
                mode="FAST_COMPILE",  # skip expensive rewrites; compile in seconds
                on_unused_input="ignore",
                name=model._likelihood.name + "_grad",
                trust_input=True,
            )
            print("  Analytic gradient compiled.")
        except Exception as exc:
            print(
                f"  Warning: analytic gradient compilation failed ({exc}); "
                "will fall back to numerical gradients."
            )

    with fn_cache.open("wb") as f:
        pickle.dump((log_prob_fn, grad_fn, input_names), f)
    print(f"  Compiled functions cached to {fn_cache}")

    return log_prob_fn, grad_fn, input_names


def profile_nll(
    log_prob_fn,
    grad_fn,
    input_names,
    model,
    mu_val,
    method="L-BFGS-B",
    ftol=1e-6,
    gtol=1e-6,
    maxiter=10000,
    x0_override: dict[str, float] | None = None,
    migrad_strategy: int = 1,
):
    """Minimize -2*log_prob over nuisance parameters with mu_HH fixed.

    Parameters
    ----------
    log_prob_fn : pytensor compiled function returning the scalar NLL.
    grad_fn : pytensor compiled function returning a list of NLL gradients
        (one per free input, in the same order as input_names), or None to
        fall back to numerical gradients.
    input_names : list[str] of parameter names in the order both compiled
        functions expect them (as returned by compile_log_prob).
    model : pyhs3.Model with .free_params and .data
    mu_val : float, the fixed value of mu_HH for this scan point
    method : str, scipy minimization method (default L-BFGS-B)
    ftol : float, function-value tolerance passed to the optimizer.
        For L-BFGS-B this is a *relative* criterion: halts when
        |ΔF|/max(|F|,1) < ftol.  For SLSQP it is an *absolute* |ΔF|.
        Default 1e-6.
    gtol : float, gradient-norm tolerance (L-BFGS-B only; ignored by SLSQP).
        Default 1e-6.
    maxiter : int, maximum number of iterations for minimization
    x0_override : dict mapping NP name → starting value, or None.
        When provided, each free NP is initialised from this dict; any NP
        not present in the dict falls back to ``model.free_params``.  Use
        the optimised NP values from a neighbouring scan point to warm-start
        the minimisation and dramatically reduce iteration counts.

    Returns
    -------
    (result, free_names) : (scipy.optimize.OptimizeResult, list[str])
        result.x[k] corresponds to free_names[k]; use free_names for labelling.
    """
    pinned = {**model.data, "mu_HH": mu_val}

    # Build a template: pinned entries filled in, free entries left as None.
    # We track which indices are free so the optimizer can fill them.
    template = []
    free_names = []
    free_input_indices = []
    bounds = []

    model_bounds = {axis.name: (axis.min, axis.max) for axis in model.domain.axes}
    for i, name in enumerate(input_names):
        if name in pinned:
            template.append(np.asarray(pinned[name]))
        else:
            template.append(None)
            free_names.append(name)
            free_input_indices.append(i)
            bounds.append(model_bounds[name])

    # Starting point: use x0_override values when provided, fall back to nominal.
    x0 = np.array(
        [
            x0_override.get(name, model.free_params[name])
            if x0_override is not None
            else model.free_params[name]
            for name in free_names
        ],
        dtype=float,
    )
    # Clip to bounds so a warm-start value that drifted outside is still valid.
    for k, (lo, hi) in enumerate(bounds):
        if lo is not None:
            x0[k] = max(x0[k], lo)
        if hi is not None:
            x0[k] = min(x0[k], hi)

    def _fill_template(x):
        """Return the full input list with free slots filled from x."""
        vals = list(template)
        for idx, xi in zip(free_input_indices, x, strict=True):
            vals[idx] = xi
        return [np.asarray(v) for v in vals]

    def nll(x):
        # log_prob_fn already returns -2*log_prob (see compile_log_prob).
        return float(log_prob_fn(*_fill_template(x))[0])

    # Jacobian for gradient-based methods.
    # If an analytic grad_fn is available use it directly (exact, cheap).
    # Otherwise fall back to scipy's built-in '3-point' central differences,
    # which have O(h²) error vs O(h) for the default forward differences.
    # Central differences matter here: the NLL has absolute value ~141 800, so
    # forward-difference noise (~NLL x eps / h) can swamp the true gradient
    # signal in flat NP directions.
    #
    # Derivative-free methods (Nelder-Mead, Powell, COBYLA) ignore jac entirely.
    jac: str | None = None
    if grad_fn is not None:

        def jac(x):  # type: ignore[misc]
            grads = grad_fn(*_fill_template(x))
            return np.array([float(grads[i]) for i in free_input_indices])
    elif method not in ("Nelder-Mead", "Powell", "COBYLA"):
        # Use central differences for all gradient-based methods without an
        # analytic grad_fn.  2xN_free extra NLL calls per step, but much more
        # accurate than forward differences on a large-valued function.
        jac = "3-point"

    # -----------------------------------------------------------------------
    # iminuit path — MIGRAD and IMPROVE
    #
    # iminuit wraps CERN's Minuit2, the same algorithm used inside ROOT/TMinuit.
    # MIGRAD uses a variable-metric (quasi-Newton) method with:
    #   • internal numerical gradient tuned for HEP likelihoods
    #   • automatic parameter scaling so NPs with very different magnitudes
    #     receive comparable step sizes
    #   • a covariance-matrix update (HESSE) that is more stable than
    #     scipy's L-BFGS-B quasi-Newton approximation
    # IMPROVE runs MIGRAD, then SIMPLEX (Nelder-Mead, derivative-free) to
    # explore away from the found minimum, then MIGRAD again to re-converge;
    # this helps escape shallow local minima at the cost of extra evaluations.
    # -----------------------------------------------------------------------
    if method in ("MIGRAD", "IMPROVE"):
        try:
            from iminuit import Minuit  # noqa: PLC0415
        except ImportError as exc:
            msg = (
                f"iminuit is required for --method {method}. "
                "Install with: pixi add iminuit  or  pip install iminuit"
            )
            raise ImportError(msg) from exc

        def _nll_minuit(*args: float) -> float:
            return float(log_prob_fn(*_fill_template(args))[0])

        m = Minuit(_nll_minuit, *x0, name=free_names)
        # errordef=1.0: our NLL is -2*log_L, so 1-sigma corresponds to ΔNLL=1.
        m.errordef = Minuit.LEAST_SQUARES  # = 1.0
        # strategy: 0=fast/inaccurate, 1=default, 2=slow/accurate gradient steps.
        m.strategy = migrad_strategy
        # Set parameter limits from the model domain.
        m.limits = [
            (
                float(lo) if lo is not None else None,
                float(hi) if hi is not None else None,
            )
            for lo, hi in bounds
        ]

        # Use analytic gradient if available; otherwise let Minuit compute its
        # own (which is better tuned than scipy's default finite differences).
        if grad_fn is not None:

            def _grad_minuit(*args: float) -> list[float]:
                g = grad_fn(*_fill_template(args))
                return [float(g[i]) for i in free_input_indices]

            m.grad = _grad_minuit

        # ncall budget: each MIGRAD step needs ~2*n_params evaluations for the
        # numerical gradient, so a budget of maxiter *per free parameter* gives
        # approximately maxiter actual MIGRAD steps.  This matches iminuit's own
        # internal default of 200*n_params and avoids premature call-limit exits
        # when maxiter (designed for scipy iteration counts) is too small.
        _n = len(free_names)
        _ncall = maxiter * max(_n, 1)

        m.migrad(ncall=_ncall)
        if method == "IMPROVE":
            # iminuit v2 does not expose the old Minuit1 IMPROVE command.
            # The standard iminuit equivalent is: run SIMPLEX (Nelder-Mead)
            # to escape the local basin, then re-run MIGRAD to converge
            # precisely.  SIMPLEX is derivative-free and explores more broadly,
            # so it can find a lower basin that MIGRAD alone would miss.
            m.simplex(ncall=_ncall)
            m.migrad(ncall=_ncall)

        # Diagnose non-convergence so the user knows what to fix.
        _msg = "MIGRAD converged"
        if not m.valid:
            _fmin = m.fmin
            _reasons: list[str] = []
            if _fmin.has_reached_call_limit:
                _reasons.append(f"call limit ({m.nfcn} calls; raise --maxiter)")
            if _fmin.is_above_max_edm:
                _reasons.append(
                    f"EDM={_fmin.edm:.3g} > goal {m.tol * 0.002:.3g} "
                    "(flat direction or noisy gradient)"
                )
            if not _fmin.has_posdef_covariance:
                _reasons.append(
                    "covariance not positive definite (try --migrad-strategy 2)"
                )
            if _fmin.hesse_failed:
                _reasons.append("HESSE failed")
            _msg = "MIGRAD did not converge: " + (
                "; ".join(_reasons) if _reasons else "unknown reason"
            )

        result = OptimizeResult(
            x=np.array(list(m.values)),
            fun=float(m.fval),
            success=bool(m.valid),
            # Minuit doesn't expose an iteration count separate from nfcn.
            nit=int(m.nfcn),
            nfev=int(m.nfcn),
            message=_msg,
        )
        return result, free_names

    # -----------------------------------------------------------------------
    # scipy path — L-BFGS-B, SLSQP, TNC, trust-constr, …
    # -----------------------------------------------------------------------

    # L-BFGS-B uses ftol as a *relative* criterion: stops when
    #   |ΔF| / max(|F|, 1) ≤ ftol   →  at NLL≈2100, ftol=1e-9 → |ΔF| < ~2e-6.
    # It also has an independent gradient-norm criterion (gtol, default 1e-5).
    # SLSQP treats ftol as an *absolute* tolerance on |ΔF|; gtol is N/A.
    #
    # maxls (L-BFGS-B only): max line-search steps per iteration (default 20).
    # Increasing to 50 reduces the chance of premature step rejection when NPs
    # have very different scales or the Hessian approximation is poor early on.
    extra_opts: dict = {}
    if method == "L-BFGS-B":
        extra_opts["gtol"] = gtol
        extra_opts["maxls"] = 50
    result = minimize(
        nll,
        x0,
        method=method,
        jac=jac,
        options={"maxiter": maxiter, "ftol": ftol, **extra_opts},
        bounds=bounds,
    )
    return result, free_names


def make_run_dir() -> Path:
    """Create a timestamped directory under 'runs/' for this scan's output.

    Returns the created Path, e.g. runs/2026-05-23_143022/.
    A symlink 'runs/latest' is kept pointing to the most recent run.
    """
    runs_root = Path("runs")
    runs_root.mkdir(exist_ok=True)

    tag = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_dir = runs_root / tag
    run_dir.mkdir()

    # Update (or create) the 'latest' symlink.
    latest = runs_root / "latest"
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(tag)

    print(f"  Results will be written to {run_dir}/")
    print(f"  (symlinked as {latest})")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--workspace",
        "-w",
        type=Path,
        default=None,
        metavar="WS.json",
        help="Workspace JSON to load (default: data/<WS_FILE>, then skhep-testdata)",
    )
    parser.add_argument(
        "--method",
        "-m",
        default="SLSQP",
        metavar="METHOD",
        help=(
            "Minimization method (default: %(default)s).\n"
            "iminuit methods (require: pixi add iminuit):\n"
            "  MIGRAD     — Minuit2 MIGRAD: the same algorithm used by ROOT/TMinuit.\n"
            "               Variable-metric quasi-Newton with internal gradient tuning\n"
            "               and automatic parameter scaling. Best for HEP likelihoods.\n"
            "  IMPROVE    — MIGRAD, then SIMPLEX (Nelder-Mead) to escape local minima,\n"
            "               then MIGRAD again to re-converge precisely. Slower but\n"
            "               more robust than MIGRAD alone.\n"
            "scipy methods:\n"
            "  SLSQP      — Sequential Least Squares; handles bounds and constraints;\n"
            "               ftol is absolute |ΔF|. Good general-purpose default.\n"
            "  L-BFGS-B   — Quasi-Newton with bounds; ftol is relative |ΔF|/|F|;\n"
            "               efficient for many nuisance parameters.\n"
            "  trust-constr — Trust-region interior-point; more robust than L-BFGS-B\n"
            "               on ill-conditioned problems; slower per iteration.\n"
            "  TNC        — Truncated Newton with bounds; gradient-based like\n"
            "               L-BFGS-B but uses a conjugate-gradient inner loop.\n"
            "  Nelder-Mead — Derivative-free simplex; robust when the NLL surface\n"
            "               is noisy or non-smooth; slower convergence.\n"
            "  Powell     — Derivative-free direction-set method; faster than\n"
            "               Nelder-Mead on smooth surfaces; no bounds support.\n"
            "See: https://docs.scipy.org/doc/scipy/reference/generated/"
            "scipy.optimize.minimize.html\n"
        ),
    )
    parser.add_argument(
        "--start-from",
        default=None,
        metavar="PARAM_SET[,PARAM_SET,…]",
        help=(
            "Comma-separated list of workspace parameter-point names to use as\n"
            "the starting NP values for the scan (default: model nominal values).\n"
            "Example: --start-from unconditionalNuis_muhat,unconditionalGlobs_muhat\n"
            "These NPs come from an actual fit to data so they are close to the\n"
            "true minimum for any mu_HH near the best-fit, greatly reducing the\n"
            "number of optimizer iterations needed.\n"
            "With --no-warm-start, every scan point restarts from these values\n"
            "rather than from the previous point's solution.\n"
            "Available sets are printed when an unknown name is requested."
        ),
    )
    parser.add_argument(
        "--ftol",
        type=float,
        default=1e-9,
        help="Function-value tolerance (default: 1e-9). For L-BFGS-B this is a\n"
        "relative criterion: |ΔF|/max(|F|,1) < ftol; at NLL≈2100 this means\n"
        "|ΔF| < ~2e-6. For SLSQP it is an absolute |ΔF| criterion.",
    )
    parser.add_argument(
        "--scan-start",
        type=float,
        default=0.9,
        metavar="MU",
        help="mu_HH value closest to the expected best-fit; the bidirectional scan\n"
        "starts here and moves outward so every step has a close warm-start.\n"
        "Default: 0.9 (near the reference minimum for this workspace).\n"
        "Use --scan-start -0.5 --no-warm-start to recover the original\n"
        "sequential left-to-right cold-start behaviour.",
    )
    parser.add_argument(
        "--no-warm-start",
        action="store_true",
        help="Disable warm-starting: every scan point begins from the nominal\n"
        "parameter values instead of the previous point's solution.\n"
        "Combined with --scan-start <grid_min> this recovers the original\n"
        "sequential cold-start scan (e.g. --scan-start -0.5 --no-warm-start).",
    )
    parser.add_argument(
        "--gtol",
        type=float,
        default=1e-6,
        help="Gradient-norm tolerance for L-BFGS-B (default: 1e-6, ignored by SLSQP).",
    )
    parser.add_argument(
        "--migrad-strategy",
        type=int,
        default=1,
        choices=[0, 1, 2],
        metavar="{0,1,2}",
        help=(
            "Minuit strategy for MIGRAD/IMPROVE (default: 1).\n"
            "  0 — fastest; least accurate gradient/Hessian.\n"
            "  1 — default balance between speed and accuracy.\n"
            "  2 — most accurate; uses finer finite-difference steps for the\n"
            "      gradient, which helps when NPs are near bounds or the NLL\n"
            "      surface is not smooth. More function calls per iteration.\n"
            "Ignored for scipy methods."
        ),
    )
    parser.add_argument(
        "--maxiter",
        type=int,
        default=10000,
        help="Maximum number of optimizer iterations per scan point (default: 10000).",
    )
    parser.add_argument(
        "--analytic-grad",
        action="store_true",
        help="Compile an analytic gradient function via PyTensor autodiff.\n"
        "Gives exact gradients but compilation can take hours for large models.\n"
        "Default: off (use scipy '3-point' central-difference numerical gradients).",
    )
    parser.add_argument(
        "--jax",
        action="store_true",
        help="Enable the JAX/XLA backend instead of PyTensor compiled functions.\n"
        "JAX compiles the NLL via XLA and provides automatic differentiation,\n"
        "but can use very large amounts of memory (>64 GB) for complex models.\n"
        "Default: off.",
    )
    args = parser.parse_args()

    ws_path, cache_dir = resolve_workspace(args.workspace)
    print(f"Cache directory: {cache_dir}/")

    model = build_model(ws_path, cache_dir)

    # Build the NP starting-point override from workspace parameter sets.
    # The workspace is re-parsed here (fast: just JSON); no cache invalidation needed.
    x0_from_args: dict[str, float] | None = None
    if args.start_from:
        print(f"\nLoading starting-point NPs from '{args.start_from}' ...")
        _ws_params = pyhs3.Workspace.load(ws_path)
        # NamedCollection iterates over ParameterSet objects (not name strings).
        available = [ps.name for ps in _ws_params.parameter_points]
        x0_from_args = {}
        for _pp_name in args.start_from.split(","):
            _pp_name = _pp_name.strip()
            if _pp_name not in _ws_params.parameter_points:
                print(
                    f"  Warning: '{_pp_name}' not found in workspace.\n"
                    f"  Available parameter sets: {available}"
                )
                continue
            # ws.parameter_points[name] → ParameterSet; iterate over ParameterPoint.
            _pset = _ws_params.parameter_points[_pp_name]
            for _p in _pset:  # ParameterSet.__iter__ yields ParameterPoint objects
                x0_from_args[_p.name] = float(_p.value)
            print(f"  '{_pp_name}': {len(_pset)} parameters loaded.")
        if not x0_from_args:
            print("  No valid parameter sets found; falling back to nominal values.")
            x0_from_args = None
        del _ws_params  # free memory

    print("\nCompiling log_prob ...")
    t0 = time.perf_counter()
    log_prob_fn, grad_fn, input_names = compile_log_prob(
        model,
        cache_dir,
        analytic_grad=args.analytic_grad,
        use_jax=args.jax,
    )
    print(f"  Done in {time.perf_counter() - t0:.1f}s")
    print(f"  {len(input_names)} free inputs to compiled function")
    if grad_fn is not None:
        print(
            "  Analytic gradient available — numerical finite differences not needed."
        )

    run_dir = make_run_dir()

    # Record the run configuration so results are reproducible.
    run_config = {
        "workspace": str(ws_path),
        "method": args.method,
        "ftol": args.ftol,
        "gtol": args.gtol,
        "maxiter": args.maxiter,
        "use_jax": args.jax,
        "start_from": args.start_from,
        "no_warm_start": args.no_warm_start,
        "scan_start": args.scan_start,
    }
    with (run_dir / "run_config.json").open("w") as f:
        json.dump(run_config, f, indent=2)
    print(
        f"  Run config: method={args.method}, ftol={args.ftol}, "
        f"gtol={args.gtol}, maxiter={args.maxiter}"
    )

    # Write the starting-point values for all free parameters once per run.
    # compare_postfit.py reads this to add an "initial" column to its table.
    init_path = run_dir / "initial_values.txt"
    with init_path.open("w") as f:
        for name, value in model.free_params.items():
            f.write(f"{name}: {value}\n")
    print(f"  Initial values written to {init_path.name}")

    MU_GRID = [
        -0.5,
        -0.4,
        -0.3,
        -0.2,
        -0.1,
        0.0,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0,
        1.1,
        1.2,
        1.3,
        1.4,
        1.5,
        1.6,
        1.7,
        1.8,
        1.9,
        2.0,
        2.1,
        2.2,
        2.3,
        2.4,
        2.5,
    ]
    mu_arr = np.array(MU_GRID)
    summary_path = run_dir / "scan_summary.json"

    # -----------------------------------------------------------------------
    # Bidirectional warm-start scan
    #
    # Starting from near the expected profile minimum (--scan-start, default
    # mu_HH≈0.9) and scanning outward in both directions means every scan point
    # inherits its NP starting values from a converged neighbour.  The profile
    # minimum moves smoothly with mu_HH, so a warm-started point typically
    # converges in < 20 iterations rather than > 100 from the cold nominal.
    #
    # Pass order:
    #   right pass:  [start_idx, start_idx+1, ..., n-1]  (increasing mu)
    #   left  pass:  [start_idx-1, start_idx-2, ..., 0]  (decreasing mu)
    # The left pass is seeded with the converged NP values from start_idx.
    # -----------------------------------------------------------------------
    _start_idx = int(np.argmin(np.abs(mu_arr - args.scan_start)))
    _right_pass = list(range(_start_idx, len(MU_GRID)))
    _left_pass = list(range(_start_idx - 1, -1, -1))

    # results_by_idx[i] = {"mu": ..., "nll": ..., "x": np.ndarray, ...}
    results_by_idx: dict[int, dict] = {}
    _free_names_global: list[str] | None = None  # identical for every scan point

    def _flush_summary() -> None:
        """Write scan_summary.json sorted by MU_GRID order after every point."""
        sorted_entries = [
            {k: v for k, v in results_by_idx[i].items() if k != "x"}
            for i in sorted(results_by_idx)
        ]
        with summary_path.open("w") as _f:
            json.dump(sorted_entries, _f, indent=2)

    def _run_scan_pass(
        indices: list[int],
        warm_x0: dict[str, float] | None,
    ) -> dict[str, float] | None:
        """Run one directional pass over the listed grid indices.

        Returns the NP values from the last point for seeding the next pass.
        """
        nonlocal _free_names_global
        last_x0 = warm_x0
        for idx in indices:
            mu = MU_GRID[idx]
            t0 = time.perf_counter()
            result, free_names_local = profile_nll(
                log_prob_fn,
                grad_fn,
                input_names,
                model,
                mu_val=mu,
                method=args.method,
                ftol=args.ftol,
                gtol=args.gtol,
                maxiter=args.maxiter,
                x0_override=last_x0,
                migrad_strategy=args.migrad_strategy,
            )
            dt = time.perf_counter() - t0
            if _free_names_global is None:
                _free_names_global = free_names_local

            # If the warm-started fit failed to converge, retry once from the
            # nominal starting point (cold start).  Accept whichever gives the
            # lower NLL so we never leave a clearly-stuck result on the table.
            # Skip when --no-warm-start is set: we already cold-started.
            cold_tried = False
            if not result.success and last_x0 is not None and not args.no_warm_start:
                t1 = time.perf_counter()
                result_cold, _ = profile_nll(
                    log_prob_fn,
                    grad_fn,
                    input_names,
                    model,
                    mu_val=mu,
                    method=args.method,
                    ftol=args.ftol,
                    gtol=args.gtol,
                    maxiter=args.maxiter,
                    x0_override=None,
                    migrad_strategy=args.migrad_strategy,
                )
                cold_tried = True
                if result_cold.fun < result.fun:
                    result = result_cold
                    print(
                        f"  (cold-start retry: NLL={result.fun:.4f}, "
                        f"{time.perf_counter() - t1:.1f}s)"
                    )

            # Update warm start from the current (best) result regardless of
            # convergence flag — even a non-converged result is a better start
            # than the cold nominal for the next neighbouring scan point.
            # When --no-warm-start is set, keep last_x0=None so every point
            # cold-starts from the nominal values.
            if not args.no_warm_start:
                last_x0 = dict(zip(free_names_local, result.x, strict=False))

            converge_str = "✓" if result.success else ("?" if cold_tried else "✗")
            print(
                f"mu={mu:+.1f}  -2ln(L)={result.fun:.6f}  "
                f"{converge_str} ({result.nit} iter, {result.nfev} fn evals, {dt:.1f}s)"
            )
            if not result.success and hasattr(result, "message"):
                print(f"  ↳ {result.message}")

            out_path = run_dir / f"results_mu_{mu}.txt"
            with out_path.open("w") as f:
                for name, value in zip(free_names_local, result.x, strict=True):
                    f.write(f"{name}: {value}\n")

            results_by_idx[idx] = {
                "mu": mu,
                "nll": result.fun,
                "converged": bool(result.success),
                "nit": int(result.nit),
                "nfev": int(result.nfev),
                "time_s": round(dt, 2),
                "x": result.x,  # kept in-memory only; not written to JSON
            }
            _flush_summary()

        return last_x0

    print(f"\nBidirectional scan: starting at mu={MU_GRID[_start_idx]:+.1f} ...")
    if x0_from_args:
        print(f"  Initial NPs from --start-from ({len(x0_from_args)} values).")
    _right_last = _run_scan_pass(_right_pass, warm_x0=x0_from_args)

    # Seed the leftward pass.
    # • With warm-starting: use the start-index solution (closest neighbour).
    # • With --no-warm-start: reuse x0_from_args (or None for cold nominal)
    #   so every left-pass point also starts from the requested values.
    if args.no_warm_start:
        _seed_left = x0_from_args
    elif _free_names_global is not None and _start_idx in results_by_idx:
        _seed_left = dict(
            zip(_free_names_global, results_by_idx[_start_idx]["x"], strict=False)
        )
    else:
        _seed_left = x0_from_args
    _run_scan_pass(_left_pass, warm_x0=_seed_left)

    # Build ordered arrays for plotting (MU_GRID order, not scan order).
    computed_nlls_ordered = [results_by_idx[i]["nll"] for i in range(len(MU_GRID))]

    provided_nll = _REFERENCE["nll"]
    provided_nll_shifted = [2 * (v - min(provided_nll)) for v in provided_nll]
    computed_nll_shifted = [
        v - min(computed_nlls_ordered) for v in computed_nlls_ordered
    ]
    plt.figure()
    plt.scatter(mu_arr, provided_nll_shifted, label="provided nll", marker="o")
    plt.scatter(mu_arr, computed_nll_shifted, label="computed nll", marker="x")
    plt.xlabel("mu_HH")
    plt.ylabel(r"$-2\,\Delta(\ln\Lambda)$")
    plt.legend()
    plot_path = run_dir / "nll_comparison_minimization.pdf"
    plt.savefig(plot_path)
    print(f"\nPlot saved to {plot_path}")


if __name__ == "__main__":
    main()
