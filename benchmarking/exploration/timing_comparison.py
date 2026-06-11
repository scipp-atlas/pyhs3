#!/usr/bin/env python3
"""
Timing comparison script for logpdf evaluation: ROOT vs pyhs3 (FAST_RUN, NUMBA, JAX modes).

This script compares the performance of logpdf evaluation between:
- ROOT (RooFit implementation)
- pyhs3 with mode='FAST_RUN' (default PyTensor optimized mode)
- pyhs3 with mode='NUMBA' (Numba-compiled mode)
- pyhs3 with mode='JAX' (JAX-compiled mode)

The comparison uses the rf501_simultaneouspdf example for consistency.
"""

import timeit
import numpy as np
from pathlib import Path

import pytensor.compile
from pytensor.graph.basic import explicit_graph_inputs
from pytensor.graph.fg import FunctionGraph
from pytensor.link.jax.dispatch import jax_funcify
import jax.numpy as jnp
import jax

# Try to import ROOT
try:
    import ROOT
    HAS_ROOT = True
except ImportError:
    HAS_ROOT = False
    print("Warning: ROOT not available, skipping ROOT timing")

# Import pyhs3
try:
    import pyhs3
    from pyhs3 import Workspace
    HAS_PYHS3 = True
except ImportError:
    HAS_PYHS3 = False
    print("Error: pyhs3 not available")
    exit(1)


# Configuration
N_EVALUATIONS = 1000
WORKSPACE_PATH = Path(__file__).parent / "tests/test_pdf/rf501_simultaneouspdf.json"

# Evaluation point for the "model" distribution
EVAL_POINT = {
    "x": 0.0,
    "f": 0.2,
    "mean": 0.0,
    "sigma": 0.3,
    "mean2": 0.0,
    "sigma2": 0.3,
}


def build_root_model():
    """Build ROOT model once and return model components.

    Returns a dict containing all ROOT objects to keep them alive.
    """
    if not HAS_ROOT:
        return None

    # Build ROOT model (based on rf501_simultaneouspdf.py)
    x = ROOT.RooRealVar("x", "x", -8, 8)
    mean = ROOT.RooRealVar("mean", "mean", 0, -8, 8)
    sigma = ROOT.RooRealVar("sigma", "sigma", 0.3, 0.1, 10)
    gx = ROOT.RooGaussian("gx", "gx", x, mean, sigma)

    mean2 = ROOT.RooRealVar("mean2", "mean2", 0, -3, 3)
    sigma2 = ROOT.RooRealVar("sigma2", "sigma2", 0.3, 0.1, 10)
    px = ROOT.RooGaussian("px", "px", x, mean2, sigma2)

    f = ROOT.RooRealVar("f", "f", 0.2, 0.0, 1.0)
    model = ROOT.RooAddPdf("model", "model", [gx, px], [f])

    # Set parameter values
    x.setVal(EVAL_POINT["x"])
    f.setVal(EVAL_POINT["f"])
    mean.setVal(EVAL_POINT["mean"])
    sigma.setVal(EVAL_POINT["sigma"])
    mean2.setVal(EVAL_POINT["mean2"])
    sigma2.setVal(EVAL_POINT["sigma2"])

    obs_set = ROOT.RooArgSet(x)

    # Return all objects to keep them alive
    return {
        "model": model,
        "obs_set": obs_set,
        "x": x,
        "mean": mean,
        "sigma": sigma,
        "gx": gx,
        "mean2": mean2,
        "sigma2": sigma2,
        "px": px,
        "f": f
    }


def time_root_logpdf(model, obs_set, n_evals: int, cache: bool = False) -> tuple[float, float]:
    """
    Time ROOT logpdf evaluation.

    Args:
        model: ROOT model to evaluate
        obs_set: ROOT RooArgSet for observables
        n_evals: Number of evaluations to time

    Returns:
        tuple[float, float]: (total_time, time_per_eval)
    """
    if model is None or obs_set is None:
        return None, None

    ROOT.RooAbsArg.setDirtyInhibit(not cache)

    # Evaluation function
    def evaluate_pdf():
        # Get PDF value (ROOT will cache the result since parameters don't change)
        return model.getValV(obs_set)

    # Warmup
    _ = evaluate_pdf()

    # Time evaluations
    total_time = timeit.timeit(evaluate_pdf, number=n_evals)
    time_per_eval = total_time / n_evals

    return total_time, time_per_eval


def build_pyhs3_model(mode: str):
    """Build pyhs3 model once and return compiled function and parameters."""
    if not HAS_PYHS3:
        return None, None, None

    try:
        # Load workspace and build model
        workspace = Workspace.load(WORKSPACE_PATH)
        model = workspace.model(mode=mode if mode != 'JAX_TRANSPILE' else 'FAST_RUN', progress=False)

        # Get the compiled function and parameter order
        dist = model.distributions["model"]

        # Get compiled function (this is what pdf() does internally)
        compiled_func = model._get_compiled_function("model")

        # Get inputs in the order expected by the compiled function
        inputs = [var for var in explicit_graph_inputs([dist])]

        # Build positional argument list in the correct order (once, outside timing loop)
        # Convert to numpy arrays for proper PyTensor/JAX/Numba handling
        positional_values = tuple(np.array(EVAL_POINT[var.name]) for var in inputs)

        return model, compiled_func, positional_values

    except Exception as e:
        print(f"  ❌ {mode} model build failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def time_pyhs3_logpdf(compiled_func, positional_values, mode: str, n_evals: int) -> tuple[float, float]:
    """
    Time pyhs3 logpdf evaluation for a given mode.

    Args:
        compiled_func: Compiled PyTensor function
        positional_values: Tuple of positional argument values
        mode: PyTensor compilation mode ('FAST_RUN', 'NUMBA', 'JAX')
        n_evals: Number of evaluations to time

    Returns:
        tuple[float, float]: (total_time, time_per_eval)
    """
    if compiled_func is None or positional_values is None:
        return None, None

    # Evaluation function - just calls the compiled function
    def evaluate_pdf():
        # Call the compiled function directly with positional arguments
        return compiled_func(*positional_values)

    # Warmup - this triggers compilation for JAX/NUMBA
    print(f"  {mode} warming up (compiling)...")
    _ = evaluate_pdf()

    # Time evaluations
    total_time = timeit.timeit(evaluate_pdf, number=n_evals)
    time_per_eval = total_time / n_evals

    return total_time, time_per_eval

def time_jax_transpile_logpdf(model: pyhs3.Model, name: str, positional_values, n_evals: int) -> tuple[float, float]:

    dist = model.distributions[name]
    inputs: list[TensorVar] = [
        var for var in explicit_graph_inputs([dist])
    ]

    fgraph = FunctionGraph(inputs=inputs, outputs=[dist], clone=True)
    pytensor.compile.mode.JAX.optimizer.rewrite(fgraph)

    compiled_func = jax_funcify(fgraph)
    positional_values_jnp = tuple(jnp.array(value) for value in positional_values)

    # Evaluation function - just calls the compiled function
    @jax.jit
    def evaluate_pdf():
        # Call the compiled function directly with positional arguments
        return compiled_func(*positional_values_jnp)

    # Warmup - this triggers compilation for JAX/NUMBA
    print(f"  warming up (compiling)...")
    _ = evaluate_pdf()

    # Time evaluations
    total_time = timeit.timeit(evaluate_pdf, number=n_evals)
    time_per_eval = total_time / n_evals

    return total_time, time_per_eval

def main():
    """Main timing comparison."""
    print("=" * 60)
    print("logpdf() Timing Comparison: ROOT vs pyhs3")
    print("=" * 60)
    print(f"Workspace: {WORKSPACE_PATH}")
    print(f"Distribution: model")
    print(f"Evaluation point: {EVAL_POINT}")
    print(f"Number of evaluations: {N_EVALUATIONS:,}")
    print()

    # Build models once
    print("=" * 60)
    print("BUILDING MODELS")
    print("=" * 60)

    root_components = None
    pyhs3_models = {}

    if HAS_ROOT:
        print("Building ROOT model...")
        root_components = build_root_model()
        print("  ✅ ROOT model built")

    if HAS_PYHS3:
        for mode in ["FAST_RUN", "NUMBA", "JAX", "JAX_TRANSPILE"]:
            print(f"Building pyhs3 {mode} model...")
            model, compiled_func, positional_values = build_pyhs3_model(mode)
            if compiled_func is not None:
                pyhs3_models[mode] = (model, compiled_func, positional_values)
                print(f"  ✅ pyhs3 {mode} model built")
    print()

    # Validate that ROOT and pyhs3 compute the same values
    print("=" * 60)
    print("VALIDATION: Checking ROOT vs pyhs3 PDF values match")
    print("=" * 60)

    if root_components is not None and "FAST_RUN" in pyhs3_models:
        root_pdf = root_components["model"].getValV(root_components["obs_set"])
        model, compiled_func, positional_values = pyhs3_models["FAST_RUN"]
        pyhs3_pdf = float(compiled_func(*positional_values))

        print(f"ROOT PDF value:  {root_pdf:.15f}")
        print(f"pyhs3 PDF value: {pyhs3_pdf:.15f}")
        print(f"Difference: {abs(root_pdf - pyhs3_pdf):.2e}")
        print(f"Relative difference: {abs(root_pdf - pyhs3_pdf) / root_pdf * 100:.6f}%")

        if np.allclose(root_pdf, pyhs3_pdf, rtol=1e-10):
            print("✅ Values match! Proceeding with timing comparison...")
        else:
            print("❌ WARNING: Values DO NOT match! Timing comparison may not be fair.")
        print()

    # Results storage
    results = {}

    for cache in [True, False]:
        # Test ROOT
        if root_components is not None:
            print(f"Testing ROOT{' (cached)' if cache else ''}:")
            total_time, time_per_eval = time_root_logpdf(root_components["model"], root_components["obs_set"], N_EVALUATIONS, cache=cache)

            if total_time is not None:
                results[("ROOT", "native" if cache else "uncached")] = {
                    "total_time": total_time,
                    "time_per_eval": time_per_eval,
                    "evaluations_per_sec": N_EVALUATIONS / total_time
                }
                print(f"  ✅ {N_EVALUATIONS:,} evaluations: {total_time:.4f}s total, {time_per_eval*1000:.4f}ms per eval")
                print(f"  📈 {results[('ROOT', 'native')]['evaluations_per_sec']:.0f} evaluations/second")
            else:
                print(f"  ❌ ROOT failed")
            print()

    # Test pyhs3 modes
    for mode in ["FAST_RUN", "NUMBA", "JAX", "JAX_TRANSPILE"]:
        if mode not in pyhs3_models:
            continue

        print(f"Testing pyhs3 {mode} mode:")
        model, compiled_func, positional_values = pyhs3_models[mode]
        if mode == 'JAX_TRANSPILE':
            total_time, time_per_eval = time_jax_transpile_logpdf(model, "model", positional_values, N_EVALUATIONS)
        else:
            total_time, time_per_eval = time_pyhs3_logpdf(compiled_func, positional_values, mode, N_EVALUATIONS)

        if total_time is not None:
            results[("pyhs3", mode)] = {
                "total_time": total_time,
                "time_per_eval": time_per_eval,
                "evaluations_per_sec": N_EVALUATIONS / total_time
            }
            print(f"  ✅ {N_EVALUATIONS:,} evaluations: {total_time:.4f}s total, {time_per_eval*1000:.4f}ms per eval")
            print(f"  📈 {results[('pyhs3', mode)]['evaluations_per_sec']:.0f} evaluations/second")
        print()

    # Summary comparison
    print("=" * 60)
    print("PERFORMANCE SUMMARY")
    print("=" * 60)

    if results:
        # Sort by performance (fastest first)
        sorted_results = sorted(results.items(), key=lambda x: x[1]["time_per_eval"])

        print("Ranked by performance (fastest first):")
        print()

        fastest_time = sorted_results[0][1]["time_per_eval"]

        for i, ((package, mode), result) in enumerate(sorted_results, 1):
            speedup = result["time_per_eval"] / fastest_time
            speedup_label = "(baseline)" if speedup == 1.0 else f"{speedup:.1f}x slower"
            print(f"{i}. {package:15} | {mode:15} | {result['time_per_eval']*1000:8.4f}ms/eval | "
                  f"{result['evaluations_per_sec']:8.0f} eval/s | {speedup_label}")

        print()
        print("Key findings:")
        best_mode = sorted_results[0][0]
        worst_mode = sorted_results[-1][0]
        performance_ratio = sorted_results[-1][1]["time_per_eval"] / fastest_time

        print(f"✅ Fastest: {best_mode[0]} ({best_mode[1]}) - {sorted_results[0][1]['time_per_eval']*1000:.4f}ms per evaluation")
        print(f"🐌 Slowest: {worst_mode[0]} ({worst_mode[1]}) - {sorted_results[-1][1]['time_per_eval']*1000:.4f}ms per evaluation")
        print(f"📊 Performance ratio: {performance_ratio:.1f}x difference between fastest and slowest")
    else:
        print("❌ No successful timing results obtained")


if __name__ == "__main__":
    main()
