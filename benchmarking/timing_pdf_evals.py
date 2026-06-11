#!/usr/bin/env python3
"""
Timing comparison script for PyHS3 Gaussian implementations vs numba-stats.

Compares PyHS3 Gaussian PDF evaluation performance across different modes:
- FAST_RUN (default optimized mode)
- FAST_COMPILE (faster compilation, slower execution)
- JAX (if available)

Also compares against numba-stats norm.pdf as baseline.
"""

import time
import timeit
import numpy as np
import pyhs3
from contextlib import contextmanager
from pytensor.compile.function import function
import pytensor.tensor as pt
import math

# Import numba-stats for comparison
try:
    from numba_stats import norm as numba_norm
    HAS_NUMBA_STATS = True
except ImportError:
    HAS_NUMBA_STATS = False
    print("Warning: numba-stats not available, skipping numba-stats comparison")

# Test parameters
x = np.linspace(-10, 10, 1000)
mu = 2.0
sigma = 3.0
N_EVALUATIONS = 100000

@contextmanager
def time_block(label):
    """Context manager for timing code blocks."""
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    print(f"{label}: {end - start:.4f} seconds")


def setup_pytensor_graph():
    """Create pytensor Gaussian distribution."""

    x = pt.vector("x")
    mean = pt.vector("mean")
    sigma = pt.vector("sigma")

    norm_const = 1.0 / (
        pt.sqrt(2 * math.pi) * sigma
    )
    exponent = pt.exp( -0.5 * ( ( x - mean) / sigma) ** 2)
    return norm_const * exponent, [x, mean, sigma]


def setup_pyhs3_workspace():
    """Create PyHS3 workspace with Gaussian distribution."""

    workspace_data = {
      "distributions": [
        {
          "mean": "mu",
          "name": "norm",
          "sigma": "sigma",
          "type": "gaussian_dist",
          "x": "x"
        }
      ],
      "domains": [
        {
          "axes": [
            {
              "max": 8.0,
              "min": -8.0,
              "name": "x"
            },
            {
              "max": 8.0,
              "min": -8.0,
              "name": "mu"
            },
            {
              "max": 10.0,
              "min": 0.1,
              "name": "sigma"
            }
          ],
          "name": "default_domain",
          "type": "product_domain"
        }
      ],
      "parameter_points": [
        {
          "name": "default_values",
          "parameters": [
            {
              "name": "x",
              "value": 0.0
            },
            {
              "name": "mu",
              "value": 0.0
            },
            {
              "name": "sigma",
              "value": 0.3
            }
          ]
        }
      ]
    }

    return pyhs3.Workspace(workspace_data)

def time_pytensor_mode(dist, inputs, mode, x_vals, mu_val, sigma_val, n_evals):
    func = function(
        inputs=inputs,
        outputs=dist,
        mode=mode,
        on_unused_input="ignore",
    )

    params = {"mean": mu_val, "sigma": sigma_val, "x": x_vals}

    def evaluate_pdf():
        return func(**params)

    total_time = timeit.timeit(evaluate_pdf, setup=evaluate_pdf, number=n_evals)
    time_per_eval = total_time / n_evals

    return total_time, time_per_eval

def time_pyhs3_mode(ws, mode, x_vals, mu_val, sigma_val, n_evals):
    """Time PyHS3 evaluation for a specific mode."""
    try:
        # Create model with specified mode
        model = ws.model(mode=mode)

        # Parameter values
        params = {"mu": mu_val, "sigma": sigma_val, "x": x_vals}

        # Setup function for timeit (includes JIT warm-up)
        def setup():
            print(f"  {mode} warming up JIT compilation...")
            model.pdf("norm", **params)

        # Timing function for timeit
        def evaluate_pdf():
            return model.pdf("norm", **params)

        # Time multiple evaluations with proper setup
        total_time = timeit.timeit(evaluate_pdf, setup=setup, number=n_evals)
        time_per_eval = total_time / n_evals

        return total_time, time_per_eval

    except Exception as e:
        print(f"  {mode} failed: {e}")
        return None, None

def time_numba_stats(x_vals, mu_val, sigma_val, n_evals):
    """Time numba-stats evaluation."""
    if not HAS_NUMBA_STATS:
        return None, None

    # Setup function for timeit (includes JIT warm-up)
    def setup():
        print("  numba-stats warming up JIT compilation...")
        numba_norm.pdf(x_vals, mu_val, sigma_val)

    # Timing function for timeit
    def evaluate_pdf():
        return numba_norm.pdf(x_vals, mu_val, sigma_val)

    # Time multiple evaluations with proper setup
    total_time = timeit.timeit(evaluate_pdf, setup=setup, number=n_evals)
    time_per_eval = total_time / n_evals

    return total_time, time_per_eval

def main():
    """Main timing comparison."""
    print("=" * 60)
    print("PyHS3 Gaussian PDF Timing Comparison")
    print("=" * 60)
    print(f"Test parameters:")
    print(f"  x: {len(x)} points from {x[0]:.1f} to {x[-1]:.1f}")
    print(f"  mu: {mu}")
    print(f"  sigma: {sigma}")
    print(f"  Evaluations: {N_EVALUATIONS:,}")
    print()

    # Setup PyHS3 workspace
    print("Setting up PyHS3 workspace...")
    ws = setup_pyhs3_workspace()
    graph, inputs = setup_pytensor_graph()
    print("✅ PyHS3 workspace created")
    print()

    # Results storage
    results = {}

    # Test PyHS3 modes
    pytensor_modes = ["FAST_RUN", "JAX", "NUMBA"]

    for mode in pytensor_modes:
        print(f"Testing PyTensor {mode} mode:")
        total_time, time_per_eval = time_pytensor_mode(graph, inputs, mode, x, [mu]*len(x), [sigma]*len(x), N_EVALUATIONS)

        if total_time is not None:
            results[('pytensor', mode)] = {
                "total_time": total_time,
                "time_per_eval": time_per_eval,
                "evaluations_per_sec": N_EVALUATIONS / total_time
            }
            print(f"  ✅ {N_EVALUATIONS:,} evaluations: {total_time:.4f}s total, {time_per_eval*1000:.4f}ms per eval")
            print(f"  📈 {results[('pytensor', mode)]['evaluations_per_sec']:.0f} evaluations/second")
        else:
            print(f"  ❌ {mode} mode failed or unavailable")
        print()


    for mode in pytensor_modes:
        print(f"Testing PyHS3 {mode} mode:")
        total_time, time_per_eval = time_pyhs3_mode(ws, mode, x, [mu]*len(x), [sigma]*len(x), N_EVALUATIONS)

        if total_time is not None:
            results[('pyhs3', mode)] = {
                "total_time": total_time,
                "time_per_eval": time_per_eval,
                "evaluations_per_sec": N_EVALUATIONS / total_time
            }
            print(f"  ✅ {N_EVALUATIONS:,} evaluations: {total_time:.4f}s total, {time_per_eval*1000:.4f}ms per eval")
            print(f"  📈 {results[('pyhs3', mode)]['evaluations_per_sec']:.0f} evaluations/second")
        else:
            print(f"  ❌ {mode} mode failed or unavailable")
        print()

    # Test numba-stats
    if HAS_NUMBA_STATS:
        print("Testing numba-stats baseline:")
        total_time, time_per_eval = time_numba_stats(x, mu, sigma, N_EVALUATIONS)

        if total_time is not None:
            results[("numba-stats", "NUMBA")] = {
                "total_time": total_time,
                "time_per_eval": time_per_eval,
                "evaluations_per_sec": N_EVALUATIONS / total_time
            }
            print(f"  ✅ {N_EVALUATIONS:,} evaluations: {total_time:.4f}s total, {time_per_eval*1000:.4f}ms per eval")
            print(f"  📈 {results[('numba-stats', 'NUMBA')]['evaluations_per_sec']:.0f} evaluations/second")
        else:
            print(f"  ❌ numba-stats failed")
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
            speedup = fastest_time / result["time_per_eval"]
            print(f"{i}. {package:15} | {mode:15} | {result['time_per_eval']*1000:8.4f}ms/eval | "
                  f"{result['evaluations_per_sec']:8.0f} eval/s | "
                  f"{speedup:5.1f}x {'(baseline)' if speedup == 1.0 else 'slower'}")

        print()
        print("Key findings:")
        best_mode = sorted_results[0][0]
        worst_mode = sorted_results[-1][0]
        performance_ratio = sorted_results[-1][1]["time_per_eval"] / fastest_time

        print(f"✅ Fastest: {best_mode} ({sorted_results[0][1]['time_per_eval']*1000:.4f}ms per evaluation)")
        print(f"🐌 Slowest: {worst_mode} ({sorted_results[-1][1]['time_per_eval']*1000:.4f}ms per evaluation)")
        print(f"📊 Performance ratio: {performance_ratio:.1f}x difference between fastest and slowest")

    else:
        print("❌ No successful timing results obtained")

if __name__ == "__main__":
    main()
