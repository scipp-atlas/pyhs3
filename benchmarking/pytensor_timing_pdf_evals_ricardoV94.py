
#!/usr/bin/env python3
"""
Timing comparison script for pytensor/mode Gaussian implementations vs numba-stats.

Compares PyTensor Gaussian PDF evaluation performance across different modes:
- FAST_RUN (default optimized mode)
- JAX (if available)
- NUMBA (if available)

Also compares against numba-stats norm.pdf as baseline.
"""

import time
import timeit
import numpy as np
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
mu = np.array(2.0)
sigma = np.array(3.0)
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
    mean = pt.scalar("mean")
    sigma = pt.scalar("sigma")

    norm_const = 1.0 / (
        pt.sqrt(2 * math.pi) * sigma
    )
    exponent = pt.exp( -0.5 * ( ( x - mean) / sigma) ** 2)
    return norm_const * exponent, [x, mean, sigma]


def time_pytensor_mode(dist, inputs, mode, x_vals, mu_val, sigma_val, n_evals):
    func = function(
        inputs=inputs,
        outputs=dist,
        mode=mode,
        on_unused_input="ignore",
        trust_input=True,
    )

    def evaluate_pdf():
        return func(x_vals, mu_val, sigma_val)

    total_time = timeit.timeit(evaluate_pdf, setup=evaluate_pdf, number=n_evals)
    time_per_eval = total_time / n_evals

    return total_time, time_per_eval

def time_numba_stats(x_vals, mu_val, sigma_val, n_evals):
    """Time numba-stats evaluation."""
    if not HAS_NUMBA_STATS:
        return None, None

    mu_val = mu_val.item()
    sigma_val = sigma_val.item()

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
    print("Gaussian PDF Timing Comparison")
    print("=" * 60)
    print(f"Test parameters:")
    print(f"  x: {len(x)} points from {x[0]:.1f} to {x[-1]:.1f}")
    print(f"  mu: {mu}")
    print(f"  sigma: {sigma}")
    print(f"  Evaluations: {N_EVALUATIONS:,}")
    print()

    print("Setting up pytensor graph...")
    graph, inputs = setup_pytensor_graph()
    print("✅ pytensor graph created")
    print()

    # Results storage
    results = {}

    # Test pytensor modes
    pytensor_modes = ["FAST_RUN", "NUMBA", "JAX"]

    for mode in pytensor_modes:
        print(f"Testing PyTensor {mode} mode:")
        total_time, time_per_eval = time_pytensor_mode(graph, inputs, mode, np.array(x), np.array(mu), np.array(sigma), N_EVALUATIONS)

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
