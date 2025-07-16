#!/usr/bin/env python3
"""
Example usage of PyHS3 with compilation optimization and visualization features.

This script demonstrates:
1. FAST_COMPILE mode (minimal optimization)
2. FAST_RUN mode for better performance (default behavior)
3. JAX mode for JAX-compiled execution
4. Graph visualization and inspection
5. Performance comparison between modes
"""

import json
import time
from contextlib import contextmanager
from pathlib import Path

from skhep_testdata import data_path as skhep_testdata_path

import pyhs3 as hs3


@contextmanager
def time_block(label):
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    print(f"{label}: {end - start:.4f} seconds")


def main():
    """Main example function demonstrating PyHS3 features."""
    print("=== PyHS3 Example: Compilation and Visualization ===\n")

    # Load test data
    print("Loading test data...")
    with time_block("Loading and parsing JSON"):
        fpath = Path(skhep_testdata_path("test_hs3_unbinned_pyhs3_validation_issue41.json"))
        ws_json = json.loads(fpath.read_text(encoding="utf-8"))
        ws = hs3.Workspace(ws_json)

    # Example 1: Basic Usage (FAST_COMPILE Mode)
    print("1. Basic Usage (FAST_COMPILE Mode)")
    print("=" * 40)

    # Use FAST_COMPILE mode for comparison
    with time_block("Creating FAST_COMPILE model"):
        model_interpreted = ws.model(mode="FAST_COMPILE")
    print(f"Model created: {model_interpreted}")

    # Prepare parameter values
    with time_block("Preparing parameter values"):
        parametervalues = {par.name: par.value for par in model_interpreted.parameterset}

    # Evaluate PDF
    with time_block("FAST_COMPILE mode evaluation"):
        result1 = model_interpreted.logpdf('_model_Run2HM_1', **parametervalues)

    print(f"LogPDF result: {result1}")
    print()

    # Example 2: FAST_RUN Mode (Optimized)
    print("2. FAST_RUN Mode (Optimized)")
    print("=" * 40)

    # Use default behavior (FAST_RUN mode)
    with time_block("Creating FAST_RUN model"):
        model_compiled = ws.model()  # mode="FAST_RUN" by default
    print(f"Model created: {model_compiled}")

    # First call compiles the function
    with time_block("First FAST_RUN evaluation (includes compilation)"):
        result2 = model_compiled.logpdf('_model_Run2HM_1', **parametervalues)
    print(f"LogPDF result (first call): {result2}")

    # Subsequent calls use cached compiled function
    with time_block("Cached FAST_RUN evaluation"):
        result3 = model_compiled.logpdf('_model_Run2HM_1', **parametervalues)
    print(f"LogPDF result (cached): {result3}")
    print()

    # Example 3: JAX Mode (if available)
    print("3. JAX Mode (if available)")
    print("=" * 40)
    
    # Check if JAX is available by trying to create a JAX model
    jax_available = False
    model_jax = None
    try:
        with time_block("Creating JAX model"):
            model_jax = ws.model(mode="JAX")
        print(f"JAX model created: {model_jax}")
        
        # Test JAX evaluation
        with time_block("First JAX evaluation (includes compilation)"):
            result_jax = model_jax.logpdf('_model_Run2HM_1', **parametervalues)
        print(f"JAX LogPDF result: {result_jax}")
        
        # Cached JAX evaluation
        with time_block("Cached JAX evaluation"):
            result_jax_cached = model_jax.logpdf('_model_Run2HM_1', **parametervalues)
        print(f"JAX LogPDF result (cached): {result_jax_cached}")
        
        jax_available = True
        
    except Exception as e:
        print(f"JAX mode not available: {e}")
        print("To enable JAX mode, install: pip install jax jaxlib")
    
    print()

    # Example 4: Performance Comparison
    print("4. Performance Comparison")
    print("=" * 40)

    # Test with a different parameter value
    modified_params = {**parametervalues, 'atlas_invMass_Run2HM_1': 110.250}

    # FAST_COMPILE mode
    with time_block("FAST_COMPILE mode (different parameters)"):
        result_interpreted = model_interpreted.logpdf('_model_Run2HM_1', **modified_params)

    # FAST_RUN mode (cached)
    with time_block("FAST_RUN mode (different parameters)"):
        result_compiled = model_compiled.logpdf('_model_Run2HM_1', **modified_params)

    print(f"FAST_COMPILE result: {result_interpreted}")
    print(f"FAST_RUN result:     {result_compiled}")
    
    # JAX mode if available
    if jax_available:
        with time_block("JAX mode (different parameters)"):
            result_jax_diff = model_jax.logpdf('_model_Run2HM_1', **modified_params)
        print(f"JAX result:          {result_jax_diff}")

    # Performance benchmark with different evaluation counts
    print("\nPerformance Benchmark:")
    if jax_available:
        print("Comparing FAST_COMPILE (10 evals), FAST_RUN (10000 evals), and JAX (10000 evals)...")
    else:
        print("Comparing FAST_COMPILE (10 evals) and FAST_RUN (10000 evals)...")
    print("-" * 40)

    # Generate parameter variations for different numbers of evaluations
    import numpy as np
    base_mass = parametervalues.get('atlas_invMass_Run2HM_1', 125.0)
    
    # 10 variations for FAST_COMPILE
    mass_variations_10 = [base_mass + i * 0.5 for i in range(10)]
    
    # 10000 variations for FAST_RUN and JAX (reuse same 10 variations in a loop)
    mass_variations_10000 = mass_variations_10 * 1000

    # Benchmark FAST_COMPILE mode (10 evaluations)
    interpreted_results = []
    start_time = time.perf_counter()
    for mass in mass_variations_10:
        params = {**parametervalues, 'atlas_invMass_Run2HM_1': mass}
        result = model_interpreted.logpdf('_model_Run2HM_1', **params)
        interpreted_results.append(result)
    interpreted_time = time.perf_counter() - start_time
    avg_interpreted_time = interpreted_time / len(mass_variations_10)
    print(f"FAST_COMPILE mode: {len(mass_variations_10)} evaluations, avg {avg_interpreted_time:.6f} sec/eval")

    # Benchmark FAST_RUN mode (10000 evaluations)
    compiled_results = []
    start_time = time.perf_counter()
    for mass in mass_variations_10000:
        params = {**parametervalues, 'atlas_invMass_Run2HM_1': mass}
        result = model_compiled.logpdf('_model_Run2HM_1', **params)
        compiled_results.append(result)
    compiled_time = time.perf_counter() - start_time
    avg_compiled_time = compiled_time / len(mass_variations_10000)
    print(f"FAST_RUN mode: {len(mass_variations_10000)} evaluations, avg {avg_compiled_time:.6f} sec/eval")

    # Benchmark JAX mode if available (10000 evaluations)
    jax_results = []
    if jax_available:
        start_time = time.perf_counter()
        for mass in mass_variations_10000:
            params = {**parametervalues, 'atlas_invMass_Run2HM_1': mass}
            result = model_jax.logpdf('_model_Run2HM_1', **params)
            jax_results.append(result)
        jax_time = time.perf_counter() - start_time
        avg_jax_time = jax_time / len(mass_variations_10000)
        print(f"JAX mode: {len(mass_variations_10000)} evaluations, avg {avg_jax_time:.6f} sec/eval")

    # Verify results are consistent (compare first 10 results)
    max_diff_fast = max(abs(i - c) for i, c in zip(interpreted_results, compiled_results[:10]))
    print(f"Maximum difference (FAST_COMPILE vs FAST_RUN): {max_diff_fast:.2e}")
    
    if jax_available:
        max_diff_jax = max(abs(c - j) for c, j in zip(compiled_results[:10], jax_results[:10]))
        print(f"Maximum difference (FAST_RUN vs JAX): {max_diff_jax:.2e}")
    print()

    # Example 5: Multiple Distribution Analysis
    print("5. Multiple Distribution Analysis")
    print("=" * 40)

    print("Available distributions:")
    for dist_name in model_compiled.distributions.keys():
        print(f"  - {dist_name}")

    # Analyze the first few distributions with different modes
    models_to_analyze = [("FAST_RUN", model_compiled)]
    if jax_available:
        models_to_analyze.append(("JAX", model_jax))
    
    for mode_name, model in models_to_analyze:
        print(f"\nAnalyzing with {mode_name} mode:")
        for i, dist_name in enumerate(list(model.distributions.keys())[:2]):  # Analyze fewer to save space
            print(f"\n{dist_name} ({mode_name}):")
            try:
                summary = model.graph_summary(dist_name)
                # Print just the key metrics
                lines = summary.strip().split('\n')
                for line in lines[1:4]:  # Skip the title, show first 3 metrics
                    print(f"  {line.strip()}")
            except Exception as e:
                print(f"  Error analyzing {dist_name}: {e}")

    print("\n=== Example Complete ===")


if __name__ == "__main__":
    main()
