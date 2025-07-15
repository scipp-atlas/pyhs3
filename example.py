#!/usr/bin/env python3
"""
Example usage of PyHS3 with compilation optimization and visualization features.

This script demonstrates:
1. Interpreted mode (compile=False)
2. Compiled mode for better performance (default behavior)
3. Graph visualization and inspection
4. Performance comparison between modes
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

    # Example 1: Basic Usage (Interpreted Mode)
    print("1. Basic Usage (Interpreted Mode)")
    print("=" * 40)

    # Explicitly disable compilation for comparison
    with time_block("Creating interpreted model"):
        model_interpreted = ws.model(compile=False)
    print(f"Model created: {model_interpreted}")

    # Prepare parameter values
    with time_block("Preparing parameter values"):
        parametervalues = {par.name: par.value for par in model_interpreted.parameterset}

    # Evaluate PDF
    with time_block("Interpreted mode evaluation"):
        result1 = model_interpreted.logpdf('_model_Run2HM_1', **parametervalues)

    print(f"LogPDF result: {result1}")
    print()

    # Example 2: Compiled Mode (Optimized)
    print("2. Compiled Mode (Optimized)")
    print("=" * 40)

    # Use default behavior (compilation enabled)
    with time_block("Creating compiled model"):
        model_compiled = ws.model()  # compile=True by default
    print(f"Model created: {model_compiled}")

    # First call compiles the function
    with time_block("First compiled evaluation (includes compilation)"):
        result2 = model_compiled.logpdf('_model_Run2HM_1', **parametervalues)
    print(f"LogPDF result (first call): {result2}")

    # Subsequent calls use cached compiled function
    with time_block("Cached compiled evaluation"):
        result3 = model_compiled.logpdf('_model_Run2HM_1', **parametervalues)
    print(f"LogPDF result (cached): {result3}")
    print()

    # Example 3: Performance Comparison
    print("3. Performance Comparison")
    print("=" * 40)

    # Test with a different parameter value
    modified_params = {**parametervalues, 'atlas_invMass_Run2HM_1': 110.250}

    # Interpreted mode
    with time_block("Interpreted mode (different parameters)"):
        result_interpreted = model_interpreted.logpdf('_model_Run2HM_1', **modified_params)

    # Compiled mode (cached)
    with time_block("Compiled mode (different parameters)"):
        result_compiled = model_compiled.logpdf('_model_Run2HM_1', **modified_params)

    print(f"Interpreted result: {result_interpreted}")
    print(f"Compiled result:    {result_compiled}")

    # Performance benchmark with multiple evaluations
    print("\nPerformance Benchmark (10 evaluations):")
    print("-" * 40)

    # Generate some parameter variations
    import numpy as np
    base_mass = parametervalues.get('atlas_invMass_Run2HM_1', 125.0)
    mass_variations = [base_mass + i * 0.5 for i in range(10)]

    # Benchmark interpreted mode
    interpreted_results = []
    with time_block("Interpreted mode (10 evaluations)"):
        for mass in mass_variations:
            params = {**parametervalues, 'atlas_invMass_Run2HM_1': mass}
            result = model_interpreted.logpdf('_model_Run2HM_1', **params)
            interpreted_results.append(result)

    # Benchmark compiled mode
    compiled_results = []
    with time_block("Compiled mode (10 evaluations)"):
        for mass in mass_variations:
            params = {**parametervalues, 'atlas_invMass_Run2HM_1': mass}
            result = model_compiled.logpdf('_model_Run2HM_1', **params)
            compiled_results.append(result)

    # Verify results are consistent
    max_diff = max(abs(i - c) for i, c in zip(interpreted_results, compiled_results))
    print(f"Maximum difference between modes: {max_diff:.2e}")
    print()

    # Example 4: Graph Visualization and Inspection
    print("4. Graph Visualization and Inspection")
    print("=" * 40)

    # Get model overview (like numpy arrays)
    print("Model Overview:")
    print(model_compiled)
    print()

    # Get detailed graph information
    print("Graph Summary:")
    print(model_compiled.graph_summary('_model_Run2HM_1'))

    # Visualize the computation graph
    try:
        print("Generating graph visualization...")
        with time_block("Creating SVG graph"):
            output_file = model_compiled.visualize_graph('_model_Run2HM_1', format='svg')
        print(f"Graph saved to: {output_file}")

        # Also create a PNG version
        with time_block("Creating PNG graph"):
            png_file = model_compiled.visualize_graph('_model_Run2HM_1', format='png')
        print(f"PNG graph saved to: {png_file}")

    except ImportError as e:
        print(f"Graph visualization not available: {e}")
        print("To enable visualization, install: pip install pydot")

    print()

    # Example 5: Multiple Distribution Analysis
    print("5. Multiple Distribution Analysis")
    print("=" * 40)

    print("Available distributions:")
    for dist_name in model_compiled.distributions.keys():
        print(f"  - {dist_name}")

    # Analyze the first few distributions
    for i, dist_name in enumerate(list(model_compiled.distributions.keys())[:3]):
        print(f"\n{dist_name}:")
        try:
            summary = model_compiled.graph_summary(dist_name)
            # Print just the key metrics
            lines = summary.strip().split('\n')
            for line in lines[1:4]:  # Skip the title, show first 3 metrics
                print(f"  {line.strip()}")
        except Exception as e:
            print(f"  Error analyzing {dist_name}: {e}")

    print("\n=== Example Complete ===")


if __name__ == "__main__":
    main()
