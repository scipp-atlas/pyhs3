#!/usr/bin/env python3
"""
Debug script to understand PyTensor optimization for vectorized vs individual Poisson calculations.

This script explores:
1. Numerical equivalence between vectorized and individual Poisson calculations
2. Computational graph differences and optimization behavior
3. Performance characteristics of different approaches
"""

import numpy as np
import pytensor.tensor as pt
from pytensor import function
from pytensor.printing import debugprint
import time
from typing import List, Callable


def create_vectorized_poisson(rates_tensor: pt.TensorVariable, x_value: float) -> pt.TensorVariable:
    """Create vectorized Poisson calculation: prod(Poisson(x | rates))"""
    x = pt.constant(x_value)

    # Vectorized Poisson log probabilities: x * log(rates) - rates - log(x!)
    log_probs = x * pt.log(rates_tensor) - rates_tensor - pt.gammaln(x + 1)

    # Convert to probabilities and take product
    probs = pt.exp(log_probs)
    result = pt.prod(probs)

    return result


def create_individual_poissons(rates_list: List[float], x_value: float) -> pt.TensorVariable:
    """Create individual Poisson calculations: prod([Poisson(x | rate) for rate in rates])"""
    x = pt.constant(x_value)

    # Individual Poisson probabilities
    individual_probs = []
    for rate in rates_list:
        rate_tensor = pt.constant(rate)
        log_prob = x * pt.log(rate_tensor) - rate_tensor - pt.gammaln(x + 1)
        prob = pt.exp(log_prob)
        individual_probs.append(prob)

    # Product of individual probabilities
    result = pt.prod(pt.stack(individual_probs))

    return result


def create_hybrid_approach(rates_tensor: pt.TensorVariable, x_value: float) -> pt.TensorVariable:
    """Create hybrid approach: exp(sum(log_probs)) instead of prod(exp(log_probs))"""
    x = pt.constant(x_value)

    # Vectorized Poisson log probabilities
    log_probs = x * pt.log(rates_tensor) - rates_tensor - pt.gammaln(x + 1)

    # Sum log probabilities and exponentiate (more numerically stable)
    result = pt.exp(pt.sum(log_probs))

    return result


def create_parameterized_vectorized(x_value: float) -> tuple[pt.TensorVariable, pt.TensorVariable]:
    """Create parameterized vectorized version for performance testing"""
    rates_var = pt.dvector("rates")
    x = pt.constant(x_value)

    log_probs = x * pt.log(rates_var) - rates_var - pt.gammaln(x + 1)
    result = pt.exp(pt.sum(log_probs))

    return result, rates_var


def analyze_computational_graphs():
    """Analyze the computational graphs for different approaches."""
    print("=" * 60)
    print("COMPUTATIONAL GRAPH ANALYSIS")
    print("=" * 60)

    # Test parameters
    rates = [1.0, 2.0, 3.0, 4.0, 5.0]
    x_value = 3.0

    rates_tensor = pt.constant(rates)

    print(f"Test case: rates = {rates}, x = {x_value}")
    print()

    # Create different approaches
    vectorized = create_vectorized_poisson(rates_tensor, x_value)
    individual = create_individual_poissons(rates, x_value)
    hybrid = create_hybrid_approach(rates_tensor, x_value)

    # Print computational graphs
    print("1. VECTORIZED APPROACH: pt.prod(pt.exp(x * log(rates) - rates - gammaln(x+1)))")
    print("-" * 40)
    debugprint(vectorized)
    print()

    print("2. INDIVIDUAL APPROACH: pt.prod([pt.exp(x * log(rate) - rate - gammaln(x+1)) for rate in rates])")
    print("-" * 40)
    debugprint(individual)
    print()

    print("3. HYBRID APPROACH: pt.exp(pt.sum(x * log(rates) - rates - gammaln(x+1)))")
    print("-" * 40)
    debugprint(hybrid)
    print()

    return vectorized, individual, hybrid


def test_numerical_equivalence():
    """Test numerical equivalence between different approaches."""
    print("=" * 60)
    print("NUMERICAL EQUIVALENCE TESTING")
    print("=" * 60)

    # Test cases with different characteristics
    test_cases = [
        ([1.0, 2.0, 3.0, 4.0, 5.0], 3.0, "Basic case"),
        ([0.1, 0.5, 1.0, 2.0, 10.0], 1.0, "Wide range of rates"),
        ([1.0] * 10, 5.0, "Identical rates (10 dims)"),
        ([0.01, 0.02, 0.03], 0.0, "Small rates, x=0"),
        ([100.0, 200.0, 300.0], 150.0, "Large values"),
    ]

    for rates, x_value, description in test_cases:
        print(f"Test case: {description}")
        print(f"  rates = {rates}, x = {x_value}")

        rates_tensor = pt.constant(rates)

        # Create expressions
        vectorized = create_vectorized_poisson(rates_tensor, x_value)
        individual = create_individual_poissons(rates, x_value)
        hybrid = create_hybrid_approach(rates_tensor, x_value)

        # Compile and evaluate
        f_vectorized = function([], vectorized)
        f_individual = function([], individual)
        f_hybrid = function([], hybrid)

        result_vectorized = f_vectorized()
        result_individual = f_individual()
        result_hybrid = f_hybrid()

        # Check equivalence
        diff_vec_ind = abs(result_vectorized - result_individual)
        diff_vec_hyb = abs(result_vectorized - result_hybrid)
        diff_ind_hyb = abs(result_individual - result_hybrid)

        print(f"  Vectorized result: {result_vectorized:.2e}")
        print(f"  Individual result: {result_individual:.2e}")
        print(f"  Hybrid result:     {result_hybrid:.2e}")
        print(f"  |Vectorized - Individual|: {diff_vec_ind:.2e}")
        print(f"  |Vectorized - Hybrid|:     {diff_vec_hyb:.2e}")
        print(f"  |Individual - Hybrid|:     {diff_ind_hyb:.2e}")

        # Check if they're equivalent within machine precision
        tolerance = 1e-14
        equivalent = all([
            diff_vec_ind < tolerance,
            diff_vec_hyb < tolerance,
            diff_ind_hyb < tolerance
        ])
        print(f"  Numerically equivalent: {equivalent}")
        print()


def benchmark_performance():
    """Benchmark performance of different approaches."""
    print("=" * 60)
    print("PERFORMANCE BENCHMARKING")
    print("=" * 60)

    # Test different dimensionalities (avoid 500+ to prevent compilation errors)
    dimensions = [5, 10, 50, 100]
    x_value = 3.0
    num_trials = 1000

    print(f"Number of trials per test: {num_trials}")
    print(f"x_value: {x_value}")
    print()

    results = []

    for n_dims in dimensions:
        print(f"Testing {n_dims} dimensions...")

        # Create test rates
        rates = np.random.uniform(0.5, 5.0, n_dims).tolist()

        # Create parameterized version for fair comparison
        vectorized_expr, rates_var = create_parameterized_vectorized(x_value)
        f_vectorized = function([rates_var], vectorized_expr)

        # Individual approach (compiled once)
        individual_expr = create_individual_poissons(rates, x_value)
        f_individual = function([], individual_expr)

        # Benchmark vectorized approach
        start_time = time.time()
        for _ in range(num_trials):
            result_vec = f_vectorized(rates)
        time_vectorized = time.time() - start_time

        # Benchmark individual approach
        start_time = time.time()
        for _ in range(num_trials):
            result_ind = f_individual()
        time_individual = time.time() - start_time

        # Calculate speedup
        speedup = time_individual / time_vectorized if time_vectorized > 0 else float('inf')

        print(f"  Vectorized time: {time_vectorized:.4f}s ({time_vectorized/num_trials*1000:.3f}ms per call)")
        print(f"  Individual time: {time_individual:.4f}s ({time_individual/num_trials*1000:.3f}ms per call)")
        print(f"  Speedup (individual/vectorized): {speedup:.2f}x")
        print(f"  Result difference: {abs(result_vec - result_ind):.2e}")
        print()

        results.append({
            'dimensions': n_dims,
            'time_vectorized': time_vectorized,
            'time_individual': time_individual,
            'speedup': speedup
        })

    # Summary
    print("PERFORMANCE SUMMARY:")
    print("Dims\tVectorized(ms)\tIndividual(ms)\tSpeedup")
    print("-" * 50)
    for r in results:
        print(f"{r['dimensions']}\t{r['time_vectorized']/num_trials*1000:.3f}\t\t{r['time_individual']/num_trials*1000:.3f}\t\t{r['speedup']:.2f}x")


def analyze_optimization_differences():
    """Analyze how PyTensor optimizes different approaches."""
    print("=" * 60)
    print("OPTIMIZATION ANALYSIS")
    print("=" * 60)

    rates = [1.0, 2.0, 3.0, 4.0, 5.0]
    x_value = 3.0
    rates_tensor = pt.constant(rates)

    # Create expressions
    vectorized = create_vectorized_poisson(rates_tensor, x_value)
    individual = create_individual_poissons(rates, x_value)
    hybrid = create_hybrid_approach(rates_tensor, x_value)

    # Compile with different optimization modes
    print("Analyzing compiled function properties...")
    print()

    # Default compilation
    f_vec_default = function([], vectorized)
    f_ind_default = function([], individual)
    f_hyb_default = function([], hybrid)

    # Fast compilation (less optimization)
    f_vec_fast = function([], vectorized, mode='FAST_COMPILE')
    f_ind_fast = function([], individual, mode='FAST_COMPILE')
    f_hyb_fast = function([], hybrid, mode='FAST_COMPILE')

    # Analyze function properties
    def analyze_function(func, name):
        print(f"{name}:")
        print(f"  Number of apply nodes: {len(func.maker.fgraph.apply_nodes)}")
        print(f"  Number of variables: {len(func.maker.fgraph.variables)}")
        print(f"  Function creation time estimate: Check compilation logs")

        # Try to get more detailed info about the computation graph
        try:
            # This might not work in all PyTensor versions
            ops = [node.op.__class__.__name__ for node in func.maker.fgraph.apply_nodes]
            op_counts = {}
            for op in ops:
                op_counts[op] = op_counts.get(op, 0) + 1
            print(f"  Operations used: {dict(sorted(op_counts.items()))}")
        except Exception as e:
            print(f"  Could not analyze operations: {e}")
        print()

    print("DEFAULT COMPILATION MODE:")
    analyze_function(f_vec_default, "Vectorized")
    analyze_function(f_ind_default, "Individual")
    analyze_function(f_hyb_default, "Hybrid")

    print("FAST_COMPILE MODE:")
    analyze_function(f_vec_fast, "Vectorized (fast)")
    analyze_function(f_ind_fast, "Individual (fast)")
    analyze_function(f_hyb_fast, "Hybrid (fast)")


def test_gradient_computation():
    """Test gradient computation efficiency for different approaches."""
    print("=" * 60)
    print("GRADIENT COMPUTATION ANALYSIS")
    print("=" * 60)

    # Create parameterized versions for gradient testing
    rates_var = pt.dvector("rates")
    x_value = 3.0

    # Vectorized approach with parameter
    x = pt.constant(x_value)
    log_probs_vec = x * pt.log(rates_var) - rates_var - pt.gammaln(x + 1)
    result_vec = pt.exp(pt.sum(log_probs_vec))

    # Compute gradients
    grad_vec = pt.grad(result_vec, rates_var)

    # Compile functions
    f_func = function([rates_var], result_vec)
    f_grad = function([rates_var], grad_vec)

    # Test gradients
    test_rates = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    print(f"Test rates: {test_rates}")
    print(f"Function value: {f_func(test_rates):.6e}")
    print(f"Gradient: {f_grad(test_rates)}")
    print()

    # Numerical gradient check
    def numerical_gradient(rates, h=1e-8):
        grad = np.zeros_like(rates)
        for i in range(len(rates)):
            rates_plus = rates.copy()
            rates_minus = rates.copy()
            rates_plus[i] += h
            rates_minus[i] -= h
            grad[i] = (f_func(rates_plus) - f_func(rates_minus)) / (2 * h)
        return grad

    numerical_grad = numerical_gradient(test_rates)
    analytical_grad = f_grad(test_rates)

    print("Gradient verification:")
    print(f"  Analytical: {analytical_grad}")
    print(f"  Numerical:  {numerical_grad}")
    print(f"  Difference: {np.abs(analytical_grad - numerical_grad)}")
    print(f"  Max diff:   {np.max(np.abs(analytical_grad - numerical_grad)):.2e}")


def main():
    """Run all analyses."""
    print("PyTensor Poisson Optimization Analysis")
    print("=" * 60)
    print()

    # Run analyses
    try:
        vectorized, individual, hybrid = analyze_computational_graphs()
        test_numerical_equivalence()
        benchmark_performance()
        analyze_optimization_differences()
        test_gradient_computation()

    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

    print("=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print()
    print("Key takeaways:")
    print("1. Check numerical equivalence results above")
    print("2. Compare performance scaling with dimensions")
    print("3. Review computational graph differences")
    print("4. Consider gradient computation efficiency for optimization")


if __name__ == "__main__":
    main()