#!/usr/bin/env python3
"""
Compare optimized computational graphs between vectorized and individual approaches.
"""

import numpy as np
import pytensor.tensor as pt
from pytensor import function
from pytensor.printing import debugprint
from pytensor.graph.rewriting.db import RewriteDatabaseQuery
from pytensor.compile.mode import get_default_mode


def create_small_test_case():
    """Create a small test case to examine optimization."""
    print("=" * 60)
    print("OPTIMIZED GRAPH COMPARISON (Small Case)")
    print("=" * 60)

    # Use small size to avoid compilation limits
    rates = [1.0, 2.0, 3.0]
    x_value = 3.0

    # Vectorized approach
    rates_tensor = pt.constant(rates)
    x = pt.constant(x_value)
    log_probs_vec = x * pt.log(rates_tensor) - rates_tensor - pt.gammaln(x + 1)
    vectorized = pt.prod(pt.exp(log_probs_vec))

    # Individual approach
    individual_probs = []
    for rate in rates:
        rate_const = pt.constant(rate)
        log_prob = x * pt.log(rate_const) - rate_const - pt.gammaln(x + 1)
        prob = pt.exp(log_prob)
        individual_probs.append(prob)
    individual = pt.prod(pt.stack(individual_probs))

    print("BEFORE OPTIMIZATION:")
    print("\n1. Vectorized approach:")
    debugprint(vectorized)
    print("\n2. Individual approach:")
    debugprint(individual)

    # Compile functions to see optimized graphs
    print("\n" + "=" * 60)
    print("AFTER COMPILATION/OPTIMIZATION:")
    print("=" * 60)

    f_vec = function([], vectorized)
    f_ind = function([], individual)

    # Access the optimized graphs
    print("\n1. Vectorized optimized graph:")
    debugprint(f_vec.maker.fgraph.outputs[0])

    print("\n2. Individual optimized graph:")
    debugprint(f_ind.maker.fgraph.outputs[0])

    # Compare results
    result_vec = f_vec()
    result_ind = f_ind()

    print(f"\nResults:")
    print(f"Vectorized: {result_vec:.10e}")
    print(f"Individual: {result_ind:.10e}")
    print(f"Difference: {abs(result_vec - result_ind):.2e}")

    return f_vec, f_ind


def analyze_optimization_stages():
    """Analyze optimization at different stages."""
    print("\n" + "=" * 60)
    print("OPTIMIZATION STAGE ANALYSIS")
    print("=" * 60)

    rates = [1.0, 2.0, 3.0]
    x_value = 3.0

    # Create individual approach
    individual_probs = []
    x = pt.constant(x_value)
    for rate in rates:
        rate_const = pt.constant(rate)
        log_prob = x * pt.log(rate_const) - rate_const - pt.gammaln(x + 1)
        prob = pt.exp(log_prob)
        individual_probs.append(prob)
    individual = pt.prod(pt.stack(individual_probs))

    print("1. BEFORE ANY OPTIMIZATION:")
    debugprint(individual)

    # Apply specific optimization passes manually
    from pytensor.graph.rewriting.basic import out2in
    from pytensor.tensor.rewriting.basic import constant_folding
    from pytensor.graph.fg import FunctionGraph

    # Create a function graph
    fg = FunctionGraph([], [individual])

    print(f"\nNodes before optimization: {len(fg.apply_nodes)}")

    # Apply constant folding
    try:
        out2in(constant_folding).optimize(fg)
        print(f"Nodes after constant folding: {len(fg.apply_nodes)}")
        print("\n2. AFTER CONSTANT FOLDING:")
        debugprint(fg.outputs[0])
    except Exception as e:
        print(f"Constant folding failed: {e}")

    # Try with default optimizations
    print("\n3. WITH FULL COMPILATION:")
    try:
        f_full = function([], individual)
        print(f"Nodes after full optimization: {len(f_full.maker.fgraph.apply_nodes)}")
        debugprint(f_full.maker.fgraph.outputs[0])
    except Exception as e:
        print(f"Full compilation failed: {e}")


def compare_with_parameters():
    """Compare optimized graphs when using parameters instead of constants."""
    print("\n" + "=" * 60)
    print("PARAMETER-BASED GRAPH OPTIMIZATION")
    print("=" * 60)

    # Using parameters for fair comparison
    rates_var = pt.dvector("rates")
    x_value = 3.0
    x = pt.constant(x_value)

    # Vectorized with parameters
    log_probs_vec = x * pt.log(rates_var) - rates_var - pt.gammaln(x + 1)
    vectorized_param = pt.prod(pt.exp(log_probs_vec))

    # Individual with parameters (small size)
    individual_probs_param = []
    for i in range(3):
        rate_i = rates_var[i]
        log_prob = x * pt.log(rate_i) - rate_i - pt.gammaln(x + 1)
        prob = pt.exp(log_prob)
        individual_probs_param.append(prob)
    individual_param = pt.prod(pt.stack(individual_probs_param))

    print("BEFORE OPTIMIZATION (with parameters):")
    print("\n1. Vectorized:")
    debugprint(vectorized_param)
    print("\n2. Individual:")
    debugprint(individual_param)

    # Compile and check optimized graphs
    f_vec_param = function([rates_var], vectorized_param)
    f_ind_param = function([rates_var], individual_param)

    print("\nAFTER OPTIMIZATION (with parameters):")
    print("\n1. Vectorized optimized:")
    debugprint(f_vec_param.maker.fgraph.outputs[0])
    print("\n2. Individual optimized:")
    debugprint(f_ind_param.maker.fgraph.outputs[0])

    # Test with actual data
    test_rates = np.array([1.0, 2.0, 3.0])
    result_vec_param = f_vec_param(test_rates)
    result_ind_param = f_ind_param(test_rates)

    print(f"\nResults with parameters:")
    print(f"Vectorized: {result_vec_param:.10e}")
    print(f"Individual: {result_ind_param:.10e}")
    print(f"Difference: {abs(result_vec_param - result_ind_param):.2e}")

    # Compare graph complexities
    print(f"\nGraph complexity comparison:")
    print(f"Vectorized nodes: {len(f_vec_param.maker.fgraph.apply_nodes)}")
    print(f"Individual nodes: {len(f_ind_param.maker.fgraph.apply_nodes)}")

    return f_vec_param, f_ind_param


def detailed_node_analysis():
    """Detailed analysis of what nodes are present in each graph."""
    print("\n" + "=" * 60)
    print("DETAILED NODE ANALYSIS")
    print("=" * 60)

    rates_var = pt.dvector("rates")
    x = pt.constant(3.0)

    # Create both approaches
    log_probs_vec = x * pt.log(rates_var) - rates_var - pt.gammaln(x + 1)
    vectorized = pt.prod(pt.exp(log_probs_vec))

    individual_probs = []
    for i in range(3):
        rate_i = rates_var[i]
        log_prob = x * pt.log(rate_i) - rate_i - pt.gammaln(x + 1)
        individual_probs.append(pt.exp(log_prob))
    individual = pt.prod(pt.stack(individual_probs))

    # Compile
    f_vec = function([rates_var], vectorized)
    f_ind = function([rates_var], individual)

    def analyze_nodes(fgraph, name):
        print(f"\n{name} - Node Analysis:")
        print(f"Total nodes: {len(fgraph.apply_nodes)}")

        # Count node types
        node_types = {}
        for node in fgraph.apply_nodes:
            op_name = type(node.op).__name__
            node_types[op_name] = node_types.get(op_name, 0) + 1

        print("Node types:")
        for op_type, count in sorted(node_types.items()):
            print(f"  {op_type}: {count}")

        # Show the operations in order
        print("Operations in graph:")
        for i, node in enumerate(fgraph.apply_nodes):
            inputs_info = [str(inp.type) for inp in node.inputs]
            print(f"  {i}: {type(node.op).__name__} - inputs: {len(node.inputs)}")

    analyze_nodes(f_vec.maker.fgraph, "VECTORIZED")
    analyze_nodes(f_ind.maker.fgraph, "INDIVIDUAL")

    # Check if they're structurally equivalent
    print(f"\nStructural comparison:")
    vec_ops = [type(node.op).__name__ for node in f_vec.maker.fgraph.apply_nodes]
    ind_ops = [type(node.op).__name__ for node in f_ind.maker.fgraph.apply_nodes]

    print(f"Vectorized ops: {vec_ops}")
    print(f"Individual ops: {ind_ops}")
    print(f"Operations identical: {vec_ops == ind_ops}")


if __name__ == "__main__":
    # Run all analyses
    create_small_test_case()
    analyze_optimization_stages()
    compare_with_parameters()
    detailed_node_analysis()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("Check the output above to see if PyTensor optimizes")
    print("individual operations down to the same graph structure")
    print("as vectorized operations.")