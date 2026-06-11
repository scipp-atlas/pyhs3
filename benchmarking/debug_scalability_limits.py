#!/usr/bin/env python3
"""
Test scalability limits of individual vs vectorized approaches.
"""

import numpy as np
import pytensor.tensor as pt
from pytensor import function
import time

def test_vectorized_scalability():
    """Test how vectorized operations scale with size."""
    print("=" * 60)
    print("VECTORIZED APPROACH SCALABILITY")
    print("=" * 60)

    sizes = [100, 500, 1000, 5000, 10000]
    x_value = 3.0

    for size in sizes:
        print(f"Testing vectorized with {size} dimensions...")

        try:
            # Create large tensor
            rates_var = pt.dvector("rates")
            x = pt.constant(x_value)

            # Vectorized approach - should work for any size
            log_probs = x * pt.log(rates_var) - rates_var - pt.gammaln(x + 1)
            result = pt.exp(pt.sum(log_probs))

            # Try to compile
            f = function([rates_var], result)

            # Test with actual data
            test_rates = np.random.uniform(0.5, 5.0, size)
            result_val = f(test_rates)

            print(f"  ✅ Success: {size} dimensions compiled and executed")
            print(f"      Result: {result_val:.2e}")

        except Exception as e:
            print(f"  ❌ Failed: {size} dimensions - {str(e)[:100]}...")

        print()

def test_individual_scalability():
    """Test how individual operations scale with size."""
    print("=" * 60)
    print("INDIVIDUAL APPROACH SCALABILITY")
    print("=" * 60)

    sizes = [10, 50, 100, 200, 300, 400, 500]  # More granular to find the limit
    x_value = 3.0

    for size in sizes:
        print(f"Testing individual operations with {size} dimensions...")

        try:
            # Individual approach - create separate operations for each element
            rates_list = [pt.dscalar(f"rate_{i}") for i in range(size)]
            x = pt.constant(x_value)

            # Create individual Poisson probabilities
            individual_probs = []
            for i, rate in enumerate(rates_list):
                log_prob = x * pt.log(rate) - rate - pt.gammaln(x + 1)
                prob = pt.exp(log_prob)
                individual_probs.append(prob)

            # Product of all probabilities
            result = pt.prod(pt.stack(individual_probs))

            # Try to compile - this is where it should fail for large sizes
            inputs = rates_list
            f = function(inputs, result)

            # Test with actual data
            test_rates = np.random.uniform(0.5, 5.0, size).tolist()
            result_val = f(*test_rates)

            print(f"  ✅ Success: {size} individual operations compiled and executed")
            print(f"      Result: {result_val:.2e}")

        except Exception as e:
            print(f"  ❌ Failed: {size} individual operations")
            print(f"      Error: {str(e)[:200]}...")
            if "bracket nesting" in str(e):
                print(f"      → This is the C++ compiler bracket limit!")
            break  # Stop testing larger sizes once we hit the limit

        print()

def test_makevector_limit():
    """Test the specific MakeVector limit that caused the original failure."""
    print("=" * 60)
    print("MAKEVECTOR APPROACH (Original Failure)")
    print("=" * 60)

    sizes = [100, 200, 300, 400, 500, 600]
    x_value = 3.0

    for size in sizes:
        print(f"Testing MakeVector approach with {size} dimensions...")

        try:
            # This mimics what the original script was doing
            rates = np.random.uniform(0.5, 5.0, size).tolist()
            x = pt.constant(x_value)

            # Create individual operations for each rate (as constants)
            individual_probs = []
            for rate in rates:
                rate_const = pt.constant(rate)
                log_prob = x * pt.log(rate_const) - rate_const - pt.gammaln(x + 1)
                prob = pt.exp(log_prob)
                individual_probs.append(prob)

            # This creates MakeVector with many inputs
            result = pt.prod(pt.stack(individual_probs))

            # Try to compile
            f = function([], result)
            result_val = f()

            print(f"  ✅ Success: {size} MakeVector elements compiled and executed")
            print(f"      Result: {result_val:.2e}")

        except Exception as e:
            print(f"  ❌ Failed: {size} MakeVector elements")
            print(f"      Error: {str(e)[:200]}...")
            if "bracket nesting" in str(e):
                print(f"      → Found the bracket nesting limit at {size} elements!")
            break

        print()

if __name__ == "__main__":
    test_vectorized_scalability()
    test_individual_scalability()
    test_makevector_limit()