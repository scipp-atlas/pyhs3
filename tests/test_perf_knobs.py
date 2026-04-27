"""
Regression guards for performance-critical pytensor compilation settings.

See issue #115 for benchmark data showing the impact of these flags.
"""

from __future__ import annotations

import numpy as np
import pytest

import pyhs3 as hs3


@pytest.fixture
def simple_workspace():
    """Minimal workspace with one Gaussian distribution."""
    return hs3.Workspace(
        metadata={"hs3_version": "0.2"},
        distributions=[
            {
                "name": "gauss",
                "type": "gaussian_dist",
                "x": "x",
                "mean": "mu",
                "sigma": "sigma",
            }
        ],
        parameter_points=[
            {
                "name": "default_values",
                "parameters": [
                    {"name": "x", "value": 0.0},
                    {"name": "mu", "value": 0.0},
                    {"name": "sigma", "value": 1.0},
                ],
            }
        ],
        domains=[
            {
                "name": "default_domain",
                "type": "product_domain",
                "axes": [
                    {"name": "x", "min": -5.0, "max": 5.0},
                    {"name": "mu", "min": -2.0, "max": 2.0},
                    {"name": "sigma", "min": 0.1, "max": 3.0},
                ],
            }
        ],
    )


class TestCompiledFunctionFlags:
    """Regression guards for pytensor compiled-function performance flags."""

    def test_trust_input_is_true(self, simple_workspace):
        """Compiled functions must have trust_input=True (issue #115).

        trust_input=True skips per-call type-checking in the pytensor VM,
        reducing FAST_RUN overhead from ~11x to ~2x vs ROOT. This test
        prevents accidental regression to trust_input=False.
        """
        model = simple_workspace.model()
        # Force compilation by calling pdf once
        model.pdf_unsafe(
            "gauss", x=np.array([0.0]), mu=np.array(0.0), sigma=np.array(1.0)
        )

        fn = model._compiled_functions["gauss"]
        # pytensor Function objects expose trust_input on their vm
        assert fn.trust_input is True

    def test_compiled_function_accepts_numpy_array_inputs(self, simple_workspace):
        """pdf_unsafe returns finite values when given numpy array inputs."""
        model = simple_workspace.model()
        result = model.pdf_unsafe(
            "gauss", x=np.float64(0.0), mu=np.float64(0.0), sigma=np.float64(1.0)
        )
        assert np.all(np.isfinite(result))

    def test_compiled_function_caches_across_calls(self, simple_workspace):
        """Second pdf_unsafe call reuses the cached compiled function object."""
        model = simple_workspace.model()
        model.pdf_unsafe(
            "gauss", x=np.array([0.0]), mu=np.array(0.0), sigma=np.array(1.0)
        )
        fn_first = model._compiled_functions.get("gauss")
        assert fn_first is not None
        model.pdf_unsafe(
            "gauss", x=np.array([1.0]), mu=np.array(0.0), sigma=np.array(1.0)
        )
        fn_second = model._compiled_functions.get("gauss")
        assert fn_second is not None
        # Should be the exact same object (no recompilation)
        assert fn_first is fn_second
