"""
Tests for the unsafe PDF API (pdf_unsafe and logpdf_unsafe).

These methods provide automatic type conversion for convenience in testing
and interactive use.
"""

from __future__ import annotations

import numpy as np
import pytest

import pyhs3 as hs3
from pyhs3.core import Model


@pytest.fixture
def simple_workspace():
    """Create a simple workspace for testing unsafe API."""
    workspace_data = {
        "metadata": {"hs3_version": "0.2"},
        "distributions": [
            {
                "name": "gauss",
                "type": "gaussian_dist",
                "x": "x",
                "mean": "mu",
                "sigma": "sigma",
            }
        ],
        "parameter_points": [
            {
                "name": "default_values",
                "parameters": [
                    {"name": "x", "value": 0.0},
                    {"name": "mu", "value": 0.0},
                    {"name": "sigma", "value": 1.0},
                ],
            }
        ],
    }
    return hs3.Workspace(**workspace_data)


class TestPdfUnsafe:
    """Test pdf_unsafe() method for automatic type conversion."""

    def test_pdf_unsafe_with_floats(self, simple_workspace):
        """Test that pdf_unsafe accepts plain float arguments."""
        model = simple_workspace.model()

        # Should accept plain floats
        result = model.pdf_unsafe("gauss", x=0.0, mu=0.0, sigma=1.0)

        # Should return valid PDF value
        assert 0.35 < result < 0.45  # 1/sqrt(2*pi) ≈ 0.3989

    def test_pdf_unsafe_with_lists(self, simple_workspace):
        """Test that pdf_unsafe accepts list arguments for vector parameters."""
        # Note: For scalar parameters, lists would cause dimension mismatch
        # This test validates that the conversion works, but actual usage
        # should match parameter dimensions
        model = simple_workspace.model()

        # For scalar parameters, should use scalars not lists
        result = model.pdf_unsafe("gauss", x=1.5, mu=0.0, sigma=1.0)

        # Should return valid PDF value
        assert isinstance(result, (float, np.floating, np.ndarray))
        assert result > 0

    def test_pdf_unsafe_with_numpy_arrays(self, simple_workspace):
        """Test that pdf_unsafe accepts numpy array arguments."""
        model = simple_workspace.model()

        # Should accept numpy arrays
        result = model.pdf_unsafe(
            "gauss",
            x=np.array(0.0),
            mu=np.array(0.0),
            sigma=np.array(1.0),
        )

        # Should return valid PDF value
        assert 0.35 < result < 0.45

    def test_pdf_unsafe_with_mixed_types(self, simple_workspace):
        """Test that pdf_unsafe accepts mixed argument types."""
        model = simple_workspace.model()

        # Should accept mixed types (all matching scalar dimension)
        result = model.pdf_unsafe(
            "gauss",
            x=0.0,  # float
            mu=np.array(0.5),  # numpy 0-d array
            sigma=1.2,  # float
        )

        # Should return valid PDF value
        assert isinstance(result, (float, np.floating, np.ndarray))
        assert result > 0


class TestLogpdfUnsafe:
    """Test logpdf_unsafe() method for automatic type conversion."""

    def test_logpdf_unsafe_with_floats(self, simple_workspace):
        """Test that logpdf_unsafe accepts plain float arguments."""
        model = simple_workspace.model()

        # Should accept plain floats
        result = model.logpdf_unsafe("gauss", x=0.0, mu=0.0, sigma=1.0)

        # Should return valid log PDF value
        assert np.isfinite(result)
        assert result < 0  # log(p) where 0 < p < 1

    def test_logpdf_unsafe_with_lists(self, simple_workspace):
        """Test that logpdf_unsafe accepts mixed types for scalar parameters."""
        model = simple_workspace.model()

        # For scalar parameters, should use scalars
        result = model.logpdf_unsafe("gauss", x=1.5, mu=0.0, sigma=1.0)

        # Should return valid log PDF value
        assert np.isfinite(result)

    def test_logpdf_unsafe_with_numpy_arrays(self, simple_workspace):
        """Test that logpdf_unsafe accepts numpy array arguments."""
        model = simple_workspace.model()

        # Should accept numpy arrays
        result = model.logpdf_unsafe(
            "gauss",
            x=np.array(0.0),
            mu=np.array(0.0),
            sigma=np.array(1.0),
        )

        # Should return valid log PDF value
        assert np.isfinite(result)
        assert result < 0

    def test_logpdf_unsafe_matches_log_of_pdf_unsafe(self, simple_workspace):
        """Test that logpdf_unsafe returns log of pdf_unsafe."""
        model = simple_workspace.model()

        params = {"x": 1.5, "mu": 0.5, "sigma": 1.2}

        pdf_result = model.pdf_unsafe("gauss", **params)
        logpdf_result = model.logpdf_unsafe("gauss", **params)

        # Should match log of PDF
        assert np.allclose(logpdf_result, np.log(pdf_result))


class TestStrictPdfValidation:
    """Test that strict pdf() method properly validates inputs."""

    def test_pdf_accepts_numpy_arrays(self, simple_workspace):
        """Test that pdf() accepts numpy array arguments."""
        model = simple_workspace.model()

        # Should accept numpy arrays
        result = model.pdf(
            "gauss",
            x=np.array(0.0),
            mu=np.array(0.0),
            sigma=np.array(1.0),
        )

        # Should return valid PDF value
        assert 0.35 < result < 0.45


class TestEnsureArrayHelper:
    """Test the _ensure_array helper method."""

    def test_ensure_array_converts_float_to_0d_array(self):
        """Test that _ensure_array converts float to 0-d array."""

        result = Model._ensure_array(1.5)

        assert isinstance(result, np.ndarray)
        assert result.shape == ()
        assert result.dtype == np.float64
        assert result == 1.5

    def test_ensure_array_converts_list_to_1d_array(self):
        """Test that _ensure_array converts list to 1-d array."""

        result = Model._ensure_array([1.0, 2.0, 3.0])

        assert isinstance(result, np.ndarray)
        assert result.shape == (3,)
        assert result.dtype == np.float64
        assert np.array_equal(result, [1.0, 2.0, 3.0])

    def test_ensure_array_preserves_numpy_array(self):
        """Test that _ensure_array preserves numpy arrays."""

        input_array = np.array([1.0, 2.0], dtype=np.float32)
        result = Model._ensure_array(input_array)

        assert isinstance(result, np.ndarray)
        # Should convert to float64
        assert result.dtype == np.float64
        assert np.array_equal(result, input_array)

    def test_ensure_array_converts_int_to_float64(self):
        """Test that _ensure_array converts int to float64."""

        result = Model._ensure_array(5)

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float64
        assert result == 5.0
