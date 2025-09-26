"""
Tests for HistFactory distribution implementation.

Test the HistFactoryDist class with various modifiers and configurations.
"""

from __future__ import annotations

import numpy as np
import pytensor.tensor as pt
import pytest
from pytensor import function

from pyhs3.context import Context
from pyhs3.distributions import HistFactoryDist
from pyhs3.distributions.histfactory import (
    apply_interpolation,
    interpolate_lin,
    interpolate_log,
    interpolate_parabolic,
    interpolate_poly6,
)


class TestHistFactoryDist:
    """Test the HistFactoryDist distribution."""

    def test_basic_creation(self):
        """Test basic HistFactoryDist creation."""
        axes = [{"name": "x", "min": 0.0, "max": 10.0, "nbins": 2}]
        samples = [
            {
                "name": "signal",
                "data": {"contents": [5.0, 3.0], "errors": [1.0, 1.0]},
                "modifiers": [],
            }
        ]

        dist = HistFactoryDist(name="test_channel", axes=axes, samples=samples)

        assert dist.name == "test_channel"
        assert dist.type == "histfactory_dist"
        assert len(dist.axes) == 1
        assert len(dist.samples) == 1

    def test_normfactor_modifier(self):
        """Test HistFactoryDist with normfactor modifier."""
        axes = [{"name": "x", "min": 0.0, "max": 10.0, "nbins": 2}]
        samples = [
            {
                "name": "signal",
                "data": {"contents": [5.0, 3.0], "errors": [1.0, 1.0]},
                "modifiers": [{"type": "normfactor", "parameter": "mu_signal"}],
            }
        ]

        dist = HistFactoryDist(name="test_channel", axes=axes, samples=samples)

        # Check that the distribution can be created
        assert dist.name == "test_channel"
        assert len(dist.samples[0]["modifiers"]) == 1
        assert dist.samples[0]["modifiers"][0]["type"] == "normfactor"

    def test_normsys_modifier(self):
        """Test HistFactoryDist with normsys modifier."""
        axes = [{"name": "x", "min": 0.0, "max": 10.0, "nbins": 2}]
        samples = [
            {
                "name": "background",
                "data": {"contents": [10.0, 8.0], "errors": [2.0, 2.0]},
                "modifiers": [
                    {
                        "type": "normsys",
                        "parameter": "bkg_norm_sys",
                        "constraint": "Gauss",
                        "data": {"hi": 1.1, "lo": 0.9},
                    }
                ],
            }
        ]

        dist = HistFactoryDist(name="test_channel", axes=axes, samples=samples)

        # Check that the distribution can be created
        assert dist.name == "test_channel"
        modifier = dist.samples[0]["modifiers"][0]
        assert modifier["type"] == "normsys"
        assert modifier["constraint"] == "Gauss"
        assert modifier["data"]["hi"] == 1.1
        assert modifier["data"]["lo"] == 0.9

    def test_histosys_modifier(self):
        """Test HistFactoryDist with histosys modifier."""
        axes = [{"name": "x", "min": 0.0, "max": 10.0, "nbins": 2}]
        samples = [
            {
                "name": "signal",
                "data": {"contents": [5.0, 3.0], "errors": [1.0, 1.0]},
                "modifiers": [
                    {
                        "type": "histosys",
                        "parameter": "signal_shape_sys",
                        "constraint": "Gauss",
                        "data": {
                            "hi": {"contents": [0.5, 0.3]},
                            "lo": {"contents": [-0.4, -0.2]},
                        },
                    }
                ],
            }
        ]

        dist = HistFactoryDist(name="test_channel", axes=axes, samples=samples)

        # Check that the distribution can be created
        assert dist.name == "test_channel"
        modifier = dist.samples[0]["modifiers"][0]
        assert modifier["type"] == "histosys"
        assert modifier["constraint"] == "Gauss"
        assert len(modifier["data"]["hi"]["contents"]) == 2
        assert len(modifier["data"]["lo"]["contents"]) == 2

    def test_multiple_modifiers(self):
        """Test HistFactoryDist with multiple modifiers on one sample."""
        axes = [{"name": "x", "min": 0.0, "max": 10.0, "nbins": 2}]
        samples = [
            {
                "name": "signal",
                "data": {"contents": [5.0, 3.0], "errors": [1.0, 1.0]},
                "modifiers": [
                    {"type": "normfactor", "parameter": "mu_signal"},
                    {
                        "type": "normsys",
                        "parameter": "lumi_sys",
                        "constraint": "Gauss",
                        "data": {"hi": 1.05, "lo": 0.95},
                    },
                ],
            }
        ]

        dist = HistFactoryDist(name="test_channel", axes=axes, samples=samples)

        # Check that the distribution can be created
        assert dist.name == "test_channel"
        assert len(dist.samples[0]["modifiers"]) == 2
        assert dist.samples[0]["modifiers"][0]["type"] == "normfactor"
        assert dist.samples[0]["modifiers"][1]["type"] == "normsys"

    def test_multiple_samples(self):
        """Test HistFactoryDist with multiple samples."""
        axes = [{"name": "x", "min": 0.0, "max": 10.0, "nbins": 2}]
        samples = [
            {
                "name": "signal",
                "data": {"contents": [5.0, 3.0], "errors": [1.0, 1.0]},
                "modifiers": [{"type": "normfactor", "parameter": "mu_signal"}],
            },
            {
                "name": "background",
                "data": {"contents": [10.0, 8.0], "errors": [2.0, 2.0]},
                "modifiers": [
                    {
                        "type": "normsys",
                        "parameter": "bkg_norm_sys",
                        "constraint": "Gauss",
                        "data": {"hi": 1.1, "lo": 0.9},
                    }
                ],
            },
        ]

        dist = HistFactoryDist(name="test_channel", axes=axes, samples=samples)

        # Check that the distribution can be created
        assert dist.name == "test_channel"
        assert len(dist.samples) == 2
        assert dist.samples[0]["name"] == "signal"
        assert dist.samples[1]["name"] == "background"

    def test_bin_count_calculation(self):
        """Test calculation of total bins from axes."""
        # Single axis
        dist1 = HistFactoryDist(
            name="test1",
            axes=[{"name": "x", "min": 0.0, "max": 10.0, "nbins": 5}],
            samples=[
                {
                    "name": "s1",
                    "data": {"contents": [1, 2, 3, 4, 5], "errors": [1, 1, 1, 1, 1]},
                    "modifiers": [],
                }
            ],
        )
        assert dist1._get_total_bins() == 5

        # Multiple axes (should multiply)
        dist2 = HistFactoryDist(
            name="test2",
            axes=[
                {"name": "x", "min": 0.0, "max": 10.0, "nbins": 3},
                {"name": "y", "min": 0.0, "max": 5.0, "nbins": 2},
            ],
            samples=[
                {
                    "name": "s1",
                    "data": {
                        "contents": [1, 2, 3, 4, 5, 6],
                        "errors": [1, 1, 1, 1, 1, 1],
                    },
                    "modifiers": [],
                }
            ],
        )
        assert dist2._get_total_bins() == 6  # 3 * 2

    def test_invalid_bin_count(self):
        """Test error handling for mismatched bin counts."""
        axes = [{"name": "x", "min": 0.0, "max": 10.0, "nbins": 2}]
        samples = [
            {
                "name": "signal",
                "data": {
                    "contents": [5.0, 3.0, 2.0],
                    "errors": [1.0, 1.0, 1.0],
                },  # 3 bins instead of 2
                "modifiers": [],
            }
        ]

        dist = HistFactoryDist(name="test_channel", axes=axes, samples=samples)

        # This should raise an error when trying to process the sample

        context = Context({})  # Empty context for testing

        with pytest.raises(ValueError, match="has 3 bins, expected 2"):
            dist._process_sample(context, samples[0], 2)


class TestInterpolationFunctions:
    """Test the interpolation functions used by HistFactory modifiers."""

    def test_linear_interpolation(self):
        """Test linear interpolation function."""
        alpha = pt.dscalar("alpha")
        nom = pt.constant(1.0)
        hi = pt.constant(1.2)
        lo = pt.constant(0.8)

        result = interpolate_lin(alpha, nom, hi, lo)
        f = function([alpha], result)

        # Test key points
        assert np.isclose(f(0.0), 1.0)  # At nominal
        assert np.isclose(f(1.0), 1.2)  # At hi
        assert np.isclose(f(-1.0), 0.8)  # At lo
        assert np.isclose(f(0.5), 1.1)  # Halfway to hi
        assert np.isclose(f(-0.5), 0.9)  # Halfway to lo

    def test_log_interpolation(self):
        """Test logarithmic interpolation function."""
        alpha = pt.dscalar("alpha")
        nom = pt.constant(1.0)
        hi = pt.constant(1.2)
        lo = pt.constant(0.8)

        result = interpolate_log(alpha, nom, hi, lo)
        f = function([alpha], result)

        # Test key points
        assert np.isclose(f(0.0), 1.0)  # At nominal
        assert np.isclose(f(1.0), 1.2)  # At hi
        assert np.isclose(f(-1.0), 0.8)  # At lo

    def test_parabolic_interpolation(self):
        """Test parabolic interpolation function."""
        alpha = pt.dscalar("alpha")
        nom = pt.constant(1.0)
        hi = pt.constant(1.2)
        lo = pt.constant(0.8)

        result = interpolate_parabolic(alpha, nom, hi, lo)
        f = function([alpha], result)

        # Test key points
        assert np.isclose(f(0.0), 1.0)  # At nominal
        # Note: parabolic may not exactly hit hi/lo at ±1

    def test_poly6_interpolation(self):
        """Test 6th-order polynomial interpolation function."""
        alpha = pt.dscalar("alpha")
        nom = pt.constant(1.0)
        hi = pt.constant(1.2)
        lo = pt.constant(0.8)

        result = interpolate_poly6(alpha, nom, hi, lo)
        f = function([alpha], result)

        # Test key points
        assert np.isclose(f(0.0), 1.0)  # At nominal

    def test_apply_interpolation_method_selection(self):
        """Test that apply_interpolation selects the correct method."""
        alpha = pt.dscalar("alpha")
        nom = pt.constant(1.0)
        hi = pt.constant(1.2)
        lo = pt.constant(0.8)

        # Test each method
        for method in ["lin", "log", "parabolic", "poly6"]:
            result = apply_interpolation(method, alpha, nom, hi, lo)
            f = function([alpha], result)
            # Should at least work at nominal point
            assert np.isclose(f(0.0), 1.0)

        # Test default (unknown method should fall back to linear)
        result = apply_interpolation("unknown", alpha, nom, hi, lo)
        f = function([alpha], result)
        assert np.isclose(f(0.0), 1.0)


class TestHistFactoryExpression:
    """Test HistFactory distribution expression evaluation."""

    def test_simple_expression_evaluation(self):
        """Test that we can evaluate a simple HistFactory expression."""

        # Create a simple 2-bin HistFactory model
        axes = [{"name": "x", "min": 0.0, "max": 10.0, "nbins": 2}]
        samples = [
            {
                "name": "signal",
                "data": {"contents": [5.0, 3.0], "errors": [1.0, 1.0]},
                "modifiers": [{"type": "normfactor", "parameter": "mu_signal"}],
            }
        ]

        dist = HistFactoryDist(name="test_channel", axes=axes, samples=samples)

        # Create context with parameters
        mu_signal = pt.dscalar("mu_signal")
        observed_data = pt.dvector("observed_data")

        context = Context(
            {"mu_signal": mu_signal, "test_channel_observed": observed_data}
        )

        # Get the expression
        expr = dist.expression(context)

        # Compile function
        f = function([mu_signal, observed_data], expr)

        # Test evaluation with some values
        mu_val = 1.2  # Signal strength
        obs_val = np.array([6.0, 4.0])  # Observed counts

        result = f(mu_val, obs_val)

        # Should return a finite log probability
        assert np.isfinite(result)
        assert isinstance(result, (float, np.floating, np.ndarray))

    def test_expression_with_constraints(self):
        """Test HistFactory expression with constrained modifiers."""

        axes = [{"name": "x", "min": 0.0, "max": 10.0, "nbins": 2}]
        samples = [
            {
                "name": "background",
                "data": {"contents": [10.0, 8.0], "errors": [2.0, 2.0]},
                "modifiers": [
                    {
                        "type": "normsys",
                        "parameter": "bkg_norm_sys",
                        "constraint": "Gauss",
                        "data": {"hi": 1.1, "lo": 0.9},
                    }
                ],
            }
        ]

        dist = HistFactoryDist(name="test_channel", axes=axes, samples=samples)

        # Create context with parameters
        bkg_norm_sys = pt.dscalar("bkg_norm_sys")
        observed_data = pt.dvector("observed_data")

        context = Context(
            {"bkg_norm_sys": bkg_norm_sys, "test_channel_observed": observed_data}
        )

        # Get the expression
        expr = dist.expression(context)

        # Compile function
        f = function([bkg_norm_sys, observed_data], expr)

        # Test evaluation
        norm_sys_val = 0.5  # 0.5 sigma systematic variation
        obs_val = np.array([11.0, 9.0])  # Observed counts

        result = f(norm_sys_val, obs_val)

        # Should return a finite log probability that includes constraint penalty
        assert np.isfinite(result)

        # Check that constraint penalty affects result
        result_nominal = f(0.0, obs_val)  # No systematic variation
        result_varied = f(1.0, obs_val)  # 1 sigma systematic variation

        # With constraint, moving away from nominal should reduce likelihood
        assert result_nominal > result_varied  # Higher log prob at nominal

    def test_multiple_samples_expression(self):
        """Test HistFactory with multiple samples."""

        axes = [{"name": "x", "min": 0.0, "max": 10.0, "nbins": 2}]
        samples = [
            {
                "name": "signal",
                "data": {"contents": [2.0, 1.5], "errors": [0.5, 0.5]},
                "modifiers": [{"type": "normfactor", "parameter": "mu_signal"}],
            },
            {
                "name": "background",
                "data": {"contents": [8.0, 6.5], "errors": [1.5, 1.5]},
                "modifiers": [],
            },
        ]

        dist = HistFactoryDist(name="test_channel", axes=axes, samples=samples)

        # Create context
        mu_signal = pt.dscalar("mu_signal")
        observed_data = pt.dvector("observed_data")

        context = Context(
            {"mu_signal": mu_signal, "test_channel_observed": observed_data}
        )

        # Get the expression
        expr = dist.expression(context)

        # Compile function
        f = function([mu_signal, observed_data], expr)

        # Test evaluation
        mu_val = 1.0
        obs_val = np.array([10.0, 8.0])  # Total expected: [10.0, 8.0] with mu=1

        result = f(mu_val, obs_val)
        assert np.isfinite(result)
