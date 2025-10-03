"""
Tests for HistFactory distribution implementation.

Test the HistFactoryDist class with various modifiers and configurations.
"""

from __future__ import annotations

import platform

import numpy as np
import pytensor.tensor as pt
import pytest
from pytensor import function

try:
    import pyhf

    HAS_PYHF = True
except ImportError:
    HAS_PYHF = False

from pyhs3.context import Context
from pyhs3.distributions import HistFactoryDist


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
                "modifiers": [
                    {
                        "name": "mu_signal",
                        "type": "normfactor",
                        "parameter": "mu_signal",
                    }
                ],
            }
        ]

        dist = HistFactoryDist(name="test_channel", axes=axes, samples=samples)

        # Check that the distribution can be created
        assert dist.name == "test_channel"
        assert len(dist.samples[0].modifiers) == 1
        assert dist.samples[0].modifiers[0].type == "normfactor"

    def test_normsys_modifier(self):
        """Test HistFactoryDist with normsys modifier."""
        axes = [{"name": "x", "min": 0.0, "max": 10.0, "nbins": 2}]
        samples = [
            {
                "name": "background",
                "data": {"contents": [10.0, 8.0], "errors": [2.0, 2.0]},
                "modifiers": [
                    {
                        "name": "bkg_norm_sys",
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
        modifier = dist.samples[0].modifiers[0]
        assert modifier.type == "normsys"
        assert modifier.constraint == "Gauss"
        assert modifier.data.hi == 1.1
        assert modifier.data.lo == 0.9

    def test_histosys_modifier(self):
        """Test HistFactoryDist with histosys modifier."""
        axes = [{"name": "x", "min": 0.0, "max": 10.0, "nbins": 2}]
        samples = [
            {
                "name": "signal",
                "data": {"contents": [5.0, 3.0], "errors": [1.0, 1.0]},
                "modifiers": [
                    {
                        "name": "signal_shape_sys",
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
        modifier = dist.samples[0].modifiers[0]
        assert modifier.type == "histosys"
        assert modifier.constraint == "Gauss"
        assert len(modifier.data.hi.contents) == 2
        assert len(modifier.data.lo.contents) == 2

    def test_multiple_modifiers(self):
        """Test HistFactoryDist with multiple modifiers on one sample."""
        axes = [{"name": "x", "min": 0.0, "max": 10.0, "nbins": 2}]
        samples = [
            {
                "name": "signal",
                "data": {"contents": [5.0, 3.0], "errors": [1.0, 1.0]},
                "modifiers": [
                    {
                        "name": "mu_signal",
                        "type": "normfactor",
                        "parameter": "mu_signal",
                    },
                    {
                        "name": "lumi_sys",
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
        assert len(dist.samples[0].modifiers) == 2
        assert dist.samples[0].modifiers[0].type == "normfactor"
        assert dist.samples[0].modifiers[1].type == "normsys"

    def test_multiple_samples(self):
        """Test HistFactoryDist with multiple samples."""
        axes = [{"name": "x", "min": 0.0, "max": 10.0, "nbins": 2}]
        samples = [
            {
                "name": "signal",
                "data": {"contents": [5.0, 3.0], "errors": [1.0, 1.0]},
                "modifiers": [
                    {
                        "name": "mu_signal",
                        "type": "normfactor",
                        "parameter": "mu_signal",
                    }
                ],
            },
            {
                "name": "background",
                "data": {"contents": [10.0, 8.0], "errors": [2.0, 2.0]},
                "modifiers": [
                    {
                        "name": "bkg_norm_sys",
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
        assert dist.samples[0].name == "signal"
        assert dist.samples[1].name == "background"

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
            dist._process_sample(context, dist.samples[0], 2)


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
                "modifiers": [
                    {
                        "name": "mu_signal",
                        "type": "normfactor",
                        "parameter": "mu_signal",
                    }
                ],
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


@pytest.mark.skipif(not HAS_PYHF, reason="pyhf not available")
class TestPyhfPrecisionValidation:
    """Test precision validation against pyhf to prevent regressions."""

    def test_normsys_precision_vs_pyhf(self):
        """Test that normsys constraint achieves perfect precision vs pyhf."""

        # Test parameters that previously achieved 0.0 difference
        mu = 1.0
        bkg_norm = 0.5
        observed = 15.0

        # pyhf model
        pyhf_spec = {
            "channels": [
                {
                    "name": "singlechannel",
                    "samples": [
                        {
                            "name": "signal",
                            "data": [10],
                            "modifiers": [
                                {"name": "mu", "type": "normfactor", "data": None}
                            ],
                        },
                        {
                            "name": "background",
                            "data": [20],
                            "modifiers": [
                                {
                                    "name": "bkg_norm",
                                    "type": "normsys",
                                    "data": {"hi": 1.1, "lo": 0.9},
                                }
                            ],
                        },
                    ],
                }
            ]
        }

        pyhf_model = pyhf.Model(pyhf_spec)
        pyhf_params = [mu, bkg_norm]
        pyhf_data = [int(observed), *pyhf_model.config.auxdata]

        pyhf_expected = pyhf_model.expected_actualdata(pyhf_params)
        pyhf_logpdf = pyhf_model.logpdf(pyhf_params, pyhf_data).item()

        # pyhs3 model
        axes = [{"name": "observable", "min": 0.0, "max": 1.0, "nbins": 1}]
        samples = [
            {
                "name": "signal",
                "data": {"contents": [10.0], "errors": [1.0]},
                "modifiers": [{"name": "mu", "type": "normfactor", "parameter": "mu"}],
            },
            {
                "name": "background",
                "data": {"contents": [20.0], "errors": [1.0]},
                "modifiers": [
                    {
                        "name": "bkg_norm",
                        "type": "normsys",
                        "parameter": "bkg_norm",
                        "constraint": "Gauss",
                        "data": {"hi": 1.1, "lo": 0.9},
                    }
                ],
            },
        ]

        dist = HistFactoryDist(name="singlechannel", axes=axes, samples=samples)

        # Create context
        mu_var = pt.dscalar("mu")
        bkg_norm_var = pt.dscalar("bkg_norm")
        observed_data_var = pt.dvector("singlechannel_observed")

        context = Context(
            {
                "mu": mu_var,
                "bkg_norm": bkg_norm_var,
                "singlechannel_observed": observed_data_var,
            }
        )

        # Get pyhs3 results
        expected_rates = dist._compute_expected_rates(context, 1)
        total_expr = dist.log_expression(context)

        rates_func = function([mu_var, bkg_norm_var], expected_rates)
        total_func = function([mu_var, bkg_norm_var, observed_data_var], total_expr)

        pyhs3_expected = rates_func(mu, bkg_norm)
        pyhs3_logpdf = total_func(mu, bkg_norm, np.array([observed]))

        # Validate precision (use tight tolerances to prevent regression)
        # Use more lenient tolerance on Windows due to numerical precision differences
        tolerance = 1e-12 if platform.system() == "Windows" else 1e-14

        assert pyhs3_expected[0] == pytest.approx(pyhf_expected[0], abs=tolerance), (
            f"Expected rates differ: pyhf={pyhf_expected[0]}, pyhs3={pyhs3_expected[0]}"
        )
        assert float(pyhs3_logpdf) == pytest.approx(pyhf_logpdf, abs=tolerance), (
            f"Log PDF differs: pyhf={pyhf_logpdf}, pyhs3={float(pyhs3_logpdf)}"
        )

    def test_histosys_precision_vs_pyhf(self):
        """Test that histosys constraint achieves perfect precision vs pyhf."""

        # Test parameters that should achieve high precision
        mu = 1.0
        alpha = 0.5
        observed = 12.0

        # pyhf model with histosys
        pyhf_spec = {
            "channels": [
                {
                    "name": "singlechannel",
                    "samples": [
                        {
                            "name": "signal",
                            "data": [10],
                            "modifiers": [
                                {"name": "mu", "type": "normfactor", "data": None}
                            ],
                        },
                        {
                            "name": "background",
                            "data": [20],
                            "modifiers": [
                                {
                                    "name": "alpha",
                                    "type": "histosys",
                                    "data": {"hi_data": [25], "lo_data": [15]},
                                }
                            ],
                        },
                    ],
                }
            ]
        }

        pyhf_model = pyhf.Model(pyhf_spec)
        # pyhf parameter order is ['alpha', 'mu'], not ['mu', 'alpha']
        pyhf_params = [alpha, mu]  # Correct order: [alpha, mu]
        pyhf_data = [int(observed), *pyhf_model.config.auxdata]

        pyhf_expected = pyhf_model.expected_actualdata(pyhf_params)
        pyhf_logpdf = pyhf_model.logpdf(pyhf_params, pyhf_data).item()

        # pyhs3 model with histosys
        axes = [{"name": "observable", "min": 0.0, "max": 1.0, "nbins": 1}]
        samples = [
            {
                "name": "signal",
                "data": {"contents": [10.0], "errors": [1.0]},
                "modifiers": [{"name": "mu", "type": "normfactor", "parameter": "mu"}],
            },
            {
                "name": "background",
                "data": {"contents": [20.0], "errors": [1.0]},
                "modifiers": [
                    {
                        "name": "alpha",
                        "type": "histosys",
                        "parameter": "alpha",
                        "constraint": "Gauss",
                        "data": {
                            "hi": {"contents": [25.0]},
                            "lo": {"contents": [15.0]},
                        },
                    }
                ],
            },
        ]

        dist = HistFactoryDist(name="singlechannel", axes=axes, samples=samples)

        # Create context
        mu_var = pt.dscalar("mu")
        alpha_var = pt.dscalar("alpha")
        observed_data_var = pt.dvector("singlechannel_observed")

        context = Context(
            {
                "mu": mu_var,
                "alpha": alpha_var,
                "singlechannel_observed": observed_data_var,
            }
        )

        # Get pyhs3 results
        expected_rates = dist._compute_expected_rates(context, 1)
        total_expr = dist.log_expression(context)

        rates_func = function([mu_var, alpha_var], expected_rates)
        total_func = function([mu_var, alpha_var, observed_data_var], total_expr)

        pyhs3_expected = rates_func(mu, alpha)
        pyhs3_logpdf = total_func(mu, alpha, np.array([observed]))

        # Validate precision (use tight tolerances to prevent regression)
        assert pyhs3_expected[0] == pytest.approx(pyhf_expected[0], abs=1e-12), (
            f"Expected rates differ: pyhf={pyhf_expected[0]}, pyhs3={pyhs3_expected[0]}"
        )
        assert float(pyhs3_logpdf) == pytest.approx(pyhf_logpdf, abs=1e-12), (
            f"Log PDF differs: pyhf={pyhf_logpdf}, pyhs3={float(pyhs3_logpdf)}"
        )

    def test_staterror_precision_vs_pyhf(self):
        """Test that staterror constraint achieves perfect precision vs pyhf."""

        # Test parameters that should achieve high precision (use less extreme values)
        mu = 1.0
        staterror_bin_0 = 1.0  # Use nominal value to avoid extreme constraint
        observed = 30.0  # Use expected value to avoid extreme Poisson

        # pyhf model with staterror
        pyhf_spec = {
            "channels": [
                {
                    "name": "singlechannel",
                    "samples": [
                        {
                            "name": "signal",
                            "data": [10],
                            "modifiers": [
                                {"name": "mu", "type": "normfactor", "data": None}
                            ],
                        },
                        {
                            "name": "background",
                            "data": [20],
                            "modifiers": [
                                {
                                    "name": "staterror_bin_0",
                                    "type": "staterror",
                                    "data": [0.1],
                                }
                            ],
                        },
                    ],
                }
            ]
        }

        pyhf_model = pyhf.Model(pyhf_spec)
        pyhf_params = [mu, staterror_bin_0]
        pyhf_data = [int(observed), *pyhf_model.config.auxdata]

        pyhf_expected = pyhf_model.expected_actualdata(pyhf_params)
        pyhf_logpdf = pyhf_model.logpdf(pyhf_params, pyhf_data).item()

        # pyhs3 model with staterror
        axes = [{"name": "observable", "min": 0.0, "max": 1.0, "nbins": 1}]
        samples = [
            {
                "name": "signal",
                "data": {"contents": [10.0], "errors": [1.0]},
                "modifiers": [{"name": "mu", "type": "normfactor", "parameter": "mu"}],
            },
            {
                "name": "background",
                "data": {"contents": [20.0], "errors": [2.0]},
                "modifiers": [
                    {
                        "name": "staterror_bin_0",
                        "type": "staterror",
                        "parameters": ["staterror_bin_0"],
                        "constraint": "Gauss",
                        "data": {"uncertainties": [0.1]},
                    }
                ],
            },
        ]

        dist = HistFactoryDist(name="singlechannel", axes=axes, samples=samples)

        # Create context
        mu_var = pt.dscalar("mu")
        staterror_bin_0_var = pt.dscalar("staterror_bin_0")
        observed_data_var = pt.dvector("singlechannel_observed")

        context = Context(
            {
                "mu": mu_var,
                "staterror_bin_0": staterror_bin_0_var,
                "singlechannel_observed": observed_data_var,
            }
        )

        # Get pyhs3 results
        expected_rates = dist._compute_expected_rates(context, 1)
        total_expr = dist.log_expression(context)

        rates_func = function([mu_var, staterror_bin_0_var], expected_rates)
        total_func = function(
            [mu_var, staterror_bin_0_var, observed_data_var], total_expr
        )

        pyhs3_expected = rates_func(mu, staterror_bin_0)
        pyhs3_logpdf = total_func(mu, staterror_bin_0, np.array([observed]))

        # Validate precision (use reasonable tolerances for staterror)
        assert pyhs3_expected[0] == pytest.approx(pyhf_expected[0], abs=1e-12), (
            f"Expected rates differ: pyhf={pyhf_expected[0]}, pyhs3={pyhs3_expected[0]}"
        )
        assert float(pyhs3_logpdf) == pytest.approx(pyhf_logpdf, abs=1e-12), (
            f"Log PDF differs: pyhf={pyhf_logpdf}, pyhs3={float(pyhs3_logpdf)}"
        )

    def test_shapesys_precision_vs_pyhf(self):
        """Test that shapesys constraint achieves perfect precision vs pyhf."""

        # Test parameters that should achieve high precision
        mu = 1.0
        gamma_0 = 0.9
        observed = 16.0

        # pyhf model with shapesys
        pyhf_spec = {
            "channels": [
                {
                    "name": "singlechannel",
                    "samples": [
                        {
                            "name": "signal",
                            "data": [10],
                            "modifiers": [
                                {"name": "mu", "type": "normfactor", "data": None}
                            ],
                        },
                        {
                            "name": "background",
                            "data": [20],
                            "modifiers": [
                                {"name": "shape_bkg", "type": "shapesys", "data": [25]}
                            ],
                        },
                    ],
                }
            ]
        }

        pyhf_model = pyhf.Model(pyhf_spec)
        pyhf_params = [mu, gamma_0]
        pyhf_data = [int(observed), *pyhf_model.config.auxdata]

        pyhf_expected = pyhf_model.expected_actualdata(pyhf_params)
        pyhf_logpdf = pyhf_model.logpdf(pyhf_params, pyhf_data).item()

        # pyhs3 model with shapesys
        axes = [{"name": "observable", "min": 0.0, "max": 1.0, "nbins": 1}]
        samples = [
            {
                "name": "signal",
                "data": {"contents": [10.0], "errors": [1.0]},
                "modifiers": [{"name": "mu", "type": "normfactor", "parameter": "mu"}],
            },
            {
                "name": "background",
                "data": {"contents": [20.0], "errors": [1.0]},
                "modifiers": [
                    {
                        "name": "shape_bkg",
                        "type": "shapesys",
                        "parameters": ["gamma_0"],
                        "constraint": "Poisson",
                        "data": {"vals": [25]},
                    }
                ],
            },
        ]

        dist = HistFactoryDist(name="singlechannel", axes=axes, samples=samples)

        # Create context
        mu_var = pt.dscalar("mu")
        gamma_0_var = pt.dscalar("gamma_0")
        observed_data_var = pt.dvector("singlechannel_observed")

        context = Context(
            {
                "mu": mu_var,
                "gamma_0": gamma_0_var,
                "singlechannel_observed": observed_data_var,
            }
        )

        # Get pyhs3 results
        expected_rates = dist._compute_expected_rates(context, 1)
        total_expr = dist.log_expression(context)

        rates_func = function([mu_var, gamma_0_var], expected_rates)
        total_func = function([mu_var, gamma_0_var, observed_data_var], total_expr)

        pyhs3_expected = rates_func(mu, gamma_0)
        pyhs3_logpdf = total_func(mu, gamma_0, np.array([observed]))

        # Validate precision (use reasonable tolerances for shapesys)
        assert pyhs3_expected[0] == pytest.approx(pyhf_expected[0], abs=1e-12), (
            f"Expected rates differ: pyhf={pyhf_expected[0]}, pyhs3={pyhs3_expected[0]}"
        )
        assert float(pyhs3_logpdf) == pytest.approx(pyhf_logpdf, abs=1e-12), (
            f"Log PDF differs: pyhf={pyhf_logpdf}, pyhs3={float(pyhs3_logpdf)}"
        )

    def test_expression_with_constraints(self):
        """Test HistFactory expression with constrained modifiers."""

        axes = [{"name": "x", "min": 0.0, "max": 10.0, "nbins": 2}]
        samples = [
            {
                "name": "background",
                "data": {"contents": [10.0, 8.0], "errors": [2.0, 2.0]},
                "modifiers": [
                    {
                        "name": "bkg_norm_sys",
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
                "modifiers": [
                    {
                        "name": "mu_signal",
                        "type": "normfactor",
                        "parameter": "mu_signal",
                    }
                ],
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
