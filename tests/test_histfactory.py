"""
Tests for HistFactory distribution implementation.

Test the HistFactoryDist class with various modifiers and configurations.
"""

from __future__ import annotations

import json
import platform
import warnings

import numpy as np
import pytensor.tensor as pt
import pytest
from pytensor import function

try:
    import pyhf

    HAS_PYHF = True
except ImportError:
    HAS_PYHF = False

import pyhs3
from pyhs3.context import Context
from pyhs3.distributions import HistFactoryDistChannel
from pyhs3.distributions.histfactory.modifiers import (
    HistoSysModifier,
    NormSysModifier,
    ShapeFactorModifier,
    StatErrorModifier,
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

        dist = HistFactoryDistChannel(name="test_channel", axes=axes, samples=samples)

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

        dist = HistFactoryDistChannel(name="test_channel", axes=axes, samples=samples)

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

        dist = HistFactoryDistChannel(name="test_channel", axes=axes, samples=samples)

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

        dist = HistFactoryDistChannel(name="test_channel", axes=axes, samples=samples)

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

        dist = HistFactoryDistChannel(name="test_channel", axes=axes, samples=samples)

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

        dist = HistFactoryDistChannel(name="test_channel", axes=axes, samples=samples)

        # Check that the distribution can be created
        assert dist.name == "test_channel"
        assert len(dist.samples) == 2
        assert dist.samples[0].name == "signal"
        assert dist.samples[1].name == "background"

    def test_bin_count_calculation(self):
        """Test calculation of total bins from axes."""
        # Single axis
        dist1 = HistFactoryDistChannel(
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
        dist2 = HistFactoryDistChannel(
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

        dist = HistFactoryDistChannel(name="test_channel", axes=axes, samples=samples)

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

        dist = HistFactoryDistChannel(name="test_channel", axes=axes, samples=samples)

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

        dist = HistFactoryDistChannel(name="singlechannel", axes=axes, samples=samples)

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

        precision = 1e-10 if platform.system() == "Windows" else 1e-14

        # Validate precision (use tight tolerances to prevent regression)
        assert pyhs3_expected[0] == pytest.approx(pyhf_expected[0], abs=precision), (
            f"Expected rates differ: pyhf={pyhf_expected[0]}, pyhs3={pyhs3_expected[0]}"
        )
        assert float(pyhs3_logpdf) == pytest.approx(pyhf_logpdf, abs=precision), (
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

        dist = HistFactoryDistChannel(name="singlechannel", axes=axes, samples=samples)

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

        dist = HistFactoryDistChannel(name="singlechannel", axes=axes, samples=samples)

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

        dist = HistFactoryDistChannel(name="singlechannel", axes=axes, samples=samples)

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

        dist = HistFactoryDistChannel(name="test_channel", axes=axes, samples=samples)

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

        dist = HistFactoryDistChannel(name="test_channel", axes=axes, samples=samples)

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


@pytest.mark.parametrize(
    ("pars"),
    [[i, j, k] for i in [0.1, 1.0] for j in [0.1, 1.0] for k in [0.1, 1.0]],
)
def test_simplemodel_uncorrelated_pyhf(pars, datadir):
    """
    To convert the pyhf simplemodel in HiFa JSON to HS3 JSON:

        $ pyhf xml2json <hifa.json> --output-dir hs3
        $ cd hs3
        $ hist2workspace FitConfig.xml
        $ root -b config/FitConfig_combined_measurement_model.root
        root [1] combined
        (RooWorkspace *) 0x1362aac00
        root [2] auto mytool = RooJSONFactoryWSTool(*combined);
        root [3] mytool.exportJSON("<hs3.json>")
        (bool) true
    """
    ws_pyhf = pyhf.Workspace(
        json.loads(
            datadir.joinpath(
                "simplemodel_uncorrelated-background_hifa.json"
            ).read_text()
        )
    )
    model_pyhf = ws_pyhf.model()
    data_pyhf = ws_pyhf.data(model_pyhf)

    ws_pyhs3 = pyhs3.Workspace(
        **json.loads(
            datadir.joinpath("simplemodel_uncorrelated-background_hs3.json").read_text()
        )
    )
    model_pyhs3 = ws_pyhs3.model()

    # Get observed data
    obs_data = None
    for data_item in ws_pyhs3.data.root:
        if data_item.name == "obsData_singlechannel":
            obs_data = data_item.contents
            break

    # Map pyhf parameters to pyhs3 parameters
    # pyhf_params = model_pyhf.config.par_names
    # ['mu', 'uncorr_bkguncrt[0]', 'uncorr_bkguncrt[1]']

    # Get default parameter values from workspace
    default_values = {}
    if ws_pyhs3.parameter_points:
        for param_obj in ws_pyhs3.parameter_points[0].parameters:
            default_values[param_obj.name] = param_obj.value

    # Create parameter dictionary for pyhs3
    pyhs3_params = {
        "model_singlechannel_observed": np.array(obs_data),
        "mu": np.array(pars[0]),  # pyhf: 'mu' -> pyhs3: 'mu'
        "uncorr_bkguncrt_0": np.array(
            pars[1]
        ),  # pyhf: 'uncorr_bkguncrt[0]' -> pyhs3: 'uncorr_bkguncrt_0'
        "uncorr_bkguncrt_1": np.array(
            pars[2]
        ),  # pyhf: 'uncorr_bkguncrt[1]' -> pyhs3: 'uncorr_bkguncrt_1'
        "Lumi": np.array(1.0),
    }

    pyhf_result = model_pyhf.pdf(pars, data_pyhf)
    pyhs3_result = model_pyhs3.pdf("model_singlechannel", **pyhs3_params)

    # Compare PDF values - extract scalar from arrays if needed
    pyhf_scalar = (
        pyhf_result[0]
        if hasattr(pyhf_result, "__len__") and len(pyhf_result) == 1
        else pyhf_result
    )
    pyhs3_scalar = (
        float(pyhs3_result)
        if hasattr(pyhs3_result, "shape") and pyhs3_result.shape == ()
        else pyhs3_result
    )

    assert pyhs3_scalar == pytest.approx(pyhf_scalar)

    # Test logpdf (suppress warnings for log(0) = -inf)
    pyhf_logresult = model_pyhf.logpdf(pars, data_pyhf)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        pyhs3_logresult = model_pyhs3.logpdf("model_singlechannel", **pyhs3_params)

    # Compare logpdf values - extract scalar from arrays if needed
    pyhf_log_scalar = (
        pyhf_logresult[0]
        if hasattr(pyhf_logresult, "__len__") and len(pyhf_logresult) == 1
        else pyhf_logresult
    )
    pyhs3_log_scalar = (
        float(pyhs3_logresult)
        if hasattr(pyhs3_logresult, "shape") and pyhs3_logresult.shape == ()
        else pyhs3_logresult
    )

    assert pyhs3_log_scalar == pytest.approx(pyhf_log_scalar)


@pytest.mark.parametrize(
    ("pars"),
    [[i, j] for i in [0.1, 1.0] for j in [-1.0, 0.0, 1.0]],
)
def test_simplemodel_correlated_pyhf(pars, datadir):
    """
    Test correlated background model with histosys modifier.

    This covers the additive modifier path in HistFactoryDistChannel._process_sample.
    """
    ws_pyhf = pyhf.Workspace(
        json.loads(
            datadir.joinpath("simplemodel_correlated-background_hifa.json").read_text()
        )
    )
    model_pyhf = ws_pyhf.model()
    data_pyhf = ws_pyhf.data(model_pyhf)

    ws_pyhs3 = pyhs3.Workspace(
        **json.loads(
            datadir.joinpath("simplemodel_correlated-background_hs3.json").read_text()
        )
    )
    model_pyhs3 = ws_pyhs3.model()

    # Get observed data
    obs_data = None
    for data_item in ws_pyhs3.data.root:
        if data_item.name == "obsData_singlechannel":
            obs_data = data_item.contents
            break

    # Map pyhf parameters to pyhs3 parameters
    # pyhf_params = model_pyhf.config.par_names
    # ['correlated_bkg_uncertainty', 'mu']

    # Get default parameter values from workspace
    default_values = {}
    if ws_pyhs3.parameter_points:
        for param_obj in ws_pyhs3.parameter_points[0].parameters:
            default_values[param_obj.name] = param_obj.value

    # Create parameter dictionary for pyhs3
    pyhs3_params = {
        "model_singlechannel_observed": np.array(obs_data),
        "alpha_correlated_bkg_uncertainty": np.array(pars[1]),  # histosys parameter
        "mu": np.array(pars[0]),  # signal strength
        "Lumi": np.array(1.0),
    }

    # pyhf expects parameters in a specific order
    pyhf_pars = [pars[1], pars[0]]  # [correlated_bkg_uncertainty, mu]
    pyhf_result = model_pyhf.pdf(pyhf_pars, data_pyhf)
    pyhs3_result = model_pyhs3.pdf("model_singlechannel", **pyhs3_params)

    # Compare PDF values - extract scalar from arrays if needed
    pyhf_scalar = (
        pyhf_result[0]
        if hasattr(pyhf_result, "__len__") and len(pyhf_result) == 1
        else pyhf_result
    )
    pyhs3_scalar = (
        float(pyhs3_result)
        if hasattr(pyhs3_result, "shape") and pyhs3_result.shape == ()
        else pyhs3_result
    )

    assert pyhs3_scalar == pytest.approx(pyhf_scalar)

    # Test logpdf (suppress warnings for log(0) = -inf)
    pyhf_logresult = model_pyhf.logpdf(pyhf_pars, data_pyhf)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        pyhs3_logresult = model_pyhs3.logpdf("model_singlechannel", **pyhs3_params)

    # Compare logpdf values - extract scalar from arrays if needed
    pyhf_log_scalar = (
        pyhf_logresult[0]
        if hasattr(pyhf_logresult, "__len__") and len(pyhf_logresult) == 1
        else pyhf_logresult
    )
    pyhs3_log_scalar = (
        float(pyhs3_logresult)
        if hasattr(pyhs3_logresult, "shape") and pyhs3_logresult.shape == ()
        else pyhs3_logresult
    )

    assert pyhs3_log_scalar == pytest.approx(pyhf_log_scalar)


class TestHistFactoryAdditiveModifierPath:
    """Test the additive modifier path in HistFactoryDistChannel._process_sample."""

    def test_additive_modifier_from_context(self):
        """Test that additive modifiers are correctly applied when pre-computed in context."""
        axes = [{"name": "x", "min": 0.0, "max": 10.0, "nbins": 2}]
        samples = [
            {
                "name": "background",
                "data": {"contents": [50.0, 52.0], "errors": [1.0, 1.0]},
                "modifiers": [
                    {
                        "name": "shape_sys",
                        "type": "histosys",
                        "parameter": "alpha_shape",
                        "constraint": "Gauss",
                        "data": {
                            "hi": {"contents": [55.0, 45.0]},
                            "lo": {"contents": [45.0, 55.0]},
                        },
                    }
                ],
            }
        ]

        dist = HistFactoryDistChannel(name="test_channel", axes=axes, samples=samples)

        # Create context with pre-computed modifier result
        # This simulates the dependency graph having already computed the modifier's contribution
        modifier_graph_name = "test_channel/background/histosys/shape_sys"
        observed_data = pt.dvector("test_channel_observed")

        # Pre-compute the additive variation (this would normally be done by the dependency graph)
        # At alpha=0.5, interpolation gives halfway between nominal and hi
        additive_variation = pt.constant([2.5, -3.5])  # (55-50)/2, (45-52)/2

        context = Context(
            {
                "test_channel_observed": observed_data,
                "alpha_shape": pt.constant(0.5),
                modifier_graph_name: additive_variation,  # Pre-computed result in context
            }
        )

        # Get the expression - this should use the pre-computed value from context
        expr = dist.log_expression(context)

        # Compile function
        f = function([observed_data], expr)

        # Test evaluation
        obs_val = np.array([53.0, 49.0])  # Close to expected [52.5, 48.5]
        result = f(obs_val)

        # Should return a finite log probability
        assert np.isfinite(result)


class TestModifierExpressions:
    """Test modifier expression methods for dependency graph evaluation."""

    def test_normsys_expression(self):
        """Test NormSysModifier.expression() method."""

        modifier = NormSysModifier(
            name="test_normsys",
            parameter="alpha_test",
            data={"hi": 1.1, "lo": 0.9},
        )

        # Create context with parameter
        alpha_test = pt.dscalar("alpha_test")
        context = Context({"alpha_test": alpha_test})

        # Get expression
        expr = modifier.expression(context)

        # Compile and test
        f = function([alpha_test], expr)

        # Test at different parameter values
        assert f(0.0) == pytest.approx(1.0)  # nominal
        assert f(1.0) == pytest.approx(1.1)  # +1 sigma
        assert f(-1.0) == pytest.approx(0.9)  # -1 sigma

    def test_histosys_expression(self):
        """Test HistoSysModifier.expression() method."""

        modifier = HistoSysModifier(
            name="test_histosys",
            parameter="alpha_shape",
            data={
                "hi": {"contents": [6.0, 4.0]},
                "lo": {"contents": [4.0, 6.0]},
            },
        )

        # Create context with parameter
        alpha_shape = pt.dscalar("alpha_shape")
        context = Context({"alpha_shape": alpha_shape})

        # Get expression (should return parameter value for dependency tracking)
        expr = modifier.expression(context)

        # Compile and test
        f = function([alpha_shape], expr)

        # HistoSys expression returns the parameter value for dependency tracking
        assert f(0.0) == pytest.approx(0.0)
        assert f(0.5) == pytest.approx(0.5)
        assert f(1.0) == pytest.approx(1.0)

    def test_shapefactor_expression(self):
        """Test ShapeFactorModifier.expression() method."""

        modifier = ShapeFactorModifier(
            name="test_shapefactor",
            parameters=["gamma_0", "gamma_1"],
        )

        # Create context with parameters
        gamma_0 = pt.dscalar("gamma_0")
        gamma_1 = pt.dscalar("gamma_1")
        context = Context({"gamma_0": gamma_0, "gamma_1": gamma_1})

        # Get expression
        expr = modifier.expression(context)

        # Compile and test
        f = function([gamma_0, gamma_1], expr)

        # Test with various values
        result = f(1.0, 1.0)
        assert len(result) == 2
        assert result[0] == pytest.approx(1.0)
        assert result[1] == pytest.approx(1.0)

        result = f(0.8, 1.2)
        assert result[0] == pytest.approx(0.8)
        assert result[1] == pytest.approx(1.2)

    def test_staterror_expression(self):
        """Test StatErrorModifier.expression() method."""

        modifier = StatErrorModifier(
            name="test_staterror",
            parameters=["staterror_bin_0", "staterror_bin_1"],
            data={"uncertainties": [0.1, 0.15]},
        )

        # Create context with parameters
        staterror_bin_0 = pt.dscalar("staterror_bin_0")
        staterror_bin_1 = pt.dscalar("staterror_bin_1")
        context = Context(
            {
                "staterror_bin_0": staterror_bin_0,
                "staterror_bin_1": staterror_bin_1,
            }
        )

        # Get expression
        expr = modifier.expression(context)

        # Compile and test
        f = function([staterror_bin_0, staterror_bin_1], expr)

        # Test with various values
        result = f(1.0, 1.0)
        assert len(result) == 2
        assert result[0] == pytest.approx(1.0)
        assert result[1] == pytest.approx(1.0)

        result = f(0.95, 1.05)
        assert result[0] == pytest.approx(0.95)
        assert result[1] == pytest.approx(1.05)
