"""
Tests for HistFactory distribution implementation.

Test the HistFactoryDist class with various modifiers and configurations.
"""

from __future__ import annotations

import json
import math
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

from pydantic import ValidationError

import pyhs3
from pyhs3.axes import BinnedAxes
from pyhs3.context import Context
from pyhs3.distributions import HistFactoryDistChannel
from pyhs3.distributions.basic import GaussianDist, PoissonDist
from pyhs3.distributions.histfactory.modifiers import (
    HistoSysModifier,
    NormFactorModifier,
    NormSysModifier,
    ShapeFactorModifier,
    ShapeSysModifier,
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

        dist = HistFactoryDistChannel(
            name="singlechannel",
            axes=axes,
            samples=samples,
            barlow_beeston_method="full",
        )

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

    def test_staterror_lite_precision_vs_pyhf(self):
        """Test that BB-lite staterror (multi-sample, shared gamma) achieves perfect
        precision vs pyhf.

        pyhf's staterror is itself Barlow-Beeston lite with a shared Gaussian constraint
        over samples with the same modifier name.  pyhs3 lite mode with matching errors
        must produce identical logpdf and expected rates.
        """
        mu = 1.0
        gamma = 1.0  # staterror gamma at nominal
        observed = 30.0  # signal(10) + bkg1(15) + bkg2(5) = 30

        # pyhf: two backgrounds with the same staterror name
        # pyhf combines sigma_total = sqrt(3^2 + 4^2) = 5 over nu_total = 15+5 = 20
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
                            "name": "background1",
                            "data": [15],
                            "modifiers": [
                                {
                                    "name": "stat_error",
                                    "type": "staterror",
                                    "data": [3.0],
                                }
                            ],
                        },
                        {
                            "name": "background2",
                            "data": [5],
                            "modifiers": [
                                {
                                    "name": "stat_error",
                                    "type": "staterror",
                                    "data": [4.0],
                                }
                            ],
                        },
                    ],
                }
            ]
        }

        pyhf_model = pyhf.Model(pyhf_spec)
        pyhf_params = [mu, gamma]
        pyhf_data = [int(observed), *pyhf_model.config.auxdata]

        pyhf_expected = pyhf_model.expected_actualdata(pyhf_params)
        pyhf_logpdf = pyhf_model.logpdf(pyhf_params, pyhf_data).item()

        # pyhs3 lite mode: signal + 2 backgrounds with shared staterror parameters
        # errors match pyhf staterror data so sigma_combined = sqrt(3^2+4^2) = 5
        axes = [{"name": "observable", "min": 0.0, "max": 1.0, "nbins": 1}]
        samples = [
            {
                "name": "signal",
                "data": {"contents": [10.0], "errors": [1.0]},
                "modifiers": [{"name": "mu", "type": "normfactor", "parameter": "mu"}],
            },
            {
                "name": "background1",
                "data": {"contents": [15.0], "errors": [3.0]},
                "modifiers": [
                    {
                        "name": "stat_error",
                        "type": "staterror",
                        "parameters": ["stat_error"],
                        "constraint": "Gauss",
                    }
                ],
            },
            {
                "name": "background2",
                "data": {"contents": [5.0], "errors": [4.0]},
                "modifiers": [
                    {
                        "name": "stat_error",
                        "type": "staterror",
                        "parameters": ["stat_error"],
                        "constraint": "Gauss",
                    }
                ],
            },
        ]

        dist = HistFactoryDistChannel(
            name="singlechannel",
            axes=axes,
            samples=samples,
            barlow_beeston_method="lite",
        )

        # Create symbolic context
        mu_var = pt.dscalar("mu")
        gamma_var = pt.dscalar("stat_error")
        observed_data_var = pt.dvector("singlechannel_observed")

        context = Context(
            {
                "mu": mu_var,
                "stat_error": gamma_var,
                "singlechannel_observed": observed_data_var,
            }
        )

        expected_rates = dist._compute_expected_rates(context, 1)
        total_expr = dist.log_expression(context)

        rates_func = function([mu_var, gamma_var], expected_rates)
        total_func = function([mu_var, gamma_var, observed_data_var], total_expr)

        pyhs3_expected = rates_func(mu, gamma)
        pyhs3_logpdf = total_func(mu, gamma, np.array([observed]))

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
    model_pyhs3 = ws_pyhs3.model(0)

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
    model_pyhs3 = ws_pyhs3.model(0)

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

    def test_normfactor_apply_equals_rates_times_expression(self):
        """apply(ctx, rates) must equal rates * expression(ctx) for NormFactorModifier."""
        modifier = NormFactorModifier(name="mu", parameter="mu")

        mu_var = pt.dscalar("mu")
        rates_var = pt.dvector("rates")
        context = Context({"mu": mu_var})

        expr_apply = modifier.apply(context, rates_var)
        expr_product = rates_var * modifier.expression(context)

        f_apply = function([mu_var, rates_var], expr_apply)
        f_product = function([mu_var, rates_var], expr_product)

        np.testing.assert_allclose(
            f_apply(2.0, [10.0, 20.0]), f_product(2.0, [10.0, 20.0])
        )
        np.testing.assert_allclose(
            f_apply(0.5, [10.0, 20.0]), f_product(0.5, [10.0, 20.0])
        )

    def test_normsys_apply_equals_rates_times_expression(self):
        """apply(ctx, rates) must equal rates * expression(ctx) for NormSysModifier."""
        modifier = NormSysModifier(
            name="alpha", parameter="alpha", data={"hi": 1.2, "lo": 0.8}
        )

        alpha_var = pt.dscalar("alpha")
        rates_var = pt.dvector("rates")
        context = Context({"alpha": alpha_var})

        expr_apply = modifier.apply(context, rates_var)
        expr_product = rates_var * modifier.expression(context)

        f_apply = function([alpha_var, rates_var], expr_apply)
        f_product = function([alpha_var, rates_var], expr_product)

        np.testing.assert_allclose(
            f_apply(0.5, [10.0, 20.0]), f_product(0.5, [10.0, 20.0])
        )
        np.testing.assert_allclose(
            f_apply(-0.5, [10.0, 20.0]), f_product(-0.5, [10.0, 20.0])
        )
        np.testing.assert_allclose(
            f_apply(0.0, [10.0, 20.0]), f_product(0.0, [10.0, 20.0])
        )

    def test_shapefactor_apply_equals_rates_times_expression(self):
        """apply(ctx, rates) must equal rates * expression(ctx) for ShapeFactorModifier."""
        modifier = ShapeFactorModifier(name="gamma", parameters=["gamma_0", "gamma_1"])

        g0 = pt.dscalar("gamma_0")
        g1 = pt.dscalar("gamma_1")
        rates_var = pt.dvector("rates")
        context = Context({"gamma_0": g0, "gamma_1": g1})

        expr_apply = modifier.apply(context, rates_var)
        expr_product = rates_var * modifier.expression(context)

        f_apply = function([g0, g1, rates_var], expr_apply)
        f_product = function([g0, g1, rates_var], expr_product)

        np.testing.assert_allclose(
            f_apply(1.2, 0.8, [10.0, 20.0]), f_product(1.2, 0.8, [10.0, 20.0])
        )

    def test_shapesys_apply_equals_rates_times_expression(self):
        """apply(ctx, rates) must equal rates * expression(ctx) for ShapeSysModifier."""
        modifier = ShapeSysModifier(
            name="staterr",
            parameters=["gamma_0", "gamma_1"],
            data={"vals": [0.1, 0.15]},
        )

        g0 = pt.dscalar("gamma_0")
        g1 = pt.dscalar("gamma_1")
        rates_var = pt.dvector("rates")
        context = Context({"gamma_0": g0, "gamma_1": g1})

        expr_apply = modifier.apply(context, rates_var)
        expr_product = rates_var * modifier.expression(context)

        f_apply = function([g0, g1, rates_var], expr_apply)
        f_product = function([g0, g1, rates_var], expr_product)

        np.testing.assert_allclose(
            f_apply(1.1, 0.9, [10.0, 20.0]), f_product(1.1, 0.9, [10.0, 20.0])
        )


class TestExtendedLikelihoodConstraintDedup:
    """Test that extended_likelihood dedupes constraints by parameter name."""

    def test_shared_normsys_parameter_emits_one_constraint(self):
        """Two normsys modifiers on different samples sharing one parameter
        must emit exactly ONE Gaussian factor, not one per modifier.

        At alpha=1 the ratio extended_likelihood(alpha=1) / extended_likelihood(alpha=0)
        equals exp(-0.5) for one Gaussian factor or exp(-1) for two.
        """
        axes = [{"name": "x", "min": 0.0, "max": 10.0, "nbins": 2}]
        samples = [
            {
                "name": "signal",
                "data": {"contents": [5.0, 3.0], "errors": [1.0, 1.0]},
                "modifiers": [
                    {
                        "name": "lumi",
                        "type": "normsys",
                        "parameter": "alpha_lumi",
                        "data": {"hi": 1.1, "lo": 0.9},
                        "constraint": "Gauss",
                    }
                ],
            },
            {
                "name": "background",
                "data": {"contents": [10.0, 8.0], "errors": [1.0, 1.0]},
                "modifiers": [
                    {
                        "name": "lumi",
                        "type": "normsys",
                        "parameter": "alpha_lumi",
                        "data": {"hi": 1.1, "lo": 0.9},
                        "constraint": "Gauss",
                    }
                ],
            },
        ]

        channel = HistFactoryDistChannel(name="ch", axes=axes, samples=samples)

        alpha = pt.dscalar("alpha_lumi")
        obs = pt.dvector("ch_observed")
        context = Context({"alpha_lumi": alpha, "ch_observed": obs})

        el_expr = channel.extended_likelihood(context)
        f = function([alpha], el_expr)

        val_at_0 = f(0.0)
        val_at_1 = f(1.0)

        # Ratio for ONE Gaussian factor N(alpha | 0, 1) at x=0:
        #   exp(-0.5 * 1^2) / exp(0) = exp(-0.5)
        # Ratio for TWO factors (the bug): exp(-1.0)
        ratio = val_at_1 / val_at_0
        assert ratio == pytest.approx(np.exp(-0.5), rel=1e-6)


class TestHistFactoryChannelHistConversion:
    """Tests for HistFactoryDistChannel.to_hist() method."""

    def test_to_hist_single_sample_1d(self):
        """Test HistFactoryDistChannel.to_hist() with single sample and 1D binning."""
        axes = [{"name": "mass", "min": 100.0, "max": 150.0, "nbins": 5}]
        samples = [
            {
                "name": "signal",
                "data": {
                    "contents": [10.0, 20.0, 15.0, 25.0, 18.0],
                    "errors": [3.0, 4.0, 3.5, 4.5, 4.0],
                },
                "modifiers": [],
            }
        ]

        channel = HistFactoryDistChannel(name="SR", axes=axes, samples=samples)
        h = channel.to_hist()

        # Check that we have 2 axes: categorical sample + mass
        assert len(h.axes) == 2

        # Check first axis is categorical with sample names (labeled "process")
        assert h.axes[0].name == "process"
        assert list(h.axes[0]) == ["signal"]

        # Check second axis is the mass axis
        assert h.axes[1].name == "mass"
        assert h.axes[1].size == 5

        # Check values for the single sample
        assert np.array_equal(h["signal", :].values(), [10.0, 20.0, 15.0, 25.0, 18.0])

        # Check variances
        expected_variances = np.square([3.0, 4.0, 3.5, 4.5, 4.0])
        assert np.allclose(h["signal", :].variances(), expected_variances)

    def test_to_hist_multiple_samples_1d(self):
        """Test HistFactoryDistChannel.to_hist() with multiple samples and 1D binning."""
        axes = [{"name": "observable", "min": 0.0, "max": 4.0, "nbins": 4}]
        samples = [
            {
                "name": "signal",
                "data": {
                    "contents": [5.0, 8.0, 12.0, 7.0],
                    "errors": [2.0, 2.5, 3.0, 2.3],
                },
                "modifiers": [],
            },
            {
                "name": "background",
                "data": {
                    "contents": [15.0, 18.0, 14.0, 16.0],
                    "errors": [3.5, 4.0, 3.7, 3.9],
                },
                "modifiers": [],
            },
        ]

        channel = HistFactoryDistChannel(name="channel1", axes=axes, samples=samples)
        h = channel.to_hist()

        # Check axes
        assert len(h.axes) == 2
        assert h.axes[0].name == "process"
        assert list(h.axes[0]) == ["signal", "background"]
        assert h.axes[1].name == "observable"

        # Check values for both samples
        assert np.array_equal(h["signal", :].values(), [5.0, 8.0, 12.0, 7.0])
        assert np.array_equal(h["background", :].values(), [15.0, 18.0, 14.0, 16.0])

        # Check variances for both samples
        assert np.allclose(h["signal", :].variances(), np.square([2.0, 2.5, 3.0, 2.3]))
        assert np.allclose(
            h["background", :].variances(), np.square([3.5, 4.0, 3.7, 3.9])
        )

    def test_to_hist_multiple_samples_2d(self):
        """Test HistFactoryDistChannel.to_hist() with multiple samples and 2D binning."""
        # 2x3 = 6 bins
        axes = [
            {"name": "mass", "min": 100.0, "max": 150.0, "nbins": 2},
            {"name": "pt", "min": 0.0, "max": 300.0, "nbins": 3},
        ]
        samples = [
            {
                "name": "signal",
                "data": {
                    "contents": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                    "errors": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                },
                "modifiers": [],
            },
            {
                "name": "background",
                "data": {
                    "contents": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
                    "errors": [2.0, 2.1, 2.2, 2.3, 2.4, 2.5],
                },
                "modifiers": [],
            },
        ]

        channel = HistFactoryDistChannel(name="SR_2D", axes=axes, samples=samples)
        h = channel.to_hist()

        # Check we have 3 axes: process + mass + pt
        assert len(h.axes) == 3
        assert h.axes[0].name == "process"
        assert h.axes[1].name == "mass"
        assert h.axes[2].name == "pt"

        # Check shape: (2 samples, 2 mass bins, 3 pt bins)
        assert h.values().shape == (2, 2, 3)

        # Check signal sample values (reshape to 2x3)
        expected_signal = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        assert np.array_equal(h["signal", :, :].values(), expected_signal)

        # Check background sample values
        expected_background = np.array([[10.0, 11.0, 12.0], [13.0, 14.0, 15.0]])
        assert np.array_equal(h["background", :, :].values(), expected_background)

    def test_to_hist_irregular_binning(self):
        """Test HistFactoryDistChannel.to_hist() with irregular (variable-width) binning."""
        axes = [{"name": "pt", "edges": [0.0, 10.0, 50.0, 100.0, 200.0]}]
        samples = [
            {
                "name": "signal",
                "data": {
                    "contents": [5.0, 15.0, 25.0, 10.0],
                    "errors": [2.0, 3.0, 4.0, 2.5],
                },
                "modifiers": [],
            },
            {
                "name": "background",
                "data": {
                    "contents": [50.0, 60.0, 40.0, 30.0],
                    "errors": [7.0, 8.0, 6.0, 5.0],
                },
                "modifiers": [],
            },
        ]

        channel = HistFactoryDistChannel(name="VarWidth", axes=axes, samples=samples)
        h = channel.to_hist()

        # Check axes
        assert len(h.axes) == 2
        assert h.axes[0].name == "process"
        assert h.axes[1].name == "pt"

        # Check that pt axis has the correct edges
        expected_edges = [0.0, 10.0, 50.0, 100.0, 200.0]
        assert np.array_equal(h.axes[1].edges, expected_edges)

        # Check values
        assert np.array_equal(h["signal", :].values(), [5.0, 15.0, 25.0, 10.0])
        assert np.array_equal(h["background", :].values(), [50.0, 60.0, 40.0, 30.0])

    def test_to_hist_sample_axis_is_first(self):
        """Test that the process axis is the first axis (for intuitive slicing)."""
        axes = [{"name": "x", "min": 0.0, "max": 5.0, "nbins": 5}]
        samples = [
            {
                "name": "A",
                "data": {
                    "contents": [1.0, 2.0, 3.0, 4.0, 5.0],
                    "errors": [0.5, 0.5, 0.5, 0.5, 0.5],
                },
                "modifiers": [],
            },
            {
                "name": "B",
                "data": {
                    "contents": [10.0, 20.0, 30.0, 40.0, 50.0],
                    "errors": [1.0, 1.0, 1.0, 1.0, 1.0],
                },
                "modifiers": [],
            },
        ]

        channel = HistFactoryDistChannel(name="test", axes=axes, samples=samples)
        h = channel.to_hist()

        # Verify that axis 0 is the categorical process axis
        assert h.axes[0].name == "process"

        # Verify we can slice by sample name as first index
        sample_a_hist = h["A", :]
        assert np.array_equal(sample_a_hist.values(), [1.0, 2.0, 3.0, 4.0, 5.0])

        sample_b_hist = h["B", :]
        assert np.array_equal(sample_b_hist.values(), [10.0, 20.0, 30.0, 40.0, 50.0])

    def test_to_hist_values_match_samples(self):
        """Test that histogram values match individual Sample.to_hist() results."""
        axes_def = [{"name": "mass", "min": 100.0, "max": 140.0, "nbins": 4}]
        samples = [
            {
                "name": "signal",
                "data": {
                    "contents": [10.0, 20.0, 15.0, 25.0],
                    "errors": [3.0, 4.0, 3.5, 4.5],
                },
                "modifiers": [],
            },
            {
                "name": "background",
                "data": {
                    "contents": [50.0, 60.0, 55.0, 65.0],
                    "errors": [7.0, 8.0, 7.5, 8.5],
                },
                "modifiers": [],
            },
        ]

        channel = HistFactoryDistChannel(name="channel", axes=axes_def, samples=samples)
        h_channel = channel.to_hist()

        axes_obj = BinnedAxes(axes_def)

        # Convert each sample individually and compare
        for sample in channel.samples:
            h_sample = sample.to_hist(axes_obj)

            # Check that channel histogram slice matches individual sample histogram
            assert np.array_equal(h_channel[sample.name, :].values(), h_sample.values())
            assert np.allclose(
                h_channel[sample.name, :].variances(), h_sample.variances()
            )

    def test_to_hist_variances_preserved(self):
        """Test that error uncertainties are properly preserved as variances."""
        axes = [{"name": "x", "min": 0.0, "max": 3.0, "nbins": 3}]
        samples = [
            {
                "name": "data",
                "data": {
                    "contents": [100.0, 150.0, 120.0],
                    "errors": [10.0, 12.0, 11.0],
                },
                "modifiers": [],
            }
        ]

        channel = HistFactoryDistChannel(name="channel", axes=axes, samples=samples)
        h = channel.to_hist()

        # Errors should be stored as variances (errors squared)
        expected_variances = np.square([10.0, 12.0, 11.0])
        assert np.allclose(h["data", :].variances(), expected_variances)

        # We should be able to recover the errors by taking sqrt of variances
        recovered_errors = np.sqrt(h["data", :].variances())
        assert np.allclose(recovered_errors, [10.0, 12.0, 11.0])


class TestHistoSysNominalRates:
    """Tests for issue #219: histosys variations must be computed against sample nominal.

    The HistFactory formula is lambda = (N + sum(delta_histosys(N))) * prod(kappa_multiplicative).
    Each histosys variation is relative to the sample nominal N, not to the
    incrementally-modified rates. When a multiplicative modifier (normfactor,
    normsys) runs before histosys in the modifier list, the buggy sequential
    application computes δ against the already-scaled rates instead of N.
    """

    def _make_channel(self, modifiers: list[dict]) -> HistFactoryDistChannel:
        axes = [{"name": "x", "min": 0.0, "max": 10.0, "nbins": 2}]
        samples = [
            {
                "name": "signal",
                "data": {"contents": [10.0, 20.0], "errors": [1.0, 1.0]},
                "modifiers": modifiers,
            }
        ]
        return HistFactoryDistChannel(name="ch", axes=axes, samples=samples)

    def test_histosys_with_normfactor_uses_nominal(self):
        """Histosys variation must be against nominal, not normfactor-scaled rates.

        nominal=[10, 20], histosys hi=[15, 25] lo=[5, 15], normfactor mu=2, alpha=0.5.
        Correct: (N + variation_against_N) * mu = [12.5, 22.5] * 2 = [25.0, 45.0].
        Buggy (normfactor first): rates=[20, 40], variation computed against [20, 40].
        """
        # normfactor listed first — this is the ordering that exposes the bug
        dist = self._make_channel(
            [
                {"name": "mu", "type": "normfactor", "parameter": "mu"},
                {
                    "name": "alpha",
                    "type": "histosys",
                    "parameter": "alpha",
                    "constraint": "Gauss",
                    "data": {
                        "hi": {"contents": [15.0, 25.0]},
                        "lo": {"contents": [5.0, 15.0]},
                    },
                },
            ]
        )

        mu_var = pt.dscalar("mu")
        alpha_var = pt.dscalar("alpha")
        context = Context({"mu": mu_var, "alpha": alpha_var})

        expr = dist._compute_expected_rates(context, 2)
        f = function([mu_var, alpha_var], expr)
        result = f(2.0, 0.5)

        # hi/lo are symmetric around nominal ([15-10]==[10-5]==5), so at alpha=0.5
        # the variation equals 0.5*(hi-N)=[2.5, 2.5] for all interpolation methods.
        # Correct: (N + [2.5, 2.5]) * 2 = [12.5, 22.5] * 2 = [25, 45].
        np.testing.assert_allclose(result, [25.0, 45.0], rtol=1e-12)

    def test_two_histosys_variations_sum_against_nominal(self):
        """Two histosys modifiers on the same sample must both compute against the nominal.

        If variations chain against each other's output, the second modifier
        uses a different 'nominal' than intended.
        nominal=[10.0], histosys1 hi=[15] lo=[5], histosys2 hi=[12] lo=[8], both alpha=0.5.
        Correct: 10 + 0.5*(15-10) + 0.5*(12-10) = 10 + 2.5 + 1.0 = 13.5.
        """
        axes = [{"name": "x", "min": 0.0, "max": 10.0, "nbins": 1}]
        samples = [
            {
                "name": "signal",
                "data": {"contents": [10.0], "errors": [1.0]},
                "modifiers": [
                    {
                        "name": "alpha1",
                        "type": "histosys",
                        "parameter": "alpha1",
                        "constraint": "Gauss",
                        "data": {
                            "hi": {"contents": [15.0]},
                            "lo": {"contents": [5.0]},
                        },
                    },
                    {
                        "name": "alpha2",
                        "type": "histosys",
                        "parameter": "alpha2",
                        "constraint": "Gauss",
                        "data": {
                            "hi": {"contents": [12.0]},
                            "lo": {"contents": [8.0]},
                        },
                    },
                ],
            }
        ]
        dist = HistFactoryDistChannel(name="ch", axes=axes, samples=samples)

        alpha1_var = pt.dscalar("alpha1")
        alpha2_var = pt.dscalar("alpha2")
        context = Context({"alpha1": alpha1_var, "alpha2": alpha2_var})

        expr = dist._compute_expected_rates(context, 1)
        f = function([alpha1_var, alpha2_var], expr)
        result = f(0.5, 0.5)

        np.testing.assert_allclose(result, [13.5], rtol=1e-12)

    def test_modifier_order_invariant(self):
        """Expected rates are the same regardless of histosys/normfactor ordering.

        normfactor * (N + histosys_variation) must equal (N + histosys_variation) * normfactor.
        The current sequential approach makes order matter, which is wrong.
        Also verifies the same invariance holds for normsys (also multiplicative).
        """
        histosys_spec = {
            "name": "alpha",
            "type": "histosys",
            "parameter": "alpha",
            "constraint": "Gauss",
            "data": {
                "hi": {"contents": [15.0, 25.0]},
                "lo": {"contents": [5.0, 15.0]},
            },
        }
        normfactor_spec = {"name": "mu", "type": "normfactor", "parameter": "mu"}
        normsys_spec = {
            "name": "mu",
            "type": "normsys",
            "parameter": "mu",
            "constraint": "Gauss",
            "data": {"hi": 1.2, "lo": 0.8},
        }

        dist_hf_first = self._make_channel([histosys_spec, normfactor_spec])
        dist_nf_first = self._make_channel([normfactor_spec, histosys_spec])

        mu_var = pt.dscalar("mu")
        alpha_var = pt.dscalar("alpha")
        context = Context({"mu": mu_var, "alpha": alpha_var})

        expr_hf = dist_hf_first._compute_expected_rates(context, 2)
        expr_nf = dist_nf_first._compute_expected_rates(context, 2)

        f_hf = function([mu_var, alpha_var], expr_hf)
        f_nf = function([mu_var, alpha_var], expr_nf)

        result_hf = f_hf(2.0, 0.5)
        result_nf = f_nf(2.0, 0.5)

        np.testing.assert_allclose(result_hf, result_nf, rtol=1e-12)

        # Same invariance for normsys (multiplicative, like normfactor, but with
        # hi/lo interpolation rather than direct scaling).
        dist_ns_first = self._make_channel([normsys_spec, histosys_spec])
        dist_sn_first = self._make_channel([histosys_spec, normsys_spec])

        expr_ns = dist_ns_first._compute_expected_rates(context, 2)
        expr_sn = dist_sn_first._compute_expected_rates(context, 2)

        f_ns = function([mu_var, alpha_var], expr_ns)
        f_sn = function([mu_var, alpha_var], expr_sn)

        result_ns = f_ns(0.5, 0.5)
        result_sn = f_sn(0.5, 0.5)

        np.testing.assert_allclose(result_ns, result_sn, rtol=1e-12)


class TestBarlowBeestonLite:
    """Test Barlow-Beeston lite implementation for staterror modifier."""

    def test_staterror_modifier_without_data(self):
        """Test that StatErrorModifier accepts data=None for lite mode."""
        modifier = StatErrorModifier(
            name="stat_error",
            type="staterror",
            parameters=["gamma_bin0", "gamma_bin1"],
            constraint="Gauss",
            data=None,
        )
        assert modifier.data is None
        assert modifier.parameters == ["gamma_bin0", "gamma_bin1"]

    def test_staterror_modifier_poisson_constraint(self):
        """Test that StatErrorModifier accepts constraint='Poisson'."""
        modifier = StatErrorModifier(
            name="stat_error",
            type="staterror",
            parameters=["gamma_bin0", "gamma_bin1"],
            constraint="Poisson",
            data=None,
        )
        assert modifier.constraint == "Poisson"

    def test_lite_field_accepted(self):
        """Test that HistFactoryDistChannel accepts barlow_beeston_method='lite'."""
        axes = [{"name": "x", "min": 0.0, "max": 10.0, "nbins": 2}]
        samples = [
            {
                "name": "signal",
                "data": {"contents": [5.0, 3.0], "errors": [1.0, 1.0]},
                "modifiers": [],
            }
        ]
        channel = HistFactoryDistChannel(
            name="test_channel",
            axes=axes,
            samples=samples,
            barlow_beeston_method="lite",
        )
        assert channel.barlow_beeston_method == "lite"

    def test_lite_default(self):
        """Test that barlow_beeston_method defaults to 'lite'."""
        axes = [{"name": "x", "min": 0.0, "max": 10.0, "nbins": 2}]
        samples = [
            {
                "name": "signal",
                "data": {"contents": [5.0, 3.0], "errors": [1.0, 1.0]},
                "modifiers": [],
            }
        ]
        channel = HistFactoryDistChannel(
            name="test_channel",
            axes=axes,
            samples=samples,
        )
        assert channel.barlow_beeston_method == "lite"

    def test_lite_combined_uncertainties(self):
        """Test that BB-lite computes combined uncertainties correctly.

        Combined uncertainty should be: sigma = sqrt(sum(errors^2))
        Relative error should be: relerr = sigma / total_nominal
        """
        axes = [{"name": "x", "min": 0.0, "max": 10.0, "nbins": 2}]
        samples = [
            {
                "name": "signal",
                "data": {"contents": [10.0, 20.0], "errors": [3.0, 4.0]},
                "modifiers": [
                    {
                        "name": "stat_error",
                        "type": "staterror",
                        "parameters": ["gamma_bin0", "gamma_bin1"],
                        "constraint": "Gauss",
                    }
                ],
            },
            {
                "name": "background",
                "data": {"contents": [20.0, 30.0], "errors": [4.0, 5.0]},
                "modifiers": [
                    {
                        "name": "stat_error",
                        "type": "staterror",
                        "parameters": ["gamma_bin0", "gamma_bin1"],
                        "constraint": "Gauss",
                    }
                ],
            },
        ]
        channel = HistFactoryDistChannel(
            name="test_channel",
            axes=axes,
            samples=samples,
            barlow_beeston_method="lite",
        )

        # Expected combined values:
        # total_nominal = [10+20, 20+30] = [30, 50]
        # total_sigma = [sqrt(9+16), sqrt(16+25)] = [5, sqrt(41)]
        # relerr = [5/30, sqrt(41)/50] = [0.1667, 0.1281]

        # At gamma=1.0 the exponent is 0: N(1|1, relerr) = 1/(relerr * sqrt(2*pi))
        context = Context(
            {"gamma_bin0": pt.constant(1.0), "gamma_bin1": pt.constant(1.0)}
        )
        constraint = channel._make_barlow_beeston_lite_constraint(context)
        assert constraint is not None
        constraint_val = float(constraint.eval())

        total_nominal = np.array([30.0, 50.0])
        total_sigma = np.array([5.0, np.sqrt(41.0)])
        relerr = total_sigma / total_nominal
        expected = float(np.prod(1.0 / (relerr * np.sqrt(2 * np.pi))))
        np.testing.assert_allclose(constraint_val, expected, rtol=1e-6)

    def test_lite_poisson_constraint(self):
        """Test BB-lite Poisson constraint with tau = (nu/sigma)^2."""
        axes = [{"name": "x", "min": 0.0, "max": 10.0, "nbins": 2}]
        samples = [
            {
                "name": "signal",
                "data": {"contents": [100.0, 200.0], "errors": [10.0, 14.14]},
                "modifiers": [
                    {
                        "name": "stat_error",
                        "type": "staterror",
                        "parameters": ["gamma_bin0", "gamma_bin1"],
                        "constraint": "Poisson",
                    }
                ],
            },
        ]
        channel = HistFactoryDistChannel(
            name="test_channel",
            axes=axes,
            samples=samples,
            barlow_beeston_method="lite",
        )

        # For Poisson: tau = (nu/sigma)^2
        # Bin 0: tau = (100/10)^2 = 100
        # Bin 1: tau = (200/14.14)^2 ≈ 200.06

        # At gamma=1.0: Poisson(tau | gamma*tau=tau) = exp(tau*log(tau) - tau - lgamma(tau+1))
        context = Context(
            {"gamma_bin0": pt.constant(1.0), "gamma_bin1": pt.constant(1.0)}
        )
        constraint = channel._make_barlow_beeston_lite_constraint(context)
        assert constraint is not None
        constraint_val = float(constraint.eval())

        nu = np.array([100.0, 200.0])
        sigma = np.array([10.0, 14.14])
        tau = (nu / sigma) ** 2
        # Generalized Poisson PMF: exp(x*log(mu) - mu - lgamma(x+1)) with x=mu=tau
        log_factors = (
            tau * np.log(tau)
            - tau
            - np.array([math.lgamma(float(t) + 1.0) for t in tau])
        )
        expected = float(np.exp(np.sum(log_factors)))
        # PyTensor stores integer-valued tau (100.0) as float32, which reduces gammaln
        # precision slightly compared to the float64 math.lgamma reference.  rtol=1e-4
        # covers this implementation detail while still catching formula errors.
        np.testing.assert_allclose(constraint_val, expected, rtol=1e-4)

    def test_lite_gaussian_constraint(self):
        """Test BB-lite Gaussian constraint with relerr = sigma/nu."""
        axes = [{"name": "x", "min": 0.0, "max": 10.0, "nbins": 2}]
        samples = [
            {
                "name": "signal",
                "data": {"contents": [100.0, 200.0], "errors": [10.0, 20.0]},
                "modifiers": [
                    {
                        "name": "stat_error",
                        "type": "staterror",
                        "parameters": ["gamma_bin0", "gamma_bin1"],
                        "constraint": "Gauss",
                    }
                ],
            },
        ]
        channel = HistFactoryDistChannel(
            name="test_channel",
            axes=axes,
            samples=samples,
            barlow_beeston_method="lite",
        )

        # For Gaussian: relerr = sigma/nu
        # Bin 0: relerr = 10/100 = 0.1
        # Bin 1: relerr = 20/200 = 0.1

        # At off-nominal gamma=[0.9, 1.1], the Gaussian exponent is non-trivial:
        # N(1.0 | gamma, relerr) = exp(-0.5*((1.0-gamma)/relerr)^2) / (relerr*sqrt(2*pi))
        context = Context(
            {"gamma_bin0": pt.constant(0.9), "gamma_bin1": pt.constant(1.1)}
        )
        constraint = channel._make_barlow_beeston_lite_constraint(context)
        assert constraint is not None
        constraint_val = float(constraint.eval())

        relerr = 0.1
        gamma_vals = np.array([0.9, 1.1])
        factors = np.exp(-0.5 * ((1.0 - gamma_vals) / relerr) ** 2) / (
            relerr * np.sqrt(2 * np.pi)
        )
        expected = float(np.prod(factors))
        np.testing.assert_allclose(constraint_val, expected, rtol=1e-6)

    def test_lite_gamma_modifies_rates(self):
        """Test that shared gamma parameters scale all sample rates in BB-lite."""
        axes = [{"name": "x", "min": 0.0, "max": 10.0, "nbins": 2}]
        samples = [
            {
                "name": "signal",
                "data": {"contents": [10.0, 20.0], "errors": [3.0, 4.0]},
                "modifiers": [
                    {
                        "name": "stat_error",
                        "type": "staterror",
                        "parameters": ["gamma_bin0", "gamma_bin1"],
                        "constraint": "Gauss",
                    }
                ],
            },
            {
                "name": "background",
                "data": {"contents": [20.0, 30.0], "errors": [4.0, 5.0]},
                "modifiers": [
                    {
                        "name": "stat_error",
                        "type": "staterror",
                        "parameters": ["gamma_bin0", "gamma_bin1"],
                        "constraint": "Gauss",
                    }
                ],
            },
        ]
        channel = HistFactoryDistChannel(
            name="test_channel",
            axes=axes,
            samples=samples,
            barlow_beeston_method="lite",
        )

        # With gamma_bin0 = 1.2, both samples should be scaled by 1.2 in bin 0
        context = Context({"gamma_bin0": 1.2, "gamma_bin1": 1.0})
        total_bins = channel._get_total_bins()
        rates = channel._compute_expected_rates(context, total_bins)

        # Expected: signal[0] = 10 * 1.2 = 12, background[0] = 20 * 1.2 = 24
        # Total rate in bin 0 should be 12 + 24 = 36
        # Total rate in bin 1 should be 20 + 30 = 50 (gamma_bin1 = 1.0)
        rates_eval = rates.eval()
        assert np.isclose(rates_eval[0], 36.0), f"Expected 36.0, got {rates_eval[0]}"
        assert np.isclose(rates_eval[1], 50.0), f"Expected 50.0, got {rates_eval[1]}"

    def test_lite_skips_per_sample_constraint(self):
        """Test that per-sample make_constraint() is not called in lite mode."""
        axes = [{"name": "x", "min": 0.0, "max": 10.0, "nbins": 2}]
        samples = [
            {
                "name": "signal",
                "data": {"contents": [10.0, 20.0], "errors": [3.0, 4.0]},
                "modifiers": [
                    {
                        "name": "stat_error",
                        "type": "staterror",
                        "parameters": ["gamma_bin0", "gamma_bin1"],
                        "constraint": "Gauss",
                    }
                ],
            },
        ]
        channel = HistFactoryDistChannel(
            name="test_channel",
            axes=axes,
            samples=samples,
            barlow_beeston_method="lite",
        )

        # In lite mode, make_constraint() must NOT be called on the data=None modifier —
        # that would raise ValueError.  Success proves the per-sample path was skipped.
        context = Context(
            {
                "gamma_bin0": pt.constant(1.0),
                "gamma_bin1": pt.constant(1.0),
            }
        )

        # extended_likelihood must complete without raising ValueError
        result = channel.extended_likelihood(context)
        result_val = float(result.eval())

        # The result must equal the lite channel-level constraint (no per-sample term)
        lite_constraint = channel._make_barlow_beeston_lite_constraint(context)
        assert lite_constraint is not None
        np.testing.assert_allclose(
            result_val, float(lite_constraint.eval()), rtol=1e-10
        )

    def test_full_mode_backward_compatible(self):
        """Test that barlow_beeston_method='full' uses original behavior."""
        axes = [{"name": "x", "min": 0.0, "max": 10.0, "nbins": 2}]
        samples = [
            {
                "name": "signal",
                "data": {"contents": [10.0, 20.0], "errors": [3.0, 4.0]},
                "modifiers": [
                    {
                        "name": "stat_error",
                        "type": "staterror",
                        "parameters": ["gamma_bin0", "gamma_bin1"],
                        "constraint": "Gauss",
                        "data": {
                            "uncertainties": [0.3, 0.2],  # relative uncertainties
                        },
                    }
                ],
            },
        ]
        channel = HistFactoryDistChannel(
            name="test_channel",
            axes=axes,
            samples=samples,
            barlow_beeston_method="full",
        )

        # Full mode should use per-sample constraints as before
        assert channel.barlow_beeston_method == "full"

    def test_lite_with_other_modifiers(self):
        """Test that BB-lite works alongside normsys and normfactor modifiers."""
        axes = [{"name": "x", "min": 0.0, "max": 10.0, "nbins": 2}]
        samples = [
            {
                "name": "signal",
                "data": {"contents": [10.0, 20.0], "errors": [3.0, 4.0]},
                "modifiers": [
                    {
                        "name": "mu",
                        "type": "normfactor",
                        "parameter": "mu",
                    },
                    {
                        "name": "stat_error",
                        "type": "staterror",
                        "parameters": ["gamma_bin0", "gamma_bin1"],
                        "constraint": "Gauss",
                    },
                ],
            },
            {
                "name": "background",
                "data": {"contents": [20.0, 30.0], "errors": [4.0, 5.0]},
                "modifiers": [
                    {
                        "name": "bkg_norm",
                        "type": "normsys",
                        "parameter": "bkg_norm",
                        "constraint": "Gauss",
                        "data": {"hi": 1.1, "lo": 0.9},
                    },
                    {
                        "name": "stat_error",
                        "type": "staterror",
                        "parameters": ["gamma_bin0", "gamma_bin1"],
                        "constraint": "Gauss",
                    },
                ],
            },
        ]
        channel = HistFactoryDistChannel(
            name="test_channel",
            axes=axes,
            samples=samples,
            barlow_beeston_method="lite",
        )

        # In lite mode: extended_likelihood = normsys_constraint * lite_staterror_constraint
        # Crucially, make_constraint() is NOT called on the data=None staterror modifiers —
        # that would raise ValueError.  The test succeeds only if the staterror was skipped.
        context = Context(
            {
                "mu": pt.constant(1.0),
                "bkg_norm": pt.constant(0.0),
                "gamma_bin0": pt.constant(1.0),
                "gamma_bin1": pt.constant(1.0),
            }
        )
        result_val = float(channel.extended_likelihood(context).eval())

        # normsys at alpha=0: N(0 | 0, 1) = 1/sqrt(2*pi)
        normsys_val = 1.0 / np.sqrt(2 * np.pi)

        # lite staterror (both samples have staterror):
        # total_nominal=[10+20, 20+30]=[30,50], total_sigma=[sqrt(9+16), sqrt(16+25)]=[5,sqrt(41)]
        total_sigma = np.array([5.0, np.sqrt(41.0)])
        total_nominal = np.array([30.0, 50.0])
        relerr = total_sigma / total_nominal
        lite_val = float(np.prod(1.0 / (relerr * np.sqrt(2 * np.pi))))

        expected = normsys_val * lite_val
        np.testing.assert_allclose(result_val, expected, rtol=1e-6)

    def test_lite_zero_yield_bin(self):
        """Test that bins with nu=0 or sigma=0 are skipped gracefully."""
        axes = [{"name": "x", "min": 0.0, "max": 10.0, "nbins": 3}]
        samples = [
            {
                "name": "signal",
                "data": {
                    "contents": [10.0, 0.0, 20.0],  # bin 1 has zero yield
                    "errors": [3.0, 0.0, 4.0],
                },
                "modifiers": [
                    {
                        "name": "stat_error",
                        "type": "staterror",
                        "parameters": ["gamma_bin0", "gamma_bin1", "gamma_bin2"],
                        "constraint": "Gauss",
                    }
                ],
            },
        ]
        channel = HistFactoryDistChannel(
            name="test_channel",
            axes=axes,
            samples=samples,
            barlow_beeston_method="lite",
        )

        # Bin 1 (nu=0, sigma=0) must be skipped; only bins 0 and 2 contribute.
        context = Context(
            {
                "gamma_bin0": pt.constant(1.0),
                "gamma_bin1": pt.constant(1.0),  # skipped — not used
                "gamma_bin2": pt.constant(1.0),
            }
        )
        constraint = channel._make_barlow_beeston_lite_constraint(context)
        assert constraint is not None  # bins 0 and 2 are still valid

        constraint_val = float(constraint.eval())

        # relerr_0 = 3/10 = 0.3, relerr_2 = 4/20 = 0.2; exponent=0 at gamma=1
        expected = float(
            (1.0 / (0.3 * np.sqrt(2 * np.pi))) * (1.0 / (0.2 * np.sqrt(2 * np.pi)))
        )
        np.testing.assert_allclose(constraint_val, expected, rtol=1e-6)

    def test_lite_constraint_none_without_staterror(self):
        """_make_barlow_beeston_lite_constraint returns None when no staterror modifier
        is present; extended_likelihood falls back to pt.constant(1.0)."""
        axes = [{"name": "x", "min": 0.0, "max": 10.0, "nbins": 2}]
        samples = [
            {
                "name": "signal",
                "data": {"contents": [5.0, 3.0], "errors": [1.0, 1.0]},
                "modifiers": [{"name": "mu", "type": "normfactor", "parameter": "mu"}],
            }
        ]
        channel = HistFactoryDistChannel(
            name="test_channel",
            axes=axes,
            samples=samples,
            barlow_beeston_method="lite",
        )

        context = Context({"mu": pt.constant(1.0)})

        # No staterror modifier in any sample
        assert channel._find_staterror_modifier() is None
        # The private method returns None — staterror_mod is None guard fires
        assert channel._make_barlow_beeston_lite_constraint(context) is None

        # With no constraints at all, extended_likelihood returns constant 1.0
        result_val = float(channel.extended_likelihood(context).eval())
        assert result_val == pytest.approx(1.0)

    def test_lite_constraint_none_all_bins_skipped(self):
        """_make_barlow_beeston_lite_constraint returns None when every bin satisfies
        the nu<=0 or sigma<=0 skip condition (the 'if not dists: return None' branch)."""
        axes = [{"name": "x", "min": 0.0, "max": 10.0, "nbins": 2}]
        samples = [
            {
                "name": "signal",
                # All contents are zero → total_nominal=0 for every bin → all skipped
                "data": {"contents": [0.0, 0.0], "errors": [1.0, 1.0]},
                "modifiers": [
                    {
                        "name": "stat_error",
                        "type": "staterror",
                        "parameters": ["gamma_bin0", "gamma_bin1"],
                        "constraint": "Gauss",
                    }
                ],
            }
        ]
        channel = HistFactoryDistChannel(
            name="test_channel",
            axes=axes,
            samples=samples,
            barlow_beeston_method="lite",
        )

        context = Context(
            {
                "gamma_bin0": pt.constant(1.0),
                "gamma_bin1": pt.constant(1.0),
            }
        )

        # All bins have nu=0; every bin is skipped → method returns None
        assert channel._make_barlow_beeston_lite_constraint(context) is None

        # extended_likelihood with no valid constraints returns constant 1.0
        result_val = float(channel.extended_likelihood(context).eval())
        assert result_val == pytest.approx(1.0)

    def test_lite_excludes_non_staterror_samples(self):
        """BB-lite only accumulates uncertainty from samples carrying a staterror
        modifier; samples without staterror are excluded from the combined sigma."""
        axes = [{"name": "x", "min": 0.0, "max": 10.0, "nbins": 1}]
        samples = [
            {
                "name": "signal",
                "data": {"contents": [10.0], "errors": [2.0]},
                "modifiers": [
                    {
                        "name": "stat_error",
                        "type": "staterror",
                        "parameters": ["gamma_bin0"],
                        "constraint": "Gauss",
                    }
                ],
            },
            {
                "name": "background",
                # errors=[5.0] would change the combined sigma if included
                "data": {"contents": [20.0], "errors": [5.0]},
                "modifiers": [],  # no staterror — must NOT contribute
            },
        ]
        channel = HistFactoryDistChannel(
            name="test_channel",
            axes=axes,
            samples=samples,
            barlow_beeston_method="lite",
        )

        context = Context({"gamma_bin0": pt.constant(1.0)})
        constraint = channel._make_barlow_beeston_lite_constraint(context)
        assert constraint is not None
        constraint_val = float(constraint.eval())

        # Only signal contributes: nu=10, sigma=2, relerr=0.2
        # N(1|1, 0.2) = 1 / (0.2 * sqrt(2*pi))
        expected = 1.0 / (0.2 * np.sqrt(2 * np.pi))
        np.testing.assert_allclose(constraint_val, expected, rtol=1e-6)

        # Verify: if background (errors=5) were included, the result would differ.
        # total_nominal would be 10+20=30 (both samples), total_sigma=sqrt(4+25)=sqrt(29)
        relerr_if_included = np.sqrt(2.0**2 + 5.0**2) / (10.0 + 20.0)
        wrong_expected = 1.0 / (relerr_if_included * np.sqrt(2 * np.pi))
        assert not np.isclose(constraint_val, wrong_expected, rtol=1e-3)

    def test_find_staterror_modifier_none(self):
        """_find_staterror_modifier() returns None when no sample carries a staterror
        modifier, and returns the modifier when one is present."""
        axes = [{"name": "x", "min": 0.0, "max": 10.0, "nbins": 1}]

        # Channel without staterror
        channel_no_stat = HistFactoryDistChannel(
            name="no_stat",
            axes=axes,
            samples=[
                {
                    "name": "signal",
                    "data": {"contents": [10.0], "errors": [1.0]},
                    "modifiers": [
                        {"name": "mu", "type": "normfactor", "parameter": "mu"}
                    ],
                }
            ],
        )
        assert channel_no_stat._find_staterror_modifier() is None

        # Channel with staterror
        channel_with_stat = HistFactoryDistChannel(
            name="with_stat",
            axes=axes,
            samples=[
                {
                    "name": "signal",
                    "data": {"contents": [10.0], "errors": [2.0]},
                    "modifiers": [
                        {
                            "name": "stat_error",
                            "type": "staterror",
                            "parameters": ["gamma_bin0"],
                            "constraint": "Gauss",
                        }
                    ],
                }
            ],
        )
        found = channel_with_stat._find_staterror_modifier()
        assert found is not None
        assert found.type == "staterror"
        assert found.parameters == ["gamma_bin0"]

    def test_extended_likelihood_full_mode(self):
        """In full mode, extended_likelihood calls per-sample make_constraint on each
        staterror modifier (not the channel-level lite path)."""
        axes = [{"name": "x", "min": 0.0, "max": 10.0, "nbins": 1}]
        samples = [
            {
                "name": "signal",
                "data": {"contents": [100.0], "errors": [2.0]},
                "modifiers": [
                    {
                        "name": "stat_error",
                        "type": "staterror",
                        "parameters": ["gamma_bin0"],
                        "constraint": "Gauss",
                        "data": {"uncertainties": [2.0]},  # required for full mode
                    }
                ],
            }
        ]
        channel = HistFactoryDistChannel(
            name="test_channel",
            axes=axes,
            samples=samples,
            barlow_beeston_method="full",
        )

        context = Context({"gamma_bin0": pt.constant(1.0)})
        result_val = float(channel.extended_likelihood(context).eval())

        # Full mode: N(1.0 | gamma=1.0, sigma_value) where sigma_value = unc/nominal
        # sigma_value = 2.0 / 100.0 = 0.02; at gamma=1 the exponent is 0
        sigma_value = 2.0 / 100.0
        expected = 1.0 / (sigma_value * np.sqrt(2 * np.pi))
        np.testing.assert_allclose(result_val, expected, rtol=1e-6)

    def test_lite_positive_yield_zero_sigma_not_skipped(self):
        """A bin with nu > 0 and sigma = 0 must not be silently dropped from the constraint.

        If `sigma <= 0` fires when `nu > 0`, the gamma parameter floats free:
        _compute_expected_rates() still scales rates by gamma but no constraint is added.
        The parsing layer prevents sigma=0 with nu>0, so this check is dead code that
        masks validation failures rather than surfacing them.
        """
        axes = [{"name": "x", "min": 0.0, "max": 10.0, "nbins": 1}]
        samples = [
            {
                "name": "signal",
                "data": {"contents": [10.0], "errors": [0.0]},  # nu=10, sigma=0
                "modifiers": [
                    {
                        "name": "stat_error",
                        "type": "staterror",
                        "parameters": ["gamma_bin0"],
                        "constraint": "Gauss",
                    }
                ],
            }
        ]
        channel = HistFactoryDistChannel(
            name="test_channel",
            axes=axes,
            samples=samples,
            barlow_beeston_method="lite",
        )

        context = Context({"gamma_bin0": pt.constant(1.0)})

        # With the old condition `nu <= 0 or sigma <= 0`, this bin (nu=10, sigma=0) was
        # silently dropped → method returned None.  After the fix (`nu <= 0` only), the
        # bin is included and the method returns a TensorVar (not None).
        result = channel._make_barlow_beeston_lite_constraint(context)
        assert result is not None

    def test_full_mode_default_parameter_names(self):
        """BB-full: omitting 'parameters' defaults to staterror_{channel}_bin{i}."""
        axes = [{"name": "x", "min": 0.0, "max": 10.0, "nbins": 2}]
        samples = [
            {
                "name": "signal",
                "data": {"contents": [10.0, 20.0], "errors": [1.0, 2.0]},
                "modifiers": [
                    {
                        "name": "staterror",
                        "type": "staterror",
                        "constraint": "Gauss",
                        "data": {"uncertainties": [0.1, 0.2]},
                    }
                ],
            }
        ]
        channel = HistFactoryDistChannel(
            name="test_channel",
            axes=axes,
            samples=samples,
            barlow_beeston_method="full",
        )
        mod = channel._find_staterror_modifier()
        assert mod is not None
        assert mod.parameters == [
            "staterror_test_channel_bin0",
            "staterror_test_channel_bin1",
        ]

    def test_full_mode_missing_data_raises(self):
        """BB-full: a staterror with no modifier 'data' is an invalid spec."""
        axes = [{"name": "x", "min": 0.0, "max": 10.0, "nbins": 2}]
        samples = [
            {
                "name": "signal",
                "data": {"contents": [10.0, 20.0], "errors": [1.0, 2.0]},
                "modifiers": [
                    {
                        "name": "staterror",
                        "type": "staterror",
                        "parameters": ["gamma_bin0", "gamma_bin1"],
                        "constraint": "Gauss",
                    }
                ],
            }
        ]
        with pytest.raises(ValidationError, match="requires 'data'"):
            HistFactoryDistChannel(
                name="test_channel",
                axes=axes,
                samples=samples,
                barlow_beeston_method="full",
            )

    def test_lite_mode_default_parameter_names(self):
        """BB-lite: omitting 'parameters' defaults to staterror_{channel}_bin{i}."""
        axes = [{"name": "x", "min": 0.0, "max": 10.0, "nbins": 2}]
        samples = [
            {
                "name": "signal",
                "data": {"contents": [10.0, 20.0], "errors": [1.0, 2.0]},
                "modifiers": [
                    {
                        "name": "staterror",
                        "type": "staterror",
                        "constraint": "Gauss",
                    }
                ],
            }
        ]
        channel = HistFactoryDistChannel(
            name="test_channel",
            axes=axes,
            samples=samples,
        )
        mod = channel._find_staterror_modifier()
        assert mod is not None
        assert mod.parameters == [
            "staterror_test_channel_bin0",
            "staterror_test_channel_bin1",
        ]

    def test_lite_mode_with_data_raises(self):
        """BB-lite: a staterror with modifier 'data' is an invalid spec."""
        axes = [{"name": "x", "min": 0.0, "max": 10.0, "nbins": 2}]
        samples = [
            {
                "name": "signal",
                "data": {"contents": [10.0, 20.0], "errors": [1.0, 2.0]},
                "modifiers": [
                    {
                        "name": "staterror",
                        "type": "staterror",
                        "parameters": ["gamma_bin0", "gamma_bin1"],
                        "constraint": "Gauss",
                        "data": {"uncertainties": [0.1, 0.2]},
                    }
                ],
            }
        ]
        with pytest.raises(ValidationError, match="must not specify 'data'"):
            HistFactoryDistChannel(
                name="test_channel",
                axes=axes,
                samples=samples,
            )

    def test_lite_default_names_shared_across_samples(self):
        """BB-lite: bare staterror in two samples gets same channel-scoped parameter names."""
        axes = [{"name": "x", "min": 0.0, "max": 10.0, "nbins": 2}]
        samples = [
            {
                "name": "signal",
                "data": {"contents": [10.0, 20.0], "errors": [1.0, 2.0]},
                "modifiers": [
                    {
                        "name": "staterror",
                        "type": "staterror",
                        "constraint": "Gauss",
                    }
                ],
            },
            {
                "name": "background",
                "data": {"contents": [5.0, 8.0], "errors": [0.5, 0.8]},
                "modifiers": [
                    {
                        "name": "staterror",
                        "type": "staterror",
                        "constraint": "Gauss",
                    }
                ],
            },
        ]
        channel = HistFactoryDistChannel(
            name="SR",
            axes=axes,
            samples=samples,
        )
        expected = ["staterror_SR_bin0", "staterror_SR_bin1"]
        for sample in channel.samples:
            for mod in sample.modifiers:
                if isinstance(mod, StatErrorModifier):
                    assert mod.parameters == expected


class TestBarlowBeestonLiteVectorizedConstraint:
    """The channel-level BB-lite constraint builds ONE constraint distribution
    per channel call (vectorized over bins), not one scalar distribution per
    bin.  Structural regression test for the #230 review perf item."""

    N_BINS = 8

    @pytest.mark.parametrize("constraint_type", ["Gauss", "Poisson"])
    def test_builds_one_dist_per_call(self, monkeypatch, constraint_type):
        dist_cls = GaussianDist if constraint_type == "Gauss" else PoissonDist
        calls = {"count": 0}
        original_init = dist_cls.__init__

        def wrapped(self, *args, **kwargs):
            calls["count"] += 1
            original_init(self, *args, **kwargs)

        monkeypatch.setattr(dist_cls, "__init__", wrapped)

        params = [f"gamma_bin{i}" for i in range(self.N_BINS)]
        axes = [
            {"name": "x", "min": 0.0, "max": float(self.N_BINS), "nbins": self.N_BINS}
        ]
        samples = [
            {
                "name": "signal",
                "data": {
                    "contents": [10.0 + i for i in range(self.N_BINS)],
                    "errors": [1.0 + 0.1 * i for i in range(self.N_BINS)],
                },
                "modifiers": [
                    {
                        "name": "stat_error",
                        "type": "staterror",
                        "parameters": params,
                        "constraint": constraint_type,
                    }
                ],
            }
        ]
        channel = HistFactoryDistChannel(
            name="test_channel",
            axes=axes,
            samples=samples,
            barlow_beeston_method="lite",
        )
        context = Context({p: pt.constant(1.0, dtype="float64") for p in params})

        channel._make_barlow_beeston_lite_constraint(context)
        assert calls["count"] == 1

        calls["count"] = 0
        channel._make_barlow_beeston_lite_log_constraint(context)
        assert calls["count"] == 1
