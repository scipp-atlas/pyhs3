"""Normalization observables are matched by graph dependence (issues #214, #396)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from pyhs3 import Workspace


def _workspace(
    data_axes: list[str],
    dist_mean: str,
    extra_distributions: list[dict] | None = None,
    root: str = "model",
) -> Workspace:
    """Gaussian in ``x`` whose mean is ``dist_mean``; unbinned data over ``data_axes``."""
    return Workspace(
        metadata={"hs3_version": "0.2"},
        functions=[
            {
                "name": "fy",
                "type": "generic_function",
                "expression": "a0 - a1*sqrt(10*abs(y))",
            }
        ],
        distributions=[
            {
                "name": "model",
                "type": "gaussian_dist",
                "x": "x",
                "mean": dist_mean,
                "sigma": 3.0,
            },
            *(extra_distributions or []),
        ],
        data=[
            {
                "name": "modelData",
                "type": "unbinned",
                "axes": [{"name": name, "min": -5.0, "max": 5.0} for name in data_axes],
                "entries": [[0.5] * len(data_axes), [1.5] * len(data_axes)],
            }
        ],
        likelihoods=[{"name": "L", "distributions": [root], "data": ["modelData"]}],
        analyses=[
            {
                "name": "A",
                "likelihood": "L",
                "domains": ["d"],
                "init": "p",
            }
        ],
        domains=[
            {
                "name": "d",
                "type": "product_domain",
                "axes": [
                    {"name": "a0", "min": -10.0, "max": 10.0},
                    {"name": "a1", "min": -10.0, "max": 10.0},
                ],
            }
        ],
        parameter_points=[
            {
                "name": "p",
                "parameters": [
                    {"name": "a0", "value": -3.5},
                    {"name": "a1", "value": 1.0},
                    {"name": "m", "value": 0.25, "const": True},
                    *(
                        []
                        if "y" in data_axes
                        else [{"name": "y", "value": 2.0, "const": True}]
                    ),
                ],
            }
        ],
    )


def _truncated_gaussian_logpdf(x, mean, sigma, lower, upper):
    z = norm.cdf(upper, mean, sigma) - norm.cdf(lower, mean, sigma)
    return norm.logpdf(x, mean, sigma) - np.log(z)


class TestObservableDependenceThroughFunction:
    """An observable reaching a distribution only through a function is matched."""

    def test_rf309_style_model_raises_nd_error(self):
        """Gaussian in x with a mean built from y has two observables (x, y)."""
        ws = _workspace(["x", "y"], "fy")
        with pytest.raises(NotImplementedError, match="issues/214"):
            ws.model("A", parameter_set="p")

    def test_error_names_the_observables(self):
        """The error lists the observables and points at the conditional question."""
        ws = _workspace(["x", "y"], "fy")
        with pytest.raises(NotImplementedError, match=r"\['x', 'y'\]") as info:
            ws.model("A", parameter_set="p")
        assert "issues/396" in str(info.value)

    def test_no_second_observable_is_unchanged(self):
        """A mean that depends on no observable keeps 1-D normalization."""
        ws = _workspace(["x"], "m")
        model = ws.model("A", parameter_set="p")
        x = np.array([0.5, 1.5])
        value = model.logpdf("model", x=x, m=np.asarray(0.25, dtype=np.float64))
        expected = _truncated_gaussian_logpdf(x, 0.25, 3.0, -5.0, 5.0)
        np.testing.assert_allclose(np.ravel(value), expected, rtol=1e-6)

    def test_conditional_parameter_is_unchanged(self):
        """A ``y`` that is a parameter, not a data axis, stays outside the normalization."""
        ws = _workspace(["x"], "fy")
        model = ws.model("A", parameter_set="p")
        x = np.array([0.5, 1.5])
        mean = -3.5 - 1.0 * np.sqrt(10 * 2.0)
        value = model.logpdf(
            "model",
            x=x,
            y=np.asarray(2.0, dtype=np.float64),
            a0=np.asarray(-3.5, dtype=np.float64),
            a1=np.asarray(1.0, dtype=np.float64),
        )
        expected = _truncated_gaussian_logpdf(x, mean, 3.0, -5.0, 5.0)
        np.testing.assert_allclose(np.ravel(value), expected, rtol=1e-6)


class TestCompositeDistributions:
    """Product components are normalized densities, so the product is not re-normalized."""

    def test_product_over_separate_observables_builds(self):
        """A product of a Gaussian in x and a Gaussian in y has no N-D error."""
        ws = _workspace(
            ["x", "y"],
            "m",
            extra_distributions=[
                {
                    "name": "gy",
                    "type": "gaussian_dist",
                    "x": "y",
                    "mean": 0.0,
                    "sigma": 2.0,
                },
                {"name": "prod", "type": "product_dist", "factors": ["model", "gy"]},
            ],
            root="prod",
        )
        model = ws.model("A", parameter_set="p")
        x = np.array([0.5, 1.5])
        y = np.array([0.0, 1.0])
        value = model.logpdf("prod", x=x, y=y, m=np.asarray(0.25, dtype=np.float64))
        expected = _truncated_gaussian_logpdf(
            x, 0.25, 3.0, -5.0, 5.0
        ) + _truncated_gaussian_logpdf(y, 0.0, 2.0, -5.0, 5.0)
        np.testing.assert_allclose(np.ravel(value), expected, rtol=1e-6)
