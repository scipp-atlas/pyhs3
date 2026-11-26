"""Tests for Model.expected_data and Model.sample_data on HistFactory channels.

expected_data gives the expected Poisson means; sample_data draws toy
datasets around them.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.stats import poisson

import pyhs3
from pyhs3 import model as model_module


def _channel(name: str, signal: list[float], background: list[float]) -> dict:
    """HistFactory channel: signal scaled by ``mu``, background by normsys ``alpha``."""
    return {
        "name": name,
        "type": "histfactory_dist",
        "axes": [{"name": f"obs_{name}", "min": 0.0, "max": 1.0, "nbins": len(signal)}],
        "samples": [
            {
                "name": "signal",
                "data": {"contents": signal},
                "modifiers": [{"name": "mu", "type": "normfactor", "parameter": "mu"}],
            },
            {
                "name": "background",
                "data": {"contents": background},
                "modifiers": [
                    {
                        "name": "alpha",
                        "type": "normsys",
                        "parameter": "alpha",
                        "constraint": "Gauss",
                        "data": {"hi": 1.1, "lo": 0.9, "interpolation": "code4"},
                    },
                    {"name": "k", "type": "normfactor", "parameter": "k"},
                ],
            },
        ],
    }


def _binned(name: str, contents: list[float]) -> dict:
    return {
        "name": f"obs_data_{name}",
        "type": "binned",
        "contents": contents,
        "axes": [
            {"name": f"obs_{name}", "min": 0.0, "max": 1.0, "nbins": len(contents)}
        ],
    }


def _workspace(*, gaussian: bool = False) -> pyhs3.Workspace:
    """Channels SR (2 bins) and CR (3 bins); ``k`` is const at 2.

    With ``gaussian``, the likelihood also holds an unbinned Gaussian channel.
    """
    distributions: list[dict[str, Any]] = [
        _channel("SR", [5.0, 10.0], [50.0, 60.0]),
        _channel("CR", [1.0, 2.0, 3.0], [100.0, 80.0, 60.0]),
    ]
    data: list[dict[str, Any]] = [
        _binned("SR", [110.0, 135.0]),
        _binned("CR", [200.0, 165.0, 125.0]),
    ]
    if gaussian:
        distributions.append(
            {
                "name": "gauss",
                "type": "gaussian_dist",
                "x": "x",
                "mean": "mu",
                "sigma": 1.0,
            }
        )
        data.append(
            {
                "name": "events",
                "type": "unbinned",
                "axes": [{"name": "x", "min": -5.0, "max": 5.0}],
                "entries": [[0.5], [1.5]],
            }
        )
    return pyhs3.Workspace(
        metadata={"hs3_version": "0.2"},
        distributions=distributions,
        data=data,
        domains=[
            {
                "name": "main",
                "type": "product_domain",
                "axes": [
                    {"name": "mu", "min": 0.0, "max": 10.0},
                    {"name": "alpha", "min": -5.0, "max": 5.0},
                    {"name": "k", "min": 0.0, "max": 10.0},
                ],
            }
        ],
        parameter_points=[
            {
                "name": "init",
                "parameters": [
                    {"name": "mu", "value": 1.0},
                    {"name": "alpha", "value": 0.0},
                    {"name": "k", "value": 2.0, "const": True},
                ],
            }
        ],
        likelihoods=[
            {
                "name": "L",
                "distributions": [d["name"] for d in distributions],
                "data": [d["name"] for d in data],
            }
        ],
        analyses=[
            {"name": "A", "likelihood": "L", "domains": ["main"], "init": "init"}
        ],
    )


def _gaussian_workspace() -> pyhs3.Workspace:
    """A single Gaussian in ``x`` with mean ``mu``: no HistFactory channel."""
    return pyhs3.Workspace(
        metadata={"hs3_version": "0.2"},
        distributions=[
            {
                "name": "gauss",
                "type": "gaussian_dist",
                "x": "x",
                "mean": "mu",
                "sigma": 1.0,
            }
        ],
        parameter_points=[
            {
                "name": "init",
                "parameters": [
                    {"name": "x", "value": 0.0},
                    {"name": "mu", "value": 0.0},
                ],
            }
        ],
    )


@pytest.fixture
def model():
    ws = _workspace()
    return ws.model(ws.analyses["A"], progress=False)


class TestExpectedData:
    def test_per_channel_means(self, model):
        # alpha = 1 scales the background by exactly hi = 1.1; k = 2 is const.
        expected = model.expected_data(mu=3.0, alpha=1.0)
        assert list(expected) == ["SR", "CR"]
        assert_allclose(expected["SR"], [3 * 5.0 + 2.2 * 50.0, 3 * 10.0 + 2.2 * 60.0])
        assert_allclose(
            expected["CR"],
            [3 * 1.0 + 2.2 * 100.0, 3 * 2.0 + 2.2 * 80.0, 3 * 3.0 + 2.2 * 60.0],
        )

    def test_shapes_and_dtype(self, model):
        expected = model.expected_data(**model.free_params)
        assert expected["SR"].shape == (2,)
        assert expected["CR"].shape == (3,)
        assert expected["SR"].dtype == np.float64

    def test_accepts_numpy_scalars(self, model):
        expected = model.expected_data(mu=np.float64(0.0), alpha=np.array(-1.0))
        assert_allclose(expected["SR"], [2 * 0.9 * 50.0, 2 * 0.9 * 60.0])

    def test_extra_parameters_are_ignored(self, model):
        # A full parameter point, including the const k and an unrelated name.
        expected = model.expected_data(mu=1.0, alpha=0.0, k=2.0, unrelated=7.0)
        assert_allclose(expected["SR"], [5.0 + 100.0, 10.0 + 120.0])

    def test_const_parameter_is_baked(self, model):
        # A value passed for the const k does not override the baked k = 2.
        expected = model.expected_data(mu=1.0, alpha=0.0, k=5.0)
        assert_allclose(expected["SR"], [5.0 + 100.0, 10.0 + 120.0])

    def test_missing_parameters_raise(self, model):
        with pytest.raises(ValueError, match=r"alpha, mu"):
            model.expected_data()

    def test_compiles_once(self, model, monkeypatch):
        calls = []
        compile_function = model_module.function

        def counting_function(*args, **kwargs):
            calls.append(kwargs.get("name"))
            return compile_function(*args, **kwargs)

        monkeypatch.setattr(model_module, "function", counting_function)
        first = model.expected_data(mu=1.0, alpha=0.0)
        second = model.expected_data(mu=2.0, alpha=0.5)
        assert calls == ["expected_data"]
        assert not np.allclose(first["SR"], second["SR"])

    def test_unbinned_channel_is_skipped(self):
        ws = _workspace(gaussian=True)
        model = ws.model(ws.analyses["A"], progress=False)
        expected = model.expected_data(mu=1.0, alpha=0.0)
        assert list(expected) == ["SR", "CR"]

    def test_model_without_histfactory_channel_raises(self):
        ws = _gaussian_workspace()
        model = ws.model(0, progress=False)
        with pytest.raises(ValueError, match="no HistFactory channel"):
            model.expected_data(mu=0.0)


class TestSampleData:
    def test_keys_shapes_and_dtype(self, model):
        toys = model.sample_data(4, rng=np.random.default_rng(0), mu=1.0, alpha=0.0)
        assert list(toys) == list(model.observed_data)
        assert toys["SR"].shape == (4, 2)
        assert toys["CR"].shape == (4, 3)
        assert toys["SR"].dtype == np.float64
        # Poisson counts: whole numbers.
        assert_array_equal(toys["CR"], np.round(toys["CR"]))

    def test_seeded_reproducibility(self, model):
        first = model.sample_data(5, rng=np.random.default_rng(42), mu=1.0, alpha=0.0)
        second = model.sample_data(5, rng=np.random.default_rng(42), mu=1.0, alpha=0.0)
        other = model.sample_data(5, rng=np.random.default_rng(43), mu=1.0, alpha=0.0)
        for name in first:
            assert_array_equal(first[name], second[name])
        assert not all(np.array_equal(first[name], other[name]) for name in first)

    def test_generator_state_advances(self, model):
        rng = np.random.default_rng(7)
        first = model.sample_data(5, rng=rng, mu=1.0, alpha=0.0)
        second = model.sample_data(5, rng=rng, mu=1.0, alpha=0.0)
        assert not np.array_equal(first["CR"], second["CR"])

    def test_mean_approaches_expected_data(self, model):
        n = 20_000
        params = {"mu": 3.0, "alpha": 0.5}
        toys = model.sample_data(n, rng=np.random.default_rng(1), **params)
        expected = model.expected_data(**params)
        for name, counts in toys.items():
            # Standard error of the mean of n Poisson draws: sqrt(lambda / n),
            # shape (B,); a seeded draw within 5 standard errors.
            stderr = np.sqrt(expected[name] / n)
            assert np.all(np.abs(counts.mean(axis=0) - expected[name]) < 5 * stderr)
            # The variance of a Poisson draw equals its mean.
            assert_allclose(counts.var(axis=0), expected[name], rtol=0.05)

    def test_toys_feed_compiled_nll(self, model):
        nll = model.compile_nll(data_inputs=True)
        x = np.array(nll.initial)
        params = dict(zip(nll.parameter_names, x, strict=True))
        toys = model.sample_data(3, rng=np.random.default_rng(2), **params)
        rates = model.expected_data(**params)
        observed = model.observed_data
        reference = nll.value(x)
        for i in range(3):
            toy = {name: counts[i] for name, counts in toys.items()}
            # Only the main Poisson terms depend on the data, so the NLL
            # changes by the difference of their log-pmfs.
            shift = sum(
                poisson.logpmf(observed[name], rates[name]).sum()
                - poisson.logpmf(toy[name], rates[name]).sum()
                for name in toy
            )
            assert nll.value(x, data=toy) == pytest.approx(reference + shift, rel=1e-10)

    def test_negative_expected_rate_raises(self, model):
        # mu = -30 makes both SR means negative: 5 * -30 + 100 and 10 * -30 + 120.
        with pytest.raises(ValueError, match="'SR'"):
            model.sample_data(2, rng=np.random.default_rng(0), mu=-30.0, alpha=0.0)

    def test_model_without_histfactory_channel_raises(self):
        ws = _gaussian_workspace()
        model = ws.model(0, progress=False)
        with pytest.raises(ValueError, match="no HistFactory channel"):
            model.sample_data(2, rng=np.random.default_rng(0), mu=0.0)
