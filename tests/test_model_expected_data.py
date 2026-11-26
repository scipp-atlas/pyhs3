"""Tests for Model.expected_data: expected Poisson means of HistFactory channels."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose

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
        ws = pyhs3.Workspace(
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
        model = ws.model(0, progress=False)
        with pytest.raises(ValueError, match="no HistFactory channel"):
            model.expected_data(mu=0.0)
