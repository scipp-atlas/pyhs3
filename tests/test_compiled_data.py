"""Tests for Model.observed_data and a compiled NLL evaluated on swapped data."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose
from test_compiled import _workspace as _unbinned_workspace

import pyhs3

# CR's observed counts equal its only sample's contents, so the graph holds
# two equal constants that must stay distinct when the data are swapped.
_SPEC: dict[str, Any] = {
    "channels": [
        {
            "name": "SR",
            "samples": [
                {
                    "name": "signal",
                    "data": [5.0, 10.0],
                    "modifiers": [{"name": "mu", "type": "normfactor", "data": None}],
                },
                {
                    "name": "background",
                    "data": [50.0, 60.0],
                    "modifiers": [
                        {
                            "name": "bkg_norm",
                            "type": "normsys",
                            "data": {"hi": 1.1, "lo": 0.9},
                        }
                    ],
                },
            ],
        },
        {
            "name": "CR",
            "samples": [
                {
                    "name": "background",
                    "data": [100.0, 80.0, 60.0],
                    "modifiers": [
                        {
                            "name": "bkg_norm",
                            "type": "normsys",
                            "data": {"hi": 1.1, "lo": 0.9},
                        },
                        {
                            "name": "stat_CR",
                            "type": "staterror",
                            "data": [5.0, 4.0, 3.0],
                        },
                    ],
                },
            ],
        },
    ],
    "observations": [
        {"name": "SR", "data": [52.0, 75.0]},
        {"name": "CR", "data": [100.0, 80.0, 60.0]},
    ],
    "measurements": [{"name": "meas", "config": {"poi": "mu", "parameters": []}}],
    "version": "1.0.0",
}

# Swapped data: non-integer (Asimov-like) counts and a zero count.
_SWAPPED = {
    "model_SR": np.array([61.5, 0.0]),
    "model_CR": np.array([97.25, 83.0, 58.0]),
}


def _model(observations: dict[str, np.ndarray] | None = None) -> pyhs3.Model:
    """The two-channel model, with the observed counts replaced if given."""
    spec = copy.deepcopy(_SPEC)
    for observation in spec["observations"]:
        key = f"model_{observation['name']}"
        if observations is not None:
            observation["data"] = observations[key].tolist()
    ws = pyhs3.from_hifa(spec)
    return ws.model(ws.analyses["meas"], progress=False)


@pytest.fixture
def model():
    return _model()


@pytest.fixture
def points(model):
    """Two parameter vectors of the compiled NLL, shape (P,) each."""
    nll = model.compile_nll()
    other = nll.initial.copy()
    other[:] = np.linspace(0.8, 1.3, len(other))
    return [nll.initial, other]


class TestObservedData:
    def test_keys_shapes_and_values(self, model):
        """One (B,) array per HistFactory channel, keyed by distribution name."""
        observed = model.observed_data

        assert list(observed) == ["model_SR", "model_CR"]
        assert observed["model_SR"].shape == (2,)
        assert observed["model_CR"].shape == (3,)
        assert observed["model_SR"].dtype == np.float64
        assert_allclose(observed["model_SR"], [52.0, 75.0], rtol=0)
        assert_allclose(observed["model_CR"], [100.0, 80.0, 60.0], rtol=0)

    def test_returns_copies(self, model):
        model.observed_data["model_SR"][0] = -1.0
        assert model.observed_data["model_SR"][0] == 52.0

    def test_no_binned_channel(self):
        ws = _unbinned_workspace()
        assert ws.model(ws.analyses["A"], progress=False).observed_data == {}

    def test_requires_a_likelihood(self):
        ws = _unbinned_workspace()
        with pytest.raises(RuntimeError, match="likelihood"):
            _ = ws.model(0, progress=False).observed_data


class TestSwappedData:
    def test_default_data_is_the_observed_data(self, model, points):
        baked = model.compile_nll()
        swappable = model.compile_nll(data_inputs=True)

        for x in points:
            value, grad = swappable.value_and_grad(x)
            assert value == pytest.approx(baked.value(x), rel=1e-12)
            assert_allclose(grad, baked.grad(x), rtol=1e-10)
            assert swappable.value(x, data=None) == value

    def test_matches_a_model_built_with_that_data(self, model, points):
        swappable = model.compile_nll(data_inputs=True)
        reference = _model(_SWAPPED).compile_nll()

        assert swappable.parameter_names == reference.parameter_names
        for x in points:
            value, grad = swappable.value_and_grad(x, data=_SWAPPED)
            assert value == pytest.approx(reference.value(x), rel=1e-12)
            assert_allclose(grad, reference.grad(x), rtol=1e-10)
            assert swappable.value(x, data=_SWAPPED) == value
            assert_allclose(swappable.grad(x, data=_SWAPPED), grad, rtol=0)

    def test_observed_data_round_trips(self, model, points):
        swappable = model.compile_nll(data_inputs=True)
        x = points[1]
        assert swappable.value(x, data=model.observed_data) == swappable.value(x)

    def test_switching_back_to_the_default_data(self, model, points):
        """Results are cached per parameter vector and data, not per vector alone."""
        swappable = model.compile_nll(data_inputs=True)
        x = points[1]
        default = swappable.value(x)

        swapped = swappable.value(x, data=_SWAPPED)

        assert swapped != pytest.approx(default)
        assert swappable.value(x) == default

    def test_accepts_lists(self, model, points):
        swappable = model.compile_nll(data_inputs=True)
        as_lists = {name: values.tolist() for name, values in _SWAPPED.items()}
        x = points[0]
        assert swappable.value(x, data=as_lists) == swappable.value(x, data=_SWAPPED)

    def test_cache_entry_is_shared_across_observed_data(self, tmp_path, points):
        """The cached function does not hold the data; each model passes its own."""
        first = _model().compile_nll(data_inputs=True, cache=tmp_path)
        second = _model(_SWAPPED).compile_nll(data_inputs=True, cache=tmp_path)

        assert (first.from_cache, second.from_cache) == (False, True)
        reference = _model(_SWAPPED).compile_nll()
        x = points[1]
        assert second.value(x) == pytest.approx(reference.value(x), rel=1e-12)


class TestSwappedDataHessian:
    def test_default_data_is_the_observed_data(self, model, points):
        baked = model.compile_nll()
        swappable = model.compile_nll(data_inputs=True)

        for x in points:
            assert_allclose(swappable.hessian(x), baked.hessian(x), rtol=1e-10)

    def test_matches_a_model_built_with_that_data(self, model, points):
        swappable = model.compile_nll(data_inputs=True)
        reference = _model(_SWAPPED).compile_nll()

        for x in points:
            assert_allclose(
                swappable.hessian(x, data=_SWAPPED), reference.hessian(x), rtol=1e-10
            )

    def test_switching_back_to_the_default_data(self, model, points):
        """Hessians are cached per parameter vector and data, like the values."""
        swappable = model.compile_nll(data_inputs=True)
        x = points[1]
        default = swappable.hessian(x)

        swapped = swappable.hessian(x, data=_SWAPPED)

        assert not np.allclose(swapped, default)
        assert_allclose(swappable.hessian(x), default, rtol=0)

    def test_data_needs_data_inputs(self, model, points):
        nll = model.compile_nll()
        with pytest.raises(ValueError, match="data_inputs=True"):
            nll.hessian(points[0], data=_SWAPPED)


class TestSwappedDataErrors:
    def test_data_needs_data_inputs(self, model, points):
        nll = model.compile_nll()
        with pytest.raises(ValueError, match="data_inputs=True"):
            nll.value(points[0], data=_SWAPPED)

    def test_unknown_channel(self, model, points):
        nll = model.compile_nll(data_inputs=True)
        data = {**_SWAPPED, "model_VR": np.array([1.0])}
        with pytest.raises(KeyError, match="model_VR"):
            nll.value(points[0], data=data)

    def test_missing_channel(self, model, points):
        nll = model.compile_nll(data_inputs=True)
        with pytest.raises(KeyError, match="model_CR"):
            nll.value(points[0], data={"model_SR": _SWAPPED["model_SR"]})

    @pytest.mark.parametrize(
        "values",
        [
            pytest.param(np.array([1.0, 2.0, 3.0]), id="too_many_bins"),
            pytest.param(np.array([[1.0, 2.0]]), id="two_dimensional"),
            pytest.param(np.array(1.0), id="scalar"),
        ],
    )
    def test_wrong_shape(self, model, points, values):
        nll = model.compile_nll(data_inputs=True)
        with pytest.raises(ValueError, match=r"model_SR.*\(2,\)"):
            nll.value(points[0], data={**_SWAPPED, "model_SR": values})

    def test_no_binned_channel(self):
        ws = _unbinned_workspace()
        model = ws.model(ws.analyses["A"], progress=False)
        with pytest.raises(ValueError, match="binned"):
            model.compile_nll(data_inputs=True)
