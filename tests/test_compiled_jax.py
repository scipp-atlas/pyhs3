"""Tests for Model.compile_nll(backend="jax"); skipped without JAX."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import minimize

jax = pytest.importorskip("jax")

# Reuse the workspace builder and reference of the C-backend tests.
from test_compiled import _EVENTS, _reference_nll, _workspace


@pytest.fixture
def model():
    ws = _workspace()
    return ws.model(ws.analyses["A"], progress=False)


class TestJaxBackend:
    def test_value_and_gradient_match_the_c_backend(self, model):
        c_nll = model.compile_nll()
        jax_nll = model.compile_nll(backend="jax")
        x = c_nll.to_vector({"mu": 0.7, "sigma": 1.4})

        value, grad = jax_nll.value_and_grad(x)

        assert jax_nll.parameter_names == c_nll.parameter_names
        assert value == pytest.approx(c_nll.value(x), rel=1e-12)
        np.testing.assert_allclose(grad, c_nll.grad(x), rtol=1e-10)
        assert value == pytest.approx(_reference_nll(0.7, 1.4), rel=1e-8)

    def test_value_only(self, model):
        nll = model.compile_nll(backend="jax", gradient=False)
        x = nll.to_vector({"mu": 0.7, "sigma": 1.4})
        assert nll.value(x) == pytest.approx(_reference_nll(0.7, 1.4), rel=1e-8)
        with pytest.raises(RuntimeError, match="gradient=False"):
            nll.value_and_grad(x)

    def test_results_are_float64(self, model):
        """Likelihoods need double precision; the backend enables it."""
        nll = model.compile_nll(backend="jax")
        _, grad = nll.value_and_grad(nll.initial)
        assert grad.dtype == np.float64

    def test_no_free_parameters(self):
        ws = _workspace()
        for name in ("mu", "sigma"):
            ws.parameter_points["init"][name].const = True
        nll = ws.model(ws.analyses["A"], progress=False).compile_nll(backend="jax")

        value, grad = nll.value_and_grad(np.array([]))

        assert value == pytest.approx(_reference_nll(1.0, 2.0), rel=1e-8)
        assert grad.shape == (0,)

    @pytest.mark.parametrize(
        "kwargs", [{"fuse": True}, {"mode": "FAST_RUN"}], ids=["fuse", "mode"]
    )
    def test_c_only_options_are_rejected(self, model, kwargs):
        with pytest.raises(ValueError, match="backend='jax'"):
            model.compile_nll(backend="jax", **kwargs)

    def test_cache_round_trip(self, tmp_path):
        ws = _workspace()
        first = ws.model(ws.analyses["A"], progress=False).compile_nll(
            backend="jax", cache=tmp_path
        )
        second = ws.model(ws.analyses["A"], progress=False).compile_nll(
            backend="jax", cache=tmp_path
        )
        x = first.to_vector({"mu": 0.7, "sigma": 1.4})

        assert (first.from_cache, second.from_cache) == (False, True)
        assert second.value(x) == pytest.approx(first.value(x), rel=1e-14)
        np.testing.assert_allclose(second.grad(x), first.grad(x), rtol=1e-14)

    def test_cache_entries_differ_by_backend(self, tmp_path):
        ws = _workspace()
        ws.model(ws.analyses["A"], progress=False).compile_nll(cache=tmp_path)
        ws.model(ws.analyses["A"], progress=False).compile_nll(
            backend="jax", cache=tmp_path
        )
        assert len([p for p in tmp_path.iterdir() if p.is_file()]) == 2

    def test_scipy_fit_recovers_the_maximum_likelihood_estimate(self, model):
        nll = model.compile_nll(backend="jax")

        result = minimize(
            nll.value_and_grad,
            nll.initial,
            jac=True,
            bounds=nll.bounds,
            method="L-BFGS-B",
        )

        assert result.success
        fitted = nll.to_dict(result.x)
        assert fitted["mu"] == pytest.approx(np.mean(_EVENTS), abs=1e-4)
        assert fitted["sigma"] == pytest.approx(np.std(_EVENTS), abs=1e-4)
