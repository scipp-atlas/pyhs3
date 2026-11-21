"""Tests for Model.compile_nll(backend="jax"); skipped without JAX."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import minimize

jax = pytest.importorskip("jax")

# Reuse the workspace builder and reference of the C-backend tests.
from test_compiled import _EVENTS, _reference_nll, _workspace

from pyhs3.compiled import BatchedCompiledNLL, CompiledNLL


@pytest.fixture(autouse=True)
def isolated_xla_cache():
    """Start each test without a JAX compilation cache directory.

    JAX initializes its persistent compilation cache once per process, at the
    first compile after a directory is set.  Disabling the cache keeps a
    test's temporary directory from becoming that process-wide cache, while
    compile_nll can still set the directory option under test.
    """
    saved = (
        jax.config.jax_compilation_cache_dir,
        jax.config.jax_enable_compilation_cache,
    )
    jax.config.update("jax_compilation_cache_dir", None)
    jax.config.update("jax_enable_compilation_cache", False)
    yield
    jax.config.update("jax_compilation_cache_dir", saved[0])
    jax.config.update("jax_enable_compilation_cache", saved[1])


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


class TestJaxCompilationCacheDirectory:
    def test_cache_sets_the_xla_cache_directory(self, model, tmp_path):
        model.compile_nll(backend="jax", cache=tmp_path)
        assert jax.config.jax_compilation_cache_dir == str(tmp_path / "xla")

    def test_an_existing_xla_cache_directory_is_kept(self, model, tmp_path):
        jax.config.update("jax_compilation_cache_dir", str(tmp_path / "mine"))
        model.compile_nll(backend="jax", cache=tmp_path / "pyhs3")
        assert jax.config.jax_compilation_cache_dir == str(tmp_path / "mine")

    def test_without_cache_the_xla_cache_directory_is_unset(self, model):
        model.compile_nll(backend="jax")
        assert jax.config.jax_compilation_cache_dir is None


# Compiles the test workspace with cache=CACHE_DIR in a fresh process, as a
# user's second session would.  The XLA step of the toy model takes well
# under JAX's default one-second threshold for persisting an executable.
_SESSION = """
import logging
import sys

import jax

sys.path.insert(0, sys.argv[2])
from test_compiled import _workspace

jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
compiler_log = logging.getLogger("jax._src.compiler")
# Importing pyhs3 disables loggers that existed before it, such as JAX's.
compiler_log.disabled = False
compiler_log.setLevel(logging.DEBUG)
compiler_log.propagate = False
compiler_log.addHandler(logging.StreamHandler())

ws = _workspace()
nll = ws.model(ws.analyses["A"], progress=False).compile_nll(
    backend="jax", cache=sys.argv[1]
)
print(nll.from_cache, repr(nll.value(nll.to_vector({"mu": 0.7, "sigma": 1.4}))))
"""


class TestJaxCompilationCacheAcrossSessions:
    def _session(self, cache_dir):
        return subprocess.run(
            [
                sys.executable,
                "-c",
                _SESSION,
                str(cache_dir),
                str(Path(__file__).parent),
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=300,
        )

    def test_second_session_loads_the_xla_executable(self, tmp_path):
        first = self._session(tmp_path)
        entries = sorted(p.name for p in (tmp_path / "xla").iterdir())
        second = self._session(tmp_path)

        assert entries
        assert sorted(p.name for p in (tmp_path / "xla").iterdir()) == entries
        assert "Persistent compilation cache hit" not in first.stderr
        assert "Persistent compilation cache hit" in second.stderr
        first_cached, first_value = first.stdout.split()
        second_cached, second_value = second.stdout.split()
        assert (first_cached, second_cached) == ("False", "True")
        assert float(second_value) == float(first_value)
        assert float(second_value) == pytest.approx(_reference_nll(0.7, 1.4), rel=1e-8)


@pytest.fixture
def batch(model):
    """Three parameter vectors, shape (K, P) = (3, 2), as {name: value} rows."""
    nll = model.compile_nll(backend="jax")
    rows = [
        {"mu": 0.7, "sigma": 1.4},
        {"mu": 1.0, "sigma": 2.0},
        {"mu": 2.0, "sigma": 1.0},
    ]
    return np.stack([nll.to_vector(row) for row in rows]), nll


class TestJaxBatched:
    def test_returns_a_batched_compiled_nll(self, model):
        nll = model.compile_nll(backend="jax", batched=True)
        assert isinstance(nll, BatchedCompiledNLL)
        assert isinstance(nll, CompiledNLL)

    def test_rows_match_the_one_point_function(self, model, batch):
        X, one = batch
        nll = model.compile_nll(backend="jax", batched=True)

        values, grads = nll.value_and_grad(X)

        assert nll.parameter_names == one.parameter_names
        assert values.shape == (3,)
        assert grads.shape == (3, 2)
        for row, value, grad in zip(X, values, grads, strict=True):
            assert value == pytest.approx(one.value(row), rel=1e-12)
            np.testing.assert_allclose(grad, one.grad(row), rtol=1e-10)
        np.testing.assert_allclose(nll.value(X), values, rtol=0)
        np.testing.assert_allclose(nll.grad(X), grads, rtol=0)

    def test_any_batch_size(self, model, batch):
        """One compiled likelihood evaluates any number of rows."""
        X, one = batch
        nll = model.compile_nll(backend="jax", batched=True)
        big = np.tile(X, (4, 1))  # (12, 2)

        assert nll.value(X[:1]).shape == (1,)
        values = nll.value(big)

        assert values.shape == (12,)
        np.testing.assert_allclose(values, [one.value(row) for row in big], rtol=1e-12)

    def test_value_only(self, model, batch):
        X, one = batch
        nll = model.compile_nll(backend="jax", batched=True, gradient=False)
        np.testing.assert_allclose(
            nll.value(X), [one.value(row) for row in X], rtol=1e-12
        )
        with pytest.raises(RuntimeError, match="gradient=False"):
            nll.value_and_grad(X)

    def test_no_free_parameters(self):
        ws = _workspace()
        for name in ("mu", "sigma"):
            ws.parameter_points["init"][name].const = True
        nll = ws.model(ws.analyses["A"], progress=False).compile_nll(
            backend="jax", batched=True
        )

        values, grads = nll.value_and_grad(np.empty((3, 0)))

        np.testing.assert_allclose(values, [_reference_nll(1.0, 2.0)] * 3, rtol=1e-8)
        assert grads.shape == (3, 0)

    def test_one_vector_is_rejected(self, model):
        nll = model.compile_nll(backend="jax", batched=True)
        with pytest.raises(ValueError, match=r"\(K, 2\)"):
            nll.value(nll.initial)

    def test_requires_the_jax_backend(self, model):
        with pytest.raises(ValueError, match="batched=True"):
            model.compile_nll(batched=True)

    def test_cache_round_trip(self, tmp_path, batch):
        X, _ = batch
        ws = _workspace()
        first = ws.model(ws.analyses["A"], progress=False).compile_nll(
            backend="jax", batched=True, cache=tmp_path
        )
        second = ws.model(ws.analyses["A"], progress=False).compile_nll(
            backend="jax", batched=True, cache=tmp_path
        )

        assert (first.from_cache, second.from_cache) == (False, True)
        np.testing.assert_allclose(second.value(X), first.value(X), rtol=1e-14)

    def test_batched_and_one_point_cache_entries_differ(self, tmp_path):
        ws = _workspace()
        ws.model(ws.analyses["A"], progress=False).compile_nll(
            backend="jax", cache=tmp_path
        )
        ws.model(ws.analyses["A"], progress=False).compile_nll(
            backend="jax", batched=True, cache=tmp_path
        )
        assert len([p for p in tmp_path.iterdir() if p.is_file()]) == 2
