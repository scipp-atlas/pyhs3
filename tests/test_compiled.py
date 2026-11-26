"""Tests for Model.compile_nll and CompiledNLL."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest
from pytensor.graph.traversal import explicit_graph_inputs
from scipy.optimize import minimize
from scipy.stats import truncnorm

from pyhs3 import Workspace
from pyhs3 import compiled as compiled_module
from pyhs3.compiled import CompiledNLL, default_nll_mode

_EVENTS = [0.4, 1.1, 1.9, 2.3, 2.8, 3.6]


def _workspace(extra_parameters: list[dict] | None = None) -> Workspace:
    """Unbinned Gaussian with free mean and width on [-10, 10]."""
    return Workspace(
        metadata={"hs3_version": "0.2"},
        distributions=[
            {
                "name": "g",
                "type": "gaussian_dist",
                "x": "x_obs",
                "mean": "mu",
                "sigma": "sigma",
            }
        ],
        domains=[
            {
                "name": "main",
                "type": "product_domain",
                "axes": [
                    {"name": "mu", "min": -5.0, "max": 5.0},
                    {"name": "sigma", "min": 0.1, "max": 10.0},
                ],
            }
        ],
        data=[
            {
                "name": "d",
                "type": "unbinned",
                "axes": [{"name": "x_obs", "min": -10.0, "max": 10.0}],
                "entries": [[x] for x in _EVENTS],
            }
        ],
        likelihoods=[{"name": "L", "distributions": ["g"], "data": ["d"]}],
        analyses=[
            {"name": "A", "likelihood": "L", "domains": ["main"], "init": "init"}
        ],
        parameter_points=[
            {
                "name": "init",
                "parameters": [
                    {"name": "mu", "value": 1.0},
                    {"name": "sigma", "value": 2.0},
                    *(extra_parameters or []),
                ],
            }
        ],
    )


def _reference_nll(mu: float, sigma: float) -> float:
    a, b = (-10.0 - mu) / sigma, (10.0 - mu) / sigma
    return -float(np.sum(truncnorm.logpdf(_EVENTS, a, b, loc=mu, scale=sigma)))


@pytest.fixture
def model():
    ws = _workspace()
    return ws.model(ws.analyses["A"], progress=False)


class TestCompiledNLLInterface:
    def test_returns_compiled_nll(self, model):
        assert isinstance(model.compile_nll(), CompiledNLL)

    def test_parameters_initial_and_bounds(self, model):
        """Free parameters come in a fixed order with initial values and bounds."""
        nll = model.compile_nll()

        assert set(nll.parameter_names) == {"mu", "sigma"}
        point = dict(zip(nll.parameter_names, nll.initial, strict=True))
        assert point == {"mu": 1.0, "sigma": 2.0}
        bounds = dict(zip(nll.parameter_names, nll.bounds, strict=True))
        assert bounds == {"mu": (-5.0, 5.0), "sigma": (0.1, 10.0)}
        assert nll.initial.shape == (2,)

    def test_vector_and_dict_round_trip(self, model):
        nll = model.compile_nll()
        point = {"mu": 0.3, "sigma": 1.7}

        x = nll.to_vector(point)

        assert x.shape == (2,)
        assert nll.to_dict(x) == point

    def test_to_vector_rejects_unknown_or_missing_names(self, model):
        nll = model.compile_nll()
        with pytest.raises(KeyError, match="sigma"):
            nll.to_vector({"mu": 0.3})
        with pytest.raises(KeyError, match="nope"):
            nll.to_vector({"mu": 0.3, "sigma": 1.0, "nope": 2.0})

    def test_length_is_the_number_of_free_parameters(self, model):
        assert len(model.compile_nll()) == 2

    def test_no_free_parameters(self):
        """With every parameter const the vector is empty and the gradient (0,)."""
        ws = _workspace()
        for name in ("mu", "sigma"):
            ws.parameter_points["init"][name].const = True
        nll = ws.model(ws.analyses["A"], progress=False).compile_nll()
        empty = np.array([])

        value, grad = nll.value_and_grad(empty)

        assert nll.parameter_names == ()
        assert value == pytest.approx(_reference_nll(1.0, 2.0), rel=1e-8)
        assert grad.shape == (0,)

    def test_wrong_length_vector_raises(self, model):
        nll = model.compile_nll()
        with pytest.raises(ValueError, match="expected 2"):
            nll.value(np.array([1.0, 2.0, 3.0]))


class TestCompiledNLLValues:
    def test_value_matches_reference(self, model):
        """value(x) is the NLL -sum(log_prob) with data baked in."""
        nll = model.compile_nll()
        x = nll.to_vector({"mu": 0.7, "sigma": 1.4})

        assert nll.value(x) == pytest.approx(_reference_nll(0.7, 1.4), rel=1e-8)

    def test_value_matches_log_prob(self, model):
        nll = model.compile_nll()
        inputs = [v for v in explicit_graph_inputs([model.log_prob]) if v.name]
        log_prob = pytensor.function(inputs, model.log_prob)
        point = {"mu": -0.4, "sigma": 2.5}

        expected = -float(np.sum(log_prob(**{**model.data, **point})))

        assert nll.value(nll.to_vector(point)) == pytest.approx(expected, rel=1e-12)

    def test_value_and_grad_matches_finite_differences(self, model):
        nll = model.compile_nll()
        x = nll.to_vector({"mu": 0.7, "sigma": 1.4})
        step = 1e-6

        value, grad = nll.value_and_grad(x)

        assert value == pytest.approx(nll.value(x), rel=1e-14)
        assert grad.shape == (2,)
        for i in range(2):
            up, down = x.copy(), x.copy()
            up[i] += step
            down[i] -= step
            fd = (nll.value(up) - nll.value(down)) / (2 * step)
            assert grad[i] == pytest.approx(fd, rel=1e-6)

    def test_grad_matches_value_and_grad(self, model):
        nll = model.compile_nll()
        x = nll.to_vector({"mu": 0.2, "sigma": 3.0})
        np.testing.assert_array_equal(nll.grad(x), nll.value_and_grad(x)[1])

    def test_without_gradient(self, model):
        nll = model.compile_nll(gradient=False)
        x = nll.initial
        assert np.isfinite(nll.value(x))
        with pytest.raises(RuntimeError, match="gradient=False"):
            nll.value_and_grad(x)


class TestCompiledNLLBackend:
    def test_unknown_backend_is_rejected(self, model):
        with pytest.raises(ValueError, match="backend"):
            model.compile_nll(backend="numba")

    @pytest.mark.skipif(
        importlib.util.find_spec("jax") is not None, reason="JAX is installed"
    )
    def test_jax_backend_without_jax_raises(self, model):
        with pytest.raises(ImportError, match="JAX"):
            model.compile_nll(backend="jax")


class TestDefaultNLLMode:
    @pytest.mark.parametrize("fuse", [False, True])
    def test_uses_the_c_rewrite_set(self, fuse):
        """Rewrites come from the C mode, not the Numba mode "FAST_RUN" resolves to by default."""
        query = default_nll_mode(fuse=fuse).provided_optimizer
        assert "fast_run" in query.include
        assert "numba" not in query.include


class TestCompiledNLLFuse:
    def test_fuse_gives_the_same_value_and_gradient(self, model):
        """Fusion changes how the graph is compiled, not what it computes."""
        unfused = model.compile_nll()
        fused = model.compile_nll(fuse=True)
        x = unfused.to_vector({"mu": 0.7, "sigma": 1.4})

        value, grad = fused.value_and_grad(x)

        assert value == pytest.approx(unfused.value(x), rel=1e-12)
        np.testing.assert_allclose(grad, unfused.grad(x), rtol=1e-10)

    def test_fuse_cannot_be_combined_with_mode(self, model):
        with pytest.raises(ValueError, match="fuse=True"):
            model.compile_nll(fuse=True, mode="FAST_RUN")


class TestCompiledNLLModelMode:
    @pytest.fixture
    def compile_modes(self, monkeypatch):
        """Modes passed to PyTensor's ``function`` while compiling the NLL."""
        modes = []
        real_function = compiled_module.function

        def recording_function(*args, **kwargs):
            modes.append(kwargs["mode"])
            return real_function(*args, **kwargs)

        monkeypatch.setattr(compiled_module, "function", recording_function)
        return modes

    def test_model_mode_is_used(self, compile_modes):
        ws = _workspace()
        model = ws.model(ws.analyses["A"], progress=False, mode="FAST_COMPILE")

        model.compile_nll()

        assert compile_modes == ["FAST_COMPILE"]

    def test_default_model_mode_keeps_the_nll_default(self, model, compile_modes):
        model.compile_nll()

        (mode,) = compile_modes
        assert not isinstance(mode, str)
        assert "fast_run" in mode.provided_optimizer.include

    def test_explicit_mode_overrides_the_model_mode(self, compile_modes):
        ws = _workspace()
        model = ws.model(ws.analyses["A"], progress=False, mode="FAST_COMPILE")

        model.compile_nll(mode="FAST_RUN")

        assert compile_modes == ["FAST_RUN"]

    def test_fuse_uses_the_fused_default_mode(self, compile_modes):
        ws = _workspace()
        model = ws.model(ws.analyses["A"], progress=False, mode="FAST_COMPILE")

        model.compile_nll(fuse=True)

        (mode,) = compile_modes
        assert not isinstance(mode, str)


class TestCompiledNLLErrors:
    def test_batched_free_parameter_is_rejected(self):
        ws = _workspace()
        ws.parameter_points["init"]["mu"].kind = pt.vector
        with pytest.warns(UserWarning, match="kind override vector"):
            model = ws.model(ws.analyses["A"], progress=False)
        with pytest.raises(ValueError, match="mu"):
            model.compile_nll()

    def test_requires_a_likelihood(self):
        ws = _workspace()
        model = ws.model(0, progress=False)
        with pytest.raises(RuntimeError, match="likelihood"):
            model.compile_nll()


class TestCompiledNLLCache:
    def test_cache_round_trip(self, tmp_path):
        ws = _workspace()
        first = ws.model(ws.analyses["A"], progress=False).compile_nll(cache=tmp_path)
        assert not first.from_cache
        assert len(list(tmp_path.iterdir())) == 1

        second = ws.model(ws.analyses["A"], progress=False).compile_nll(cache=tmp_path)

        assert second.from_cache
        x = first.to_vector({"mu": 0.7, "sigma": 1.4})
        assert second.value(x) == first.value(x)
        np.testing.assert_array_equal(second.grad(x), first.grad(x))

    def test_cache_key_depends_on_options_and_model(self, tmp_path):
        ws = _workspace()
        ws.model(ws.analyses["A"], progress=False).compile_nll(cache=tmp_path)
        ws.model(ws.analyses["A"], progress=False).compile_nll(
            cache=tmp_path, gradient=False
        )
        ws.model(ws.analyses["A"], progress=False).compile_nll(
            cache=tmp_path, fuse=True
        )
        other = _workspace()
        other.parameter_points["init"]["sigma"].const = True
        other.model(other.analyses["A"], progress=False).compile_nll(cache=tmp_path)

        assert len(list(tmp_path.iterdir())) == 4


class TestCompiledNLLCacheLocations:
    def test_default_cache_directory(self, tmp_path, monkeypatch):
        """cache=True writes under $XDG_CACHE_HOME/pyhs3."""
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
        ws = _workspace()

        nll = ws.model(ws.analyses["A"], progress=False).compile_nll(cache=True)

        assert not nll.from_cache
        assert len(list((tmp_path / "pyhs3").iterdir())) == 1

    @pytest.mark.parametrize(
        "mode",
        [
            pytest.param("FAST_RUN", id="mode_name"),
            pytest.param(None, id="mode_object"),
        ],
    )
    def test_cache_with_explicit_mode(self, tmp_path, mode):
        """A named mode and a Mode object each get a reusable cache entry."""
        mode = default_nll_mode(fuse=True) if mode is None else mode
        ws = _workspace()
        first = ws.model(ws.analyses["A"], progress=False).compile_nll(
            cache=tmp_path, mode=mode
        )
        second = ws.model(ws.analyses["A"], progress=False).compile_nll(
            cache=tmp_path, mode=mode
        )

        assert (first.from_cache, second.from_cache) == (False, True)
        assert len(list(tmp_path.iterdir())) == 1

    def test_failed_cache_write_leaves_no_files(self, tmp_path, monkeypatch):
        """If pickling fails, the writer removes its own temporary file."""

        def fail(*_args, **_kwargs):
            msg = "cannot pickle"
            raise RuntimeError(msg)

        monkeypatch.setattr(compiled_module.pickle, "dump", fail)
        ws = _workspace()

        with pytest.raises(RuntimeError, match="cannot pickle"):
            ws.model(ws.analyses["A"], progress=False).compile_nll(cache=tmp_path)

        assert list(tmp_path.iterdir()) == []


class TestCompiledNLLFit:
    def test_scipy_fit_recovers_the_maximum_likelihood_estimate(self, model):
        """End to end: L-BFGS-B with the analytic gradient finds the MLE."""
        nll = model.compile_nll()

        result = minimize(
            nll.value_and_grad,
            nll.initial,
            jac=True,
            bounds=nll.bounds,
            method="L-BFGS-B",
        )

        assert result.success
        fitted = nll.to_dict(result.x)
        # The data sit far inside [-10, 10], so the truncated-Gaussian MLE is
        # the sample mean and the (biased) sample standard deviation.
        assert fitted["mu"] == pytest.approx(np.mean(_EVENTS), abs=1e-4)
        assert fitted["sigma"] == pytest.approx(np.std(_EVENTS), abs=1e-4)


def _poi_model(pois: list[str], **const: bool):
    """The _workspace model with the analysis listing ``pois`` as parameters of interest."""
    ws = _workspace()
    ws.analyses["A"].parameters_of_interest = pois
    for name, flag in const.items():
        ws.parameter_points["init"][name].const = flag
    return ws.model(ws.analyses["A"], progress=False)


class TestCompiledNLLParametersOfInterest:
    def test_names_and_indices_from_the_analysis(self):
        nll = _poi_model(["sigma", "mu"]).compile_nll()

        assert nll.parameters_of_interest == ("sigma", "mu")
        assert tuple(nll.parameter_names[i] for i in nll.poi_indices) == (
            "sigma",
            "mu",
        )

    def test_no_parameters_of_interest(self, model):
        nll = model.compile_nll()

        assert nll.parameters_of_interest == ()
        assert nll.poi_indices == ()

    def test_const_parameter_of_interest_is_not_free(self):
        """A POI held const is baked into the function, so it has no index."""
        nll = _poi_model(["mu"], mu=True).compile_nll()

        assert nll.parameter_names == ("sigma",)
        assert nll.parameters_of_interest == ()
        assert nll.poi_indices == ()

    def test_kept_when_loaded_from_cache(self, tmp_path):
        _poi_model(["mu"]).compile_nll(cache=tmp_path)
        nll = _poi_model(["mu"]).compile_nll(cache=tmp_path)

        assert nll.from_cache
        assert nll.parameters_of_interest == ("mu",)
        assert nll.parameter_names[nll.poi_indices[0]] == "mu"

    def test_compile_nll_function_argument(self, model):
        log_prob = model.log_prob
        mu, sigma, x_obs = sorted(
            (v for v in explicit_graph_inputs([log_prob]) if v.name),
            key=lambda v: v.name,
        )
        nll = compiled_module.compile_nll(
            log_prob,
            [sigma, mu],
            {x_obs: np.asarray(_EVENTS)},
            [2.0, 1.0],
            [(None, None), (None, None)],
            parameters_of_interest=["mu", "lumi"],
        )

        assert nll.parameters_of_interest == ("mu",)
        assert nll.poi_indices == (1,)


def _const_model(**const: float):
    """The _workspace model with each named parameter const at the given value."""
    ws = _workspace()
    for name, value in const.items():
        ws.parameter_points["init"][name].value = value
        ws.parameter_points["init"][name].const = True
    return ws.model(ws.analyses["A"], progress=False)


class TestCompiledNLLFixed:
    def test_matches_a_model_with_the_parameter_const(self, model):
        nll = model.compile_nll(fixed={"mu": 0.5})
        reference = _const_model(mu=0.5).compile_nll()

        assert nll.parameter_names == reference.parameter_names == ("sigma",)
        for sigma in (0.7, 1.4, 3.0):
            x = np.array([sigma])
            value, grad = nll.value_and_grad(x)
            assert value == pytest.approx(reference.value(x), rel=1e-12)
            np.testing.assert_allclose(grad, reference.grad(x), rtol=1e-10)
            assert value == pytest.approx(_reference_nll(0.5, sigma), rel=1e-8)

    def test_initial_and_bounds_drop_the_fixed_parameter(self, model):
        nll = model.compile_nll(fixed={"sigma": 1.5})

        assert nll.parameter_names == ("mu",)
        assert nll.initial.tolist() == [1.0]
        assert nll.bounds == [(-5.0, 5.0)]

    def test_every_parameter_fixed(self, model):
        nll = model.compile_nll(fixed={"mu": 0.3, "sigma": 1.2})

        value, grad = nll.value_and_grad(np.array([]))

        assert nll.parameter_names == ()
        assert value == pytest.approx(_reference_nll(0.3, 1.2), rel=1e-8)
        assert grad.shape == (0,)

    def test_fixed_parameter_of_interest_is_not_listed(self):
        """Like a const one, a fixed POI is baked into the function, so it has no index."""
        nll = _poi_model(["mu"]).compile_nll(fixed={"mu": 0.0})

        assert nll.parameters_of_interest == ()
        assert nll.poi_indices == ()

    def test_the_model_is_unchanged(self, model):
        model.compile_nll(fixed={"mu": 0.5})
        assert set(model.compile_nll().parameter_names) == {"mu", "sigma"}

    def test_value_outside_the_domain_warns(self, model):
        with pytest.warns(UserWarning, match="outside domain"):
            model.compile_nll(fixed={"mu": 7.0})

    def test_unknown_name_is_rejected(self, model):
        with pytest.raises(ValueError, match="nope"):
            model.compile_nll(fixed={"nope": 1.0})

    def test_observable_is_rejected(self, model):
        with pytest.raises(ValueError, match="x_obs"):
            model.compile_nll(fixed={"x_obs": 1.0})

    def test_const_parameter_is_rejected(self):
        model = _const_model(sigma=2.0)
        with pytest.raises(ValueError, match=r"sigma.*const"):
            model.compile_nll(fixed={"sigma": 1.0})

    def test_profile_likelihood_with_scipy(self, model):
        """Profiling sigma at fixed mu gives the closed-form conditional MLE."""
        for mu in (0.5, 1.5, 2.5):
            nll = model.compile_nll(fixed={"mu": mu})

            result = minimize(
                nll.value_and_grad,
                nll.initial,
                jac=True,
                bounds=nll.bounds,
                method="L-BFGS-B",
            )

            assert result.success
            # The data sit far inside [-10, 10], so the truncated-Gaussian
            # conditional MLE of sigma is the RMS deviation from mu.
            expected = np.sqrt(np.mean((np.asarray(_EVENTS) - mu) ** 2))
            assert nll.to_dict(result.x)["sigma"] == pytest.approx(expected, abs=1e-4)
