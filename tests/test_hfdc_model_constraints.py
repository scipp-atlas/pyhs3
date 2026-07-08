"""Tests for HFDC constraint deduplication and log_prob integration.

PR #2: When multiple HistFactoryDistChannel (HFDC) instances share nuisance
parameters, each channel's extended_likelihood previously emitted its own
constraint term, causing double-counting in the joint NLL.

Fix: _build_distribution_node stores only dist.likelihood(context) for HFDC
(Poisson term only). Model.log_prob collects constraint terms once per unique
nuisance parameter across all HFDC channels.
"""

from __future__ import annotations

import math
import types

import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest

from pyhs3.context import Context
from pyhs3.data import BinnedData, Data, UnbinnedData
from pyhs3.distributions import Distributions, HistFactoryDistChannel
from pyhs3.distributions.histfactory.modifiers import (
    HasConstraint,
    ParameterModifier,
    ParametersModifier,
    SingleParamConstraint,
)
from pyhs3.domains import Domains, ProductDomain
from pyhs3.exceptions import WorkspaceValidationError
from pyhs3.likelihoods import Likelihood, Likelihoods
from pyhs3.metadata import Metadata
from pyhs3.parameter_points import ParameterPoint, ParameterPoints, ParameterSet
from pyhs3.workspace import Workspace

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_channel(name: str, contents: list[float], modifiers: list[dict]) -> dict:
    """Return a JSON-like dict for HistFactoryDistChannel constructor.

    The observable axis is named after the channel (e.g., "x_SR") so that
    multi-channel workspaces pass the unique-axis-name validation.
    """
    obs_name = f"x_{name}"
    return {
        "name": name,
        "axes": [{"name": obs_name, "min": 0.0, "max": 10.0, "nbins": len(contents)}],
        "samples": [
            {
                "name": "signal",
                "data": {"contents": contents, "errors": [1.0] * len(contents)},
                "modifiers": modifiers,
            }
        ],
    }


def _simple_workspace(channels: list[dict], params: list[dict]) -> Workspace:
    """Build a minimal Workspace with HFDC channel(s) and BinnedData."""
    distributions = []
    data = []
    likelihood_dists = []
    likelihood_data = []

    for ch in channels:
        dist = HistFactoryDistChannel(**ch)
        distributions.append(dist)
        likelihood_dists.append(dist)

        nbins = len(ch["samples"][0]["data"]["contents"])
        obs_name = ch["axes"][0]["name"]  # already set per-channel by _make_channel
        binned = BinnedData(
            name=f"{ch['name']}_data",
            axes=[{"name": obs_name, "min": 0.0, "max": 10.0, "nbins": nbins}],
            contents=ch["samples"][0]["data"]["contents"],
        )
        data.append(binned)
        likelihood_data.append(binned)

    param_points = [ParameterPoint(name=p["name"], value=p["value"]) for p in params]

    return Workspace(
        metadata=Metadata(hs3_version="0.3.0"),
        distributions=Distributions(distributions),
        data=Data(data),
        likelihoods=Likelihoods(
            [
                Likelihood(
                    name="L",
                    distributions=likelihood_dists,
                    data=likelihood_data,
                )
            ]
        ),
        domains=Domains([ProductDomain(name="default")]),
        parameter_points=ParameterPoints(
            [
                ParameterSet(
                    name="default",
                    parameters=param_points,
                )
            ]
        ),
    )


# ---------------------------------------------------------------------------
# Unit tests: constraint_specs()
# ---------------------------------------------------------------------------


class TestConstraintModifiers:
    """Unit tests for HistFactoryDistChannel.constraint_specs()."""

    def test_empty_no_constraints(self):
        """Channel with no HasConstraint modifiers yields nothing."""
        ch = HistFactoryDistChannel(
            **_make_channel(
                "ch",
                [10.0, 20.0],
                [{"name": "mu", "type": "normfactor", "parameter": "mu"}],
            )
        )
        assert list(ch.constraint_specs()) == []

    def test_normsys_is_in_single(self):
        """normsys (ParameterModifier + HasConstraint) yields a string dedup_key."""
        ch = HistFactoryDistChannel(
            **_make_channel(
                "ch",
                [10.0],
                [
                    {
                        "name": "lumi",
                        "type": "normsys",
                        "parameter": "lumi",
                        "constraint": "Gauss",
                        "data": {"hi": 1.05, "lo": 0.95},
                    }
                ],
            )
        )
        specs = list(ch.constraint_specs())
        assert len(specs) == 1
        dedup_key, modifier, _sample_data = specs[0]
        assert dedup_key == "lumi"
        assert isinstance(modifier, HasConstraint)
        assert isinstance(modifier, ParameterModifier)

    def test_shapesys_is_in_multi(self):
        """shapesys (ParametersModifier + HasConstraint) yields dedup_key=None."""
        ch = HistFactoryDistChannel(
            **_make_channel(
                "ch",
                [10.0, 20.0],
                [
                    {
                        "name": "stat",
                        "type": "shapesys",
                        "parameters": ["gamma_0", "gamma_1"],
                        "constraint": "Poisson",
                        "data": {"vals": [2.0, 4.0]},
                    }
                ],
            )
        )
        specs = list(ch.constraint_specs())
        assert len(specs) == 1
        dedup_key, modifier, _sample_data = specs[0]
        assert dedup_key is None
        assert isinstance(modifier, HasConstraint)
        assert isinstance(modifier, ParametersModifier)

    def test_mixed_returns_both(self):
        """Channel with both normsys and shapesys yields both kinds of specs."""
        ch = HistFactoryDistChannel(
            **_make_channel(
                "ch",
                [10.0, 20.0],
                [
                    {
                        "name": "lumi",
                        "type": "normsys",
                        "parameter": "lumi",
                        "constraint": "Gauss",
                        "data": {"hi": 1.05, "lo": 0.95},
                    },
                    {
                        "name": "stat",
                        "type": "shapesys",
                        "parameters": ["gamma_0", "gamma_1"],
                        "constraint": "Poisson",
                        "data": {"vals": [2.0, 4.0]},
                    },
                ],
            )
        )
        specs = list(ch.constraint_specs())
        assert len(specs) == 2
        keys = [key for key, _, _ in specs]
        assert "lumi" in keys
        assert None in keys

    def test_duplicate_normsys_parameter_in_same_channel_yields_both(self):
        """Two samples with the same normsys parameter yield two specs with the same key.

        The caller (extended_likelihood, _build_distribution_node) is responsible for
        deduping using the key — constraint_specs() yields all modifiers without dedup.
        """
        dist = HistFactoryDistChannel(
            name="ch",
            axes=[{"name": "x", "min": 0.0, "max": 10.0, "nbins": 1}],
            samples=[
                {
                    "name": "sig",
                    "data": {"contents": [10.0], "errors": [1.0]},
                    "modifiers": [
                        {
                            "name": "lumi",
                            "type": "normsys",
                            "parameter": "lumi",
                            "constraint": "Gauss",
                            "data": {"hi": 1.05, "lo": 0.95},
                        }
                    ],
                },
                {
                    "name": "bkg",
                    "data": {"contents": [5.0], "errors": [1.0]},
                    "modifiers": [
                        {
                            "name": "lumi",
                            "type": "normsys",
                            "parameter": "lumi",
                            "constraint": "Gauss",
                            "data": {"hi": 1.05, "lo": 0.95},
                        }
                    ],
                },
            ],
        )
        specs = list(dist.constraint_specs())
        keys = [key for key, _, _ in specs]
        # Both specs have the same dedup_key; callers apply the seen-set dedup.
        assert keys == ["lumi", "lumi"]

    def test_log_extended_likelihood_dedups_shared_parameter(self):
        """Two samples sharing a normsys parameter contribute one log-constraint.

        Mirrors the spec-level test above: log_extended_likelihood must apply
        the seen-set dedup, so the shared Gaussian constraint appears once
        (log N(0|0,1) = -0.5*log(2*pi)), not twice.
        """
        dist = HistFactoryDistChannel(
            name="ch",
            axes=[{"name": "x", "min": 0.0, "max": 10.0, "nbins": 1}],
            samples=[
                {
                    "name": "sig",
                    "data": {"contents": [10.0], "errors": [1.0]},
                    "modifiers": [
                        {
                            "name": "lumi",
                            "type": "normsys",
                            "parameter": "lumi",
                            "constraint": "Gauss",
                            "data": {"hi": 1.05, "lo": 0.95},
                        }
                    ],
                },
                {
                    "name": "bkg",
                    "data": {"contents": [5.0], "errors": [1.0]},
                    "modifiers": [
                        {
                            "name": "lumi",
                            "type": "normsys",
                            "parameter": "lumi",
                            "constraint": "Gauss",
                            "data": {"hi": 1.05, "lo": 0.95},
                        }
                    ],
                },
            ],
        )
        lumi = pt.dscalar("lumi")
        context = Context({"lumi": lumi})
        expr = dist.log_extended_likelihood(context)
        fn = pytensor.function([lumi], expr)
        val = float(fn(0.0))
        assert abs(val - (-0.5 * math.log(2 * math.pi))) < 1e-9

    def test_log_extended_likelihood_empty_constraints_is_zero(self):
        """A channel with no constrained modifiers returns exactly 0.0.

        Covers the early-return branch in log_extended_likelihood taken when
        constraint_specs() yields nothing and BB-lite is not active — e.g. a
        channel whose only modifier is an unconstrained normfactor. Also checks
        that log_expression reduces to log_likelihood alone in this case.
        """
        dist = HistFactoryDistChannel(
            **_make_channel(
                "ch",
                [10.0, 20.0],
                [{"name": "mu", "type": "normfactor", "parameter": "mu"}],
            )
        )
        mu = pt.dscalar("mu")
        observed = pt.constant(np.array([10.0, 20.0]))
        context = Context({"mu": mu, "ch_observed": observed})

        expr = dist.log_extended_likelihood(context)
        # expr does not depend on mu (there are no constrained modifiers to
        # tie mu's value to the graph).
        fn = pytensor.function([mu], expr, on_unused_input="ignore")
        assert fn(1.0) == 0.0

        log_lik = dist.log_likelihood(context)
        log_expr = dist.log_expression(context)
        fn_expr = pytensor.function([mu], [log_lik, log_expr])
        lik_val, expr_val = fn_expr(1.0)
        assert expr_val == pytest.approx(lik_val)


# ---------------------------------------------------------------------------
# Unit tests: Model._try_bake_hfdc_observed
# ---------------------------------------------------------------------------


class TestTryBakeHFDCObserved:
    """Unit tests for Model._try_bake_hfdc_observed."""

    def test_skips_non_binned_data_entries(self):
        """Non-BinnedData likelihood entries are skipped; matching BinnedData is baked.

        Exercises the ``continue`` branch where a datum is not a BinnedData —
        the search must carry on to find the correct BinnedData for the channel.
        """
        ws = _simple_workspace(
            channels=[_make_channel("SR", [10.0, 20.0], [])],
            params=[],
        )
        model = ws.model(next(iter(ws.likelihoods)), progress=False)

        sr_channel = next(iter(ws.distributions))
        binned = BinnedData(
            name="SR_data",
            axes=[{"name": "x_SR", "min": 0.0, "max": 10.0, "nbins": 2}],
            contents=[10.0, 20.0],
        )
        unbinned = UnbinnedData(
            name="other_data",
            entries=[[1.0]],
            axes=[{"name": "y", "min": 0.0, "max": 5.0}],
        )
        # Unbinned datum comes first so the loop hits `continue` at least once
        # before finding the matching BinnedData.
        model._likelihood = types.SimpleNamespace(
            distributions=["other_dist", sr_channel],
            data=[unbinned, binned],
        )

        result = model._try_bake_hfdc_observed("SR_observed")
        assert result is not None
        np.testing.assert_array_equal(result.data, [10.0, 20.0])


# ---------------------------------------------------------------------------
# Integration tests: log_prob includes HFDC Poisson term
# ---------------------------------------------------------------------------


class TestHFDCLogProb:
    """log_prob for HFDC workspaces includes Poisson and constraint terms."""

    def _eval_log_prob(self, ws: Workspace) -> float:
        """Build a model from the first likelihood and evaluate log_prob at nominal params."""
        likelihood = next(iter(ws.likelihoods))
        model = ws.model(likelihood, progress=False)
        lp = model.log_prob
        inputs = {
            v.name: v
            for v in pytensor.graph.traversal.explicit_graph_inputs([lp])
            if v.name
        }
        fn = pytensor.function(list(inputs.values()), lp)
        return float(fn(**model.data, **model.nominal_params).item())

    def test_log_prob_is_finite_for_hfdc(self):
        """log_prob must be a finite number, not zero (which would mean HFDC was skipped)."""
        ws = _simple_workspace(
            channels=[_make_channel("SR", [10.0, 20.0], [])],
            params=[],
        )
        likelihood = next(iter(ws.likelihoods))
        model = ws.model(likelihood, progress=False)
        lp = model.log_prob
        inputs = {
            v.name: v
            for v in pytensor.graph.traversal.explicit_graph_inputs([lp])
            if v.name
        }
        fn = pytensor.function(list(inputs.values()), lp)
        val = float(fn(**model.data, **model.nominal_params).item())
        assert math.isfinite(val)
        assert val != 0.0

    def test_log_prob_no_constraint_matches_poisson(self):
        """Single HFDC with no constraints: log_prob equals sum of Poisson log-probs.

        When observed == expected, Poisson log-prob for bin k is
          obs_k * log(exp_k) - exp_k - log(obs_k!)
        """
        contents = [10.0, 20.0]
        ws = _simple_workspace(
            channels=[_make_channel("SR", contents, [])],
            params=[],
        )
        likelihood = next(iter(ws.likelihoods))
        model = ws.model(likelihood, progress=False)
        lp = model.log_prob
        inputs = {
            v.name: v
            for v in pytensor.graph.traversal.explicit_graph_inputs([lp])
            if v.name
        }
        fn = pytensor.function(list(inputs.values()), lp)
        # Nominal: obs == exp for each bin
        obs = np.array(contents)
        val = float(fn(**model.data, **model.nominal_params).item())
        expected = float(
            np.sum(
                obs * np.log(obs) - obs - np.array([math.lgamma(o + 1) for o in obs])
            )
        )
        assert abs(val - expected) < 1e-6

    def test_log_prob_single_channel_with_normsys(self):
        """Single HFDC with normsys: log_prob = Poisson + constraint.

        At alpha=0, the Gaussian constraint log-prob is -0.5*log(2*pi) ≈ -0.9189.
        """
        contents = [10.0]
        ws = _simple_workspace(
            channels=[
                _make_channel(
                    "SR",
                    contents,
                    [
                        {
                            "name": "lumi",
                            "type": "normsys",
                            "parameter": "lumi",
                            "constraint": "Gauss",
                            "data": {"hi": 1.05, "lo": 0.95},
                        }
                    ],
                )
            ],
            params=[{"name": "lumi", "value": 0.0}],
        )
        likelihood = next(iter(ws.likelihoods))
        model = ws.model(likelihood, progress=False)
        lp = model.log_prob
        inputs = {
            v.name: v
            for v in pytensor.graph.traversal.explicit_graph_inputs([lp])
            if v.name
        }
        fn = pytensor.function(list(inputs.values()), lp)
        val = float(fn(**model.data, **model.nominal_params).item())

        # Poisson part: obs=10 at exp=10
        poisson_lp = 10.0 * math.log(10.0) - 10.0 - math.lgamma(11.0)
        # Gaussian constraint at alpha=0: log N(0|0,1) = -0.5*log(2*pi)
        gauss_lp = -0.5 * math.log(2 * math.pi)
        expected = poisson_lp + gauss_lp
        assert abs(val - expected) < 1e-6, f"got {val}, expected {expected}"


# ---------------------------------------------------------------------------
# Regression tests: BB-lite staterror channels build through Model (#255)
# ---------------------------------------------------------------------------


class TestBBLiteStaterrorModelBuild:
    """Regression tests for issue #255.

    ``Model._build_distribution_node``'s HFDC constraint loop previously
    called ``modifier.make_constraint()`` unconditionally for every
    ``constraint_specs()`` entry, including lite-mode ``StatErrorModifier``
    instances (``data=None`` by design in lite mode -- per-bin errors come
    from sample data, not modifier data). That raised ``ValueError`` at Model
    build time for the (default) lite ``barlow_beeston_method``. The loop
    also never included the channel-level BB-lite constraint that
    ``HistFactoryDistChannel.extended_likelihood`` adds.
    """

    def _lite_staterror_channel_dict(self) -> dict:
        """Two-sample channel with a shared BB-lite staterror modifier.

        Matches the combined-uncertainty numbers exercised in
        test_histfactory.py::test_lite_combined_uncertainties: total_nominal
        = [30, 50], total_sigma = [5, sqrt(41)].
        """
        staterror = {
            "name": "stat_error",
            "type": "staterror",
            "parameters": ["gamma_bin0", "gamma_bin1"],
            "constraint": "Gauss",
        }
        return {
            "name": "SR",
            "axes": [{"name": "x_SR", "min": 0.0, "max": 10.0, "nbins": 2}],
            "samples": [
                {
                    "name": "signal",
                    "data": {"contents": [10.0, 20.0], "errors": [3.0, 4.0]},
                    "modifiers": [staterror],
                },
                {
                    "name": "background",
                    "data": {"contents": [20.0, 30.0], "errors": [4.0, 5.0]},
                    "modifiers": [staterror],
                },
            ],
        }

    def _lite_staterror_workspace(self) -> Workspace:
        """Workspace with a single lite-mode (default) staterror channel."""
        dist = HistFactoryDistChannel(**self._lite_staterror_channel_dict())
        binned = BinnedData(
            name="SR_data",
            axes=[{"name": "x_SR", "min": 0.0, "max": 10.0, "nbins": 2}],
            contents=[30.0, 50.0],
        )
        return Workspace(
            metadata=Metadata(hs3_version="0.3.0"),
            distributions=Distributions([dist]),
            data=Data([binned]),
            likelihoods=Likelihoods(
                [Likelihood(name="L", distributions=[dist], data=[binned])]
            ),
            domains=Domains([ProductDomain(name="default")]),
            parameter_points=ParameterPoints(
                [
                    ParameterSet(
                        name="default",
                        parameters=[
                            ParameterPoint(name="gamma_bin0", value=1.0),
                            ParameterPoint(name="gamma_bin1", value=1.0),
                        ],
                    )
                ]
            ),
        )

    def test_model_builds_and_log_prob_is_finite(self):
        """A default (lite-mode) staterror channel must build through ws.model().

        Before the fix this raised ``ValueError`` ("data is required for
        BB-full mode") because the model-side constraint loop called
        ``make_constraint()`` on the lite-mode ``StatErrorModifier``, whose
        ``data`` is ``None`` by design.
        """
        ws = self._lite_staterror_workspace()
        likelihood = next(iter(ws.likelihoods))
        model = ws.model(likelihood, progress=False)

        lp = model.log_prob
        inputs = {
            v.name: v
            for v in pytensor.graph.traversal.explicit_graph_inputs([lp])
            if v.name
        }
        fn = pytensor.function(list(inputs.values()), lp)
        val = float(fn(**model.data, **model.nominal_params).item())
        assert math.isfinite(val)

    def test_log_prob_includes_bblite_constraint(self):
        """log_prob must include the channel-level BB-lite constraint factor.

        Guards the second half of the bug: even once the crash is fixed by
        skipping lite-mode ``StatErrorModifier`` specs, the model-side loop
        must still add
        ``HistFactoryDistChannel._make_barlow_beeston_lite_constraint()`` --
        exactly as ``extended_likelihood()`` does -- or the joint ``log_prob``
        would silently omit the gamma constraint term.
        """
        ws = self._lite_staterror_workspace()
        likelihood = next(iter(ws.likelihoods))
        model = ws.model(likelihood, progress=False)

        lp = model.log_prob
        inputs = {
            v.name: v
            for v in pytensor.graph.traversal.explicit_graph_inputs([lp])
            if v.name
        }
        fn = pytensor.function(list(inputs.values()), lp)
        val = float(fn(**model.data, **model.nominal_params).item())

        # Poisson part: obs == exp == [30, 50] at gamma=1.0 (nominal).
        obs = np.array([30.0, 50.0])
        poisson_lp = float(
            np.sum(
                obs * np.log(obs) - obs - np.array([math.lgamma(o + 1) for o in obs])
            )
        )

        # BB-lite Gaussian constraint at gamma=1.0: N(1|1,relerr) per bin,
        # log-summed. total_nominal=[30,50], total_sigma=[5, sqrt(41)].
        total_sigma = np.array([5.0, np.sqrt(41.0)])
        total_nominal = np.array([30.0, 50.0])
        relerr = total_sigma / total_nominal
        lite_lp = float(np.sum(-np.log(relerr * np.sqrt(2 * np.pi))))

        expected = poisson_lp + lite_lp
        assert val == pytest.approx(expected, rel=1e-6)

    def test_bblite_log_constraint_matches_log_of_constraint(self):
        """``_make_barlow_beeston_lite_log_constraint`` (#243 Layer 2) must equal
        ``log(_make_barlow_beeston_lite_constraint(...))`` at ordinary gamma
        values, for both the Gauss and Poisson BB-lite constraint types."""
        for constraint_type in ("Gauss", "Poisson"):
            channel_dict = self._lite_staterror_channel_dict()
            for sample in channel_dict["samples"]:
                for modifier in sample["modifiers"]:
                    modifier["constraint"] = constraint_type
            dist = HistFactoryDistChannel(**channel_dict)

            context = Context(
                {
                    "gamma_bin0": pt.constant(1.02, dtype="float64"),
                    "gamma_bin1": pt.constant(0.98, dtype="float64"),
                }
            )
            prob = float(dist._make_barlow_beeston_lite_constraint(context).eval())  # pylint: disable=protected-access
            log_prob = float(
                dist._make_barlow_beeston_lite_log_constraint(context).eval()  # pylint: disable=protected-access
            )
            assert log_prob == pytest.approx(math.log(prob), rel=1e-8), constraint_type

    def test_model_expression_matches_dist_expression(self):
        """The model's per-channel expression matches ``dist.expression()`` directly.

        ``dist.expression()`` (``Distribution._expression``) computes
        ``likelihood() * extended_likelihood()``, which already includes the
        BB-lite constraint factor. The model-assembled per-channel expression
        (Poisson term times constraint product) must equal this exactly,
        confirming the model's constraint set -- including the BB-lite factor
        -- is identical to the distribution's own.
        """
        ws = self._lite_staterror_workspace()
        likelihood = next(iter(ws.likelihoods))
        model = ws.model(likelihood, progress=False)

        dist = next(iter(ws.distributions))
        context = Context(
            {
                "gamma_bin0": pt.constant(1.0),
                "gamma_bin1": pt.constant(1.0),
                "SR_observed": pt.constant(np.array([30.0, 50.0])),
            }
        )
        expected = float(dist.expression(context).eval())

        model_expr = model.distributions["SR"]
        inputs = {
            v.name: v
            for v in pytensor.graph.traversal.explicit_graph_inputs([model_expr])
            if v.name
        }
        fn = pytensor.function(list(inputs.values()), model_expr)
        actual = float(fn(**model.data, **model.nominal_params))

        assert actual == pytest.approx(expected, rel=1e-6)

    def test_full_mode_staterror_still_builds_via_model(self):
        """BB-full mode (explicit ``barlow_beeston_method="full"``) is unaffected
        by the BB-lite channel-level addition in ``_build_distribution_node``.

        The ordinary ``constraint_specs()`` loop already includes the
        ``StatErrorModifier``'s own per-bin constraint in full mode (its
        ``data`` is required and present), so the new
        ``if dist.barlow_beeston_method == "lite":`` block must be skipped
        entirely here.
        """
        channel = {
            "name": "SR",
            "axes": [{"name": "x_SR", "min": 0.0, "max": 10.0, "nbins": 1}],
            "samples": [
                {
                    "name": "signal",
                    "data": {"contents": [10.0], "errors": [1.0]},
                    "modifiers": [
                        {
                            "name": "stat_error",
                            "type": "staterror",
                            "parameters": ["staterror_bin0"],
                            "constraint": "Gauss",
                            "data": {"uncertainties": [0.1]},
                        }
                    ],
                }
            ],
            "barlow_beeston_method": "full",
        }
        dist = HistFactoryDistChannel(**channel)
        binned = BinnedData(
            name="SR_data",
            axes=[{"name": "x_SR", "min": 0.0, "max": 10.0, "nbins": 1}],
            contents=[10.0],
        )
        ws = Workspace(
            metadata=Metadata(hs3_version="0.3.0"),
            distributions=Distributions([dist]),
            data=Data([binned]),
            likelihoods=Likelihoods(
                [Likelihood(name="L", distributions=[dist], data=[binned])]
            ),
            domains=Domains([ProductDomain(name="default")]),
            parameter_points=ParameterPoints(
                [
                    ParameterSet(
                        name="default",
                        parameters=[ParameterPoint(name="staterror_bin0", value=1.0)],
                    )
                ]
            ),
        )
        likelihood = next(iter(ws.likelihoods))
        model = ws.model(likelihood, progress=False)

        lp = model.log_prob
        inputs = {
            v.name: v
            for v in pytensor.graph.traversal.explicit_graph_inputs([lp])
            if v.name
        }
        fn = pytensor.function(list(inputs.values()), lp)
        val = float(fn(**model.data, **model.nominal_params).item())
        assert math.isfinite(val)

    def test_inactive_bblite_channel_constraint_excluded_from_log_prob(self):
        """A BB-lite channel excluded from the active likelihood must not
        contribute its channel-level constraint (or its Poisson term) to
        ``log_prob``.

        Mirrors
        ``TestConstraintDeduplication.test_inactive_channel_constraints_excluded_from_log_prob``
        for the channel-level BB-lite constraint: ``in_likelihood`` guards
        both the ordinary per-modifier constraints and the BB-lite addition.
        """
        sr_channel = HistFactoryDistChannel(**self._lite_staterror_channel_dict())
        cr_channel = HistFactoryDistChannel(**_make_channel("CR", [5.0], []))
        sr_data = BinnedData(
            name="SR_data",
            axes=[{"name": "x_SR", "min": 0.0, "max": 10.0, "nbins": 2}],
            contents=[30.0, 50.0],
        )
        cr_data = BinnedData(
            name="CR_data",
            axes=[{"name": "x_CR", "min": 0.0, "max": 10.0, "nbins": 1}],
            contents=[5.0],
        )
        ws = Workspace(
            metadata=Metadata(hs3_version="0.3.0"),
            distributions=Distributions([sr_channel, cr_channel]),
            data=Data([sr_data, cr_data]),
            likelihoods=Likelihoods(
                [
                    Likelihood(
                        name="L",
                        distributions=[cr_channel],
                        data=[cr_data],
                    )
                ]
            ),
            domains=Domains([ProductDomain(name="default")]),
            parameter_points=ParameterPoints(
                [
                    ParameterSet(
                        name="default",
                        parameters=[
                            ParameterPoint(name="gamma_bin0", value=1.0),
                            ParameterPoint(name="gamma_bin1", value=1.0),
                        ],
                    )
                ]
            ),
        )
        likelihood = next(iter(ws.likelihoods))
        model = ws.model(likelihood, progress=False)  # SR excluded from L

        lp = model.log_prob
        inputs = {
            v.name: v
            for v in pytensor.graph.traversal.explicit_graph_inputs([lp])
            if v.name
        }
        fn = pytensor.function(list(inputs.values()), lp)
        # Filter to only the free inputs -- gamma_bin0/gamma_bin1 (SR-only)
        # are not free inputs of log_prob once SR's constraint is excluded.
        param_vals = {k: v for k, v in model.nominal_params.items() if k in inputs}
        val = float(fn(**{**model.data, **param_vals}).item())

        # Only CR's Poisson term should appear.
        expected = 5.0 * math.log(5.0) - 5.0 - math.lgamma(6.0)
        assert val == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# Integration tests: HFDC subgraphs are each built exactly once
# ---------------------------------------------------------------------------


def _single_channel_normsys_workspace() -> Workspace:
    """Single-channel HFDC workspace with one normsys constraint on ``lumi``.

    Shared setup for the HFDC subgraph-build-count regression tests and the
    per-instance cache context-guard test below, all of which only need this
    minimal single-parameter-constraint channel.
    """
    return _simple_workspace(
        channels=[
            _make_channel(
                "SR",
                [10.0],
                [
                    {
                        "name": "lumi",
                        "type": "normsys",
                        "parameter": "lumi",
                        "constraint": "Gauss",
                        "data": {"hi": 1.05, "lo": 0.95},
                    }
                ],
            )
        ],
        params=[{"name": "lumi", "value": 0.0}],
    )


class TestHFDCSubgraphBuildCounts:
    """Model construction must build each HFDC subgraph exactly once.

    ``_build_distribution_node`` previously called ``dist.likelihood(context)``,
    ``dist.log_likelihood(context)``, and ``dist.log_expression(context)`` for
    the same context; the last call internally re-ran ``log_likelihood`` (which
    recomputes expected_rates) and ``log_extended_likelihood`` (which reruns
    ``make_constraint`` for every constraint spec), duplicating work already
    done for the probability-space expression and the constraint loop below it.
    """

    def test_compute_expected_rates_called_once_per_channel(self, monkeypatch):
        """``_compute_expected_rates`` must build the expected-rates subgraph
        exactly once per channel during model construction."""
        calls: list[str] = []
        original = HistFactoryDistChannel._compute_expected_rates

        def counted(
            self: HistFactoryDistChannel, context: Context, total_bins: int
        ) -> object:
            calls.append(self.name)
            return original(self, context, total_bins)

        monkeypatch.setattr(HistFactoryDistChannel, "_compute_expected_rates", counted)

        ws = _single_channel_normsys_workspace()
        likelihood = next(iter(ws.likelihoods))
        ws.model(likelihood, progress=False)

        assert calls == ["SR"], (
            f"expected exactly one _compute_expected_rates call per channel, got {calls}"
        )

    def test_make_constraint_called_once_per_spec(self, monkeypatch):
        """``make_constraint`` must build each constraint factor exactly once
        during model construction, regardless of how many places (probability-space
        expression, log-space expression, joint log_prob) reuse the result."""
        calls: list[str] = []
        original = SingleParamConstraint.make_constraint

        def counted(
            self: SingleParamConstraint, context: Context, sample_data: object
        ) -> object:
            calls.append(self.name)
            return original(self, context, sample_data)  # type: ignore[arg-type]

        monkeypatch.setattr(SingleParamConstraint, "make_constraint", counted)

        ws = _single_channel_normsys_workspace()
        likelihood = next(iter(ws.likelihoods))
        ws.model(likelihood, progress=False)

        assert calls == ["lumi"], (
            f"expected exactly one make_constraint call per constraint spec, got {calls}"
        )

    def test_log_constraint_called_once_per_spec(self, monkeypatch):
        """``log_constraint`` (#243 Layer 2) must build each log-space constraint
        factor exactly once during model construction, alongside the single
        ``make_constraint`` call asserted above."""
        calls: list[str] = []
        original = SingleParamConstraint.log_constraint

        def counted(
            self: SingleParamConstraint, context: Context, sample_data: object
        ) -> object:
            calls.append(self.name)
            return original(self, context, sample_data)  # type: ignore[arg-type]

        monkeypatch.setattr(SingleParamConstraint, "log_constraint", counted)

        ws = _single_channel_normsys_workspace()
        likelihood = next(iter(ws.likelihoods))
        ws.model(likelihood, progress=False)

        assert calls == ["lumi"], (
            f"expected exactly one log_constraint call per constraint spec, got {calls}"
        )

    def test_logpdf_matches_log_pdf_for_hfdc_with_constraint(self):
        """logpdf(channel) must equal log(pdf(channel)) for an HFDC channel with
        a constraint modifier, confirming log_distributions is assembled from
        the same pieces as the probability-space expression."""
        ws = _single_channel_normsys_workspace()
        likelihood = next(iter(ws.likelihoods))
        model = ws.model(likelihood, progress=False)

        pdf_val = float(
            np.asarray(model.pdf_unsafe("SR", lumi=0.1, SR_observed=np.array([10.0])))
        )
        logpdf_val = float(
            np.asarray(
                model.logpdf_unsafe("SR", lumi=0.1, SR_observed=np.array([10.0]))
            )
        )
        assert logpdf_val == pytest.approx(math.log(pdf_val), rel=1e-10)


# ---------------------------------------------------------------------------
# Regression tests: per-instance HFDC caches must be tied to their context
# ---------------------------------------------------------------------------


class TestHFDCContextGuardedCache:
    """``_cached_expected_rates``/``_cached_bin_log_probs`` must only be reused
    for the exact :class:`Context` object that populated them.

    ``Workspace.model()`` passes the same ``HistFactoryDistChannel`` instances
    into every ``Model`` built from that workspace (distributions are not
    copied), and each ``Model`` build supplies its own fresh ``Context`` with
    new parameter tensors.  ``likelihood()``/``log_likelihood()`` happen to be
    called back-to-back for the same context during a single ``Model`` build,
    which masks the problem when driven only through ``Workspace.model()`` +
    ``Model.logpdf()`` -- both builds still evaluate correctly because each
    build unconditionally repopulates the cache with its own context right
    before reading it back.  But ``log_likelihood()`` (and ``log_expression()``,
    which calls it) are public methods that can be invoked directly for any
    context, and a shared instance's cache does not know which context it was
    last built for -- so a direct call for a context that never populated the
    cache must not silently reuse a previous build's stale graph.
    """

    def test_two_model_builds_still_evaluate_correctly(self):
        """Sanity check: both Model builds, and re-evaluating the first after
        the second is built, must all match the analytic Poisson+Gaussian
        expectation (the cache-overwrite direction already works)."""
        ws = _single_channel_normsys_workspace()
        likelihood = next(iter(ws.likelihoods))

        poisson_lp = 10.0 * math.log(10.0) - 10.0 - math.lgamma(11.0)
        gauss_lp = -0.5 * math.log(2 * math.pi)
        expected = poisson_lp + gauss_lp

        model1 = ws.model(likelihood, progress=False)
        val1 = float(
            np.asarray(
                model1.logpdf_unsafe("SR", lumi=0.0, SR_observed=np.array([10.0]))
            )
        )
        assert val1 == pytest.approx(expected, rel=1e-10)

        model2 = ws.model(likelihood, progress=False)
        val2 = float(
            np.asarray(
                model2.logpdf_unsafe("SR", lumi=0.0, SR_observed=np.array([10.0]))
            )
        )
        assert val2 == pytest.approx(expected, rel=1e-10)

        # model1 must still work correctly after model2's build has mutated
        # the shared distribution instance's cache.
        val1_again = float(
            np.asarray(
                model1.logpdf_unsafe("SR", lumi=0.0, SR_observed=np.array([10.0]))
            )
        )
        assert val1_again == pytest.approx(expected, rel=1e-10)

    def test_log_likelihood_rebuilds_for_a_context_that_never_populated_the_cache(self):
        """Calling ``log_likelihood`` directly for a fresh, unpaired context must
        build a graph over *that* context's tensors, not silently reuse a
        previous build's cached graph.

        Before the context-identity guard, ``log_likelihood`` reused
        ``_cached_bin_log_probs`` whenever it was non-``None``, regardless of
        which context built it. Two prior ``Model`` builds leave the shared
        ``HistFactoryDistChannel`` instance's cache populated with the second
        build's tensors; calling ``log_likelihood`` for a brand new context
        that was never passed to ``likelihood()`` must not return a graph
        wired to those old tensors.
        """
        ws = _single_channel_normsys_workspace()
        likelihood = next(iter(ws.likelihoods))
        # Two builds share the same HistFactoryDistChannel instance (Workspace.model
        # does not copy distributions), leaving its per-instance cache primed by
        # whichever build ran last.
        ws.model(likelihood, progress=False)
        ws.model(likelihood, progress=False)

        dist = next(iter(ws.distributions))

        fresh_lumi = pt.dscalar("lumi")
        fresh_observed = pt.dvector("SR_observed")
        fresh_context = Context({"lumi": fresh_lumi, "SR_observed": fresh_observed})

        expr = dist.log_likelihood(fresh_context)
        inputs = set(pytensor.graph.traversal.explicit_graph_inputs([expr]))

        # Before the fix, the returned graph references stale leaves left over
        # from the second Model build, so neither fresh tensor is even present
        # as an input.
        assert fresh_lumi in inputs
        assert fresh_observed in inputs

    def test_log_likelihood_reuses_cached_expected_rates_after_bin_log_probs_failure(
        self, monkeypatch
    ):
        """A fresh instance's first ``log_likelihood`` call can populate
        ``_cached_expected_rates``/``_cached_context`` and then still fail:
        ``_compute_expected_rates`` only needs ``lumi``, so it succeeds and
        populates the cache, but ``_bin_log_probs`` then raises because
        ``SR_observed`` is missing from the context, so ``_cached_bin_log_probs``
        is never set. A second call for the *same* context object -- once the
        missing observed data is added in place via ``Context.add_parameter``
        -- fails the outer cache check (``_cached_bin_log_probs`` is still
        ``None``) but passes the inner one (``_cached_context is context`` and
        ``_cached_expected_rates is not None``), so the cached expected rates
        are reused instead of recomputed.
        """
        calls: list[str] = []
        original = HistFactoryDistChannel._compute_expected_rates

        def counted(
            self: HistFactoryDistChannel, context: Context, total_bins: int
        ) -> object:
            calls.append(self.name)
            return original(self, context, total_bins)

        monkeypatch.setattr(HistFactoryDistChannel, "_compute_expected_rates", counted)

        ws = _single_channel_normsys_workspace()
        dist = next(iter(ws.distributions))

        lumi = pt.dscalar("lumi")
        context = Context({"lumi": lumi})

        # First call: _compute_expected_rates succeeds and populates the cache,
        # but _bin_log_probs raises because "SR_observed" is missing.
        with pytest.raises(KeyError, match="SR_observed"):
            dist.log_likelihood(context)
        assert calls == ["SR"]

        # Add the missing observed data to the SAME context object in place.
        observed = pt.dvector("SR_observed")
        context.add_parameter("SR_observed", observed)

        expr = dist.log_likelihood(context)
        # _compute_expected_rates was not called again -- the cached expected
        # rates from the first call were reused.
        assert calls == ["SR"]

        fn = pytensor.function([lumi, observed], expr)
        val = float(fn(0.0, np.array([10.0])))
        expected = 10.0 * math.log(10.0) - 10.0 - math.lgamma(11.0)
        assert val == pytest.approx(expected, rel=1e-10)


# ---------------------------------------------------------------------------
# Integration tests: constraint deduplication across channels
# ---------------------------------------------------------------------------


class TestConstraintDeduplication:
    """Shared nuisance parameters must appear exactly once in the joint NLL."""

    def test_two_channels_shared_normsys_same_as_single_constraint(self):
        """Two channels sharing a normsys parameter produce the same NLL as
        manually computing: Poisson_SR + Poisson_CR + 1 * constraint(lumi).

        Before the fix, the NLL contained 2 * constraint(lumi).
        """
        normsys_mod = {
            "name": "lumi",
            "type": "normsys",
            "parameter": "lumi",
            "constraint": "Gauss",
            "data": {"hi": 1.05, "lo": 0.95},
        }
        ws = _simple_workspace(
            channels=[
                _make_channel("SR", [10.0], [normsys_mod]),
                _make_channel("CR", [50.0], [normsys_mod]),
            ],
            params=[{"name": "lumi", "value": 0.0}],
        )
        likelihood = next(iter(ws.likelihoods))
        model = ws.model(likelihood, progress=False)
        lp = model.log_prob
        inputs = {
            v.name: v
            for v in pytensor.graph.traversal.explicit_graph_inputs([lp])
            if v.name
        }
        fn = pytensor.function(list(inputs.values()), lp)
        val = float(fn(**model.data, **model.nominal_params).item())

        # Expected: sum of Poisson log-probs + 1 Gaussian constraint at alpha=0
        poisson_sr = 10.0 * math.log(10.0) - 10.0 - math.lgamma(11.0)
        poisson_cr = 50.0 * math.log(50.0) - 50.0 - math.lgamma(51.0)
        gauss_lp = -0.5 * math.log(2 * math.pi)
        expected = poisson_sr + poisson_cr + gauss_lp
        assert abs(val - expected) < 1e-6, f"got {val}, expected {expected}"

    def test_shared_normsys_within_one_channel_single_constraint(self):
        """Two samples in one channel sharing a normsys parameter yield one constraint.

        Exercises the per-channel seen-set dedup in _build_distribution_node:
        the second sample's spec carries an already-seen dedup_key, so the
        channel product must contain the Gaussian constraint exactly once.
        """
        normsys_mod = {
            "name": "lumi",
            "type": "normsys",
            "parameter": "lumi",
            "constraint": "Gauss",
            "data": {"hi": 1.05, "lo": 0.95},
        }
        channel = {
            "name": "SR",
            "axes": [{"name": "x_SR", "min": 0.0, "max": 10.0, "nbins": 1}],
            "samples": [
                {
                    "name": "signal",
                    "data": {"contents": [10.0], "errors": [1.0]},
                    "modifiers": [normsys_mod],
                },
                {
                    "name": "background",
                    "data": {"contents": [5.0], "errors": [1.0]},
                    "modifiers": [normsys_mod],
                },
            ],
        }
        ws = _simple_workspace(
            channels=[channel],
            params=[{"name": "lumi", "value": 0.0}],
        )
        likelihood = next(iter(ws.likelihoods))
        model = ws.model(likelihood, progress=False)
        lp = model.log_prob
        inputs = {
            v.name: v
            for v in pytensor.graph.traversal.explicit_graph_inputs([lp])
            if v.name
        }
        fn = pytensor.function(list(inputs.values()), lp)
        val = float(fn(**model.data, **model.nominal_params).item())

        # Observed is sample 0's contents (10); expected at lumi=0 is 10 + 5.
        poisson = 10.0 * math.log(15.0) - 15.0 - math.lgamma(11.0)
        gauss_lp = -0.5 * math.log(2 * math.pi)
        expected = poisson + gauss_lp
        assert abs(val - expected) < 1e-6, f"got {val}, expected {expected}"

    def test_two_channels_independent_normsys_both_constraints_present(self):
        """Two channels with different normsys parameters must each contribute a constraint."""
        ws = _simple_workspace(
            channels=[
                _make_channel(
                    "SR",
                    [10.0],
                    [
                        {
                            "name": "alpha_sr",
                            "type": "normsys",
                            "parameter": "alpha_sr",
                            "constraint": "Gauss",
                            "data": {"hi": 1.1, "lo": 0.9},
                        }
                    ],
                ),
                _make_channel(
                    "CR",
                    [50.0],
                    [
                        {
                            "name": "alpha_cr",
                            "type": "normsys",
                            "parameter": "alpha_cr",
                            "constraint": "Gauss",
                            "data": {"hi": 1.2, "lo": 0.8},
                        }
                    ],
                ),
            ],
            params=[
                {"name": "alpha_sr", "value": 0.0},
                {"name": "alpha_cr", "value": 0.0},
            ],
        )
        likelihood = next(iter(ws.likelihoods))
        model = ws.model(likelihood, progress=False)
        lp = model.log_prob
        inputs = {
            v.name: v
            for v in pytensor.graph.traversal.explicit_graph_inputs([lp])
            if v.name
        }
        fn = pytensor.function(list(inputs.values()), lp)
        val = float(fn(**model.data, **model.nominal_params).item())

        poisson_sr = 10.0 * math.log(10.0) - 10.0 - math.lgamma(11.0)
        poisson_cr = 50.0 * math.log(50.0) - 50.0 - math.lgamma(51.0)
        gauss_lp = -0.5 * math.log(2 * math.pi)  # each at alpha=0
        expected = poisson_sr + poisson_cr + 2 * gauss_lp  # two independent constraints
        assert abs(val - expected) < 1e-6, f"got {val}, expected {expected}"

    def test_inactive_channel_constraints_excluded_from_log_prob(self):
        """Constraints from HFDC channels not in the active likelihood must not
        appear in log_prob even though all workspace distributions are built.

        Regression: _build_distribution_node previously collected constraints
        from ALL workspace distributions regardless of the active likelihood.
        """
        normsys_sr = {
            "name": "alpha_sr",
            "type": "normsys",
            "parameter": "alpha_sr",
            "constraint": "Gauss",
            "data": {"hi": 1.1, "lo": 0.9},
        }
        normsys_cr = {
            "name": "alpha_cr",
            "type": "normsys",
            "parameter": "alpha_cr",
            "constraint": "Gauss",
            "data": {"hi": 1.2, "lo": 0.8},
        }
        sr_channel = HistFactoryDistChannel(**_make_channel("SR", [10.0], [normsys_sr]))
        cr_channel = HistFactoryDistChannel(**_make_channel("CR", [50.0], [normsys_cr]))
        sr_data = BinnedData(
            name="SR_data",
            axes=[{"name": "x_SR", "min": 0.0, "max": 10.0, "nbins": 1}],
            contents=[10.0],
        )
        ws = Workspace(
            metadata=Metadata(hs3_version="0.3.0"),
            distributions=Distributions([sr_channel, cr_channel]),
            data=Data([sr_data]),
            likelihoods=Likelihoods(
                [
                    Likelihood(
                        name="L",
                        distributions=[sr_channel],
                        data=[sr_data],
                    )
                ]
            ),
            domains=Domains([ProductDomain(name="default")]),
            parameter_points=ParameterPoints(
                [
                    ParameterSet(
                        name="default",
                        parameters=[
                            ParameterPoint(name="alpha_sr", value=0.0),
                            ParameterPoint(name="alpha_cr", value=0.0),
                        ],
                    )
                ]
            ),
        )
        likelihood = next(iter(ws.likelihoods))
        model = ws.model(likelihood, progress=False)
        lp = model.log_prob
        inputs = {
            v.name: v
            for v in pytensor.graph.traversal.explicit_graph_inputs([lp])
            if v.name
        }
        fn = pytensor.function(list(inputs.values()), lp)
        # Filter to only the free inputs (after fix, alpha_cr is not a free input).
        param_vals = {k: v for k, v in model.nominal_params.items() if k in inputs}
        val = float(fn(**{**model.data, **param_vals}).item())

        # Expected: Poisson(SR only) + 1 Gaussian(alpha_sr=0)
        # With the bug, Gaussian(alpha_cr) from inactive CR also appears.
        poisson_sr = 10.0 * math.log(10.0) - 10.0 - math.lgamma(11.0)
        gauss_lp = -0.5 * math.log(2 * math.pi)
        expected = poisson_sr + gauss_lp
        assert abs(val - expected) < 1e-6, f"got {val}, expected {expected}"

    def test_normfactor_has_no_constraint(self):
        """normfactor adds no constraint term; log_prob equals Poisson only."""
        ws = _simple_workspace(
            channels=[
                _make_channel(
                    "SR",
                    [10.0],
                    [{"name": "mu", "type": "normfactor", "parameter": "mu"}],
                )
            ],
            params=[{"name": "mu", "value": 1.0}],
        )
        likelihood = next(iter(ws.likelihoods))
        model = ws.model(likelihood, progress=False)
        lp = model.log_prob
        inputs = {
            v.name: v
            for v in pytensor.graph.traversal.explicit_graph_inputs([lp])
            if v.name
        }
        fn = pytensor.function(list(inputs.values()), lp)
        val = float(fn(**model.data, **model.nominal_params).item())
        # At mu=1: rates=10, obs=10
        expected = 10.0 * math.log(10.0) - 10.0 - math.lgamma(11.0)
        assert abs(val - expected) < 1e-6, f"got {val}, expected {expected}"


# ---------------------------------------------------------------------------
# Validator tests: constraint type consistency and per-channel uniqueness
# ---------------------------------------------------------------------------


class TestConstraintValidator:
    """Workspace validation catches conflicting constraint configurations."""

    def test_conflicting_constraint_types_raises(self):
        """Same nuisance parameter with different constraint types must raise ValueError.

        'lumi' as normsys(Gauss) in SR and normsys(LogNormal) in CR is invalid.
        """
        sr_channel = _make_channel(
            "SR",
            [10.0],
            [
                {
                    "name": "lumi",
                    "type": "normsys",
                    "parameter": "lumi",
                    "constraint": "Gauss",
                    "data": {"hi": 1.05, "lo": 0.95},
                }
            ],
        )
        cr_channel = _make_channel(
            "CR",
            [50.0],
            [
                {
                    "name": "lumi",
                    "type": "normsys",
                    "parameter": "lumi",
                    "constraint": "LogNormal",
                    "data": {"hi": 1.05, "lo": 0.95},
                }
            ],
        )
        with pytest.raises(WorkspaceValidationError, match="conflicting constraint"):
            _simple_workspace(
                channels=[sr_channel, cr_channel],
                params=[{"name": "lumi", "value": 0.0}],
            )

    def test_shapesys_shared_across_channels_raises(self):
        """ShapeSys parameter names shared across channels must raise ValueError.

        shapesys parameters are per-channel (bin yields differ per region),
        so correlation across channels is physically nonsensical.
        """
        shared_shapesys = {
            "name": "stat",
            "type": "shapesys",
            "parameters": ["gamma_0"],
            "constraint": "Poisson",
            "data": {"vals": [2.0]},
        }
        sr = _make_channel("SR", [10.0], [shared_shapesys])
        cr = _make_channel("CR", [50.0], [shared_shapesys])
        with pytest.raises(WorkspaceValidationError, match="shapesys"):
            _simple_workspace(channels=[sr, cr], params=[])

    def test_staterror_shared_across_channels_raises(self):
        """StatError parameter names shared across channels must raise ValueError."""
        shared_staterror = {
            "name": "stat",
            "type": "staterror",
            "parameters": ["gamma_0"],
        }
        sr = _make_channel("SR", [10.0], [shared_staterror])
        cr = _make_channel("CR", [50.0], [shared_staterror])
        with pytest.raises(WorkspaceValidationError, match="staterror"):
            _simple_workspace(channels=[sr, cr], params=[])

    def test_same_normsys_type_across_channels_is_valid(self):
        """Same parameter with the same constraint type across channels is fine."""
        normsys = {
            "name": "lumi",
            "type": "normsys",
            "parameter": "lumi",
            "constraint": "Gauss",
            "data": {"hi": 1.05, "lo": 0.95},
        }
        sr = _make_channel("SR", [10.0], [normsys])
        cr = _make_channel("CR", [50.0], [normsys])
        # Should not raise
        ws = _simple_workspace(
            channels=[sr, cr], params=[{"name": "lumi", "value": 0.0}]
        )
        assert ws is not None
