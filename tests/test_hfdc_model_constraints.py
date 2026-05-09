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

import numpy as np
import pytensor
import pytest

from pyhs3.data import BinnedData, Data
from pyhs3.distributions import Distributions, HistFactoryDistChannel
from pyhs3.distributions.histfactory.modifiers import (
    HasConstraint,
    ParameterModifier,
    ParametersModifier,
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
            "data": {"uncertainties": [1.0]},
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
