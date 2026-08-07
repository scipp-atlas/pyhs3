"""
Unit tests for Workspace with None collections.

Tests FK resolution behavior when optional collection fields are explicitly None,
covering branches in core.py model_post_init.
"""

from __future__ import annotations

import pytest

from pyhs3 import Model, Workspace
from pyhs3.analyses import Analyses, Analysis
from pyhs3.data import BinnedData, Data
from pyhs3.distributions import Distributions
from pyhs3.distributions.mathematical import GenericDist
from pyhs3.exceptions import WorkspaceValidationError
from pyhs3.likelihoods import Likelihood, Likelihoods
from pyhs3.metadata import Metadata


class TestWorkspaceNoneCollections:
    """Tests for Workspace FK resolution with None collections."""

    def test_workspace_with_likelihoods_none(self):
        """Test Workspace with likelihoods=None (branch coverage for line 103)."""
        workspace = Workspace(
            metadata=Metadata(hs3_version="0.1.0"),
            likelihoods=None,
        )
        assert workspace.likelihoods is None

    def test_workspace_with_analyses_none(self):
        """Test Workspace with analyses=None (branch coverage for line 108)."""
        workspace = Workspace(
            metadata=Metadata(hs3_version="0.1.0"),
            analyses=None,
        )
        assert workspace.analyses is None

    def test_workspace_with_distributions_none(self):
        """Test Workspace with distributions=None (branch coverage for line 146)."""
        workspace = Workspace(
            metadata=Metadata(hs3_version="0.1.0"),
            distributions=None,
        )
        assert workspace.distributions is None

    def test_workspace_with_data_none(self):
        """Test Workspace with data=None (branch coverage for line 159)."""
        workspace = Workspace(
            metadata=Metadata(hs3_version="0.1.0"),
            data=None,
        )
        assert workspace.data is None

    def test_workspace_with_domains_none(self):
        """Test Workspace with domains=None (branch coverage for line 182)."""
        workspace = Workspace(
            metadata=Metadata(hs3_version="0.1.0"),
            domains=None,
        )
        assert workspace.domains is None


class TestWorkspaceFKResolutionWithNoneCollections:
    """Tests for FK resolution error branches when collections are None."""

    def test_likelihood_references_distributions_when_none(self):
        """Test error when likelihood references distributions but distributions=None."""
        with pytest.raises(
            WorkspaceValidationError, match="references unknown distributions"
        ):
            Workspace(
                metadata=Metadata(hs3_version="0.1.0"),
                likelihoods=Likelihoods(
                    [
                        Likelihood(
                            name="lk1",
                            distributions=["dist1"],
                            data=["obs1"],
                        )
                    ]
                ),
                distributions=None,  # Missing distributions collection
                data=None,
            )

    def test_likelihood_references_aux_distribution_nonexistent(self):
        """aux_distributions names not in model.distributions raise error."""
        with pytest.raises(
            WorkspaceValidationError, match="references unknown aux_distribution"
        ):
            Workspace.model_validate(
                {
                    "metadata": {"hs3_version": "0.2"},
                    "distributions": [
                        {
                            "name": "gauss1",
                            "type": "gaussian_dist",
                            "x": "x_obs",
                            "mean": "mean",
                            "sigma": 1.0,
                        },
                        {
                            "name": "constraint",
                            "type": "gaussian_dist",
                            "x": "alpha",
                            "mean": 0.0,
                            "sigma": 1.0,
                        },
                    ],
                    "domains": [
                        {
                            "name": "main",
                            "type": "product_domain",
                            "axes": [
                                {"name": "mean", "min": -10.0, "max": 10.0},
                                {"name": "alpha", "min": -5.0, "max": 5.0},
                            ],
                        }
                    ],
                    "data": [
                        {
                            "name": "data1",
                            "type": "unbinned",
                            "axes": [{"name": "x_obs", "min": -10.0, "max": 10.0}],
                            "entries": [[1.0], [2.0], [3.0]],
                        }
                    ],
                    "likelihoods": [
                        {
                            "name": "L",
                            "distributions": ["gauss1"],
                            "data": ["data1"],
                            # "nonexistent" is not a distribution in the workspace.
                            "aux_distributions": ["constraint", "nonexistent"],
                        }
                    ],
                    "analyses": [
                        {
                            "name": "A",
                            "likelihood": "L",
                            "domains": ["main"],
                            "init": "params",
                        }
                    ],
                    "parameter_points": [
                        {
                            "name": "params",
                            "parameters": [
                                {"name": "mean", "value": 0.0},
                                {"name": "alpha", "value": 0.0},
                            ],
                        }
                    ],
                }
            )

    def test_likelihood_references_data_when_none(self):
        """Test error when likelihood references data but data=None."""
        with pytest.raises(WorkspaceValidationError, match="references unknown data"):
            Workspace(
                metadata=Metadata(hs3_version="0.1.0"),
                likelihoods=Likelihoods(
                    [
                        Likelihood(
                            name="lk1",
                            distributions=["dist1"],
                            data=["obs1"],
                        )
                    ]
                ),
                distributions=None,
                data=None,  # Missing data collection
            )

    def test_analysis_references_likelihood_when_none(self):
        """Test error when analysis references likelihood but likelihoods=None."""
        with pytest.raises(
            WorkspaceValidationError, match="references unknown likelihood"
        ):
            Workspace(
                metadata=Metadata(hs3_version="0.1.0"),
                analyses=Analyses(
                    [
                        Analysis(
                            name="ana1",
                            likelihood="lk1",
                            domains=["domain1"],
                        )
                    ]
                ),
                likelihoods=None,  # Missing likelihoods collection
            )

    def test_analysis_references_domains_when_none(self):
        """Test error when analysis references domains but domains=None."""
        with pytest.raises(WorkspaceValidationError, match="references unknown domain"):
            Workspace(
                metadata=Metadata(hs3_version="0.1.0"),
                likelihoods=Likelihoods(
                    [
                        Likelihood(
                            name="lk1",
                            distributions=[],
                            data=[],
                            aux_distributions=["aux"],
                        )
                    ]
                ),
                analyses=Analyses(
                    [
                        Analysis(
                            name="ana1",
                            likelihood="lk1",
                            domains=["domain1"],
                        )
                    ]
                ),
                domains=None,  # Missing domains collection
                distributions=None,
                data=None,
            )


class TestPerLikelihoodObservables:
    """Observables are resolved per likelihood, never merged across likelihoods."""

    @staticmethod
    def _make_workspace(
        likelihood_bounds: dict[str, tuple[float, float]],
    ) -> Workspace:
        """Build a workspace with one likelihood per entry, each with an 'x' axis."""
        return Workspace(
            metadata=Metadata(hs3_version="0.3.0"),
            distributions=Distributions(
                [
                    GenericDist(name=f"dist_{name}", expression="exp(-x)")
                    for name in likelihood_bounds
                ]
            ),
            data=Data(
                [
                    BinnedData(
                        name=f"data_{name}",
                        axes=[{"name": "x", "min": lo, "max": hi, "nbins": 10}],
                        contents=[1.0] * 10,
                    )
                    for name, (lo, hi) in likelihood_bounds.items()
                ]
            ),
            likelihoods=Likelihoods(
                [
                    Likelihood(
                        name=name,
                        distributions=[f"dist_{name}"],
                        data=[f"data_{name}"],
                    )
                    for name in likelihood_bounds
                ]
            ),
        )

    @staticmethod
    def _bounds(model: Model, name: str) -> tuple[float, float]:
        """Evaluate a model observable's (min, max) tensor constants to floats."""
        lower, upper = model._observables[name]
        return (float(lower.data), float(upper.data))

    def test_different_bounds_across_likelihoods_are_valid(self):
        """Same axis name with different bounds across likelihoods is legitimate:
        the workspace loads and each likelihood's model gets its own bounds."""
        ws = self._make_workspace({"lk_a": (0.0, 10.0), "lk_b": (0.0, 20.0)})
        model_a = ws.model(ws.likelihoods["lk_a"], progress=False)
        model_b = ws.model(ws.likelihoods["lk_b"], progress=False)
        assert self._bounds(model_a, "x") == pytest.approx((0.0, 10.0))
        assert self._bounds(model_b, "x") == pytest.approx((0.0, 20.0))

    def test_legacy_model_single_likelihood_uses_its_observables(self):
        """Legacy ws.model(0) on a single-likelihood workspace resolves that
        likelihood's observables."""
        ws = self._make_workspace({"lk_a": (0.0, 10.0)})
        assert ws._compute_observables() == {"x": (0.0, 10.0)}
        model = ws.model(0, progress=False)
        assert self._bounds(model, "x") == pytest.approx((0.0, 10.0))

    def test_legacy_model_agreeing_likelihoods_share_observables(self):
        """Legacy ws.model(0) still works when every likelihood implies the same
        observables (e.g. asimov and observed data over identical axes)."""
        ws = self._make_workspace({"lk_a": (0.0, 10.0), "lk_b": (0.0, 10.0)})
        assert ws._compute_observables() == {"x": (0.0, 10.0)}
        model = ws.model(0, progress=False)
        assert self._bounds(model, "x") == pytest.approx((0.0, 10.0))

    def test_legacy_model_disagreeing_likelihoods_raises(self):
        """Legacy ws.model(0) has no principled observable choice when the
        likelihoods disagree — the user must select a likelihood explicitly."""
        ws = self._make_workspace({"lk_a": (0.0, 10.0), "lk_b": (0.0, 20.0)})
        with pytest.raises(ValueError, match="Cannot determine observables"):
            ws.model(0, progress=False)


class TestWorkspaceRepr:
    """Tests for Workspace.__repr__() method."""

    def test_workspace_repr(self):
        """Test Workspace.__repr__() returns expected format."""
        workspace = Workspace(
            metadata=Metadata(hs3_version="0.1.0"),
        )
        repr_str = repr(workspace)
        # Verify the repr contains key information
        assert "Workspace" in repr_str
        # The repr should be a useful string representation
        assert isinstance(repr_str, str)
        assert len(repr_str) > 0
