"""
Unit tests for Workspace with None collections.

Tests FK resolution behavior when optional collection fields are explicitly None,
covering branches in core.py model_post_init.
"""

from __future__ import annotations

import pytest

from pyhs3 import Workspace
from pyhs3.analyses import Analyses, Analysis
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
