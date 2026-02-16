"""
Unit tests for Workspace with None collections.

Tests FK resolution behavior when optional collection fields are explicitly None,
covering branches in core.py model_post_init.
"""

from __future__ import annotations

from pyhs3 import Workspace
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
