"""
Unit tests for Workspace with None collections.

Tests FK resolution behavior when optional collection fields are explicitly None,
covering branches in core.py model_post_init.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from pyhs3 import Workspace
from pyhs3.analyses import Analyses, Analysis
from pyhs3.exceptions import WorkspaceValidationError
from pyhs3.likelihoods import Likelihood, Likelihoods
from pyhs3.metadata import Metadata, PackageInfo


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


class TestRootVersionHint:
    """Tests for ROOT version validation functionality."""

    def test_packageinfo_validator_old_version(self):
        """Test that old ROOT version (6.34.04) raises ValidationError."""
        with pytest.raises(ValidationError, match=r"ROOT version 6.34.04 is older"):
            PackageInfo(name="ROOT", version="6.34.04")

    def test_packageinfo_validator_new_version(self):
        """Test that new ROOT version (6.38.0) succeeds."""
        pkg = PackageInfo(name="ROOT", version="6.38.0")
        assert pkg.name == "ROOT"
        assert pkg.version == "6.38.0"

    def test_packageinfo_validator_exactly_min_version(self):
        """Test that exactly version 6.38 succeeds."""
        pkg = PackageInfo(name="ROOT", version="6.38")
        assert pkg.name == "ROOT"
        assert pkg.version == "6.38"

    def test_packageinfo_validator_non_root_package(self):
        """Test that non-ROOT packages succeed regardless of version."""
        pkg = PackageInfo(name="pyhf", version="0.7.6")
        assert pkg.name == "pyhf"
        # Even old version strings pass for non-ROOT packages
        pkg2 = PackageInfo(name="pyhf", version="0.1.0")
        assert pkg2.version == "0.1.0"

    def test_packageinfo_validator_invalid_version(self):
        """Test that invalid version string succeeds (can't compare)."""
        # Invalid versions can't be compared, so they pass
        pkg = PackageInfo(name="ROOT", version="invalid.version")
        assert pkg.version == "invalid.version"

    def test_metadata_propagates_root_validation_error(self):
        """Test that Metadata with old ROOT in packages raises ValidationError."""
        with pytest.raises(ValidationError, match=r"ROOT version 6.34.04 is older"):
            Metadata(
                hs3_version="1.0.0",
                packages=[PackageInfo(name="ROOT", version="6.34.04")],
            )

    def test_workspace_load_validation_error_with_old_root(self, tmp_path):
        """Test Workspace.load() with old ROOT shows ROOT version in error."""
        invalid_workspace = {
            "metadata": {
                "hs3_version": "1.0.0",
                "packages": [{"name": "ROOT", "version": "6.34.04"}],
            },
        }
        workspace_path = tmp_path / "workspace.json"
        workspace_path.write_text(json.dumps(invalid_workspace))

        with pytest.raises(WorkspaceValidationError) as exc_info:
            Workspace.load(workspace_path)

        error_msg = str(exc_info.value)
        assert "ROOT version 6.34.04 is older" in error_msg
        assert "6.38" in error_msg

    def test_workspace_load_validation_error_with_new_root(self, tmp_path):
        """Test Workspace.load() with new ROOT and other errors doesn't mention ROOT."""
        invalid_workspace = {
            "metadata": {
                "hs3_version": "1.0.0",
                "packages": [{"name": "ROOT", "version": "6.38.0"}],
            },
            "analyses": [{"invalid": "field"}],  # Invalid analysis
        }
        workspace_path = tmp_path / "workspace.json"
        workspace_path.write_text(json.dumps(invalid_workspace))

        with pytest.raises(WorkspaceValidationError) as exc_info:
            Workspace.load(workspace_path)

        error_msg = str(exc_info.value)
        # Error should be about the invalid analysis, not ROOT version
        assert "ROOT version" not in error_msg or "6.38.0" in error_msg
