"""
Unit tests for analysis implementations.

Tests for Analysis, Analyses classes including validation
of analysis specifications and collection management.
"""

from __future__ import annotations

import pytest

from pyhs3 import Workspace
from pyhs3.analyses import Analyses, Analysis
from pyhs3.domains import Domain
from pyhs3.likelihoods import Likelihood


class TestAnalysis:
    """Tests for the Analysis class."""

    def test_analysis_creation_minimal(self):
        """Test Analysis creation with minimal required fields."""
        analysis = Analysis(
            name="test_analysis", likelihood="test_likelihood", domains=["test_domain"]
        )
        assert analysis.name == "test_analysis"
        assert analysis.likelihood == "test_likelihood"
        assert analysis.domains == ["test_domain"]
        assert analysis.parameters_of_interest is None
        assert analysis.init is None
        assert analysis.prior is None

    def test_analysis_creation_complete(self):
        """Test Analysis creation with all fields."""
        analysis = Analysis(
            name="complete_analysis",
            likelihood="test_likelihood",
            domains=["test_domain1", "test_domain2"],
            parameters_of_interest=["param1", "param2"],
            init="initial_values",
            prior="prior_distribution",
        )
        assert analysis.name == "complete_analysis"
        assert analysis.likelihood == "test_likelihood"
        assert analysis.domains == ["test_domain1", "test_domain2"]
        assert analysis.parameters_of_interest == ["param1", "param2"]
        assert analysis.init == "initial_values"
        assert analysis.prior == "prior_distribution"

    def test_analysis_creation_with_parameters_of_interest_only(self):
        """Test Analysis creation with parameters of interest but no other optionals."""
        analysis = Analysis(
            name="poi_analysis",
            likelihood="test_likelihood",
            domains=["test_domain"],
            parameters_of_interest=["mu", "sigma"],
        )
        assert analysis.name == "poi_analysis"
        assert analysis.likelihood == "test_likelihood"
        assert analysis.domains == ["test_domain"]
        assert analysis.parameters_of_interest == ["mu", "sigma"]
        assert analysis.init is None
        assert analysis.prior is None

    def test_analysis_creation_with_init_only(self):
        """Test Analysis creation with init but no other optionals."""
        analysis = Analysis(
            name="init_analysis",
            likelihood="test_likelihood",
            domains=["test_domain"],
            init="starting_values",
        )
        assert analysis.name == "init_analysis"
        assert analysis.likelihood == "test_likelihood"
        assert analysis.domains == ["test_domain"]
        assert analysis.parameters_of_interest is None
        assert analysis.init == "starting_values"
        assert analysis.prior is None

    def test_analysis_creation_with_prior_only(self):
        """Test Analysis creation with prior but no other optionals."""
        analysis = Analysis(
            name="prior_analysis",
            likelihood="test_likelihood",
            domains=["test_domain"],
            prior="prior_dist",
        )
        assert analysis.name == "prior_analysis"
        assert analysis.likelihood == "test_likelihood"
        assert analysis.domains == ["test_domain"]
        assert analysis.parameters_of_interest is None
        assert analysis.init is None
        assert analysis.prior == "prior_dist"

    def test_analysis_validation_requires_name(self):
        """Test that Analysis validation requires name field."""
        with pytest.raises(ValueError, match="Field required"):
            Analysis(likelihood="test_likelihood", domains=["test_domain"])

    def test_analysis_validation_requires_likelihood(self):
        """Test that Analysis validation requires likelihood field."""
        with pytest.raises(ValueError, match="Field required"):
            Analysis(name="test_analysis", domains=["test_domain"])

    def test_analysis_validation_requires_domains(self):
        """Test that Analysis validation requires domains field."""
        with pytest.raises(ValueError, match="Field required"):
            Analysis(name="test_analysis", likelihood="test_likelihood")


class TestAnalyses:
    """Tests for the Analyses collection class."""

    def test_analyses_creation_empty(self):
        """Test empty Analyses creation."""
        analyses = Analyses([])
        assert len(analyses) == 0
        assert list(analyses) == []

    def test_analyses_creation_with_data(self):
        """Test Analyses creation with analysis data."""
        analysis1 = Analysis(
            name="analysis1", likelihood="likelihood1", domains=["domain1"]
        )
        analysis2 = Analysis(
            name="analysis2", likelihood="likelihood2", domains=["domain2"]
        )
        analyses = Analyses([analysis1, analysis2])

        assert len(analyses) == 2
        assert "analysis1" in analyses
        assert "analysis2" in analyses
        assert analyses["analysis1"] == analysis1
        assert analyses["analysis2"] == analysis2

    def test_analyses_get_by_name(self):
        """Test getting analysis by name."""
        analysis = Analysis(
            name="test_analysis", likelihood="test_likelihood", domains=["test_domain"]
        )
        analyses = Analyses([analysis])

        assert analyses.get("test_analysis") == analysis
        assert analyses.get("nonexistent") is None

        default_analysis = Analysis(
            name="default", likelihood="default_likelihood", domains=["default_domain"]
        )
        assert analyses.get("nonexistent", default_analysis) == default_analysis

    def test_analyses_get_by_index(self):
        """Test getting analysis by index."""
        analysis1 = Analysis(
            name="analysis1", likelihood="likelihood1", domains=["domain1"]
        )
        analysis2 = Analysis(
            name="analysis2", likelihood="likelihood2", domains=["domain2"]
        )
        analyses = Analyses([analysis1, analysis2])

        assert analyses[0] == analysis1
        assert analyses[1] == analysis2

    def test_analyses_iteration(self):
        """Test iteration over analyses."""
        analysis1 = Analysis(
            name="analysis1", likelihood="likelihood1", domains=["domain1"]
        )
        analysis2 = Analysis(
            name="analysis2", likelihood="likelihood2", domains=["domain2"]
        )
        analyses = Analyses([analysis1, analysis2])

        result = list(analyses)
        assert result == [analysis1, analysis2]

    def test_analyses_contains_operator(self):
        """Test 'in' operator for analyses."""
        analysis = Analysis(
            name="test_analysis", likelihood="test_likelihood", domains=["test_domain"]
        )
        analyses = Analyses([analysis])

        assert "test_analysis" in analyses
        assert "nonexistent_analysis" not in analyses

    def test_analyses_keyerror_on_missing_name(self):
        """Test KeyError when accessing non-existent analysis by name."""
        analyses = Analyses([])

        with pytest.raises(KeyError):
            _ = analyses["nonexistent"]

    def test_analyses_indexerror_on_missing_index(self):
        """Test IndexError when accessing non-existent analysis by index."""
        analyses = Analyses([])

        with pytest.raises(IndexError):
            _ = analyses[0]


class TestForeignKeyResolution:
    """Tests for FK validation and serialization in Analysis."""

    def test_analysis_accepts_string_from_json(self):
        """Test Analysis accepts string references from JSON."""
        analysis = Analysis.model_validate(
            {
                "name": "test_analysis",
                "likelihood": "test_likelihood",
                "domains": ["domain1", "domain2"],
            }
        )
        assert analysis.name == "test_analysis"
        assert analysis.likelihood == "test_likelihood"
        assert analysis.domains == ["domain1", "domain2"]

    def test_analysis_accepts_object_from_python(self):
        """Test Analysis accepts model instances from Python."""
        likelihood = Likelihood(
            name="test_likelihood", distributions=["dist1"], data=["data1"]
        )
        domain1 = Domain(name="domain1", type="constant", value="value1")
        domain2 = Domain(name="domain2", type="constant", value="value2")

        analysis = Analysis(
            name="test_analysis", likelihood=likelihood, domains=[domain1, domain2]
        )
        assert analysis.name == "test_analysis"
        assert analysis.likelihood is likelihood
        assert analysis.domains == [domain1, domain2]

    def test_analysis_rejects_dict_for_likelihood(self):
        """Test Analysis rejects dict for likelihood field."""
        with pytest.raises(TypeError, match="Embedded objects not allowed"):
            Analysis(
                name="test_analysis",
                likelihood={"name": "bad"},
                domains=["domain1"],
            )

    def test_analysis_rejects_dict_in_domains(self):
        """Test Analysis rejects dict in domains list."""
        with pytest.raises(TypeError, match="Embedded objects not allowed"):
            Analysis(
                name="test_analysis",
                likelihood="test_likelihood",
                domains=[{"name": "bad"}],
            )

    def test_analysis_serialization_to_strings(self):
        """Test Analysis serialization converts FK fields to strings."""
        likelihood = Likelihood(
            name="test_likelihood", distributions=["dist1"], data=["data1"]
        )
        domain1 = Domain(name="domain1", type="constant", value="value1")
        domain2 = Domain(name="domain2", type="constant", value="value2")

        analysis = Analysis(
            name="test_analysis", likelihood=likelihood, domains=[domain1, domain2]
        )

        dumped = analysis.model_dump()
        assert dumped["likelihood"] == "test_likelihood"
        assert dumped["domains"] == ["domain1", "domain2"]

    def test_analysis_json_schema_shows_strings(self):
        """Test Analysis JSON schema shows FK fields as strings."""
        schema = Analysis.model_json_schema()
        properties = schema["properties"]

        # likelihood should be type string
        assert properties["likelihood"]["type"] == "string"

        # domains should be array of strings
        assert properties["domains"]["type"] == "array"
        assert properties["domains"]["items"]["type"] == "string"

    def test_analysis_standalone_keeps_strings(self):
        """Test standalone Analysis keeps string references without workspace."""
        analysis = Analysis(
            name="test_analysis",
            likelihood="test_likelihood",
            domains=["domain1"],
        )
        # Strings stay as strings without workspace resolution
        assert analysis.likelihood == "test_likelihood"
        assert analysis.domains == ["domain1"]


class TestWorkspaceFKResolution:
    """Integration tests for FK resolution in Workspace context."""

    def test_analysis_likelihood_resolved_to_object(self, datadir):
        """Test analysis likelihood is resolved to Likelihood object."""
        workspace_path = datadir / "simplemodel_uncorrelated-background_hs3.json"
        workspace = Workspace.load(workspace_path)

        assert workspace.analyses is not None
        for analysis in workspace.analyses:
            assert isinstance(analysis.likelihood, Likelihood)
            # Verify it's the actual likelihood from workspace
            assert (
                analysis.likelihood is workspace.likelihoods[analysis.likelihood.name]
            )

    def test_analysis_domains_resolved_to_objects(self, datadir):
        """Test analysis domains are resolved to Domain objects."""
        workspace_path = datadir / "simplemodel_uncorrelated-background_hs3.json"
        workspace = Workspace.load(workspace_path)

        assert workspace.analyses is not None
        for analysis in workspace.analyses:
            for domain in analysis.domains:
                assert isinstance(domain, Domain)
                # Verify it's the actual domain from workspace
                assert domain is workspace.domains[domain.name]

    def test_workspace_roundtrip_preserves_strings(self, datadir):
        """Test workspace roundtrip preserves string references in JSON."""
        workspace_path = datadir / "simplemodel_uncorrelated-background_hs3.json"
        workspace = Workspace.load(workspace_path)

        # Dump back to dict
        dumped = workspace.model_dump()

        # Reload from dict
        workspace2 = Workspace.model_validate(dumped)

        # Verify analyses are resolved again
        assert workspace2.analyses is not None
        for analysis in workspace2.analyses:
            assert isinstance(analysis.likelihood, Likelihood)
            for domain in analysis.domains:
                assert isinstance(domain, Domain)
