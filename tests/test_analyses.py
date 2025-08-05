"""
Unit tests for analysis implementations.

Tests for Analysis, Analyses classes including validation
of analysis specifications and collection management.
"""

from __future__ import annotations

import pytest

from pyhs3.analyses import Analyses, Analysis


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

    def test_analyses_analysis_map_property(self):
        """Test analysis_map property."""
        analysis1 = Analysis(
            name="analysis1", likelihood="likelihood1", domains=["domain1"]
        )
        analysis2 = Analysis(
            name="analysis2", likelihood="likelihood2", domains=["domain2"]
        )
        analyses = Analyses([analysis1, analysis2])

        analysis_map = analyses.analysis_map
        assert analysis_map["analysis1"] == analysis1
        assert analysis_map["analysis2"] == analysis2
        assert len(analysis_map) == 2

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
