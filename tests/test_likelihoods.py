"""
Unit tests for likelihood implementations.

Tests for Likelihood, Likelihoods classes including validation
of likelihood specifications and collection management.
"""

from __future__ import annotations

import pytest

from pyhs3.likelihoods import Likelihood, Likelihoods


class TestLikelihood:
    """Tests for the Likelihood class."""

    def test_likelihood_creation_basic(self):
        """Test basic Likelihood creation with required fields."""
        likelihood = Likelihood(
            name="test_likelihood",
            distributions=["dist1", "dist2"],
            data=["data1", "data2"],
        )
        assert likelihood.name == "test_likelihood"
        assert likelihood.distributions == ["dist1", "dist2"]
        assert likelihood.data == ["data1", "data2"]
        assert likelihood.aux_distributions is None

    def test_likelihood_creation_with_aux_distributions(self):
        """Test Likelihood creation with auxiliary distributions."""
        likelihood = Likelihood(
            name="test_likelihood",
            distributions=["dist1", "dist2"],
            data=["data1", "data2"],
            aux_distributions=["aux_dist1", "aux_dist2"],
        )
        assert likelihood.name == "test_likelihood"
        assert likelihood.distributions == ["dist1", "dist2"]
        assert likelihood.data == ["data1", "data2"]
        assert likelihood.aux_distributions == ["aux_dist1", "aux_dist2"]

    def test_likelihood_validation_requires_name(self):
        """Test that Likelihood validation requires name field."""
        with pytest.raises(ValueError, match="Field required"):
            Likelihood(distributions=["dist1"], data=["data1"])

    def test_likelihood_validation_requires_distributions(self):
        """Test that Likelihood validation requires distributions field."""
        with pytest.raises(ValueError, match="Field required"):
            Likelihood(name="test_likelihood", data=["data1"])

    def test_likelihood_validation_requires_data(self):
        """Test that Likelihood validation requires data field."""
        with pytest.raises(ValueError, match="Field required"):
            Likelihood(name="test_likelihood", distributions=["dist1"])


class TestLikelihoods:
    """Tests for the Likelihoods collection class."""

    def test_likelihoods_creation_empty(self):
        """Test empty Likelihoods creation."""
        likelihoods = Likelihoods([])
        assert len(likelihoods) == 0
        assert list(likelihoods) == []

    def test_likelihoods_creation_with_data(self):
        """Test Likelihoods creation with likelihood data."""
        likelihood1 = Likelihood(
            name="likelihood1", distributions=["dist1"], data=["data1"]
        )
        likelihood2 = Likelihood(
            name="likelihood2", distributions=["dist2"], data=["data2"]
        )
        likelihoods = Likelihoods([likelihood1, likelihood2])

        assert len(likelihoods) == 2
        assert "likelihood1" in likelihoods
        assert "likelihood2" in likelihoods
        assert likelihoods["likelihood1"] == likelihood1
        assert likelihoods["likelihood2"] == likelihood2

    def test_likelihoods_get_by_name(self):
        """Test getting likelihood by name."""
        likelihood = Likelihood(
            name="test_likelihood", distributions=["dist1"], data=["data1"]
        )
        likelihoods = Likelihoods([likelihood])

        assert likelihoods.get("test_likelihood") == likelihood
        assert likelihoods.get("nonexistent") is None

        default_likelihood = Likelihood(
            name="default", distributions=["default_dist"], data=["default_data"]
        )
        assert likelihoods.get("nonexistent", default_likelihood) == default_likelihood

    def test_likelihoods_get_by_index(self):
        """Test getting likelihood by index."""
        likelihood1 = Likelihood(
            name="likelihood1", distributions=["dist1"], data=["data1"]
        )
        likelihood2 = Likelihood(
            name="likelihood2", distributions=["dist2"], data=["data2"]
        )
        likelihoods = Likelihoods([likelihood1, likelihood2])

        assert likelihoods[0] == likelihood1
        assert likelihoods[1] == likelihood2

    def test_likelihoods_iteration(self):
        """Test iteration over likelihoods."""
        likelihood1 = Likelihood(
            name="likelihood1", distributions=["dist1"], data=["data1"]
        )
        likelihood2 = Likelihood(
            name="likelihood2", distributions=["dist2"], data=["data2"]
        )
        likelihoods = Likelihoods([likelihood1, likelihood2])

        result = list(likelihoods)
        assert result == [likelihood1, likelihood2]

    def test_likelihoods_likelihood_map_property(self):
        """Test likelihood_map property."""
        likelihood1 = Likelihood(
            name="likelihood1", distributions=["dist1"], data=["data1"]
        )
        likelihood2 = Likelihood(
            name="likelihood2", distributions=["dist2"], data=["data2"]
        )
        likelihoods = Likelihoods([likelihood1, likelihood2])

        likelihood_map = likelihoods.likelihood_map
        assert likelihood_map["likelihood1"] == likelihood1
        assert likelihood_map["likelihood2"] == likelihood2
        assert len(likelihood_map) == 2

    def test_likelihoods_contains_operator(self):
        """Test 'in' operator for likelihoods."""
        likelihood = Likelihood(
            name="test_likelihood", distributions=["dist1"], data=["data1"]
        )
        likelihoods = Likelihoods([likelihood])

        assert "test_likelihood" in likelihoods
        assert "nonexistent_likelihood" not in likelihoods

    def test_likelihoods_keyerror_on_missing_name(self):
        """Test KeyError when accessing non-existent likelihood by name."""
        likelihoods = Likelihoods([])

        with pytest.raises(KeyError):
            _ = likelihoods["nonexistent"]

    def test_likelihoods_indexerror_on_missing_index(self):
        """Test IndexError when accessing non-existent likelihood by index."""
        likelihoods = Likelihoods([])

        with pytest.raises(IndexError):
            _ = likelihoods[0]
