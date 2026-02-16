"""
Unit tests for likelihood implementations.

Tests for Likelihood, Likelihoods classes including validation
of likelihood specifications and collection management.
"""

from __future__ import annotations

import pytest

from pyhs3 import Workspace
from pyhs3.data import Data, Datum, PointData
from pyhs3.distributions import Distributions, GaussianDist
from pyhs3.distributions.core import Distribution
from pyhs3.likelihoods import Likelihood, Likelihoods
from pyhs3.metadata import Metadata


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


class TestForeignKeyResolution:
    """Tests for FK validation and serialization in Likelihood."""

    def test_likelihood_accepts_strings_from_json(self):
        """Test Likelihood accepts string references from JSON."""
        likelihood = Likelihood.model_validate(
            {
                "name": "test_likelihood",
                "distributions": ["dist1", "dist2"],
                "data": ["obs1", "obs2"],
            }
        )
        assert likelihood.name == "test_likelihood"
        assert likelihood.distributions == ["dist1", "dist2"]
        assert likelihood.data == ["obs1", "obs2"]

    def test_likelihood_accepts_objects_from_python(self):
        """Test Likelihood accepts model instances from Python."""
        dist1 = GaussianDist(name="dist1", x="x", mean=0.0, sigma=1.0)
        datum1 = PointData(name="obs1", value=1.5)

        likelihood = Likelihood(
            name="test_likelihood", distributions=[dist1], data=[datum1]
        )
        assert likelihood.name == "test_likelihood"
        assert likelihood.distributions == [dist1]
        assert likelihood.data == [datum1]

    def test_likelihood_rejects_dict_in_distributions(self):
        """Test Likelihood rejects dict in distributions list."""
        with pytest.raises(TypeError, match="Embedded objects not allowed"):
            Likelihood(
                name="test_likelihood",
                distributions=[{"name": "bad"}],
                data=["obs1"],
            )

    def test_likelihood_rejects_dict_in_data(self):
        """Test Likelihood rejects dict in data list."""
        with pytest.raises(TypeError, match="Embedded objects not allowed"):
            Likelihood(
                name="test_likelihood",
                distributions=["dist1"],
                data=[{"name": "bad"}],
            )

    def test_likelihood_serialization_to_strings(self):
        """Test Likelihood serialization converts FK fields to strings."""
        dist1 = GaussianDist(name="dist1", x="x", mean=0.0, sigma=1.0)
        datum1 = PointData(name="obs1", value=1.5)

        likelihood = Likelihood(
            name="test_likelihood", distributions=[dist1], data=[datum1]
        )

        dumped = likelihood.model_dump()
        assert dumped["distributions"] == ["dist1"]
        assert dumped["data"] == ["obs1"]

    def test_likelihood_serialization_with_string_references(self):
        """Test Likelihood serialization with string refs (branch #4)."""
        likelihood = Likelihood(
            name="test_likelihood",
            distributions=["dist1", "dist2"],
            data=["obs1"],
        )

        dumped = likelihood.model_dump()
        assert dumped["distributions"] == ["dist1", "dist2"]
        assert dumped["data"] == ["obs1"]

    def test_likelihood_json_schema_correct(self):
        """Test Likelihood JSON schema shows FK fields as array of strings."""
        schema = Likelihood.model_json_schema()
        properties = schema["properties"]

        # distributions should be array of strings
        assert properties["distributions"]["type"] == "array"
        assert properties["distributions"]["items"]["type"] == "string"

        # data should be array of strings
        assert properties["data"]["type"] == "array"
        assert properties["data"]["items"]["type"] == "string"


class TestWorkspaceFKResolution:
    """Integration tests for FK resolution in Workspace context."""

    def test_likelihood_distributions_resolved(self, datadir):
        """Test likelihood distributions are resolved to Distribution objects."""
        workspace_path = datadir / "simplemodel_uncorrelated-background_hs3.json"
        workspace = Workspace.load(workspace_path)

        assert workspace.likelihoods is not None
        for likelihood in workspace.likelihoods:
            for dist in likelihood.distributions:
                assert isinstance(dist, Distribution)
                # Verify it's the actual distribution from workspace
                assert dist is workspace.distributions[dist.name]

    def test_likelihood_data_resolved(self, datadir):
        """Test likelihood data are resolved to Datum objects."""
        workspace_path = datadir / "simplemodel_uncorrelated-background_hs3.json"
        workspace = Workspace.load(workspace_path)

        assert workspace.likelihoods is not None
        for likelihood in workspace.likelihoods:
            for datum in likelihood.data:
                assert isinstance(datum, Datum)
                # Verify it's the actual datum from workspace
                assert datum is workspace.data[datum.name]

    def test_workspace_roundtrip_preserves_strings(self, datadir):
        """Test workspace roundtrip preserves string references in JSON."""
        workspace_path = datadir / "simplemodel_uncorrelated-background_hs3.json"
        workspace = Workspace.load(workspace_path)

        # Dump back to dict
        dumped = workspace.model_dump()

        # Reload from dict
        workspace2 = Workspace.model_validate(dumped)

        # Verify likelihoods are resolved again
        assert workspace2.likelihoods is not None
        for likelihood in workspace2.likelihoods:
            for dist in likelihood.distributions:
                assert isinstance(dist, Distribution)
            for datum in likelihood.data:
                assert isinstance(datum, Datum)

    def test_likelihood_with_preresolved_objects(self):
        """Test Likelihood with preresolved Distribution and Datum instances (branch #1)."""
        # Create Distribution and Datum instances
        dist1 = GaussianDist(name="dist1", x="x", mean=0.0, sigma=1.0)
        datum1 = PointData(name="obs1", value=1.5)

        # Build workspace programmatically with preresolved objects
        workspace = Workspace(
            metadata=Metadata(hs3_version="0.1.0"),
            distributions=Distributions([dist1]),
            data=Data([datum1]),
            likelihoods=Likelihoods(
                [
                    Likelihood(
                        name="test_likelihood",
                        distributions=[dist1],  # Already-resolved ref
                        data=[datum1],  # Already-resolved ref
                    )
                ]
            ),
        )

        # Verify references remain resolved
        assert workspace.likelihoods is not None
        likelihood = workspace.likelihoods[0]
        assert isinstance(likelihood.distributions[0], Distribution)
        assert isinstance(likelihood.data[0], Datum)
        assert likelihood.distributions[0] is dist1
        assert likelihood.data[0] is datum1


class TestWorkspaceReferentialIntegrity:
    """Tests for workspace referential integrity validation."""

    def test_workspace_unknown_distribution_raises(self):
        """Test workspace raises error for unknown distribution reference."""
        with pytest.raises(ValueError, match="unknown distribution"):
            Workspace.model_validate(
                {
                    "metadata": {"hs3_version": "0.1.0"},
                    "likelihoods": [
                        {
                            "name": "likelihood1",
                            "distributions": ["unknown_dist"],
                            "data": [],
                        }
                    ],
                    "distributions": [],
                    "data": [],
                }
            )

    def test_workspace_unknown_data_raises(self):
        """Test workspace raises error for unknown data reference."""
        with pytest.raises(ValueError, match="unknown data"):
            Workspace.model_validate(
                {
                    "metadata": {"hs3_version": "0.1.0"},
                    "likelihoods": [
                        {
                            "name": "likelihood1",
                            "distributions": [],
                            "data": ["unknown_data"],
                        }
                    ],
                    "distributions": [],
                    "data": [],
                }
            )

    def test_workspace_unknown_likelihood_raises(self):
        """Test workspace raises error for unknown likelihood reference."""
        with pytest.raises(ValueError, match="unknown likelihood"):
            Workspace.model_validate(
                {
                    "metadata": {"hs3_version": "0.1.0"},
                    "analyses": [
                        {
                            "name": "analysis1",
                            "likelihood": "unknown_likelihood",
                            "domains": [],
                        }
                    ],
                    "likelihoods": [],
                }
            )

    def test_workspace_unknown_domain_raises(self):
        """Test workspace raises error for unknown domain reference."""
        with pytest.raises(ValueError, match="unknown domain"):
            Workspace.model_validate(
                {
                    "metadata": {"hs3_version": "0.1.0"},
                    "analyses": [
                        {
                            "name": "analysis1",
                            "likelihood": "likelihood1",
                            "domains": ["unknown_domain"],
                        }
                    ],
                    "likelihoods": [
                        {
                            "name": "likelihood1",
                            "distributions": ["dist1"],
                            "data": ["obs1"],
                        }
                    ],
                    "distributions": [
                        {
                            "name": "dist1",
                            "type": "gaussian_dist",
                            "x": "x",
                            "mean": 0.0,
                            "sigma": 1.0,
                        }
                    ],
                    "data": [{"name": "obs1", "type": "point", "value": 1.0}],
                    "domains": [],
                }
            )

    def test_workspace_collects_all_errors(self):
        """Test workspace collects multiple FK resolution errors."""
        with pytest.raises(ValueError, match="unresolved references") as exc_info:
            Workspace.model_validate(
                {
                    "metadata": {"hs3_version": "0.1.0"},
                    "analyses": [
                        {
                            "name": "analysis1",
                            "likelihood": "unknown_likelihood",
                            "domains": ["unknown_domain"],
                        }
                    ],
                    "likelihoods": [
                        {
                            "name": "likelihood1",
                            "distributions": ["unknown_dist"],
                            "data": ["unknown_data"],
                        }
                    ],
                    "distributions": [],
                    "data": [],
                    "domains": [],
                }
            )

        # Verify multiple errors are reported
        error_msg = str(exc_info.value)
        assert "unknown_likelihood" in error_msg or "unknown_domain" in error_msg
        assert "unknown_dist" in error_msg or "unknown_data" in error_msg
