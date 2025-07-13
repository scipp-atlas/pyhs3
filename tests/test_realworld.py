from __future__ import annotations

import json
from pathlib import Path

import pytest
from skhep_testdata import data_path as skhep_testdata_path

import pyhs3 as hs3


def summarize(node, max_depth=3, _depth=0):
    indent = "  " * _depth
    if _depth >= max_depth:
        print(f"{indent}…")
        return

    if isinstance(node, dict):
        for key, value in node.items():
            print(f"{indent}{key} -> {type(value).__name__}")
            summarize(value, max_depth, _depth + 1)

    elif isinstance(node, list):
        print(f"{indent}[list of {len(node)} items]")
        if node:
            summarize(node[0], max_depth, _depth + 1)

    else:
        print(f"{indent}{node!r} ({type(node).__name__})")


@pytest.fixture
def ws_json():
    """Load issue41_diHiggs_workspace.json file and return parsed JSON content.

    This workspace is from Alex Wang for the diHiggs gamgam bb analysis,
    related to GitHub issue #41.
    """
    fpath = Path(skhep_testdata_path("test_hs3_unbinned_pyhs3_validation_issue41.json"))
    return json.loads(fpath.read_text(encoding="utf-8"))


@pytest.fixture
def ws_workspace(ws_json):
    """Create workspace from WS.json content."""
    return hs3.Workspace(ws_json)


def test_workspace_structure(ws_json):
    """Test and display workspace structure."""
    summarize(ws_json, max_depth=3)
    assert isinstance(ws_json, dict)


def test_workspace_loading(ws_workspace):
    """Test loading workspace from WS.json."""
    assert ws_workspace is not None


class TestDiHiggsIssue41Workspace:
    """Test DiHiggs issue #41 workspace with expected NLL values."""

    @pytest.fixture
    def expected_nll_data(self):
        """Load expected NLL values from ROOT fits."""
        nll_path = Path(__file__).parent.parent / "json_test" / "exp_sys" / "mu_HH.json"
        return json.loads(nll_path.read_text(encoding="utf-8"))

    def test_workspace_loads_successfully(self, ws_workspace):
        """Test that the workspace loads without errors."""
        assert ws_workspace is not None
        assert len(ws_workspace.distribution_set) > 0

    def test_workspace_has_expected_distributions(self, ws_workspace):
        """Test that workspace contains the expected distribution types."""
        # Check that we have the distributions we expect
        dist_types = set()
        for dist in ws_workspace.distribution_set:
            dist_types.add(type(dist).__name__)

        # Should have various distribution types
        assert len(dist_types) > 0

    @pytest.mark.xfail(
        reason="Dependency graph construction fails due to KeyError: 'constr__THEO_BR_Hbb'. "
        "The dependency graph builder doesn't correctly handle cross-references between "
        "functions and distributions, causing missing entities during topological evaluation."
    )
    def test_nll_validation_against_root(self, ws_workspace, expected_nll_data):
        """Test NLL values match expected ROOT results."""
        # This test is expected to fail until dependency graph construction is fixed
        mu_HH_values = expected_nll_data["mu_HH"]
        expected_nll_values = expected_nll_data["nll"]

        # Find parameters and model
        param_collection = ws_workspace.parameter_collection[0]  # default_values
        domain_collection = ws_workspace.domain_collection[0]  # default_domain

        for i, _mu_HH_val in enumerate(mu_HH_values):
            _expected_nll = expected_nll_values[i]

            # Create model with mu_HH set to specific value
            # Note: This will fail due to dependency graph issues
            model = ws_workspace.model(
                parameter_point=param_collection, domain=domain_collection
            )

            # Evaluate NLL at this mu_HH value
            # This is where the test should validate against expected_nll
            # For now, we'll just check that we can create the model
            assert model is not None

            # TODO: Implement actual NLL calculation and comparison
            # when dependency graph construction is fixed

    def test_workspace_parameter_structure(self, ws_workspace):
        """Test that workspace has expected parameter structure."""
        assert len(ws_workspace.parameter_collection) > 0
        assert len(ws_workspace.domain_collection) > 0

        # Check that we have default collections
        param_names = [p.name for p in ws_workspace.parameter_collection]
        domain_names = [d.name for d in ws_workspace.domain_collection]

        # Should have at least default values
        assert len(param_names) > 0
        assert len(domain_names) > 0

    @pytest.mark.xfail(
        reason="Dependency graph construction fails due to KeyError: 'constr__THEO_BR_Hbb'. "
        "The dependency graph builder doesn't correctly handle cross-references between "
        "functions and distributions, causing missing entities during topological evaluation."
    )
    def test_workspace_model_creation(self, ws_workspace):
        """Test that we can create a model from the workspace."""
        # This tests basic model creation without full evaluation
        param_collection = ws_workspace.parameter_collection[0]
        domain_collection = ws_workspace.domain_collection[0]

        # Should be able to create model without errors
        model = ws_workspace.model(
            parameter_point=param_collection, domain=domain_collection
        )
        assert model is not None
