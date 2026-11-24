"""Tests for Model.entity_type: look up what kind of entity a name refers to."""

from __future__ import annotations

import pytest
from test_model_function_families import _workspace as _family_workspace

import pyhs3
from pyhs3 import model as model_module


@pytest.fixture
def gauss_model():
    """Gaussian with a derived mean function and a literal (constant) sigma."""
    ws = pyhs3.Workspace(
        metadata={"hs3_version": "0.2"},
        distributions=[
            {
                "name": "gauss",
                "type": "gaussian_dist",
                "x": "x",
                "mean": "scaled_mu",
                "sigma": 2.0,
            }
        ],
        functions=[
            {"name": "scaled_mu", "type": "product", "factors": ["mu", "scale"]}
        ],
        parameter_points=[
            {
                "name": "default_values",
                "parameters": [
                    {"name": "x", "value": 0.0},
                    {"name": "mu", "value": 1.0},
                    {"name": "scale", "value": 3.0},
                ],
            }
        ],
    )
    return ws.model(0, progress=False)


class TestEntityType:
    def test_distribution(self, gauss_model):
        assert gauss_model.entity_type("gauss") == "distribution"

    def test_function(self, gauss_model):
        assert gauss_model.entity_type("scaled_mu") == "function"

    def test_free_parameter(self, gauss_model):
        assert gauss_model.entity_type("mu") == "parameter"

    def test_constant(self, gauss_model):
        constants = [n for n in gauss_model.parameters if n.startswith("constant_")]
        assert constants
        assert gauss_model.entity_type(constants[0]) == "constant"

    def test_unknown_name_raises_key_error(self, gauss_model):
        with pytest.raises(ValueError, match="'missing' not found in model"):
            gauss_model.entity_type("missing")

    def test_family_names_are_not_reported(self, monkeypatch):
        family_names = []
        plan_families = model_module.plan_families

        def recording_plan_families(*args, **kwargs):
            families = list(plan_families(*args, **kwargs))
            family_names.extend(family.name for family in families)
            return families

        monkeypatch.setattr(model_module, "plan_families", recording_plan_families)
        ws = _family_workspace()
        model = ws.model(ws.analyses["A"], progress=False)

        assert family_names
        for name in family_names:
            with pytest.raises(ValueError, match="not found in model"):
                model.entity_type(name)
        # Members of a family keep their own type.
        assert model.entity_type("g0_interp0") == "function"


class TestEntityTypeAfterPruning:
    @pytest.fixture
    def pruned_model(self):
        """Likelihood over ``g``; the distribution ``h`` is not part of the model."""
        ws = pyhs3.Workspace(
            metadata={"hs3_version": "0.2"},
            distributions=[
                {
                    "name": name,
                    "type": "gaussian_dist",
                    "x": "x",
                    "mean": mean,
                    "sigma": 1.0,
                }
                for name, mean in [("g", "mu"), ("h", "nu")]
            ],
            data=[
                {
                    "name": "d",
                    "type": "unbinned",
                    "axes": [{"name": "x", "min": -5.0, "max": 5.0}],
                    "entries": [[0.0], [1.0]],
                }
            ],
            likelihoods=[{"name": "L", "distributions": ["g"], "data": ["d"]}],
            parameter_points=[
                {
                    "name": "default_values",
                    "parameters": [
                        {"name": "x", "value": 0.0},
                        {"name": "mu", "value": 0.0},
                        {"name": "nu", "value": 0.0},
                    ],
                }
            ],
        )
        return ws.model(ws.likelihoods["L"], progress=False)

    def test_kept_distribution_is_reported(self, pruned_model):
        assert pruned_model.entity_type("g") == "distribution"

    def test_pruned_distribution_is_unknown(self, pruned_model):
        assert "h" not in pruned_model.distributions
        with pytest.raises(ValueError, match="'h' not found in model"):
            pruned_model.entity_type("h")

    def test_pruned_parameter_is_unknown(self, pruned_model):
        with pytest.raises(ValueError, match="'nu' not found in model"):
            pruned_model.entity_type("nu")
