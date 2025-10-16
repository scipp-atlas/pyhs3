"""Test cross-distribution parameter sharing and dependency graph integration."""

from __future__ import annotations

import pyhs3


def test_cross_distribution_parameter_sharing():
    """Test that parameters can be shared between different distribution types in dependency graph."""

    # Create workspace with cross-distribution parameter sharing
    # mean2 parameter is shared between gaussian_dist and histfactory normfactor modifier
    test_workspace_data = {
        "metadata": {"hs3_version": "0.1.0"},
        "parameter_points": [
            {
                "name": "default_values",
                "values": {
                    "mean2": 1.0,
                    "sigma2": 0.5,
                    "x": 1.5,
                    "mu": 1.0,
                    "uncorr_bkguncrt_0": 1.0,
                    "uncorr_bkguncrt_1": 1.0,
                    "model_singlechannel_observed": [50, 52],
                },
            }
        ],
        "distributions": [
            {
                "mean": "mean2",
                "name": "px",
                "sigma": "sigma2",
                "type": "gaussian_dist",
                "x": "x",
            },
            {
                "name": "model_singlechannel",
                "type": "histfactory_dist",
                "axes": [
                    {"name": "obs_x_singlechannel", "min": 0.0, "max": 2.0, "nbins": 2}
                ],
                "samples": [
                    {
                        "name": "background",
                        "data": {"contents": [50.0, 52.0], "errors": [7.0, 7.2]},
                        "modifiers": [
                            {
                                "name": "Lumi",
                                "parameter": "mean2",  # Shared parameter!
                                "type": "normfactor",
                            },
                            {
                                "name": "uncorr_bkguncrt",
                                "type": "shapesys",
                                "parameters": [
                                    "uncorr_bkguncrt_0",
                                    "uncorr_bkguncrt_1",
                                ],
                                "data": {"vals": [7.0, 7.2]},
                            },
                        ],
                    },
                    {
                        "name": "signal",
                        "data": {"contents": [12.0, 11.0], "errors": [3.5, 3.3]},
                        "modifiers": [
                            {"name": "mu", "parameter": "mu", "type": "normfactor"}
                        ],
                    },
                ],
            },
        ],
        "functions": [],
        "domains": [],
        "measurements": [
            {
                "distributions": ["model_singlechannel"],
                "index_cat": "channelCat",
                "indices": [0],
                "labels": ["singlechannel"],
            }
        ],
    }

    # Create workspace and model
    ws_pyhs3 = pyhs3.Workspace(**test_workspace_data)
    model = ws_pyhs3.model()

    # Verify model was created successfully (no circular dependencies)
    assert len(model.parameters) > 0
    assert len(model.distributions) == 2
    assert len(model.modifiers) == 3

    # Verify cross-distribution parameter sharing
    px_dist = ws_pyhs3.distributions["px"]
    hf_dist = ws_pyhs3.distributions["model_singlechannel"]

    # Both distributions should depend on mean2
    assert "mean2" in px_dist.parameters
    assert "mean2" in hf_dist.parameters

    # Verify the modifier uses the shared parameter
    lumi_modifier = None
    for modifier in hf_dist.get_internal_nodes():
        if "Lumi" in modifier.name:
            lumi_modifier = modifier
            break

    assert lumi_modifier is not None
    assert "mean2" in lumi_modifier.dependencies

    # Verify modifier has unique name in dependency graph
    assert lumi_modifier.name == "normfactor/Lumi"
    assert "normfactor/Lumi" in model.modifiers

    # Verify other modifiers are also properly named
    assert "normfactor/mu" in model.modifiers
    assert "shapesys/uncorr_bkguncrt" in model.modifiers

    # Verify parameter exists in model
    assert "mean2" in model.parameters


def test_histfactory_modifier_unique_naming():
    """Test that modifiers get unique names in dependency graph to avoid parameter conflicts."""

    test_workspace_data = {
        "metadata": {"hs3_version": "0.1.0"},
        "parameter_points": [
            {
                "name": "default_values",
                "values": {"Lumi": 1.0, "mu": 1.0, "model_test_observed": [50]},
            }
        ],
        "distributions": [
            {
                "name": "model_test",
                "type": "histfactory_dist",
                "axes": [{"name": "obs_x", "min": 0.0, "max": 1.0, "nbins": 1}],
                "samples": [
                    {
                        "name": "sample1",
                        "data": {"contents": [50.0], "errors": [7.0]},
                        "modifiers": [
                            {
                                "name": "Lumi",
                                "parameter": "Lumi",  # Same name as parameter
                                "type": "normfactor",
                            }
                        ],
                    },
                    {
                        "name": "sample2",
                        "data": {"contents": [30.0], "errors": [5.0]},
                        "modifiers": [
                            {
                                "name": "Lumi",  # Same modifier name (correlated)
                                "parameter": "Lumi",
                                "type": "normfactor",
                            },
                            {
                                "name": "mu",
                                "parameter": "mu",  # Same name as parameter
                                "type": "normfactor",
                            },
                        ],
                    },
                ],
            }
        ],
        "functions": [],
        "domains": [],
        "measurements": [
            {
                "distributions": ["model_test"],
                "index_cat": "channelCat",
                "indices": [0],
                "labels": ["test"],
            }
        ],
    }

    # Should build successfully without circular dependencies
    ws_pyhs3 = pyhs3.Workspace(**test_workspace_data)
    model = ws_pyhs3.model()

    # Verify no circular dependencies by successful model creation
    assert len(model.parameters) > 0
    assert len(model.distributions) == 1
    assert len(model.modifiers) == 2  # Only unique modifiers (Lumi and mu)

    # Verify modifiers have unique names in graph
    assert "normfactor/Lumi" in model.modifiers
    assert "normfactor/mu" in model.modifiers

    # Verify parameters exist separately
    assert "Lumi" in model.parameters
    assert "mu" in model.parameters

    # Verify the modifier objects have the correct dependencies
    hf_dist = ws_pyhs3.distributions["model_test"]
    modifier_names = {
        mod.name: mod.dependencies for mod in hf_dist.get_internal_nodes()
    }

    assert "normfactor/Lumi" in modifier_names
    assert "normfactor/mu" in modifier_names
    assert modifier_names["normfactor/Lumi"] == {"Lumi"}
    assert modifier_names["normfactor/mu"] == {"mu"}
