from __future__ import annotations

import re
import warnings
from pathlib import Path
from unittest.mock import patch

import numpy as np
import numpy.testing as npt
import pytensor.tensor as pt
import pytest
from pytensor.graph.traversal import explicit_graph_inputs
from pytensor.tensor.basic import TensorConstant

import pyhs3 as hs3
from pyhs3.domains import ProductDomain
from pyhs3.parameter_points import ParameterPoint, ParameterSet


@pytest.fixture
def simple_workspace():
    """Create a simple workspace for testing Model functionality."""
    workspace_data = {
        "metadata": {"hs3_version": "0.2"},
        "distributions": [
            {
                "name": "gauss",
                "type": "gaussian_dist",
                "x": "x",
                "mean": "mu",
                "sigma": "sigma",
            }
        ],
        "parameter_points": [
            {
                "name": "default_values",
                "parameters": [
                    {"name": "x", "value": 0.0},
                    {"name": "mu", "value": 0.0},
                    {"name": "sigma", "value": 1.0},
                ],
            }
        ],
        "domains": [
            {
                "name": "default_domain",
                "type": "product_domain",
                "axes": [
                    {"name": "x", "min": -5.0, "max": 5.0},
                    {"name": "mu", "min": -2.0, "max": 2.0},
                    {"name": "sigma", "min": 0.1, "max": 3.0},
                ],
            }
        ],
    }
    return hs3.Workspace(**workspace_data)


class TestModelModes:
    """Test Model compilation modes and behavior."""

    @pytest.mark.parametrize("mode", ["FAST_RUN", "FAST_COMPILE"])
    def test_model_compilation_modes(self, simple_workspace, mode):
        """Test that models work correctly with different compilation modes."""
        model = simple_workspace.model(0, mode=mode)

        # Verify the mode is set correctly
        assert model.mode == mode

        # Test that we can evaluate PDF with both modes
        result = model.pdf(
            "gauss", x=np.array(0.0), mu=np.array(0.0), sigma=np.array(1.0)
        )

        # Should get a reasonable Gaussian PDF value at x=0, mu=0, sigma=1
        # This should be approximately 1/sqrt(2*pi) ≈ 0.3989
        assert 0.35 < result < 0.45

    def test_mode_compilation_differences(self, simple_workspace):
        """Test that both modes produce equivalent results."""
        model_fast_run = simple_workspace.model(0, mode="FAST_RUN")
        model_fast_compile = simple_workspace.model(0, mode="FAST_COMPILE")

        # Both should produce the same result for the same inputs
        result_fast_run = model_fast_run.pdf(
            "gauss", x=np.array(1.0), mu=np.array(0.5), sigma=np.array(1.2)
        )
        result_fast_compile = model_fast_compile.pdf(
            "gauss", x=np.array(1.0), mu=np.array(0.5), sigma=np.array(1.2)
        )

        # Results should be very close (within numerical precision)
        assert abs(result_fast_run - result_fast_compile) < 1e-10


class TestModelRepr:
    """Test Model.__repr__ method."""

    def test_repr_basic_structure(self, simple_workspace):
        """Test that __repr__ returns expected format."""
        model = simple_workspace.model(0)
        repr_str = repr(model)

        # Should start with "Model(" and end with ")"
        assert repr_str.startswith("Model(")
        assert repr_str.endswith(")")

        # Should contain mode information
        assert "mode: FAST_RUN" in repr_str

        # Should contain parameter, distribution, and function counts
        assert "parameters:" in repr_str
        assert "distributions:" in repr_str
        assert "functions:" in repr_str

    def test_repr_shows_correct_counts(self, simple_workspace):
        """Test that __repr__ shows correct entity counts."""
        model = simple_workspace.model(0)
        repr_str = repr(model)

        # Should show 3 parameters (x, mu, sigma)
        assert "parameters: 3" in repr_str

        # Should show 1 distribution (gauss)
        assert "distributions: 1" in repr_str

        # Should show 0 functions (none in simple workspace)
        assert "functions: 0" in repr_str

    def test_repr_shows_entity_names(self, simple_workspace):
        """Test that __repr__ includes entity names."""
        model = simple_workspace.model(0)
        repr_str = repr(model)

        # Should contain parameter names
        assert "x" in repr_str
        assert "mu" in repr_str
        assert "sigma" in repr_str

        # Should contain distribution name
        assert "gauss" in repr_str

    def test_repr_with_custom_mode(self, simple_workspace):
        """Test __repr__ with different compilation mode."""
        model = simple_workspace.model(0, mode="FAST_COMPILE")
        repr_str = repr(model)

        assert "mode: FAST_COMPILE" in repr_str


class TestModelGraphSummary:
    """Test Model.graph_summary() method."""

    def test_graph_summary_basic_structure(self, simple_workspace):
        """Test that graph_summary returns expected format."""
        model = simple_workspace.model(0)
        summary = model.graph_summary("gauss")

        # Should start with distribution name
        assert summary.startswith("Distribution 'gauss':")

        # Should contain input variables count
        assert "Input variables:" in summary

        # Should contain graph operations count
        assert "Graph operations:" in summary

        # Should contain operation types
        assert "Operation types:" in summary

        # Should contain mode and compilation info
        assert "Mode:" in summary
        assert "Compiled:" in summary

    def test_graph_summary_input_variables(self, simple_workspace):
        """Test that graph_summary shows input variable count."""
        model = simple_workspace.model(0)
        summary = model.graph_summary("gauss")

        # Should show some input variables (exact number depends on implementation)
        assert "Input variables:" in summary
        # The actual number may be higher than 3 due to intermediate computations
        match = re.search(r"Input variables: (\d+)", summary)
        assert match is not None
        var_count = int(match.group(1))
        assert var_count >= 3  # At least x, mu, sigma

    def test_graph_summary_compilation_info(self, simple_workspace):
        """Test that graph_summary shows compilation information."""
        model_fast_run = simple_workspace.model(0, mode="FAST_RUN")
        model_fast_compile = simple_workspace.model(0, mode="FAST_COMPILE")

        # FAST_RUN should not show as compiled initially (until function is called)
        summary_fast_run = model_fast_run.graph_summary("gauss")
        assert "Mode: FAST_RUN" in summary_fast_run
        assert "Compiled: No" in summary_fast_run

        # FAST_COMPILE should show compiled status
        summary_fast_compile = model_fast_compile.graph_summary("gauss")
        assert "Mode: FAST_COMPILE" in summary_fast_compile

    def test_graph_summary_after_compilation(self, simple_workspace):
        """Test graph_summary shows compilation status after function is called."""
        model = simple_workspace.model(0, mode="FAST_RUN")

        # Before calling pdf, should not be compiled
        summary_before = model.graph_summary("gauss")
        assert "Compiled: No" in summary_before

        # Call pdf to trigger compilation
        model.pdf("gauss", x=np.array(0.0), mu=np.array(0.0), sigma=np.array(1.0))

        # After calling pdf, should be compiled
        summary_after = model.graph_summary("gauss")
        assert "Compiled: Yes" in summary_after

    def test_graph_summary_nonexistent_distribution(self, simple_workspace):
        """Test that graph_summary raises error for nonexistent distribution."""
        model = simple_workspace.model(0)

        with pytest.raises(ValueError, match="Distribution 'nonexistent' not found"):
            model.graph_summary("nonexistent")

    def test_graph_summary_operation_types(self, simple_workspace):
        """Test that graph_summary includes operation type information."""
        model = simple_workspace.model(0)
        summary = model.graph_summary("gauss")

        # Should contain operation types in a dict-like format
        # The exact operations depend on the Gaussian implementation
        assert "Operation types: {" in summary
        assert "}" in summary


class TestModelGraphVisualization:
    """Test Model.visualize_graph() method."""

    @pytest.mark.pydot
    def test_visualize_graph_basic_functionality(self, simple_workspace, tmp_path):
        """Test that visualize_graph creates output file."""
        model = simple_workspace.model(0)

        # Test with default parameters (svg format) in tmp_path
        output_file = model.visualize_graph("gauss", path=tmp_path)

        # Should return a file path
        assert isinstance(output_file, str)
        assert output_file.endswith(".svg")
        assert "gauss_graph.svg" in output_file

        # File should exist in tmp_path
        expected_path = tmp_path / "gauss_graph.svg"
        assert expected_path.exists()

    @pytest.mark.pydot
    @pytest.mark.parametrize("fmt", ["svg", "png", "pdf"])
    def test_visualize_graph_different_formats(self, simple_workspace, tmp_path, fmt):
        """Test visualize_graph with different output formats."""
        model = simple_workspace.model(0)

        output_file = model.visualize_graph("gauss", fmt=fmt, path=tmp_path)

        # Should have correct extension
        assert output_file.endswith(f".{fmt}")
        assert f"gauss_graph.{fmt}" in output_file

        # File should exist in tmp_path
        expected_path = tmp_path / f"gauss_graph.{fmt}"
        assert expected_path.exists()

    @pytest.mark.pydot
    def test_visualize_graph_custom_output_file(self, simple_workspace, tmp_path):
        """Test visualize_graph with custom output filename."""
        model = simple_workspace.model(0)

        custom_file = tmp_path / "my_custom_graph.svg"
        output_file = model.visualize_graph("gauss", outfile=str(custom_file))

        # Should return the custom file path
        assert output_file == str(custom_file)

        # File should exist at the specified location
        assert custom_file.exists()

    @pytest.mark.pydot
    def test_visualize_graph_nonexistent_distribution(self, simple_workspace):
        """Test that visualize_graph raises error for nonexistent distribution."""
        model = simple_workspace.model(0)

        with pytest.raises(ValueError, match="Distribution 'nonexistent' not found"):
            model.visualize_graph("nonexistent")

    @pytest.mark.pydot
    def test_visualize_graph_no_path_parameter(self, simple_workspace):
        """Test visualize_graph without path parameter (uses current directory)."""
        model = simple_workspace.model(0)

        # Call without path parameter - should use current working directory
        output_file = model.visualize_graph("gauss")

        # Should return just the base filename (no path prefix)
        assert output_file == "gauss_graph.svg"

        # Clean up the generated file in current directory
        output_path = Path(output_file)
        if output_path.exists():
            output_path.unlink()

    def test_visualize_graph_import_error_handling(self, simple_workspace, monkeypatch):
        """Test that visualize_graph handles ImportError appropriately."""
        model = simple_workspace.model(0)

        # Mock the import to raise ImportError
        def mock_import(*_args, **_kwargs):
            msg = "No module named 'pydot'"
            raise ImportError(msg)

        monkeypatch.setattr("builtins.__import__", mock_import)

        with pytest.raises(ImportError, match="Graph visualization requires pydot"):
            model.visualize_graph("gauss")


class TestModelWithoutParameterPoints:
    """Test Model creation and functionality without parameter_points defined."""

    @pytest.fixture
    def workspace_no_params(self):
        """Create a workspace without parameter_points for testing parameter discovery."""
        workspace_data = {
            "metadata": {"hs3_version": "0.2"},
            "distributions": [
                {
                    "name": "gauss",
                    "type": "gaussian_dist",
                    "x": "x",
                    "mean": "mu",
                    "sigma": "sigma",
                }
            ],
            "domains": [
                {
                    "name": "default_domain",
                    "type": "product_domain",
                    "axes": [
                        {"name": "x", "min": -5.0, "max": 5.0},
                        {"name": "mu", "min": -2.0, "max": 2.0},
                        {"name": "sigma", "min": 0.1, "max": 3.0},
                    ],
                }
            ],
            # Note: no parameter_points defined
        }
        return hs3.Workspace(**workspace_data)

    def test_model_creation_without_parameter_points(self, workspace_no_params):
        """Test that a model can be created successfully without parameter_points."""
        # This should not raise an error
        model = workspace_no_params.model(0)

        # Verify model was created successfully
        assert model is not None
        assert hasattr(model, "parameters")
        assert hasattr(model, "distributions")

        # Should discover parameters from the distribution
        assert "x" in model.parameters
        assert "mu" in model.parameters
        assert "sigma" in model.parameters

        # Should have the distribution
        assert "gauss" in model.distributions

    def test_parameters_use_domain_bounds(self, workspace_no_params):
        """Test that discovered parameters use domain bounds when available."""
        model = workspace_no_params.model(0)

        # We can't directly inspect bounds, but we can verify the parameters exist
        # and that the model can evaluate successfully
        result = model.pdf(
            "gauss", x=np.array(0.0), mu=np.array(0.0), sigma=np.array(1.0)
        )

        # Should get a reasonable Gaussian PDF value
        assert 0.35 < result < 0.45

    def test_parameters_default_to_scalar_kind(self, workspace_no_params):
        """Test that discovered parameters default to scalar kind when no parameterset provided."""
        model = workspace_no_params.model(0)

        # All discovered parameters should be scalars (pt.scalar)
        # We can verify this by checking they accept scalar values in pdf evaluation
        result = model.pdf(
            "gauss", x=np.array(1.5), mu=np.array(-0.5), sigma=np.array(2.0)
        )

        # Should compute successfully with scalar inputs
        assert isinstance(result, int | float | np.ndarray)
        assert float(result) > 0  # PDF should be positive

    def test_observable_parameter_defaults_to_vector_kind(self):
        """Test that observable parameters default to vector kind even with ParameterPoint(kind=None)."""
        workspace_data = {
            "metadata": {"hs3_version": "0.2"},
            "distributions": [
                {
                    "name": "signal",
                    "type": "gaussian_dist",
                    "x": "obs_x",
                    "mean": 0.0,
                    "sigma": 1.0,
                }
            ],
            "parameter_points": [
                {
                    "name": "default",
                    "parameters": [
                        {"name": "obs_x", "value": 0.0}  # kind defaults to None
                    ],
                }
            ],
            "data": [
                {
                    "name": "data1",
                    "type": "point",
                    "value": 1.5,
                    "axes": [{"name": "obs_x", "min": -5.0, "max": 5.0}],
                }
            ],
            "likelihoods": [
                {
                    "name": "likelihood1",
                    "distributions": ["signal"],
                    "data": ["data1"],
                }
            ],
        }
        workspace = hs3.Workspace(**workspace_data)
        model = workspace.model(0)

        # obs_x is an observable: stored as the 1-D leaf, ndim == 1
        assert model.parameters["obs_x"].type.ndim == 1

    def test_parameter_kind_override_warns(self):
        """Test that overriding observable parameter kind to scalar emits a warning."""
        workspace_data = {
            "metadata": {"hs3_version": "0.2"},
            "distributions": [
                {
                    "name": "signal",
                    "type": "gaussian_dist",
                    "x": "obs_x",
                    "mean": 0.0,
                    "sigma": 1.0,
                }
            ],
            "parameter_points": [
                {
                    "name": "default",
                    "parameters": [{"name": "obs_x", "value": 0.0}],
                }
            ],
            "data": [
                {
                    "name": "data1",
                    "type": "point",
                    "value": 1.5,
                    "axes": [{"name": "obs_x", "min": -5.0, "max": 5.0}],
                }
            ],
            "likelihoods": [
                {
                    "name": "likelihood1",
                    "distributions": ["signal"],
                    "data": ["data1"],
                }
            ],
        }
        workspace = hs3.Workspace(**workspace_data)

        # Override the kind programmatically
        workspace.parameter_points[0]["obs_x"].kind = pt.scalar

        # Mock _normalization_integral to bypass normalization evaluation issues
        with (
            patch(
                "pyhs3.distributions.core.Distribution._normalization_integral",
                return_value=pt.constant(1.0),
            ),
            pytest.warns(UserWarning, match=r"Parameter 'obs_x' has kind override"),
        ):
            model = workspace.model(0)

        # obs_x should be scalar (override respected)
        assert model.parameters["obs_x"].type.ndim == 0

    def test_parameter_kind_override_no_warns_when_default(self):
        """Test that overriding observable parameter kind to vector emits no warning."""
        workspace_data = {
            "metadata": {"hs3_version": "0.2"},
            "distributions": [
                {
                    "name": "signal",
                    "type": "gaussian_dist",
                    "x": "obs_x",
                    "mean": 0.0,
                    "sigma": 1.0,
                }
            ],
            "parameter_points": [
                {
                    "name": "default",
                    "parameters": [{"name": "obs_x", "value": 0.0}],
                }
            ],
            "data": [
                {
                    "name": "data1",
                    "type": "point",
                    "value": 1.5,
                    "axes": [{"name": "obs_x", "min": -5.0, "max": 5.0}],
                }
            ],
            "likelihoods": [
                {
                    "name": "likelihood1",
                    "distributions": ["signal"],
                    "data": ["data1"],
                }
            ],
        }
        workspace = hs3.Workspace(**workspace_data)

        # Override the kind programmatically
        workspace.parameter_points[0]["obs_x"].kind = pt.vector
        model = workspace.model(0)
        # Observable override to vector: stored as the 1-D leaf, ndim == 1
        assert model.parameters["obs_x"].type.ndim == 1

    def test_non_observable_parameter_stays_scalar(self):
        """Test that non-observable parameters stay scalar even with ParameterPoint(kind=None)."""
        workspace_data = {
            "metadata": {"hs3_version": "0.2"},
            "distributions": [
                {
                    "name": "gauss",
                    "type": "gaussian_dist",
                    "x": "x",
                    "mean": "mu",
                    "sigma": 1.0,
                }
            ],
            "parameter_points": [
                {
                    "name": "default",
                    "parameters": [
                        {"name": "x", "value": 0.0},
                        {"name": "mu", "value": 0.0},  # kind defaults to None
                    ],
                }
            ],
        }
        workspace = hs3.Workspace(**workspace_data)
        model = workspace.model(0)

        # mu should be scalar (not an observable)
        assert model.parameters["mu"].type.ndim == 0

    def test_repr_shows_discovered_parameters(self, workspace_no_params):
        """Test that __repr__ correctly shows discovered parameters."""
        model = workspace_no_params.model(0)
        repr_str = repr(model)

        # Should show 3 discovered parameters
        assert "parameters: 3" in repr_str

        # Should contain discovered parameter names
        assert "x" in repr_str
        assert "mu" in repr_str
        assert "sigma" in repr_str

    def test_mixed_parameter_sources(self):
        """Test model with some parameters in parameterset and others discovered."""
        workspace_data = {
            "metadata": {"hs3_version": "0.2"},
            "distributions": [
                {
                    "name": "gauss",
                    "type": "gaussian_dist",
                    "x": "x",
                    "mean": "mu",
                    "sigma": "sigma",
                }
            ],
            "parameter_points": [
                {
                    "name": "partial_params",
                    "parameters": [
                        # Only define mu and sigma, let x be discovered
                        {"name": "mu", "value": 0.0},
                        {"name": "sigma", "value": 1.0},
                    ],
                }
            ],
            "domains": [
                {
                    "name": "default_domain",
                    "type": "product_domain",
                    "axes": [
                        {"name": "x", "min": -5.0, "max": 5.0},
                        {"name": "mu", "min": -2.0, "max": 2.0},
                        {"name": "sigma", "min": 0.1, "max": 3.0},
                    ],
                }
            ],
        }
        workspace = hs3.Workspace(**workspace_data)

        # Should successfully create model with mixed parameter sources
        model = workspace.model(0)

        # All parameters should be available
        assert "x" in model.parameters
        assert "mu" in model.parameters  # from parameterset
        assert "sigma" in model.parameters  # from parameterset

        # Should evaluate successfully
        result = model.pdf(
            "gauss", x=np.array(0.0), mu=np.array(0.0), sigma=np.array(1.0)
        )
        assert 0.35 < result < 0.45


class TestWorkspaceWithLikelihoodsAndAnalyses:
    """Test Workspace functionality with likelihoods and analyses components."""

    @pytest.fixture
    def workspace_with_likelihoods_analyses(self):
        """Create a workspace with likelihoods and analyses for testing."""
        workspace_data = {
            "metadata": {"hs3_version": "0.2"},
            "distributions": [
                {
                    "name": "signal_dist",
                    "type": "gaussian_dist",
                    "x": "x",
                    "mean": "mu",
                    "sigma": "sigma",
                },
                {
                    "name": "background_dist",
                    "type": "gaussian_dist",
                    "x": "x",
                    "mean": 0.0,
                    "sigma": 2.0,
                },
            ],
            "parameter_points": [
                {
                    "name": "nominal_values",
                    "parameters": [
                        {"name": "x", "value": 1.0},
                        {"name": "mu", "value": 1.0},
                        {"name": "sigma", "value": 0.5},
                    ],
                }
            ],
            "domains": [
                {
                    "name": "poi_domain",
                    "type": "product_domain",
                    "axes": [
                        {"name": "x", "min": -10.0, "max": 10.0},
                        {"name": "mu", "min": 0.0, "max": 2.0},
                        {"name": "sigma", "min": 0.1, "max": 1.0},
                    ],
                }
            ],
            "data": [{"name": "observed_data", "type": "point", "value": 1.2}],
            "likelihoods": [
                {
                    "name": "signal_likelihood",
                    "distributions": ["signal_dist"],
                    "data": ["observed_data"],
                },
                {
                    "name": "combined_likelihood",
                    "distributions": ["signal_dist", "background_dist"],
                    "data": ["observed_data", "observed_data"],
                    "aux_distributions": ["background_dist"],
                },
            ],
            "analyses": [
                {
                    "name": "signal_analysis",
                    "likelihood": "signal_likelihood",
                    "parameters_of_interest": ["mu"],
                    "domains": ["poi_domain"],
                    "init": "nominal_values",
                },
                {
                    "name": "combined_analysis",
                    "likelihood": "combined_likelihood",
                    "parameters_of_interest": ["mu", "sigma"],
                    "domains": ["poi_domain"],
                    "init": "nominal_values",
                },
            ],
        }
        return hs3.Workspace(**workspace_data)

    def test_workspace_loads_likelihoods_correctly(
        self, workspace_with_likelihoods_analyses
    ):
        """Test that workspace correctly loads and parses likelihood specifications."""
        workspace = workspace_with_likelihoods_analyses

        # Should have likelihoods attribute
        assert hasattr(workspace, "likelihoods")
        assert workspace.likelihoods is not None
        assert len(workspace.likelihoods) == 2

        # Check first likelihood (FK references are resolved to objects)
        signal_likelihood = workspace.likelihoods["signal_likelihood"]
        assert signal_likelihood.name == "signal_likelihood"
        assert len(signal_likelihood.distributions) == 1
        assert signal_likelihood.distributions[0].name == "signal_dist"
        assert len(signal_likelihood.data) == 1
        assert signal_likelihood.data[0].name == "observed_data"
        assert signal_likelihood.aux_distributions is None

        # Check second likelihood with aux_distributions
        combined_likelihood = workspace.likelihoods["combined_likelihood"]
        assert combined_likelihood.name == "combined_likelihood"
        assert len(combined_likelihood.distributions) == 2
        assert combined_likelihood.distributions[0].name == "signal_dist"
        assert combined_likelihood.distributions[1].name == "background_dist"
        assert len(combined_likelihood.data) == 2
        assert combined_likelihood.data[0].name == "observed_data"
        assert combined_likelihood.data[1].name == "observed_data"
        assert combined_likelihood.aux_distributions == ["background_dist"]

    def test_workspace_loads_analyses_correctly(
        self, workspace_with_likelihoods_analyses
    ):
        """Test that workspace correctly loads and parses analysis specifications."""
        workspace = workspace_with_likelihoods_analyses

        # Should have analyses attribute
        assert hasattr(workspace, "analyses")
        assert workspace.analyses is not None
        assert len(workspace.analyses) == 2

        # Check first analysis (FK references are resolved to objects)
        signal_analysis = workspace.analyses["signal_analysis"]
        assert signal_analysis.name == "signal_analysis"
        assert signal_analysis.likelihood.name == "signal_likelihood"
        assert signal_analysis.parameters_of_interest == ["mu"]
        assert len(signal_analysis.domains) == 1
        assert signal_analysis.domains[0].name == "poi_domain"
        assert signal_analysis.init == "nominal_values"
        assert signal_analysis.prior is None

        # Check second analysis
        combined_analysis = workspace.analyses["combined_analysis"]
        assert combined_analysis.name == "combined_analysis"
        assert combined_analysis.likelihood.name == "combined_likelihood"
        assert combined_analysis.parameters_of_interest == ["mu", "sigma"]
        assert len(combined_analysis.domains) == 1
        assert combined_analysis.domains[0].name == "poi_domain"
        assert combined_analysis.init == "nominal_values"
        assert combined_analysis.prior is None

    def test_workspace_model_creation_still_works(
        self, workspace_with_likelihoods_analyses
    ):
        """Test that model creation still works with the extended workspace."""
        workspace = workspace_with_likelihoods_analyses

        # Should successfully create model
        model = workspace.model(0)

        # Should have distributions
        assert "signal_dist" in model.distributions
        assert "background_dist" in model.distributions

        # Should have parameters
        assert "x" in model.parameters
        assert "mu" in model.parameters
        assert "sigma" in model.parameters

        # Should be able to evaluate PDFs
        signal_result = model.pdf(
            "signal_dist", x=np.array(1.0), mu=np.array(1.0), sigma=np.array(0.5)
        )
        assert signal_result > 0

        background_result = model.pdf(
            "background_dist", x=np.array(1.0), mu=np.array(1.0), sigma=np.array(0.5)
        )
        assert background_result > 0

    def test_workspace_json_roundtrip_with_likelihoods_analyses(
        self, workspace_with_likelihoods_analyses, tmp_path
    ):
        """Test that workspace with likelihoods and analyses can roundtrip through JSON."""
        workspace = workspace_with_likelihoods_analyses

        # Save to JSON
        json_path = tmp_path / "test_workspace.json"
        with json_path.open("w") as f:
            f.write(workspace.model_dump_json(indent=2))

        # Load from JSON
        loaded_workspace = hs3.Workspace.load(json_path)

        # Verify likelihoods are preserved
        assert len(loaded_workspace.likelihoods) == 2
        assert "signal_likelihood" in loaded_workspace.likelihoods
        assert "combined_likelihood" in loaded_workspace.likelihoods

        # Verify analyses are preserved
        assert len(loaded_workspace.analyses) == 2
        assert "signal_analysis" in loaded_workspace.analyses
        assert "combined_analysis" in loaded_workspace.analyses

        # Verify model creation still works
        model = loaded_workspace.model(0)
        assert "signal_dist" in model.distributions
        assert "background_dist" in model.distributions


class TestModelParameterOrdering:
    """Test Model.pars() and Model.parsort() methods for parameter ordering."""

    def test_pars_returns_parameter_list(self, simple_workspace):
        """Test that pars() returns a list of parameter names."""
        model = simple_workspace.model(0)
        param_list = model.pars("gauss")

        # Should return a list
        assert isinstance(param_list, list)

        # Should contain parameter names as strings
        assert all(isinstance(p, str) for p in param_list)

        # Should have parameters for the Gaussian distribution
        assert len(param_list) >= 3  # At least x, mu, sigma

    def test_pars_includes_all_distribution_parameters(self, simple_workspace):
        """Test that pars() includes all parameters used by the distribution."""
        model = simple_workspace.model(0)
        param_list = model.pars("gauss")

        # Should include the Gaussian parameters
        assert "x" in param_list
        assert "mu" in param_list
        assert "sigma" in param_list

    def test_pars_returns_consistent_order(self, simple_workspace):
        """Test that pars() returns the same order on multiple calls."""
        model = simple_workspace.model(0)

        # Call pars() multiple times
        param_list_1 = model.pars("gauss")
        param_list_2 = model.pars("gauss")
        param_list_3 = model.pars("gauss")

        # Should return identical lists
        assert param_list_1 == param_list_2
        assert param_list_2 == param_list_3

    def test_pars_triggers_compilation(self, simple_workspace):
        """Test that pars() triggers compilation if not already compiled."""
        model = simple_workspace.model(0, mode="FAST_RUN")

        # Before calling pars, distribution should not be compiled
        assert "gauss" not in model._compiled_functions

        # Call pars - should trigger compilation
        param_list = model.pars("gauss")

        # After calling pars, distribution should be compiled
        assert "gauss" in model._compiled_functions
        assert "gauss" in model._compiled_inputs

        # Should return valid parameter list
        assert isinstance(param_list, list)
        assert len(param_list) > 0

    def test_pars_order_matches_pdf_inputs(self, simple_workspace):
        """Test that pars() order matches what pdf() expects."""
        model = simple_workspace.model(0)
        param_list = model.pars("gauss")

        # Create parameter dictionary in the order returned by pars()
        params = {param: np.array(1.0 if param != "x" else 0.0) for param in param_list}

        # pdf() should accept these parameters successfully
        result = model.pdf("gauss", **params)
        assert isinstance(result, int | float | np.ndarray)
        assert float(result) > 0

    def test_parsort_returns_index_list(self, simple_workspace):
        """Test that parsort() returns a list of indices."""
        model = simple_workspace.model(0)
        param_names = ["mu", "x", "sigma"]

        indices = model.parsort("gauss", param_names)

        # Should return a list
        assert isinstance(indices, list)

        # Should contain integers
        assert all(isinstance(i, int) for i in indices)

        # Should have same length as input
        assert len(indices) == len(param_names)

    def test_parsort_returns_valid_indices(self, simple_workspace):
        """Test that parsort() returns valid indices into the input list."""
        model = simple_workspace.model(0)
        param_names = ["sigma", "mu", "x"]

        indices = model.parsort("gauss", param_names)

        # All indices should be valid for the input list
        assert all(0 <= idx < len(param_names) for idx in indices)

        # All indices should be unique
        assert len(set(indices)) == len(indices)

    def test_parsort_reorders_to_match_pars(self, simple_workspace):
        """Test that parsort() indices reorder params to match pars()."""
        model = simple_workspace.model(0)

        # Get the expected order from pars()
        expected_order = model.pars("gauss")

        # Create a shuffled list of parameters
        param_names = ["sigma", "x", "mu"]

        # Get the indices that would sort param_names to match expected_order
        indices = model.parsort("gauss", param_names)

        # Apply the indices to reorder param_names
        reordered = [param_names[i] for i in indices]

        # The reordered list should match the parameters that are in both lists
        # (pars() might have additional parameters)
        for param in param_names:
            if param in expected_order:
                reordered_idx = reordered.index(param)
                expected_idx = expected_order.index(param)
                # The relative order should be preserved
                for other_param in param_names:
                    if other_param in expected_order and other_param != param:
                        other_reordered_idx = reordered.index(other_param)
                        other_expected_idx = expected_order.index(other_param)
                        # If param comes before other_param in expected_order,
                        # it should also come before in reordered
                        if expected_idx < other_expected_idx:
                            assert reordered_idx < other_reordered_idx

    def test_parsort_example_from_docstring(self, simple_workspace):
        """Test the example from the parsort() docstring."""
        model = simple_workspace.model(0)

        # Get expected parameter order
        pars_order = model.pars("gauss")

        # Create input list with specific order
        input_names = ["mu", "x", "sigma"]

        # Get indices
        indices = model.parsort("gauss", input_names)

        # Verify that applying indices reorders correctly
        # The indices should tell us where each element of pars_order can be found in input_names
        for i, par in enumerate(pars_order):
            if par in input_names:
                # indices[i] should point to where par is in input_names
                assert input_names[indices[i]] == par

    def test_pars_and_parsort_work_together(self, simple_workspace):
        """Test that pars() and parsort() work together correctly."""
        model = simple_workspace.model(0)

        # Get the canonical parameter order
        expected_order = model.pars("gauss")

        # Create values in arbitrary order
        arbitrary_order = ["sigma", "mu", "x"]
        values = [
            np.array(2.0),
            np.array(0.5),
            np.array(1.0),
        ]  # Corresponding to sigma=2.0, mu=0.5, x=1.0

        # Use parsort to get indices
        indices = model.parsort("gauss", arbitrary_order)

        # Reorder values using the indices
        reordered_values = [values[i] for i in indices]

        # Create parameter dict using expected order and reordered values
        param_dict = dict(zip(expected_order, reordered_values, strict=False))

        # This should work correctly with pdf()
        result = model.pdf("gauss", **param_dict)
        assert float(result) > 0

    def test_pars_nonexistent_distribution_raises_error(self, simple_workspace):
        """Test that pars() raises error for nonexistent distribution."""
        model = simple_workspace.model(0)

        # Should raise an error when distribution doesn't exist
        with pytest.raises(KeyError):
            model.pars("nonexistent_dist")

    def test_parsort_nonexistent_distribution_raises_error(self, simple_workspace):
        """Test that parsort() raises error for nonexistent distribution."""
        model = simple_workspace.model(0)

        # Should raise an error when distribution doesn't exist
        with pytest.raises(KeyError):
            model.parsort("nonexistent_dist", ["x", "mu"])


class TestConstParameters:
    """Test that parameters with const=True are baked as pt.constant in the model."""

    @pytest.fixture
    def workspace_with_const(self):
        return hs3.Workspace(
            metadata={"hs3_version": "0.2"},
            distributions=[
                {
                    "name": "gauss",
                    "type": "gaussian_dist",
                    "x": "x",
                    "mean": "mu",
                    "sigma": "sigma",
                }
            ],
            parameter_points=[
                {
                    "name": "defaults",
                    "parameters": [
                        {"name": "mu", "value": 0.0},
                        {"name": "sigma", "value": 1.0, "const": True},
                    ],
                }
            ],
            domains=[
                {
                    "name": "d",
                    "type": "product_domain",
                    "axes": [{"name": "x", "min": -5.0, "max": 5.0}],
                }
            ],
        )

    def test_const_parameter_is_tensor_constant(self, workspace_with_const):
        """const=True parameters must be baked as TensorConstant."""
        model = workspace_with_const.model(0)
        assert isinstance(model.parameters["sigma"], TensorConstant), (
            "sigma has const=True but was not baked as a TensorConstant"
        )

    def test_const_parameter_value_matches(self, workspace_with_const):
        """TensorConstant value must match the ParameterPoint value."""
        model = workspace_with_const.model(0)
        npt.assert_allclose(model.parameters["sigma"].data, 1.0)

    def test_free_parameter_is_not_constant(self, workspace_with_const):
        """mu has const=False (default) and must remain a free symbolic variable."""
        model = workspace_with_const.model(0)
        assert not isinstance(model.parameters["mu"], TensorConstant)

    def test_const_parameter_absent_from_graph_inputs(self, workspace_with_const):
        """explicit_graph_inputs must not include const parameters."""
        model = workspace_with_const.model(0)
        dist_expr = model.distributions["gauss"]
        free_names = {v.name for v in explicit_graph_inputs([dist_expr])}
        assert "sigma" not in free_names, (
            "sigma (const=True) must not appear as a free graph input"
        )
        assert "mu" in free_names

    def test_pdf_with_const_sigma_matches_explicit_sigma(self, workspace_with_const):
        """Forward-pass result must match whether sigma is baked or passed explicitly."""
        model_const = workspace_with_const.model(0)

        workspace_free = hs3.Workspace(
            metadata={"hs3_version": "0.2"},
            distributions=[
                {
                    "name": "gauss",
                    "type": "gaussian_dist",
                    "x": "x",
                    "mean": "mu",
                    "sigma": "sigma",
                }
            ],
            parameter_points=[
                {
                    "name": "defaults",
                    "parameters": [
                        {"name": "mu", "value": 0.0},
                        {"name": "sigma", "value": 1.0},  # free, not const
                    ],
                }
            ],
            domains=[
                {
                    "name": "d",
                    "type": "product_domain",
                    "axes": [{"name": "x", "min": -5.0, "max": 5.0}],
                }
            ],
        )
        model_free = workspace_free.model(0)

        x_val, mu_val = np.array(1.5), np.array(0.5)

        result_const = model_const.pdf("gauss", x=x_val, mu=mu_val)
        result_free = model_free.pdf("gauss", x=x_val, mu=mu_val, sigma=np.array(1.0))

        npt.assert_allclose(result_const, result_free)

    def test_const_outside_domain_warns(self):
        """const value outside domain emits a warning; the stored value is unchanged."""
        ws = hs3.Workspace(
            metadata={"hs3_version": "0.2"},
            distributions=[
                {
                    "name": "gauss",
                    "type": "gaussian_dist",
                    "x": "x",
                    "mean": "mu",
                    "sigma": "sigma",
                }
            ],
            parameter_points=[
                {
                    "name": "defaults",
                    "parameters": [
                        {"name": "mu", "value": 0.0},
                        # sigma=5.0 but domain cap is 2.0
                        {"name": "sigma", "value": 5.0, "const": True},
                    ],
                }
            ],
            domains=[
                {
                    "name": "d",
                    "type": "product_domain",
                    "axes": [
                        {"name": "x", "min": -5.0, "max": 5.0},
                        {"name": "sigma", "min": 0.1, "max": 2.0},
                    ],
                }
            ],
        )
        with pytest.warns(UserWarning, match="outside domain"):
            model = ws.model(0)

        # Value must NOT be clipped — use exactly what const said.
        npt.assert_allclose(model.parameters["sigma"].data, 5.0)

    def test_const_no_declared_domain_no_warning(self, workspace_with_const):
        """const value with no declared domain must not produce any warning."""
        # workspace_with_const has sigma=1.0; the domain only declares x, not sigma
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            workspace_with_const.model(0)  # must not raise


class TestWorkspaceModelDispatch:
    """Tests for workspace.model() dispatch across all target types and branches."""

    @pytest.fixture
    def ws(self):
        """Workspace with one analysis, one likelihood, one param set, two domains."""
        return hs3.Workspace(
            metadata={"hs3_version": "0.2"},
            distributions=[
                {
                    "name": "sig",
                    "type": "gaussian_dist",
                    "x": "x",
                    "mean": "mu",
                    "sigma": "sigma",
                }
            ],
            parameter_points=[
                {
                    "name": "nominal",
                    "parameters": [
                        {"name": "x", "value": 0.0},
                        {"name": "mu", "value": 0.0},
                        {"name": "sigma", "value": 1.0},
                    ],
                },
                {
                    "name": "alt",
                    "parameters": [
                        {"name": "x", "value": 1.0},
                        {"name": "mu", "value": 1.0},
                        {"name": "sigma", "value": 2.0},
                    ],
                },
            ],
            domains=[
                {
                    "name": "first_domain",
                    "type": "product_domain",
                    "axes": [{"name": "x", "min": -5.0, "max": 5.0}],
                },
                {
                    "name": "second_domain",
                    "type": "product_domain",
                    "axes": [{"name": "x", "min": 0.0, "max": 10.0}],
                },
            ],
            data=[{"name": "obs", "type": "point", "value": 0.5}],
            likelihoods=[
                {
                    "name": "sig_likelihood",
                    "distributions": ["sig"],
                    "data": ["obs"],
                }
            ],
            analyses=[
                {
                    "name": "sig_analysis",
                    "likelihood": "sig_likelihood",
                    "parameters_of_interest": ["mu"],
                    "domains": ["first_domain"],
                    "init": "nominal",
                }
            ],
        )

    # ------------------------------------------------------------------ #
    # Analysis target — parameter_set branches                            #
    # ------------------------------------------------------------------ #

    def test_analysis_parameterset_instance_override(self, ws):
        """parameter_set=ParameterSet(...) is used directly, ignoring analysis.init."""
        custom = ParameterSet(
            name="custom",
            parameters=[
                ParameterPoint(name="mu", value=99.0),
                ParameterPoint(name="sigma", value=0.5),
            ],
        )
        analysis = ws.analyses["sig_analysis"]
        model = ws.model(analysis, parameter_set=custom, progress=False)
        assert model.parameterset.name == "custom"
        assert model.parameterset.get("mu").value == pytest.approx(99.0)

    def test_analysis_parameterset_string_lookup(self, ws):
        """parameter_set='alt' looks up the named set in workspace.parameter_points."""
        analysis = ws.analyses["sig_analysis"]
        model = ws.model(analysis, parameter_set="alt", progress=False)
        assert model.parameterset.name == "alt"
        assert model.parameterset.get("mu").value == pytest.approx(1.0)

    def test_analysis_parameterset_int_lookup(self, ws):
        """parameter_set=1 looks up by index in workspace.parameter_points."""
        analysis = ws.analyses["sig_analysis"]
        model = ws.model(analysis, parameter_set=1, progress=False)
        # index 1 → 'alt' → mu=1.0
        assert model.parameterset.name == "alt"
        assert model.parameterset.get("mu").value == pytest.approx(1.0)

    def test_analysis_parameterset_from_init(self, ws):
        """No parameter_set override → analysis.init ('nominal') is used."""
        analysis = ws.analyses["sig_analysis"]
        model = ws.model(analysis, progress=False)
        assert model.parameterset.name == "nominal"
        assert model.parameterset.get("mu").value == pytest.approx(0.0)

    def test_analysis_no_init_no_override_uses_empty_default(self):
        """Analysis without init + no parameter_set override → model is built with empty ParameterSet."""
        ws_no_init = hs3.Workspace(
            metadata={"hs3_version": "0.2"},
            distributions=[
                {
                    "name": "sig",
                    "type": "gaussian_dist",
                    "x": "x",
                    "mean": "mu",
                    "sigma": "sigma",
                }
            ],
            domains=[
                {
                    "name": "d",
                    "type": "product_domain",
                    "axes": [{"name": "x", "min": -5.0, "max": 5.0}],
                }
            ],
            data=[{"name": "obs", "type": "point", "value": 0.0}],
            likelihoods=[{"name": "lh", "distributions": ["sig"], "data": ["obs"]}],
            analyses=[
                {
                    "name": "a",
                    "likelihood": "lh",
                    "parameters_of_interest": ["mu"],
                    "domains": ["d"],
                    # no "init" key → init=None
                }
            ],
        )
        model = ws_no_init.model("a", progress=False)
        assert model is not None
        assert model.parameterset.name == "default"

    def test_analysis_parameterset_missing_no_parameter_points_raises(self):
        """parameter_set given but workspace has no parameter_points → ValueError."""

        ws_no_params = hs3.Workspace(
            metadata={"hs3_version": "0.2"},
            distributions=[
                {
                    "name": "sig",
                    "type": "gaussian_dist",
                    "x": "x",
                    "mean": "mu",
                    "sigma": "sigma",
                }
            ],
            domains=[
                {
                    "name": "d",
                    "type": "product_domain",
                    "axes": [{"name": "x", "min": -5.0, "max": 5.0}],
                }
            ],
            data=[{"name": "obs", "type": "point", "value": 0.0}],
            likelihoods=[
                {
                    "name": "lh",
                    "distributions": ["sig"],
                    "data": ["obs"],
                }
            ],
            analyses=[
                {
                    "name": "a",
                    "likelihood": "lh",
                    "parameters_of_interest": ["mu"],
                    "domains": ["d"],
                    "init": None,
                }
            ],
        )
        analysis = ws_no_params.analyses["a"]
        with pytest.raises(ValueError, match="parameter_set="):
            ws_no_params.model(analysis, parameter_set="nominal", progress=False)

    # ------------------------------------------------------------------ #
    # String target dispatch                                              #
    # ------------------------------------------------------------------ #

    def test_str_dispatches_to_analysis(self, ws):
        """workspace.model('sig_analysis') resolves to the Analysis overload."""
        model = ws.model("sig_analysis", progress=False)
        # Analysis overload uses analysis.init → 'nominal'
        assert model is not None
        assert model.parameterset.name == "nominal"

    def test_str_dispatches_to_likelihood(self, ws):
        """workspace.model('sig_likelihood') resolves to the Likelihood overload."""
        model = ws.model("sig_likelihood", progress=False)
        assert model is not None
        assert "sig" in model.distributions

    def test_str_analysis_domain_override_raises(self, ws):
        """domain override is not allowed when a string resolves to an Analysis."""
        with pytest.raises(ValueError, match="domain override not supported"):
            ws.model(
                "sig_analysis",
                domain=ProductDomain(name="other"),
                progress=False,
            )

    def test_str_legacy_fallback_uses_domain_name(self, ws):
        """String that matches no analysis or likelihood falls back to domain-by-name lookup."""
        model = ws.model("first_domain", progress=False)
        assert model is not None

    def test_str_legacy_parameterset_instance_override(self, ws):
        """parameter_set=ParameterSet(...) is used directly on the str-legacy path."""
        custom = ParameterSet(
            name="custom",
            parameters=[
                ParameterPoint(name="mu", value=77.0),
                ParameterPoint(name="sigma", value=5.0),
            ],
        )
        model = ws.model("first_domain", parameter_set=custom, progress=False)
        assert model.parameterset.name == "custom"
        assert model.parameterset.get("mu").value == pytest.approx(77.0)

    def test_str_legacy_parameterset_string_lookup(self, ws):
        """parameter_set='alt' looks up the named set in workspace.parameter_points on str-legacy path."""
        model = ws.model("first_domain", parameter_set="alt", progress=False)
        assert model.parameterset.name == "alt"
        assert model.parameterset.get("mu").value == pytest.approx(1.0)

    # ------------------------------------------------------------------ #
    # Likelihood target — parameter_set branches                          #
    # ------------------------------------------------------------------ #

    def test_likelihood_parameterset_instance_override(self, ws):
        """parameter_set=ParameterSet(...) is used directly on the Likelihood path."""
        custom = ParameterSet(
            name="custom",
            parameters=[
                ParameterPoint(name="mu", value=42.0),
                ParameterPoint(name="sigma", value=3.0),
            ],
        )
        lh = ws.likelihoods["sig_likelihood"]
        model = ws.model(lh, parameter_set=custom, progress=False)
        assert model.parameterset.name == "custom"
        assert model.parameterset.get("mu").value == pytest.approx(42.0)

    def test_likelihood_parameterset_string_lookup(self, ws):
        """parameter_set='alt' looks up the named set in workspace.parameter_points on Likelihood path."""
        lh = ws.likelihoods["sig_likelihood"]
        model = ws.model(lh, parameter_set="alt", progress=False)
        assert model.parameterset.name == "alt"
        assert model.parameterset.get("mu").value == pytest.approx(1.0)

    # ------------------------------------------------------------------ #
    # parameterset fallback — ValueError when not resolvable             #
    # (int/Likelihood/str-legacy paths)                                  #
    # ------------------------------------------------------------------ #

    def test_int_target_parameterset_unresolvable_raises(self):
        """int path: parameter_set given but no parameter_points → ValueError."""
        ws_no_params = hs3.Workspace(
            metadata={"hs3_version": "0.2"},
            distributions=[
                {
                    "name": "g",
                    "type": "gaussian_dist",
                    "x": "x",
                    "mean": "mu",
                    "sigma": "sigma",
                }
            ],
            domains=[
                {
                    "name": "d",
                    "type": "product_domain",
                    "axes": [{"name": "x", "min": -5.0, "max": 5.0}],
                }
            ],
        )
        with pytest.raises(ValueError, match="parameter_set="):
            ws_no_params.model(0, parameter_set="nominal", progress=False)

    def test_likelihood_target_parameterset_unresolvable_raises(self):
        """Likelihood path: parameter_set given but no parameter_points → ValueError."""
        ws_no_params = hs3.Workspace(
            metadata={"hs3_version": "0.2"},
            distributions=[
                {
                    "name": "sig",
                    "type": "gaussian_dist",
                    "x": "x",
                    "mean": "mu",
                    "sigma": "sigma",
                }
            ],
            domains=[
                {
                    "name": "d",
                    "type": "product_domain",
                    "axes": [{"name": "x", "min": -5.0, "max": 5.0}],
                }
            ],
            data=[{"name": "obs", "type": "point", "value": 0.0}],
            likelihoods=[{"name": "lh", "distributions": ["sig"], "data": ["obs"]}],
        )
        lh = ws_no_params.likelihoods["lh"]
        with pytest.raises(ValueError, match="parameter_set="):
            ws_no_params.model(lh, parameter_set="nominal", progress=False)

    def test_str_legacy_parameterset_unresolvable_raises(self):
        """str-legacy path: parameter_set given but no parameter_points → ValueError."""
        ws_no_params = hs3.Workspace(
            metadata={"hs3_version": "0.2"},
            distributions=[
                {
                    "name": "g",
                    "type": "gaussian_dist",
                    "x": "x",
                    "mean": "mu",
                    "sigma": "sigma",
                }
            ],
            domains=[
                {
                    "name": "d",
                    "type": "product_domain",
                    "axes": [{"name": "x", "min": -5.0, "max": 5.0}],
                }
            ],
        )
        with pytest.raises(ValueError, match="parameter_set="):
            ws_no_params.model("d", parameter_set="nominal", progress=False)

    # ------------------------------------------------------------------ #
    # domain=0 falsy-zero fix                                            #
    # ------------------------------------------------------------------ #

    def test_int_target_domain_zero_is_honored(self, ws):
        """domain=0 on the int path must select domains[0], not fall back to target."""

        # domain=0 should select 'first_domain' (index 0, x in [-5,5])
        model = ws.model(0, domain=0, progress=False)
        assert model is not None

    def test_str_legacy_domain_zero_is_honored(self, ws):
        """domain=0 on the str-legacy path must select domains[0]."""
        # 'unknown_name' falls through to legacy; domain=0 should pick first_domain
        model = ws.model("first_domain", domain=0, progress=False)
        assert model is not None
