from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

import pyhs3 as hs3


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
        model = simple_workspace.model(mode=mode)

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
        model_fast_run = simple_workspace.model(mode="FAST_RUN")
        model_fast_compile = simple_workspace.model(mode="FAST_COMPILE")

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
        model = simple_workspace.model()
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
        model = simple_workspace.model()
        repr_str = repr(model)

        # Should show 3 parameters (x, mu, sigma)
        assert "parameters: 3" in repr_str

        # Should show 1 distribution (gauss)
        assert "distributions: 1" in repr_str

        # Should show 0 functions (none in simple workspace)
        assert "functions: 0" in repr_str

    def test_repr_shows_entity_names(self, simple_workspace):
        """Test that __repr__ includes entity names."""
        model = simple_workspace.model()
        repr_str = repr(model)

        # Should contain parameter names
        assert "x" in repr_str
        assert "mu" in repr_str
        assert "sigma" in repr_str

        # Should contain distribution name
        assert "gauss" in repr_str

    def test_repr_with_custom_mode(self, simple_workspace):
        """Test __repr__ with different compilation mode."""
        model = simple_workspace.model(mode="FAST_COMPILE")
        repr_str = repr(model)

        assert "mode: FAST_COMPILE" in repr_str


class TestModelGraphSummary:
    """Test Model.graph_summary() method."""

    def test_graph_summary_basic_structure(self, simple_workspace):
        """Test that graph_summary returns expected format."""
        model = simple_workspace.model()
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
        model = simple_workspace.model()
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
        model_fast_run = simple_workspace.model(mode="FAST_RUN")
        model_fast_compile = simple_workspace.model(mode="FAST_COMPILE")

        # FAST_RUN should not show as compiled initially (until function is called)
        summary_fast_run = model_fast_run.graph_summary("gauss")
        assert "Mode: FAST_RUN" in summary_fast_run
        assert "Compiled: No" in summary_fast_run

        # FAST_COMPILE should show compiled status
        summary_fast_compile = model_fast_compile.graph_summary("gauss")
        assert "Mode: FAST_COMPILE" in summary_fast_compile

    def test_graph_summary_after_compilation(self, simple_workspace):
        """Test graph_summary shows compilation status after function is called."""
        model = simple_workspace.model(mode="FAST_RUN")

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
        model = simple_workspace.model()

        with pytest.raises(ValueError, match="Distribution 'nonexistent' not found"):
            model.graph_summary("nonexistent")

    def test_graph_summary_operation_types(self, simple_workspace):
        """Test that graph_summary includes operation type information."""
        model = simple_workspace.model()
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
        model = simple_workspace.model()

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
        model = simple_workspace.model()

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
        model = simple_workspace.model()

        custom_file = tmp_path / "my_custom_graph.svg"
        output_file = model.visualize_graph("gauss", outfile=str(custom_file))

        # Should return the custom file path
        assert output_file == str(custom_file)

        # File should exist at the specified location
        assert custom_file.exists()

    @pytest.mark.pydot
    def test_visualize_graph_nonexistent_distribution(self, simple_workspace):
        """Test that visualize_graph raises error for nonexistent distribution."""
        model = simple_workspace.model()

        with pytest.raises(ValueError, match="Distribution 'nonexistent' not found"):
            model.visualize_graph("nonexistent")

    @pytest.mark.pydot
    def test_visualize_graph_no_path_parameter(self, simple_workspace):
        """Test visualize_graph without path parameter (uses current directory)."""
        model = simple_workspace.model()

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
        model = simple_workspace.model()

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
        model = workspace_no_params.model()

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
        model = workspace_no_params.model()

        # We can't directly inspect bounds, but we can verify the parameters exist
        # and that the model can evaluate successfully
        result = model.pdf(
            "gauss", x=np.array(0.0), mu=np.array(0.0), sigma=np.array(1.0)
        )

        # Should get a reasonable Gaussian PDF value
        assert 0.35 < result < 0.45

    def test_parameters_default_to_scalar_kind(self, workspace_no_params):
        """Test that discovered parameters default to scalar kind when no parameterset provided."""
        model = workspace_no_params.model()

        # All discovered parameters should be scalars (pt.scalar)
        # We can verify this by checking they accept scalar values in pdf evaluation
        result = model.pdf(
            "gauss", x=np.array(1.5), mu=np.array(-0.5), sigma=np.array(2.0)
        )

        # Should compute successfully with scalar inputs
        assert isinstance(result, int | float | np.ndarray)
        assert float(result) > 0  # PDF should be positive

    def test_repr_shows_discovered_parameters(self, workspace_no_params):
        """Test that __repr__ correctly shows discovered parameters."""
        model = workspace_no_params.model()
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
        model = workspace.model()

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
        model = workspace.model()

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
        model = loaded_workspace.model()
        assert "signal_dist" in model.distributions
        assert "background_dist" in model.distributions


class TestModelParameterOrdering:
    """Test Model.pars() and Model.parsort() methods for parameter ordering."""

    def test_pars_returns_parameter_list(self, simple_workspace):
        """Test that pars() returns a list of parameter names."""
        model = simple_workspace.model()
        param_list = model.pars("gauss")

        # Should return a list
        assert isinstance(param_list, list)

        # Should contain parameter names as strings
        assert all(isinstance(p, str) for p in param_list)

        # Should have parameters for the Gaussian distribution
        assert len(param_list) >= 3  # At least x, mu, sigma

    def test_pars_includes_all_distribution_parameters(self, simple_workspace):
        """Test that pars() includes all parameters used by the distribution."""
        model = simple_workspace.model()
        param_list = model.pars("gauss")

        # Should include the Gaussian parameters
        assert "x" in param_list
        assert "mu" in param_list
        assert "sigma" in param_list

    def test_pars_returns_consistent_order(self, simple_workspace):
        """Test that pars() returns the same order on multiple calls."""
        model = simple_workspace.model()

        # Call pars() multiple times
        param_list_1 = model.pars("gauss")
        param_list_2 = model.pars("gauss")
        param_list_3 = model.pars("gauss")

        # Should return identical lists
        assert param_list_1 == param_list_2
        assert param_list_2 == param_list_3

    def test_pars_triggers_compilation(self, simple_workspace):
        """Test that pars() triggers compilation if not already compiled."""
        model = simple_workspace.model(mode="FAST_RUN")

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
        model = simple_workspace.model()
        param_list = model.pars("gauss")

        # Create parameter dictionary in the order returned by pars()
        params = {param: np.array(1.0 if param != "x" else 0.0) for param in param_list}

        # pdf() should accept these parameters successfully
        result = model.pdf("gauss", **params)
        assert isinstance(result, int | float | np.ndarray)
        assert float(result) > 0

    def test_parsort_returns_index_list(self, simple_workspace):
        """Test that parsort() returns a list of indices."""
        model = simple_workspace.model()
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
        model = simple_workspace.model()
        param_names = ["sigma", "mu", "x"]

        indices = model.parsort("gauss", param_names)

        # All indices should be valid for the input list
        assert all(0 <= idx < len(param_names) for idx in indices)

        # All indices should be unique
        assert len(set(indices)) == len(indices)

    def test_parsort_reorders_to_match_pars(self, simple_workspace):
        """Test that parsort() indices reorder params to match pars()."""
        model = simple_workspace.model()

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
        model = simple_workspace.model()

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
        model = simple_workspace.model()

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
        model = simple_workspace.model()

        # Should raise an error when distribution doesn't exist
        with pytest.raises(KeyError):
            model.pars("nonexistent_dist")

    def test_parsort_nonexistent_distribution_raises_error(self, simple_workspace):
        """Test that parsort() raises error for nonexistent distribution."""
        model = simple_workspace.model()

        # Should raise an error when distribution doesn't exist
        with pytest.raises(KeyError):
            model.parsort("nonexistent_dist", ["x", "mu"])
