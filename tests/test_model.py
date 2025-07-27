from __future__ import annotations

import re

import pytest

import pyhs3 as hs3


@pytest.fixture
def simple_workspace():
    """Create a simple workspace for testing Model functionality."""
    workspace_data = {
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
    return hs3.Workspace(workspace_data)


class TestModelModes:
    """Test Model compilation modes and behavior."""

    @pytest.mark.parametrize("mode", ["FAST_RUN", "FAST_COMPILE"])
    def test_model_compilation_modes(self, simple_workspace, mode):
        """Test that models work correctly with different compilation modes."""
        model = simple_workspace.model(mode=mode)

        # Verify the mode is set correctly
        assert model.mode == mode

        # Test that we can evaluate PDF with both modes
        result = model.pdf("gauss", x=0.0, mu=0.0, sigma=1.0)

        # Should get a reasonable Gaussian PDF value at x=0, mu=0, sigma=1
        # This should be approximately 1/sqrt(2*pi) ≈ 0.3989
        assert 0.35 < result < 0.45

    def test_mode_compilation_differences(self, simple_workspace):
        """Test that both modes produce equivalent results."""
        model_fast_run = simple_workspace.model(mode="FAST_RUN")
        model_fast_compile = simple_workspace.model(mode="FAST_COMPILE")

        # Both should produce the same result for the same inputs
        result_fast_run = model_fast_run.pdf("gauss", x=1.0, mu=0.5, sigma=1.2)
        result_fast_compile = model_fast_compile.pdf("gauss", x=1.0, mu=0.5, sigma=1.2)

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
        model.pdf("gauss", x=0.0, mu=0.0, sigma=1.0)

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

    def test_visualize_graph_custom_output_file(self, simple_workspace, tmp_path):
        """Test visualize_graph with custom output filename."""
        model = simple_workspace.model()

        custom_file = tmp_path / "my_custom_graph.svg"
        output_file = model.visualize_graph("gauss", outfile=str(custom_file))

        # Should return the custom file path
        assert output_file == str(custom_file)

        # File should exist at the specified location
        assert custom_file.exists()

    def test_visualize_graph_nonexistent_distribution(self, simple_workspace):
        """Test that visualize_graph raises error for nonexistent distribution."""
        model = simple_workspace.model()

        with pytest.raises(ValueError, match="Distribution 'nonexistent' not found"):
            model.visualize_graph("nonexistent")

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
