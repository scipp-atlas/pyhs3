"""
Tests for HistFactory samples module.

Test SampleData validation and Samples collection functionality.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyhs3.distributions.histfactory.axes import Axes
from pyhs3.distributions.histfactory.samples import Sample, SampleData, Samples

hist = pytest.importorskip("hist", reason="hist not installed")


class TestSampleData:
    """Test SampleData validation and functionality."""

    def test_valid_sampledata_with_errors(self):
        """Test valid SampleData with matching contents and errors."""
        data = SampleData(contents=[1.0, 2.0, 3.0], errors=[0.1, 0.2, 0.3])
        assert data.contents == [1.0, 2.0, 3.0]
        assert data.errors == [0.1, 0.2, 0.3]

    def test_sampledata_requires_errors(self):
        """Test that SampleData requires errors field."""
        with pytest.raises(ValueError, match="Field required"):
            SampleData(contents=[1.0, 2.0, 3.0])  # Missing errors field

    def test_sampledata_mismatched_lengths_raises_error(self):
        """Test that SampleData raises error when contents and errors have different lengths."""
        with pytest.raises(
            ValueError,
            match="Sample data contents \\(3\\) and errors \\(2\\) must have same length",
        ):
            SampleData(contents=[1.0, 2.0, 3.0], errors=[0.1, 0.2])

    def test_sampledata_empty_contents_with_errors(self):
        """Test SampleData with empty contents and matching empty errors."""
        data = SampleData(contents=[], errors=[])
        assert data.contents == []
        assert data.errors == []

    def test_sampledata_empty_contents_mismatched_errors(self):
        """Test SampleData with empty contents but non-empty errors raises error."""
        with pytest.raises(
            ValueError,
            match="Sample data contents \\(0\\) and errors \\(1\\) must have same length",
        ):
            SampleData(contents=[], errors=[0.1])


class TestSample:
    """Test Sample functionality."""

    def test_sample_creation(self):
        """Test basic Sample creation."""
        sample = Sample(
            name="test_sample",
            data={"contents": [1.0, 2.0], "errors": [0.1, 0.2]},
        )
        assert sample.name == "test_sample"
        assert sample.data.contents == [1.0, 2.0]
        assert sample.data.errors == [0.1, 0.2]
        assert len(sample.modifiers) == 0


class TestSamples:
    """Test Samples collection functionality."""

    def test_samples_creation_empty(self):
        """Test empty Samples creation."""
        samples = Samples()
        assert len(samples) == 0

    def test_samples_creation_with_data(self):
        """Test Samples creation with initial data."""
        samples_data = [
            {
                "name": "signal",
                "data": {"contents": [5.0, 3.0], "errors": [1.0, 1.0]},
                "modifiers": [],
            },
            {
                "name": "background",
                "data": {"contents": [10.0, 8.0], "errors": [2.0, 2.0]},
                "modifiers": [],
            },
        ]
        samples = Samples(samples_data)
        assert len(samples) == 2

    def test_samples_getitem_by_index(self):
        """Test Samples.__getitem__ with integer index."""
        samples_data = [
            {
                "name": "signal",
                "data": {"contents": [5.0, 3.0], "errors": [1.0, 1.0]},
                "modifiers": [],
            },
            {
                "name": "background",
                "data": {"contents": [10.0, 8.0], "errors": [2.0, 2.0]},
                "modifiers": [],
            },
        ]
        samples = Samples(samples_data)

        # Test integer access
        assert samples[0].name == "signal"
        assert samples[1].name == "background"

    def test_samples_getitem_by_name(self):
        """Test Samples.__getitem__ with string name (sample name lookup)."""
        samples_data = [
            {
                "name": "signal",
                "data": {"contents": [5.0, 3.0], "errors": [1.0, 1.0]},
                "modifiers": [],
            },
            {
                "name": "background",
                "data": {"contents": [10.0, 8.0], "errors": [2.0, 2.0]},
                "modifiers": [],
            },
        ]
        samples = Samples(samples_data)

        # Test string name access
        signal_sample = samples["signal"]
        assert signal_sample.name == "signal"
        assert signal_sample.data.contents == [5.0, 3.0]

        background_sample = samples["background"]
        assert background_sample.name == "background"
        assert background_sample.data.contents == [10.0, 8.0]

    def test_samples_getitem_by_name_not_found(self):
        """Test Samples.__getitem__ with string name that doesn't exist."""
        samples_data = [
            {
                "name": "signal",
                "data": {"contents": [5.0, 3.0], "errors": [1.0, 1.0]},
                "modifiers": [],
            },
        ]
        samples = Samples(samples_data)

        # Test string name access for non-existent sample
        with pytest.raises(KeyError):
            samples["nonexistent"]

    def test_samples_contains(self):
        """Test Samples.__contains__ functionality."""
        samples_data = [
            {
                "name": "signal",
                "data": {"contents": [5.0, 3.0], "errors": [1.0, 1.0]},
                "modifiers": [],
            },
        ]
        samples = Samples(samples_data)

        assert "signal" in samples
        assert "background" not in samples

    def test_samples_iteration(self):
        """Test Samples.__iter__ functionality."""
        samples_data = [
            {
                "name": "signal",
                "data": {"contents": [5.0, 3.0], "errors": [1.0, 1.0]},
                "modifiers": [],
            },
            {
                "name": "background",
                "data": {"contents": [10.0, 8.0], "errors": [2.0, 2.0]},
                "modifiers": [],
            },
        ]
        samples = Samples(samples_data)

        names = [sample.name for sample in samples]
        assert names == ["signal", "background"]

    def test_samples_length(self):
        """Test Samples.__len__ functionality."""
        samples_data = [
            {
                "name": "signal",
                "data": {"contents": [5.0, 3.0], "errors": [1.0, 1.0]},
                "modifiers": [],
            },
            {
                "name": "background",
                "data": {"contents": [10.0, 8.0], "errors": [2.0, 2.0]},
                "modifiers": [],
            },
        ]
        samples = Samples(samples_data)

        assert len(samples) == 2


class TestSampleHistConversion:
    """Tests for Sample.to_hist() method."""

    def test_sample_to_hist_1d_regular_binning(self):
        """Test Sample.to_hist() with 1D regular binning."""
        sample = Sample(
            name="test_sample",
            data={"contents": [10.0, 20.0, 15.0], "errors": [3.0, 4.0, 2.5]},
        )
        axes = Axes([{"name": "x", "min": 0.0, "max": 3.0, "nbins": 3}])

        h = sample.to_hist(axes)

        # Check values
        assert np.array_equal(h.values(), [10.0, 20.0, 15.0])
        # Check variances (errors squared)
        expected_variances = np.square([3.0, 4.0, 2.5])
        assert np.allclose(h.variances(), expected_variances)
        # Check axis
        assert h.axes[0].name == "x"

    def test_sample_to_hist_1d_irregular_binning(self):
        """Test Sample.to_hist() with 1D irregular binning."""
        sample = Sample(
            name="test_sample",
            data={"contents": [10.0, 25.0, 5.0], "errors": [3.0, 5.0, 2.0]},
        )
        edges = [0.0, 10.0, 50.0, 100.0]
        axes = Axes([{"name": "pt", "edges": edges}])

        h = sample.to_hist(axes)

        # Check values and variances
        assert np.array_equal(h.values(), [10.0, 25.0, 5.0])
        assert np.allclose(h.variances(), np.square([3.0, 5.0, 2.0]))
        # Check edges
        assert np.array_equal(h.axes[0].edges, edges)

    def test_sample_to_hist_2d(self):
        """Test Sample.to_hist() with 2D histogram."""
        # 2x3 = 6 bins
        sample = Sample(
            name="test_2d",
            data={
                "contents": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "errors": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            },
        )
        axes = Axes(
            [
                {"name": "x", "min": 0.0, "max": 2.0, "nbins": 2},
                {"name": "y", "min": 0.0, "max": 3.0, "nbins": 3},
            ]
        )

        h = sample.to_hist(axes)

        # Check dimensions
        assert len(h.axes) == 2
        # Check reshaped values
        expected_2d = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        assert np.array_equal(h.values(), expected_2d)

    def test_sample_to_hist_empty_contents(self):
        """Test Sample.to_hist() with empty contents raises error."""
        # This will fail validation at SampleData level
        with pytest.raises(
            ValueError, match=r"Sample data contents.*must have same length"
        ):
            Sample(
                name="empty_sample",
                data={"contents": [], "errors": [1.0]},
            )

    def test_sample_to_hist_3d(self):
        """Test Sample.to_hist() with 3D histogram."""
        # 2x2x3 = 12 bins
        sample = Sample(
            name="test_3d",
            data={
                "contents": [float(i) for i in range(1, 13)],
                "errors": [0.1 * i for i in range(1, 13)],
            },
        )
        axes = Axes(
            [
                {"name": "x", "min": 0.0, "max": 2.0, "nbins": 2},
                {"name": "y", "min": 0.0, "max": 2.0, "nbins": 2},
                {"name": "z", "min": 0.0, "max": 3.0, "nbins": 3},
            ]
        )

        h = sample.to_hist(axes)

        # Check dimensions
        assert len(h.axes) == 3
        assert h.axes[0].name == "x"
        assert h.axes[1].name == "y"
        assert h.axes[2].name == "z"

        # Check that total is preserved
        assert h.values().sum() == sum(range(1, 13))
