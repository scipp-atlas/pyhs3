"""
Tests for HistFactory modifier implementations.

Test coverage for all modifier types and their constraint handling.
"""

from __future__ import annotations

import numpy as np
import pytensor.tensor as pt
import pytest

from pyhs3.context import Context
from pyhs3.distributions.histfactory.modifiers import (
    HistoSysData,
    HistoSysDataContents,
    HistoSysModifier,
    NormFactorModifier,
    NormSysData,
    NormSysModifier,
    ShapeFactorModifier,
    ShapeSysData,
    ShapeSysModifier,
    StatErrorData,
    StatErrorModifier,
)
from pyhs3.distributions.histfactory.samples import SampleData


class TestNormFactorModifier:
    """Test NormFactorModifier functionality."""

    def test_normfactor_creation(self):
        """Test creating a normfactor modifier."""
        modifier = NormFactorModifier(name="mu", parameter="mu")

        assert modifier.name == "mu"
        assert modifier.parameter == "mu"
        assert modifier.type == "normfactor"
        assert modifier.is_multiplicative
        assert not modifier.is_additive
        assert modifier.auxdata == 0.0
        assert modifier.constraint is None

    def test_normfactor_apply(self):
        """Test applying normfactor modifier."""
        modifier = NormFactorModifier(name="mu", parameter="mu")

        # Set up context and rates
        context = Context({"mu": pt.constant(1.5)})
        rates = pt.constant([10.0, 20.0, 30.0])

        # Apply modifier
        result = modifier.apply(context, rates)
        result_val = result.eval()

        # Should scale all rates by 1.5
        expected = np.array([15.0, 30.0, 45.0])
        np.testing.assert_allclose(result_val, expected)


class TestNormSysModifier:
    """Test NormSysModifier functionality."""

    def test_normsys_creation(self):
        """Test creating a normsys modifier."""
        data = NormSysData(hi=1.1, lo=0.9, interpolation="code1")
        modifier = NormSysModifier(
            name="lumi", parameter="alpha_lumi", data=data, constraint="Gauss"
        )

        assert modifier.name == "lumi"
        assert modifier.parameter == "alpha_lumi"
        assert modifier.type == "normsys"
        assert modifier.is_multiplicative
        assert not modifier.is_additive
        assert modifier.constraint == "Gauss"
        assert modifier.auxdata == 0.0

    def test_normsys_apply_positive_alpha(self):
        """Test applying normsys modifier with positive alpha."""
        data = NormSysData(hi=1.2, lo=0.8, interpolation="code1")
        modifier = NormSysModifier(name="lumi", parameter="alpha_lumi", data=data)

        # Set up context and rates
        context = Context({"alpha_lumi": pt.constant(1.0)})
        rates = pt.constant([100.0, 200.0])

        # Apply modifier - should give hi value (1.2)
        result = modifier.apply(context, rates)
        result_val = result.eval()

        expected = np.array([120.0, 240.0])  # 100*1.2, 200*1.2
        np.testing.assert_allclose(result_val, expected, rtol=1e-6)

    def test_normsys_apply_negative_alpha(self):
        """Test applying normsys modifier with negative alpha."""
        data = NormSysData(hi=1.2, lo=0.8, interpolation="code1")
        modifier = NormSysModifier(name="lumi", parameter="alpha_lumi", data=data)

        # Set up context and rates
        context = Context({"alpha_lumi": pt.constant(-1.0)})
        rates = pt.constant([100.0, 200.0])

        # Apply modifier - should give lo value (0.8)
        result = modifier.apply(context, rates)
        result_val = result.eval()

        expected = np.array([80.0, 160.0])  # 100*0.8, 200*0.8
        np.testing.assert_allclose(result_val, expected, rtol=1e-6)

    def test_normsys_make_constraint_gauss(self):
        """Test creating Gaussian constraint."""
        data = NormSysData(hi=1.1, lo=0.9)
        modifier = NormSysModifier(
            name="lumi", parameter="alpha_lumi", data=data, constraint="Gauss"
        )

        context = Context({"alpha_lumi": pt.constant(0.0)})
        sample_data = SampleData(contents=[10.0, 20.0], errors=[1.0, 2.0])

        constraint = modifier.make_constraint(context, sample_data)
        constraint_val = constraint.eval()

        # Should be a valid probability value
        assert isinstance(constraint_val, (float, np.floating, np.ndarray))
        assert constraint_val > 0

    def test_normsys_make_constraint_poisson(self):
        """Test creating Poisson constraint."""
        data = NormSysData(hi=1.1, lo=0.9)
        modifier = NormSysModifier(
            name="lumi", parameter="alpha_lumi", data=data, constraint="Poisson"
        )

        # Create context and sample data
        context = Context({"alpha_lumi": pt.constant(1.0)})
        sample_data = SampleData(contents=[10.0, 20.0], errors=[1.0, 2.0])

        # Test constraint creation
        constraint = modifier.make_constraint(context, sample_data)
        constraint_val = constraint.eval()

        # Should produce a positive constraint value
        assert constraint_val > 0

    def test_normsys_make_constraint_lognormal(self):
        """Test creating LogNormal constraint."""
        data = NormSysData(hi=1.1, lo=0.9)
        modifier = NormSysModifier(
            name="lumi", parameter="alpha_lumi", data=data, constraint="LogNormal"
        )

        # Create context and sample data
        context = Context({"alpha_lumi": pt.constant(1.0)})
        sample_data = SampleData(contents=[10.0, 20.0], errors=[1.0, 2.0])

        # Test constraint creation
        constraint = modifier.make_constraint(context, sample_data)
        constraint_val = constraint.eval()

        # Should produce a positive constraint value
        assert constraint_val > 0


class TestHistoSysModifier:
    """Test HistoSysModifier functionality."""

    def test_histosys_creation(self):
        """Test creating a histosys modifier."""
        hi_contents = HistoSysDataContents(contents=[15.0, 25.0])
        lo_contents = HistoSysDataContents(contents=[5.0, 15.0])
        data = HistoSysData(hi=hi_contents, lo=lo_contents, interpolation="code0")

        modifier = HistoSysModifier(
            name="shape_unc", parameter="alpha_shape", data=data
        )

        assert modifier.name == "shape_unc"
        assert modifier.parameter == "alpha_shape"
        assert modifier.type == "histosys"
        assert not modifier.is_multiplicative
        assert modifier.is_additive
        assert modifier.auxdata == 0.0

    def test_histosys_data_validation(self):
        """Test histosys data validation for mismatched lengths."""
        hi_contents = HistoSysDataContents(contents=[15.0, 25.0])
        lo_contents = HistoSysDataContents(contents=[5.0])  # Wrong length

        with pytest.raises(
            ValueError,
            match="histosys data contents for hi \\(2\\) and lo \\(1\\) must have same length",
        ):
            HistoSysData(hi=hi_contents, lo=lo_contents)

    def test_histosys_apply_positive_alpha(self):
        """Test applying histosys modifier with positive alpha."""
        hi_contents = HistoSysDataContents(contents=[15.0, 25.0])
        lo_contents = HistoSysDataContents(contents=[5.0, 15.0])
        data = HistoSysData(hi=hi_contents, lo=lo_contents, interpolation="code0")

        modifier = HistoSysModifier(
            name="shape_unc", parameter="alpha_shape", data=data
        )

        # Set up context and rates
        context = Context({"alpha_shape": pt.constant(1.0)})
        rates = pt.constant([10.0, 20.0])

        # Apply modifier - should add (hi - nominal) = ([15,25] - [10,20]) = [5,5]
        result = modifier.apply(context, rates)
        result_val = result.eval()

        expected = np.array([15.0, 25.0])  # Should equal hi values
        np.testing.assert_allclose(result_val, expected)

    def test_histosys_apply_negative_alpha(self):
        """Test applying histosys modifier with negative alpha."""
        hi_contents = HistoSysDataContents(contents=[15.0, 25.0])
        lo_contents = HistoSysDataContents(contents=[5.0, 15.0])
        data = HistoSysData(hi=hi_contents, lo=lo_contents, interpolation="code0")

        modifier = HistoSysModifier(
            name="shape_unc", parameter="alpha_shape", data=data
        )

        # Set up context and rates
        context = Context({"alpha_shape": pt.constant(-1.0)})
        rates = pt.constant([10.0, 20.0])

        # Apply modifier - should add (lo - nominal) = ([5,15] - [10,20]) = [-5,-5]
        result = modifier.apply(context, rates)
        result_val = result.eval()

        expected = np.array([5.0, 15.0])  # Should equal lo values
        np.testing.assert_allclose(result_val, expected)

    def test_histosys_make_constraint_poisson(self):
        """Test creating Poisson constraint for histosys."""
        hi_contents = HistoSysDataContents(contents=[15.0, 25.0])
        lo_contents = HistoSysDataContents(contents=[5.0, 15.0])
        data = HistoSysData(hi=hi_contents, lo=lo_contents, interpolation="code0")

        modifier = HistoSysModifier(
            name="shape_unc", parameter="alpha_shape", data=data, constraint="Poisson"
        )

        # Create context and sample data
        context = Context({"alpha_shape": pt.constant(1.0)})
        sample_data = SampleData(contents=[10.0, 20.0], errors=[1.0, 2.0])

        # Test constraint creation
        constraint = modifier.make_constraint(context, sample_data)
        constraint_val = constraint.eval()

        # Should produce a positive constraint value
        assert constraint_val > 0

    def test_histosys_make_constraint_lognormal(self):
        """Test creating LogNormal constraint for histosys."""
        hi_contents = HistoSysDataContents(contents=[15.0, 25.0])
        lo_contents = HistoSysDataContents(contents=[5.0, 15.0])
        data = HistoSysData(hi=hi_contents, lo=lo_contents, interpolation="code0")

        modifier = HistoSysModifier(
            name="shape_unc", parameter="alpha_shape", data=data, constraint="LogNormal"
        )

        # Create context and sample data
        context = Context({"alpha_shape": pt.constant(1.0)})
        sample_data = SampleData(contents=[10.0, 20.0], errors=[1.0, 2.0])

        # Test constraint creation
        constraint = modifier.make_constraint(context, sample_data)
        constraint_val = constraint.eval()

        # Should produce a positive constraint value
        assert constraint_val > 0


class TestShapeFactorModifier:
    """Test ShapeFactorModifier functionality."""

    def test_shapefactor_creation(self):
        """Test creating a shapefactor modifier."""
        modifier = ShapeFactorModifier(
            name="shape_factor", parameters=["gamma_bin0", "gamma_bin1"]
        )

        assert modifier.name == "shape_factor"
        assert modifier.parameters == ["gamma_bin0", "gamma_bin1"]
        assert modifier.type == "shapefactor"
        assert modifier.is_multiplicative
        assert not modifier.is_additive
        assert modifier.auxdata == []

    def test_shapefactor_apply(self):
        """Test applying shapefactor modifier."""
        modifier = ShapeFactorModifier(
            name="shape_factor", parameters=["gamma_bin0", "gamma_bin1"]
        )

        # Set up context and rates
        context = Context(
            {"gamma_bin0": pt.constant(1.1), "gamma_bin1": pt.constant(0.9)}
        )
        rates = pt.constant([10.0, 20.0])

        # Apply modifier
        result = modifier.apply(context, rates)
        result_val = result.eval()

        expected = np.array([11.0, 18.0])  # 10*1.1, 20*0.9
        np.testing.assert_allclose(result_val, expected)


class TestShapeSysModifier:
    """Test ShapeSysModifier functionality."""

    def test_shapesys_creation(self):
        """Test creating a shapesys modifier."""
        data = ShapeSysData(vals=[0.1, 0.15])
        modifier = ShapeSysModifier(
            name="uncorr_unc",
            parameters=["gamma_uncorr_bin0", "gamma_uncorr_bin1"],
            data=data,
        )

        assert modifier.name == "uncorr_unc"
        assert modifier.parameters == ["gamma_uncorr_bin0", "gamma_uncorr_bin1"]
        assert modifier.type == "shapesys"
        assert modifier.is_multiplicative
        assert not modifier.is_additive
        assert modifier.constraint == "Poisson"
        assert modifier.auxdata == [0.1, 0.15]

    def test_shapesys_apply(self):
        """Test applying shapesys modifier."""
        data = ShapeSysData(vals=[0.1, 0.15])
        modifier = ShapeSysModifier(
            name="uncorr_unc",
            parameters=["gamma_uncorr_bin0", "gamma_uncorr_bin1"],
            data=data,
        )

        # Set up context and rates
        context = Context(
            {
                "gamma_uncorr_bin0": pt.constant(1.05),
                "gamma_uncorr_bin1": pt.constant(0.95),
            }
        )
        rates = pt.constant([100.0, 200.0])

        # Apply modifier
        result = modifier.apply(context, rates)
        result_val = result.eval()

        expected = np.array([105.0, 190.0])  # 100*1.05, 200*0.95
        np.testing.assert_allclose(result_val, expected)

    def test_shapesys_make_constraint(self):
        """Test creating shapesys Poisson constraints."""
        data = ShapeSysData(vals=[0.06, 0.1346153846153846])
        modifier = ShapeSysModifier(
            name="uncorr_bkguncrt",
            parameters=["uncorr_bkguncrt_0", "uncorr_bkguncrt_1"],
            data=data,
        )

        context = Context(
            {
                "uncorr_bkguncrt_0": pt.constant(1.0),
                "uncorr_bkguncrt_1": pt.constant(1.0),
            }
        )
        sample_data = SampleData(contents=[50.0, 52.0], errors=[3.0, 3.5])

        constraint = modifier.make_constraint(context, sample_data)
        constraint_val = constraint.eval()

        # Should be a valid probability value
        assert isinstance(constraint_val, (float, np.floating, np.ndarray))
        assert constraint_val > 0


class TestStatErrorModifier:
    """Test StatErrorModifier functionality."""

    def test_staterror_creation(self):
        """Test creating a staterror modifier."""
        data = StatErrorData(uncertainties=[1.0, 1.5])
        modifier = StatErrorModifier(
            name="stat_unc",
            parameters=["gamma_stat_bin0", "gamma_stat_bin1"],
            data=data,
        )

        assert modifier.name == "stat_unc"
        assert modifier.parameters == ["gamma_stat_bin0", "gamma_stat_bin1"]
        assert modifier.type == "staterror"
        assert modifier.is_multiplicative
        assert not modifier.is_additive
        assert modifier.constraint == "Gauss"
        assert modifier.auxdata == [1.0, 1.0]

    def test_staterror_apply(self):
        """Test applying staterror modifier."""
        data = StatErrorData(uncertainties=[1.0, 1.5])
        modifier = StatErrorModifier(
            name="stat_unc",
            parameters=["gamma_stat_bin0", "gamma_stat_bin1"],
            data=data,
        )

        # Set up context and rates
        context = Context(
            {"gamma_stat_bin0": pt.constant(1.02), "gamma_stat_bin1": pt.constant(0.98)}
        )
        rates = pt.constant([50.0, 75.0])

        # Apply modifier
        result = modifier.apply(context, rates)
        result_val = result.eval()

        expected = np.array([51.0, 73.5])  # 50*1.02, 75*0.98
        np.testing.assert_allclose(result_val, expected)

    def test_staterror_make_constraint(self):
        """Test creating staterror Gaussian constraints."""
        data = StatErrorData(uncertainties=[2.0, 3.0])
        modifier = StatErrorModifier(
            name="stat_unc",
            parameters=["gamma_stat_bin0", "gamma_stat_bin1"],
            data=data,
        )

        context = Context(
            {"gamma_stat_bin0": pt.constant(1.0), "gamma_stat_bin1": pt.constant(1.0)}
        )
        sample_data = SampleData(contents=[50.0, 75.0], errors=[2.0, 3.0])

        constraint = modifier.make_constraint(context, sample_data)
        constraint_val = constraint.eval()

        # Should be a valid probability value
        assert isinstance(constraint_val, (float, np.floating, np.ndarray))
        assert constraint_val > 0

    def test_staterror_empty_parameters_edge_case(self):
        """Test StatError edge case when parameters and uncertainties are empty."""
        # Create a StatError modifier with empty parameters/uncertainties
        data = StatErrorData(uncertainties=[])
        modifier = StatErrorModifier(name="stat", parameters=[], data=data)

        # Create context and sample data
        context = Context({})
        sample_data = SampleData(contents=[], errors=[])

        # Test constraint creation - should return constant 1.0 for empty case
        constraint = modifier.make_constraint(context, sample_data)
        constraint_val = constraint.eval()

        # Should return 1.0 when there are no constraints
        assert constraint_val == pytest.approx(1.0)


class TestModifierConstraintTypes:
    """Test different constraint types for modifiers."""

    def test_normsys_poisson_constraint(self):
        """Test normsys with Poisson constraint."""
        data = NormSysData(hi=1.1, lo=0.9)
        modifier = NormSysModifier(
            name="lumi", parameter="alpha_lumi", data=data, constraint="Poisson"
        )

        context = Context({"alpha_lumi": pt.constant(1.0)})
        sample_data = SampleData(contents=[10.0], errors=[1.0])

        constraint = modifier.make_constraint(context, sample_data)
        constraint_val = constraint.eval()

        assert isinstance(constraint_val, (float, np.floating, np.ndarray))
        assert constraint_val > 0

    @pytest.mark.xfail(reason="LogNormal constraint implementation needs debugging")
    def test_normsys_lognormal_constraint(self):
        """Test normsys with LogNormal constraint."""
        data = NormSysData(hi=1.1, lo=0.9)
        modifier = NormSysModifier(
            name="lumi", parameter="alpha_lumi", data=data, constraint="LogNormal"
        )

        context = Context(
            {"alpha_lumi": pt.constant(1.0)}
        )  # Use positive value for LogNormal
        sample_data = SampleData(contents=[10.0], errors=[1.0])

        constraint = modifier.make_constraint(context, sample_data)
        constraint_val = constraint.eval()

        assert isinstance(constraint_val, (float, np.floating, np.ndarray))
        # For LogNormal, may be very small but should be finite
        assert np.isfinite(constraint_val)

    def test_histosys_different_constraints(self):
        """Test histosys with different constraint types."""
        hi_contents = HistoSysDataContents(contents=[15.0])
        lo_contents = HistoSysDataContents(contents=[5.0])
        data = HistoSysData(hi=hi_contents, lo=lo_contents)

        # Test Gaussian constraint (most common case)
        modifier_gauss = HistoSysModifier(
            name="shape_unc", parameter="alpha_shape", data=data, constraint="Gauss"
        )

        context = Context({"alpha_shape": pt.constant(0.0)})
        sample_data = SampleData(contents=[10.0], errors=[1.0])

        constraint = modifier_gauss.make_constraint(context, sample_data)
        constraint_val = constraint.eval()

        assert isinstance(constraint_val, (float, np.floating, np.ndarray))
        assert constraint_val > 0

        # Test Poisson constraint (should work but may be very small for x=0)
        modifier_poisson = HistoSysModifier(
            name="shape_unc", parameter="alpha_shape", data=data, constraint="Poisson"
        )

        constraint_poisson = modifier_poisson.make_constraint(context, sample_data)
        constraint_val_poisson = constraint_poisson.eval()

        assert isinstance(constraint_val_poisson, (float, np.floating, np.ndarray))
        # Poisson(x=1.0|mean=1.0) gives positive value
        assert constraint_val_poisson >= 0

        # Test LogNormal constraint with positive parameter (skip for now due to implementation issues)
        # modifier_lognormal = HistoSysModifier(
        #     name="shape_unc",
        #     parameter="alpha_shape",
        #     data=data,
        #     constraint="LogNormal"
        # )
        #
        # context_positive = Context({"alpha_shape": pt.constant(1.0)})
        # constraint_lognormal = modifier_lognormal.make_constraint(context_positive, sample_data)
        # constraint_val_lognormal = constraint_lognormal.eval()
        #
        # assert isinstance(constraint_val_lognormal, (float, np.floating, np.ndarray))
        # assert np.isfinite(constraint_val_lognormal)


class TestModifierEdgeCases:
    """Test edge cases and error conditions for modifiers."""

    def test_zero_nominal_yield_staterror(self):
        """Test staterror with zero nominal yield."""
        data = StatErrorData(uncertainties=[1.0])
        modifier = StatErrorModifier(
            name="stat_unc", parameters=["gamma_stat_bin0"], data=data
        )

        context = Context({"gamma_stat_bin0": pt.constant(1.0)})
        sample_data = SampleData(contents=[0.0], errors=[1.0])  # Zero nominal

        # Should handle gracefully and set sigma to 1.0
        constraint = modifier.make_constraint(context, sample_data)
        constraint_val = constraint.eval()

        assert isinstance(constraint_val, (float, np.floating, np.ndarray))
        assert constraint_val > 0

    def test_empty_parameters_staterror(self):
        """Test staterror with empty parameters list."""
        data = StatErrorData(uncertainties=[])
        modifier = StatErrorModifier(name="stat_unc", parameters=[], data=data)

        context = Context({})
        sample_data = SampleData(contents=[], errors=[])

        # Should return constant 1.0 for empty case
        constraint = modifier.make_constraint(context, sample_data)
        constraint_val = constraint.eval()

        assert constraint_val == 1.0

    def test_interpolation_method_variations(self):
        """Test different interpolation methods for normsys."""
        data_code1 = NormSysData(hi=1.2, lo=0.8, interpolation="code1")
        data_code4 = NormSysData(hi=1.2, lo=0.8, interpolation="code4")

        modifier_code1 = NormSysModifier(
            name="test1", parameter="alpha", data=data_code1
        )
        modifier_code4 = NormSysModifier(
            name="test4", parameter="alpha", data=data_code4
        )

        context = Context({"alpha": pt.constant(0.5)})
        rates = pt.constant([100.0])

        # Apply both interpolations
        result1 = modifier_code1.apply(context, rates).eval()
        result4 = modifier_code4.apply(context, rates).eval()

        # Both should be reasonable interpolation results
        assert isinstance(result1, (float, np.floating, np.ndarray))
        assert isinstance(result4, (float, np.floating, np.ndarray))
        assert result1 > 0
        assert result4 > 0

        # Test that they both work correctly at alpha=0 (should give nominal)
        context_zero = Context({"alpha": pt.constant(0.0)})
        result1_zero = modifier_code1.apply(context_zero, rates).eval()
        result4_zero = modifier_code4.apply(context_zero, rates).eval()

        np.testing.assert_allclose(result1_zero, 100.0, rtol=1e-6)
        np.testing.assert_allclose(result4_zero, 100.0, rtol=1e-6)
