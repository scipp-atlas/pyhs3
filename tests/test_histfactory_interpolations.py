"""
Tests for HistFactory interpolation functions.

Test coverage for all interpolation functions including previously uncovered ones
and comparison with pyhf implementations.
"""

from __future__ import annotations

import math

import numpy as np
import pyhf.interpolators
import pytensor.tensor as pt
import pytest
from pytensor import function

from pyhs3.distributions.histfactory.interpolations import (
    InterpolationError,
    apply_interpolation,
    interpolate_code0,
    interpolate_code1,
    interpolate_code2,
    interpolate_code4,
    interpolate_code4p,
    interpolate_exp,
    interpolate_lin,
    interpolate_log,
    interpolate_parabolic,
    interpolate_poly6,
)


class TestInterpolationFunctions:
    """Test the interpolation functions used by HistFactory modifiers."""

    def test_linear_interpolation(self):
        """Test linear interpolation function."""
        alpha = pt.dscalar("alpha")
        nom = pt.constant(1.0)
        hi = pt.constant(1.2)
        lo = pt.constant(0.8)

        result = interpolate_lin(alpha, nom, hi, lo)
        f = function([alpha], result)

        # Test key points
        assert np.isclose(f(0.0), 1.0)  # At nominal
        assert np.isclose(f(1.0), 1.2)  # At hi
        assert np.isclose(f(-1.0), 0.8)  # At lo
        assert np.isclose(f(0.5), 1.1)  # Halfway to hi
        assert np.isclose(f(-0.5), 0.9)  # Halfway to lo

    def test_log_interpolation(self):
        """Test logarithmic interpolation function."""
        alpha = pt.dscalar("alpha")
        nom = pt.constant(1.0)
        hi = pt.constant(1.2)
        lo = pt.constant(0.8)

        result = interpolate_log(alpha, nom, hi, lo)
        f = function([alpha], result)

        # Test key points
        assert np.isclose(f(0.0), 1.0)  # At nominal
        assert np.isclose(f(1.0), 1.2)  # At hi
        assert np.isclose(f(-1.0), 0.8)  # At lo

    def test_parabolic_interpolation(self):
        """Test parabolic interpolation function."""
        alpha = pt.dscalar("alpha")
        nom = pt.constant(1.0)
        hi = pt.constant(1.2)
        lo = pt.constant(0.8)

        result = interpolate_parabolic(alpha, nom, hi, lo)
        f = function([alpha], result)

        # Test key points
        assert np.isclose(f(0.0), 1.0)  # At nominal
        # Note: parabolic may not exactly hit hi/lo at ±1

    def test_poly6_interpolation(self):
        """Test 6th-order polynomial interpolation function."""
        alpha = pt.dscalar("alpha")
        nom = pt.constant(1.0)
        hi = pt.constant(1.2)
        lo = pt.constant(0.8)

        result = interpolate_poly6(alpha, nom, hi, lo)
        f = function([alpha], result)

        # Test key points
        assert np.isclose(f(0.0), 1.0)  # At nominal

    def test_apply_interpolation_method_selection(self):
        """Test that apply_interpolation selects the correct method."""
        alpha = pt.dscalar("alpha")
        nom = pt.constant(1.0)
        hi = pt.constant(1.2)
        lo = pt.constant(0.8)

        # Test each method
        for method in [
            "lin",
            "log",
            "exp",
            "parabolic",
            "poly6",
            "code0",
            "code1",
            "code2",
            "code4",
            "code4p",
        ]:
            result = apply_interpolation(method, alpha, nom, hi, lo)
            f = function([alpha], result)
            # Should at least work at nominal point
            assert np.isclose(f(0.0), 1.0)

        # Unknown method should now raise InterpolationError
        with pytest.raises(InterpolationError, match="unknown"):
            apply_interpolation("unknown", alpha, nom, hi, lo)


class TestInterpolationFunctionsCoverage:
    """Test uncovered interpolation functions."""

    def setup_method(self):
        """Set up test data."""
        self.alpha_values = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
        self.nom = 100.0
        self.hi = 120.0
        self.lo = 80.0

    def test_interpolate_exp(self):
        """Test interpolate_exp function."""
        for alpha_val in self.alpha_values:
            alpha = pt.constant(alpha_val)
            nom = pt.constant(self.nom)
            hi = pt.constant(self.hi)
            lo = pt.constant(self.lo)

            result = interpolate_exp(alpha, nom, hi, lo)
            result_val = result.eval()

            # Check that result is reasonable
            assert isinstance(result_val, (float, np.floating, np.ndarray))
            assert result_val > 0  # Should be positive for positive inputs

            # Check boundary conditions
            if alpha_val == 0:
                assert result_val == pytest.approx(self.nom)
            elif alpha_val == 1:
                assert result_val == pytest.approx(self.hi)
            elif alpha_val == -1:
                assert result_val == pytest.approx(self.lo)

    def test_interpolate_code0(self):
        """Test interpolate_code0 function (piecewise-linear additive)."""
        for alpha_val in self.alpha_values:
            alpha = pt.constant(alpha_val)
            nom = pt.constant(self.nom)
            hi = pt.constant(self.hi)
            lo = pt.constant(self.lo)

            result = interpolate_code0(alpha, nom, hi, lo)
            result_val = result.eval()

            # Check that result is reasonable
            assert isinstance(result_val, (float, np.floating, np.ndarray))

            # Check boundary conditions
            if alpha_val == 0:
                assert result_val == pytest.approx(self.nom)
            elif alpha_val == 1:
                assert result_val == pytest.approx(self.hi)
            elif alpha_val == -1:
                assert result_val == pytest.approx(self.lo)

            # Check linearity in each region
            if alpha_val >= 0:
                expected = self.nom + alpha_val * (self.hi - self.nom)
                assert result_val == pytest.approx(expected)
            else:
                expected = self.nom + alpha_val * (self.nom - self.lo)
                assert result_val == pytest.approx(expected)

    def test_interpolate_code1(self):
        """Test interpolate_code1 function (piecewise-exponential multiplicative)."""
        for alpha_val in self.alpha_values:
            alpha = pt.constant(alpha_val)
            nom = pt.constant(self.nom)
            hi = pt.constant(self.hi)
            lo = pt.constant(self.lo)

            result = interpolate_code1(alpha, nom, hi, lo)
            result_val = result.eval()

            # Check that result is reasonable
            assert isinstance(result_val, (float, np.floating, np.ndarray))
            assert result_val > 0  # Should be positive for positive inputs

            # Check boundary conditions
            if alpha_val == 0:
                assert result_val == pytest.approx(self.nom)
            elif alpha_val == 1:
                assert result_val == pytest.approx(self.hi)
            elif alpha_val == -1:
                assert result_val == pytest.approx(self.lo)

    def test_interpolate_code2(self):
        """Test interpolate_code2 function (quadratic interpolation with linear extrapolation)."""
        for alpha_val in self.alpha_values:
            alpha = pt.constant(alpha_val)
            nom = pt.constant(self.nom)
            hi = pt.constant(self.hi)
            lo = pt.constant(self.lo)

            result = interpolate_code2(alpha, nom, hi, lo)
            result_val = result.eval()

            # Check that result is reasonable
            assert isinstance(result_val, (float, np.floating, np.ndarray))

            # Check boundary conditions
            if alpha_val == 0:
                assert result_val == pytest.approx(self.nom)
            # Note: code2 (quadratic) doesn't necessarily hit exact hi/lo at ±1
            # because it uses quadratic interpolation in the central region

    def test_interpolation_vector_inputs(self):
        """Test interpolation functions with vector inputs."""
        alphas = pt.vector("alphas")
        nom = pt.constant(self.nom)
        hi = pt.constant(self.hi)
        lo = pt.constant(self.lo)

        alpha_vals = np.array([-1.0, 0.0, 1.0])

        # Test each function with vector inputs
        for func in [interpolate_exp, interpolate_code0, interpolate_code1]:
            result = func(alphas, nom, hi, lo)
            result_val = result.eval({alphas: alpha_vals})

            assert len(result_val) == 3
            assert result_val == pytest.approx(
                [self.lo, self.nom, self.hi]
            )  # alpha=[-1,0,1] should give [lo,nom,hi]

        # Test code2 separately since it doesn't hit exact hi/lo at ±1
        result = interpolate_code2(alphas, nom, hi, lo)
        result_val = result.eval({alphas: alpha_vals})
        assert len(result_val) == 3
        assert result_val[1] == pytest.approx(self.nom)  # alpha=0 should give nom

    def test_interpolation_edge_cases(self):
        """Test interpolation functions with edge cases."""
        alpha = pt.constant(0.5)

        # Test with zero nominal
        nom_zero = pt.constant(0.0)
        hi = pt.constant(1.0)
        lo = pt.constant(-1.0)

        # code0 and code2 should work fine with zero nominal
        result_code0 = interpolate_code0(alpha, nom_zero, hi, lo)
        result_code2 = interpolate_code2(alpha, nom_zero, hi, lo)

        assert result_code0.eval() == 0.5  # 0 + 0.5 * (1 - 0)
        assert isinstance(result_code2.eval(), (float, np.floating, np.ndarray))

        # exp and code1 need special handling for zero nominal
        # They should still work mathematically
        result_exp = interpolate_exp(alpha, nom_zero + 1e-10, hi, lo)
        result_code1 = interpolate_code1(alpha, nom_zero + 1e-10, hi, lo)

        assert isinstance(result_exp.eval(), (float, np.floating, np.ndarray))
        assert isinstance(result_code1.eval(), (float, np.floating, np.ndarray))

    def test_interpolation_symmetry(self):
        """Test interpolation functions preserve certain symmetries."""
        alpha_pos = pt.constant(0.5)
        alpha_neg = pt.constant(-0.5)

        # For symmetric variations (hi/nom = nom/lo), some functions should be symmetric
        # Let's test with symmetric values
        nom_sym = pt.constant(100.0)
        hi_sym = pt.constant(125.0)  # 25% increase
        lo_sym = pt.constant(80.0)  # 20% decrease (not perfectly symmetric, but close)

        for func in [interpolate_code0, interpolate_code1]:
            result_pos = func(alpha_pos, nom_sym, hi_sym, lo_sym)
            result_neg = func(alpha_neg, nom_sym, hi_sym, lo_sym)

            # Results should be reasonable
            assert isinstance(result_pos.eval(), (float, np.floating, np.ndarray))
            assert isinstance(result_neg.eval(), (float, np.floating, np.ndarray))


class TestInterpolationComparison:
    """Compare interpolation results with reference implementations."""

    def test_code0_vs_linear_interpolation(self):
        """Test that code0 matches expected piecewise linear behavior."""
        alpha_vals = [-2.0, -0.5, 0.0, 0.5, 2.0]
        nom = 100.0
        hi = 120.0
        lo = 80.0

        for alpha_val in alpha_vals:
            alpha = pt.constant(alpha_val)
            nom_t = pt.constant(nom)
            hi_t = pt.constant(hi)
            lo_t = pt.constant(lo)

            result = interpolate_code0(alpha, nom_t, hi_t, lo_t)
            result_val = result.eval()

            # Manual calculation of expected result
            if alpha_val >= 0:
                expected = nom + alpha_val * (hi - nom)
            else:
                expected = nom + alpha_val * (nom - lo)

            assert result_val == pytest.approx(expected)

    def test_code1_vs_exponential_interpolation(self):
        """Test that code1 matches expected piecewise exponential behavior."""
        alpha_vals = [-2.0, -0.5, 0.0, 0.5, 2.0]
        nom = 100.0
        hi = 120.0
        lo = 80.0

        for alpha_val in alpha_vals:
            alpha = pt.constant(alpha_val)
            nom_t = pt.constant(nom)
            hi_t = pt.constant(hi)
            lo_t = pt.constant(lo)

            result = interpolate_code1(alpha, nom_t, hi_t, lo_t)
            result_val = result.eval()

            # Manual calculation of expected result
            if alpha_val >= 0:
                expected = nom * ((hi / nom) ** alpha_val)
            else:
                expected = nom * ((lo / nom) ** (-alpha_val))

            assert result_val == pytest.approx(expected)

    def test_exp_equals_code1(self):
        """Test that interpolate_exp is equivalent to interpolate_code1."""
        alpha_vals = [-2.0, -0.5, 0.0, 0.5, 2.0]
        nom = 100.0
        hi = 120.0
        lo = 80.0

        for alpha_val in alpha_vals:
            alpha = pt.constant(alpha_val)
            nom_t = pt.constant(nom)
            hi_t = pt.constant(hi)
            lo_t = pt.constant(lo)

            result_exp = interpolate_exp(alpha, nom_t, hi_t, lo_t)
            result_code1 = interpolate_code1(alpha, nom_t, hi_t, lo_t)

            # These should be identical
            assert result_exp.eval() == pytest.approx(result_code1.eval())

    def test_code2_quadratic_behavior(self):
        """Test that code2 exhibits quadratic behavior in the central region."""
        nom = 100.0
        hi = 120.0
        lo = 80.0

        # Test quadratic behavior for |alpha| < 1
        alpha_vals = [-0.9, -0.5, 0.0, 0.5, 0.9]

        nom_t = pt.constant(nom)
        hi_t = pt.constant(hi)
        lo_t = pt.constant(lo)

        # Calculate expected quadratic coefficients
        hi_delta = hi - nom
        lo_delta = lo - nom
        a = 0.5 * (hi_delta + lo_delta)
        b = 0.5 * (hi_delta - lo_delta)

        for alpha_val in alpha_vals:
            alpha = pt.constant(alpha_val)
            result = interpolate_code2(alpha, nom_t, hi_t, lo_t)
            result_val = result.eval()

            # Expected quadratic result
            expected = nom + a * alpha_val**2 + b * alpha_val

            assert result_val == pytest.approx(expected)

    def test_code2_linear_extrapolation(self):
        """Test that code2 uses quadratic interpolation for |alpha| ≤ 1 and linear extrapolation for |alpha| > 1."""
        nom = 100.0
        hi = 120.0
        lo = 80.0

        nom_t = pt.constant(nom)
        hi_t = pt.constant(hi)
        lo_t = pt.constant(lo)

        # Calculate coefficients
        hi_delta = hi - nom
        lo_delta = lo - nom
        a = 0.5 * (hi_delta + lo_delta)
        b = 0.5 * (hi_delta - lo_delta)

        # Test alpha = 1 (boundary case, should use quadratic)
        alpha = pt.constant(1.0)
        result = interpolate_code2(alpha, nom_t, hi_t, lo_t)
        result_val = result.eval()
        # Expected quadratic result: nom + a * alpha^2 + b * alpha
        expected = nom + a * (1.0) ** 2 + b * (1.0)
        assert result_val == pytest.approx(expected)

        # Test alpha > 1 (positive extrapolation)
        alpha_vals_pos = [1.5, 2.0]
        for alpha_val in alpha_vals_pos:
            alpha = pt.constant(alpha_val)
            result = interpolate_code2(alpha, nom_t, hi_t, lo_t)
            result_val = result.eval()

            # Expected linear extrapolation, continuous at alpha=1:
            # nom + (b + 2*a) * (alpha - 1) + (hi - nom)
            expected = nom + (b + 2 * a) * (alpha_val - 1) + hi_delta
            assert result_val == pytest.approx(expected)

        # Test alpha < -1 (negative extrapolation) - note: alpha = -1 uses quadratic
        alpha_vals_neg = [-1.5, -2.0]
        for alpha_val in alpha_vals_neg:
            alpha = pt.constant(alpha_val)
            result = interpolate_code2(alpha, nom_t, hi_t, lo_t)
            result_val = result.eval()

            # Expected linear extrapolation, continuous at alpha=-1:
            # nom + (b - 2*a) * (alpha + 1) + (lo - nom)
            expected = nom + (b - 2 * a) * (alpha_val + 1) + lo_delta
            assert result_val == pytest.approx(expected)

        # Test boundary case alpha = -1 (should use quadratic)
        alpha = pt.constant(-1.0)
        result = interpolate_code2(alpha, nom_t, hi_t, lo_t)
        result_val = result.eval()
        # Expected quadratic result: nom + a * alpha^2 + b * alpha
        expected = nom + a * (-1.0) ** 2 + b * (-1.0)
        assert result_val == pytest.approx(expected)


class TestInterpolationPyhfComparison:
    """Compare interpolation results with pyhf reference implementations."""

    def test_code0_vs_pyhf(self):
        """Test that interpolate_code0 matches pyhf.interpolators.code0."""

        alpha_vals = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]
        nom = 1.0
        hi = 2.0
        lo = 0.5

        # pyhf format: [nsysts, nsamples, nvariations, nbins] where nvariations = [down, nominal, up]
        histogramssets = [[[[lo], [nom], [hi]]]]

        interpolator = pyhf.interpolators.code0(histogramssets, subscribe=False)

        for alpha_val in alpha_vals:
            # Our implementation
            alpha = pt.constant(alpha_val)
            nom_t = pt.constant(nom)
            hi_t = pt.constant(hi)
            lo_t = pt.constant(lo)

            our_result = interpolate_code0(alpha, nom_t, hi_t, lo_t)
            our_val = our_result.eval()

            # pyhf implementation returns additive deltas
            alphasets = pyhf.tensorlib.astensor([[alpha_val]])
            pyhf_deltas = interpolator(alphasets)
            pyhf_result = nom + pyhf.tensorlib.tolist(pyhf_deltas)[0][0][0][0]

            assert our_val == pytest.approx(pyhf_result), (
                f"Mismatch at alpha={alpha_val}: our={our_val}, pyhf={pyhf_result}"
            )

    def test_code1_vs_pyhf(self):
        """Test that interpolate_code1 matches pyhf.interpolators.code1."""

        alpha_vals = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]
        nom = 1.0
        hi = 1.1
        lo = 0.9

        # pyhf format: [nsysts, nsamples, nvariations, nbins] where nvariations = [down, nominal, up]
        histogramssets = [[[[lo], [nom], [hi]]]]

        interpolator = pyhf.interpolators.code1(histogramssets, subscribe=False)

        for alpha_val in alpha_vals:
            # Our implementation
            alpha = pt.constant(alpha_val)
            nom_t = pt.constant(nom)
            hi_t = pt.constant(hi)
            lo_t = pt.constant(lo)

            our_result = interpolate_code1(alpha, nom_t, hi_t, lo_t)
            our_val = our_result.eval()

            # pyhf implementation returns multiplicative factors
            alphasets = pyhf.tensorlib.astensor([[alpha_val]])
            pyhf_factors = interpolator(alphasets)
            pyhf_result = nom * pyhf.tensorlib.tolist(pyhf_factors)[0][0][0][0]

            assert our_val == pytest.approx(pyhf_result), (
                f"Mismatch at alpha={alpha_val}: our={our_val}, pyhf={pyhf_result}"
            )

    def test_code2_vs_pyhf_slow_formula(self):
        """Test that interpolate_code2 matches pyhf's code2 formula within |alpha| <= 1."""
        # Based on pyhf's _slow_code2.summand implementation which is the reference.
        # Only the central quadratic region is compared: pyhf's extrapolation for
        # |alpha| > 1 omits the (a+b)/(a-b) offsets and is discontinuous at
        # alpha=+-1 (https://github.com/scikit-hep/pyhf/issues/2729), whereas
        # pyhs3 deliberately uses ROOT's continuous extrapolation instead.

        alpha_vals = [-1.0, -0.5, 0.0, 0.5, 1.0]
        nom = 1.0
        hi = 2.0
        lo = 0.5

        # pyhf slow formula coefficients (reference implementation)
        a = 0.5 * (hi + lo) - nom  # 0.25
        b = 0.5 * (hi - lo)  # 0.75

        for alpha_val in alpha_vals:
            # Our implementation
            alpha = pt.constant(alpha_val)
            nom_t = pt.constant(nom)
            hi_t = pt.constant(hi)
            lo_t = pt.constant(lo)

            our_result = interpolate_code2(alpha, nom_t, hi_t, lo_t)
            our_val = our_result.eval()

            # Expected result based on pyhf slow formula, |alpha| <= 1 branch
            delta = a * alpha_val * alpha_val + b * alpha_val

            expected = nom + delta

            assert our_val == pytest.approx(expected), (
                f"Mismatch at alpha={alpha_val}: our={our_val}, expected={expected}"
            )

    def test_code4_vs_pyhf(self):
        """Test that interpolate_code4 matches pyhf.interpolators.code4."""

        alpha_vals = [-2.0, -1.0, -0.5, -0.35, 0.0, 0.35, 0.5, 1.0, 2.0]
        nom = 100.0
        hi = 120.0
        lo = 80.0

        # pyhf format: [nsysts, nsamples, nvariations, nbins] where nvariations = [down, nominal, up]
        histogramssets = [[[[lo], [nom], [hi]]]]

        interpolator = pyhf.interpolators.code4(histogramssets, subscribe=False)

        for alpha_val in alpha_vals:
            # Our implementation
            alpha = pt.constant(alpha_val)
            nom_t = pt.constant(nom)
            hi_t = pt.constant(hi)
            lo_t = pt.constant(lo)

            our_result = interpolate_code4(alpha, nom_t, hi_t, lo_t)
            our_val = our_result.eval()

            # pyhf implementation returns multiplicative factors
            alphasets = pyhf.tensorlib.astensor([[alpha_val]])
            pyhf_factors = interpolator(alphasets)
            pyhf_result = nom * pyhf.tensorlib.tolist(pyhf_factors)[0][0][0][0]

            assert our_val == pytest.approx(pyhf_result), (
                f"Mismatch at alpha={alpha_val}: our={our_val}, pyhf={pyhf_result}"
            )

    def test_code4_nondefault_alpha0_vs_pyhf(self):
        """Test that interpolate_code4 honors a non-default alpha0.

        Regression: the polynomial coefficient matrix was hardcoded for
        alpha0=1, so alpha0=2.0 produced a polynomial discontinuous with the
        exponential branches at alpha=+-2.  Pin against pyhf (whose code4
        builds its inversion matrix from alpha0) across interior, boundary,
        and extrapolation values, and check continuity explicitly just inside
        both boundaries.
        """

        alpha0 = 2.0
        alpha_vals = [-2.5, -2.0, -1.5, -0.5, 0.0, 0.5, 1.5, 2.0, 2.5]
        nom = 100.0
        hi = 120.0
        lo = 80.0

        # pyhf format: [nsysts, nsamples, nvariations, nbins] where nvariations = [down, nominal, up]
        histogramssets = [[[[lo], [nom], [hi]]]]

        interpolator = pyhf.interpolators.code4(
            histogramssets, subscribe=False, alpha0=alpha0
        )

        nom_t = pt.constant(nom)
        hi_t = pt.constant(hi)
        lo_t = pt.constant(lo)

        for alpha_val in alpha_vals:
            our_result = interpolate_code4(
                pt.constant(alpha_val), nom_t, hi_t, lo_t, alpha0=alpha0
            )
            our_val = our_result.eval()

            # pyhf implementation returns multiplicative factors
            alphasets = pyhf.tensorlib.astensor([[alpha_val]])
            pyhf_factors = interpolator(alphasets)
            pyhf_result = nom * pyhf.tensorlib.tolist(pyhf_factors)[0][0][0][0]

            assert our_val == pytest.approx(pyhf_result), (
                f"Mismatch at alpha={alpha_val}: our={our_val}, pyhf={pyhf_result}"
            )

        # Continuity at +-alpha0: values immediately inside the polynomial
        # region must match the exponential branch value at the boundary.
        eps = 1e-6
        for boundary, ratio in ((alpha0, hi / nom), (-alpha0, lo / nom)):
            inside_alpha = boundary - math.copysign(eps, boundary)
            inside_val = interpolate_code4(
                pt.constant(inside_alpha), nom_t, hi_t, lo_t, alpha0=alpha0
            ).eval()
            boundary_val = nom * ratio**alpha0
            assert inside_val == pytest.approx(boundary_val, rel=1e-4), (
                f"Discontinuity at alpha={boundary}: "
                f"inside={inside_val}, boundary={boundary_val}"
            )

    def test_code4p_vs_pyhf(self):
        """Test that interpolate_code4p matches pyhf.interpolators.code4p."""

        alpha_vals = [-2.0, -1.0, -0.5, -0.35, 0.0, 0.35, 0.5, 1.0, 2.0]
        nom = 100.0
        hi = 120.0
        lo = 80.0

        # pyhf format: [nsysts, nsamples, nvariations, nbins] where nvariations = [down, nominal, up]
        histogramssets = [[[[lo], [nom], [hi]]]]

        interpolator = pyhf.interpolators.code4p(histogramssets, subscribe=False)

        for alpha_val in alpha_vals:
            # Our implementation
            alpha = pt.constant(alpha_val)
            nom_t = pt.constant(nom)
            hi_t = pt.constant(hi)
            lo_t = pt.constant(lo)

            our_result = interpolate_code4p(alpha, nom_t, hi_t, lo_t)
            our_val = our_result.eval()

            # pyhf implementation returns additive deltas
            alphasets = pyhf.tensorlib.astensor([[alpha_val]])
            pyhf_deltas = interpolator(alphasets)
            pyhf_result = nom + pyhf.tensorlib.tolist(pyhf_deltas)[0][0][0][0]

            assert our_val == pytest.approx(pyhf_result), (
                f"Mismatch at alpha={alpha_val}: our={our_val}, pyhf={pyhf_result}"
            )

    def test_code4p_nondefault_alpha0_boundary_continuity(self):
        """Test that interpolate_code4p honors a non-default alpha0.

        pyhf's code4p accepts no alpha0, so the linear extrapolation branches
        themselves are the reference: values immediately inside the polynomial
        region must match the linear branch value at alpha=+-alpha0.
        Regression for polynomial coefficients hardcoded to alpha0=1.
        """

        alpha0 = 2.0
        nom = 100.0
        hi = 120.0
        lo = 80.0

        nom_t = pt.constant(nom)
        hi_t = pt.constant(hi)
        lo_t = pt.constant(lo)

        eps = 1e-6
        boundary_cases = (
            (alpha0, nom + alpha0 * (hi - nom)),
            (-alpha0, nom - (-alpha0) * (lo - nom)),
        )
        for boundary, boundary_val in boundary_cases:
            inside_alpha = boundary - math.copysign(eps, boundary)
            inside_val = interpolate_code4p(
                pt.constant(inside_alpha), nom_t, hi_t, lo_t, alpha0=alpha0
            ).eval()
            assert inside_val == pytest.approx(boundary_val, rel=1e-4), (
                f"Discontinuity at alpha={boundary}: "
                f"inside={inside_val}, boundary={boundary_val}"
            )

    def test_mathematical_equivalence_code0(self):
        """Test that interpolate_code0 implements correct piecewise-linear behavior."""
        # Code0 should implement: nom + alpha * (hi - nom) for alpha >= 0
        #                        nom + alpha * (nom - lo) for alpha < 0
        alpha_vals = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]
        nom = 100.0
        hi = 120.0
        lo = 80.0

        for alpha_val in alpha_vals:
            # Our implementation
            alpha = pt.constant(alpha_val)
            nom_t = pt.constant(nom)
            hi_t = pt.constant(hi)
            lo_t = pt.constant(lo)

            our_result = interpolate_code0(alpha, nom_t, hi_t, lo_t)
            our_val = our_result.eval()

            # Expected mathematical result (piecewise linear)
            if alpha_val >= 0:
                expected = nom + alpha_val * (hi - nom)
            else:
                expected = nom + alpha_val * (nom - lo)

            assert our_val == pytest.approx(expected), (
                f"Mismatch at alpha={alpha_val}: our={our_val}, expected={expected}"
            )

    def test_mathematical_equivalence_code1(self):
        """Test that interpolate_code1 implements correct piecewise-exponential behavior."""
        # Code1 should implement: nom * (hi/nom)^alpha for alpha >= 0
        #                        nom * (lo/nom)^(-alpha) for alpha < 0
        alpha_vals = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]
        nom = 100.0
        hi = 120.0
        lo = 80.0

        for alpha_val in alpha_vals:
            # Our implementation
            alpha = pt.constant(alpha_val)
            nom_t = pt.constant(nom)
            hi_t = pt.constant(hi)
            lo_t = pt.constant(lo)

            our_result = interpolate_code1(alpha, nom_t, hi_t, lo_t)
            our_val = our_result.eval()

            # Expected mathematical result (piecewise exponential)
            if alpha_val >= 0:
                expected = nom * ((hi / nom) ** alpha_val)
            else:
                expected = nom * ((lo / nom) ** (-alpha_val))

            assert our_val == pytest.approx(expected), (
                f"Mismatch at alpha={alpha_val}: our={our_val}, expected={expected}"
            )

    def test_mathematical_equivalence_code2(self):
        """Test that interpolate_code2 implements correct quadratic+linear behavior."""
        # Code2 is quadratic for |alpha| <= 1, linear extrapolation beyond
        alpha_vals = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
        nom = 100.0
        hi = 120.0
        lo = 80.0

        # Calculate quadratic coefficients
        hi_delta = hi - nom
        lo_delta = lo - nom
        a = 0.5 * (hi_delta + lo_delta)
        b = 0.5 * (hi_delta - lo_delta)

        for alpha_val in alpha_vals:
            # Our implementation
            alpha = pt.constant(alpha_val)
            nom_t = pt.constant(nom)
            hi_t = pt.constant(hi)
            lo_t = pt.constant(lo)

            our_result = interpolate_code2(alpha, nom_t, hi_t, lo_t)
            our_val = our_result.eval()

            # Expected mathematical result
            if abs(alpha_val) <= 1.0:
                # Quadratic interpolation
                expected = nom + a * alpha_val**2 + b * alpha_val
            elif alpha_val > 1.0:
                # Linear extrapolation beyond +1, continuous at alpha=1
                expected = nom + (b + 2 * a) * (alpha_val - 1) + hi_delta
            else:
                # Linear extrapolation beyond -1, continuous at alpha=-1
                expected = nom + (b - 2 * a) * (alpha_val + 1) + lo_delta

            assert our_val == pytest.approx(expected), (
                f"Mismatch at alpha={alpha_val}: our={our_val}, expected={expected}"
            )

    def test_interpolation_method_mappings(self):
        """Test that apply_interpolation correctly maps method names."""
        alpha = pt.constant(0.5)
        nom = pt.constant(100.0)
        hi = pt.constant(120.0)
        lo = pt.constant(80.0)

        # Test that code4 and code1 work but give different results
        # (code4 uses polynomial interpolation, code1 uses exponential)
        result_code4 = apply_interpolation("code4", alpha, nom, hi, lo)
        result_code1 = interpolate_code1(alpha, nom, hi, lo)
        # Both should give reasonable results but they should be different
        assert result_code4.eval() == pytest.approx(
            109.74, abs=0.01
        )  # polynomial result
        assert result_code1.eval() == pytest.approx(
            109.54, abs=0.01
        )  # exponential result

        # Test that code4p maps to code2 (quadratic)
        result_code4p = apply_interpolation("code4p", alpha, nom, hi, lo)
        result_code2 = interpolate_code2(alpha, nom, hi, lo)
        assert result_code4p.eval() == pytest.approx(result_code2.eval())

        # Test that exp maps to code1
        result_exp = apply_interpolation("exp", alpha, nom, hi, lo)
        assert result_exp.eval() == pytest.approx(result_code1.eval())

    def test_interpolation_boundary_conditions(self):
        """Test that all interpolations satisfy key boundary conditions."""
        nom = 100.0
        hi = 120.0
        lo = 80.0

        nom_t = pt.constant(nom)
        hi_t = pt.constant(hi)
        lo_t = pt.constant(lo)

        # Test all interpolation methods at key points
        methods = ["lin", "log", "exp", "code0", "code1", "code2"]

        for method in methods:
            # At alpha = 0, should return nominal
            alpha_0 = pt.constant(0.0)
            result_0 = apply_interpolation(method, alpha_0, nom_t, hi_t, lo_t)
            assert result_0.eval() == pytest.approx(nom), (
                f"Method {method} failed at alpha=0"
            )

            # For linear and code0 methods, check exact hi/lo at ±1
            if method in ["lin", "code0"]:
                alpha_1 = pt.constant(1.0)
                alpha_m1 = pt.constant(-1.0)
                result_1 = apply_interpolation(method, alpha_1, nom_t, hi_t, lo_t)
                result_m1 = apply_interpolation(method, alpha_m1, nom_t, hi_t, lo_t)
                assert result_1.eval() == pytest.approx(hi), (
                    f"Method {method} failed at alpha=1"
                )
                assert result_m1.eval() == pytest.approx(lo), (
                    f"Method {method} failed at alpha=-1"
                )

            # For exponential methods, check exact hi/lo at ±1
            if method in ["log", "exp", "code1"]:
                alpha_1 = pt.constant(1.0)
                alpha_m1 = pt.constant(-1.0)
                result_1 = apply_interpolation(method, alpha_1, nom_t, hi_t, lo_t)
                result_m1 = apply_interpolation(method, alpha_m1, nom_t, hi_t, lo_t)
                assert result_1.eval() == pytest.approx(hi), (
                    f"Method {method} failed at alpha=1"
                )
                assert result_m1.eval() == pytest.approx(lo), (
                    f"Method {method} failed at alpha=-1"
                )

    def test_code4p_extrapolation_region(self):
        """Test code4p linear extrapolation for |alpha| >= 1.

        code4p uses simple linear extrapolation: nom + alpha * (hi/lo - nom)
        This is different from code2's quadratic-based extrapolation.
        """
        nom = 100.0
        hi = 120.0
        lo = 80.0

        nom_t = pt.constant(nom)
        hi_t = pt.constant(hi)
        lo_t = pt.constant(lo)

        hi_delta = hi - nom  # 20
        lo_delta = lo - nom  # -20

        # Test positive extrapolation (alpha >= 1)
        # Expected: nom + alpha * hi_delta
        for alpha_val in [1.0, 1.5, 2.0]:
            alpha = pt.constant(alpha_val)
            result = interpolate_code4p(alpha, nom_t, hi_t, lo_t, alpha0=1.0)

            # At alpha >= 1, code4p uses linear extrapolation
            expected = nom + alpha_val * hi_delta

            assert result.eval() == pytest.approx(expected), (
                f"code4p linear extrapolation failed at alpha={alpha_val}"
            )

        # Test negative extrapolation (alpha <= -1)
        # Expected: nom - alpha * lo_delta (note the minus sign from bug fix!)
        for alpha_val in [-1.0, -1.5, -2.0]:
            alpha = pt.constant(alpha_val)
            result = interpolate_code4p(alpha, nom_t, hi_t, lo_t, alpha0=1.0)

            # At alpha <= -1, code4p uses: nom - alpha * lo_delta
            # This gives: nom - (-1.5) * (-20) = nom - 30 = 70 at alpha=-1.5
            expected = nom - alpha_val * lo_delta

            assert result.eval() == pytest.approx(expected), (
                f"code4p linear extrapolation failed at alpha={alpha_val}: "
                f"got {result.eval()}, expected {expected}"
            )

        # Specifically test the bug that was fixed
        # At alpha=-1.0, should give: nom - (-1.0) * lo_delta = 100 - (-1)*(-20) = 80 = lo
        alpha_m1 = pt.constant(-1.0)
        result = interpolate_code4p(alpha_m1, nom_t, hi_t, lo_t, alpha0=1.0)

        assert result.eval() == pytest.approx(lo), (
            "code4p at alpha=-1 should give lo value (bug fix verification)"
        )

        # At alpha=1.0, should give: nom + 1.0 * hi_delta = 100 + 20 = 120 = hi
        alpha_1 = pt.constant(1.0)
        result = interpolate_code4p(alpha_1, nom_t, hi_t, lo_t, alpha0=1.0)

        assert result.eval() == pytest.approx(hi), (
            "code4p at alpha=1 should give hi value"
        )

    def test_vector_interpolation_consistency(self):
        """Test that vector interpolation gives same results as scalar interpolation."""
        alphas = pt.vector("alphas")
        nom = pt.constant(100.0)
        hi = pt.constant(120.0)
        lo = pt.constant(80.0)

        alpha_vals = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])

        # Test each method
        methods = ["code0", "code1", "code2"]

        for method_name in methods:
            # Vector result
            if method_name == "code0":
                vector_result = interpolate_code0(alphas, nom, hi, lo)
            elif method_name == "code1":
                vector_result = interpolate_code1(alphas, nom, hi, lo)
            elif method_name == "code2":
                vector_result = interpolate_code2(alphas, nom, hi, lo)

            vector_vals = vector_result.eval({alphas: alpha_vals})

            # Scalar results
            scalar_vals = []
            for alpha_val in alpha_vals:
                alpha_scalar = pt.constant(alpha_val)
                if method_name == "code0":
                    scalar_result = interpolate_code0(alpha_scalar, nom, hi, lo)
                elif method_name == "code1":
                    scalar_result = interpolate_code1(alpha_scalar, nom, hi, lo)
                elif method_name == "code2":
                    scalar_result = interpolate_code2(alpha_scalar, nom, hi, lo)

                scalar_vals.append(float(scalar_result.eval()))

            assert vector_vals == pytest.approx(scalar_vals, rel=1e-6), (
                f"Vector/scalar mismatch for {method_name}"
            )


class TestCode4Regression:
    """Regression tests for HistFactory code4 interpolation."""

    @pytest.mark.parametrize("dtype", ["float32", "float64"])
    def test_code4_scalar_boundaries_and_extrapolation(self, dtype):
        alpha = pt.scalar("alpha", dtype=dtype)
        nom = pt.constant(100.0, dtype=dtype)
        hi = pt.constant(120.0, dtype=dtype)
        lo = pt.constant(80.0, dtype=dtype)

        result = interpolate_code4(alpha, nom, hi, lo)
        evaluate = function([alpha], result)

        expected = {
            -2.0: 64.0,
            -1.0: 80.0,
            0.0: 100.0,
            1.0: 120.0,
            2.0: 144.0,
        }

        tolerance = 1e-5 if dtype == "float32" else 1e-12

        for alpha_value, expected_value in expected.items():
            evaluated = evaluate(alpha_value)

            assert np.asarray(evaluated).dtype == np.dtype(dtype)
            assert evaluated == pytest.approx(
                expected_value,
                rel=tolerance,
                abs=tolerance,
            )

    @pytest.mark.parametrize("dtype", ["float32", "float64"])
    def test_code4_vector_values(self, dtype):
        alpha = pt.scalar("alpha", dtype=dtype)
        nom = pt.vector("nom", dtype=dtype)
        hi = pt.vector("hi", dtype=dtype)
        lo = pt.vector("lo", dtype=dtype)

        result = interpolate_code4(alpha, nom, hi, lo)
        evaluate = function([alpha, nom, hi, lo], result)

        nom_value = np.asarray([1.0, 2.0, 4.0], dtype=dtype)
        hi_value = np.asarray([1.2, 2.5, 5.0], dtype=dtype)
        lo_value = np.asarray([0.8, 1.5, 3.0], dtype=dtype)

        tolerance = 1e-5 if dtype == "float32" else 1e-12

        np.testing.assert_allclose(
            evaluate(0.0, nom_value, hi_value, lo_value),
            nom_value,
            rtol=tolerance,
            atol=tolerance,
        )
        np.testing.assert_allclose(
            evaluate(1.0, nom_value, hi_value, lo_value),
            hi_value,
            rtol=tolerance,
            atol=tolerance,
        )
        np.testing.assert_allclose(
            evaluate(-1.0, nom_value, hi_value, lo_value),
            lo_value,
            rtol=tolerance,
            atol=tolerance,
        )

    def test_code4_vector_alpha_matches_scalar_evaluation(self):
        alpha = pt.dvector("alpha")
        nom = pt.dvector("nom")
        hi = pt.dvector("hi")
        lo = pt.dvector("lo")

        vector_result = interpolate_code4(alpha, nom, hi, lo)

        alpha_values = np.asarray([-1.0, -0.5, 0.0, 0.5, 1.0])
        nom_values = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0])
        hi_values = nom_values * 1.2
        lo_values = nom_values * 0.8

        vector_values = vector_result.eval(
            {
                alpha: alpha_values,
                nom: nom_values,
                hi: hi_values,
                lo: lo_values,
            }
        )

        scalar_values = [
            float(
                interpolate_code4(
                    pt.constant(alpha_value),
                    pt.constant(nom_value),
                    pt.constant(hi_value),
                    pt.constant(lo_value),
                ).eval()
            )
            for alpha_value, nom_value, hi_value, lo_value in zip(
                alpha_values,
                nom_values,
                hi_values,
                lo_values,
                strict=True,
            )
        ]

        np.testing.assert_allclose(
            vector_values,
            scalar_values,
            rtol=1e-12,
            atol=1e-12,
        )

    def test_code4_matrix_alpha_broadcasts_against_vector_templates(self):
        """Cover rank alignment when alpha has more dimensions than templates."""
        alpha = pt.tensor("alpha", shape=(None, 1), dtype="float64")
        nom = pt.dvector("nom")
        hi = pt.dvector("hi")
        lo = pt.dvector("lo")

        result = interpolate_code4(alpha, nom, hi, lo)
        evaluate = function([alpha, nom, hi, lo], result)

        alpha_values = np.asarray([[-0.5], [0.0], [0.35], [0.8]])
        nom_values = np.asarray([1.0, 2.0, 4.0])
        hi_values = np.asarray([1.2, 2.6, 5.5])
        lo_values = np.asarray([0.8, 1.4, 2.7])

        actual = evaluate(alpha_values, nom_values, hi_values, lo_values)

        expected = np.asarray(
            [
                [
                    float(
                        interpolate_code4(
                            pt.constant(alpha_value),
                            pt.constant(nom_value),
                            pt.constant(hi_value),
                            pt.constant(lo_value),
                        ).eval()
                    )
                    for nom_value, hi_value, lo_value in zip(
                        nom_values, hi_values, lo_values, strict=True
                    )
                ]
                for alpha_value in alpha_values[:, 0]
            ]
        )

        assert actual.shape == (alpha_values.shape[0], nom_values.shape[0])
        np.testing.assert_allclose(
            actual,
            expected,
            rtol=1e-12,
            atol=1e-12,
        )

    def test_code4_known_interior_values(self):
        alpha = pt.dscalar("alpha")
        nom = pt.dvector("nom")
        hi = pt.dvector("hi")
        lo = pt.dvector("lo")

        result = interpolate_code4(alpha, nom, hi, lo)
        evaluate = function([alpha, nom, hi, lo], result)

        nom_value = np.asarray([1.0, 2.0, 3.0, 4.0])
        hi_value = np.asarray([1.2, 2.7, 4.1, 5.8])
        lo_value = np.asarray([0.8, 1.3, 2.2, 3.1])

        actual = evaluate(0.35, nom_value, hi_value, lo_value)

        # Use independent scalar evaluations so differently shaped hi/lo ratios
        # catch any accidental mixing of the polynomial coefficient axis.
        expected = np.asarray(
            [
                float(
                    interpolate_code4(
                        pt.constant(0.35),
                        pt.constant(nom),
                        pt.constant(hi),
                        pt.constant(lo),
                    ).eval()
                )
                for nom, hi, lo in zip(nom_value, hi_value, lo_value, strict=True)
            ]
        )

        np.testing.assert_allclose(
            actual,
            expected,
            rtol=1e-12,
            atol=1e-12,
        )


class TestCode2BoundaryContinuity:
    """Tests pinning interpolate_code2's continuous boundary behavior.

    interpolate_code2's linear extrapolation branches (|alpha| > 1) include the
    "+ (hi - nom)" / "+ (lo - nom)" offsets so the function is continuous at
    alpha=+-1, matching ROOT's FlexibleInterpVar code 2
    (RooFit::Detail::MathFuncs::flexibleInterpSingle, case 2). pyhf's code2
    interpolator omits these offsets and is discontinuous at the boundary;
    pyhs3 deliberately diverges from pyhf here in favor of continuity
    (https://github.com/scikit-hep/pyhf/issues/2729). These tests pin the
    numeric values just inside/outside the boundary so any regression back to
    the discontinuous form is caught explicitly.
    """

    def test_continuity_at_positive_boundary(self):
        """Pin values approaching alpha=1 from both sides: no jump."""
        nom = pt.constant(1.0)
        hi = pt.constant(2.0)
        lo = pt.constant(0.5)

        just_below = interpolate_code2(pt.constant(0.999), nom, hi, lo).eval()
        at_boundary = interpolate_code2(pt.constant(1.0), nom, hi, lo).eval()
        just_above = interpolate_code2(pt.constant(1.001), nom, hi, lo).eval()

        # quadratic: nom + a*alpha^2 + b*alpha with a=0.25, b=0.75
        assert just_below == pytest.approx(1.99875025)
        assert at_boundary == pytest.approx(2.0)
        # linear extrapolation continues from hi: nom + (b+2a)(alpha-1) + (a+b)
        assert just_above == pytest.approx(2.00125)
        # adjacent values stay within the expected local slope (~1.25 per unit alpha)
        assert abs(just_above - at_boundary) < 0.002
        assert abs(at_boundary - just_below) < 0.002

    def test_continuity_at_negative_boundary(self):
        """Pin values approaching alpha=-1 from both sides: no jump."""
        nom = pt.constant(1.0)
        hi = pt.constant(2.0)
        lo = pt.constant(0.5)

        just_above = interpolate_code2(pt.constant(-0.999), nom, hi, lo).eval()
        at_boundary = interpolate_code2(pt.constant(-1.0), nom, hi, lo).eval()
        just_below = interpolate_code2(pt.constant(-1.001), nom, hi, lo).eval()

        # quadratic: nom + a*alpha^2 + b*alpha with a=0.25, b=0.75
        assert just_above == pytest.approx(0.50025025)
        assert at_boundary == pytest.approx(0.5)
        # linear extrapolation continues from lo: nom + (b-2a)(alpha+1) + (a-b)
        assert just_below == pytest.approx(0.49975)
        # adjacent values stay within the expected local slope (~0.25 per unit alpha)
        assert abs(just_below - at_boundary) < 0.002
        assert abs(at_boundary - just_above) < 0.002

    def test_central_region_is_pure_quadratic(self):
        """The |alpha| <= 1 region is exactly nom + a*alpha^2 + b*alpha.

        The continuity fix only touches the extrapolation branches; the central
        quadratic region matches pyhf's code2 formula exactly.
        """
        nom_val, hi_val, lo_val = 1.0, 2.0, 0.5
        nom = pt.constant(nom_val)
        hi = pt.constant(hi_val)
        lo = pt.constant(lo_val)

        a = 0.5 * ((hi_val - nom_val) + (lo_val - nom_val))  # 0.25
        b = 0.5 * ((hi_val - nom_val) - (lo_val - nom_val))  # 0.75

        for alpha_val in [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]:
            result = interpolate_code2(pt.constant(alpha_val), nom, hi, lo).eval()
            expected = nom_val + a * alpha_val**2 + b * alpha_val
            assert result == expected, (
                f"alpha={alpha_val}: got {result}, expected {expected}"
            )

    def test_code2_equals_parabolic_everywhere(self):
        """interpolate_code2 is an alias for interpolate_parabolic: identical on a grid."""
        alphas = pt.vector("alphas")
        nom = pt.constant(100.0)
        hi = pt.constant(120.0)
        lo = pt.constant(80.0)

        alpha_grid = np.linspace(-3.0, 3.0, 121)

        code2_vals = interpolate_code2(alphas, nom, hi, lo).eval({alphas: alpha_grid})
        parabolic_vals = interpolate_parabolic(alphas, nom, hi, lo).eval(
            {alphas: alpha_grid}
        )

        np.testing.assert_array_equal(code2_vals, parabolic_vals)
