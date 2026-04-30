"""Integration tests for distribution normalization with Model and Workspace."""

from __future__ import annotations

import warnings

import numpy as np
import pytensor.tensor as pt
from pytensor.compile.function import function
from pytensor.graph.traversal import explicit_graph_inputs

from pyhs3 import Model, Workspace
from pyhs3.data import BinnedData, Data
from pyhs3.distributions import Distributions
from pyhs3.distributions.basic import GaussianDist
from pyhs3.distributions.mathematical import GenericDist
from pyhs3.domains import Domains, ProductDomain
from pyhs3.functions import Functions
from pyhs3.likelihoods import Likelihood, Likelihoods
from pyhs3.metadata import Metadata
from pyhs3.parameter_points import ParameterPoint, ParameterPoints, ParameterSet


class TestModelNormalization:
    """Tests for Model normalization integration."""

    def test_model_with_observables(self):
        """Model with observables normalizes distributions correctly."""
        # Create a simple GenericDist
        generic_dist = GenericDist(name="test_dist", expression="exp(c*x)")

        # Create Model with observables
        parameterset = ParameterSet(
            name="default",
            parameters=[
                ParameterPoint(name="c", value=-0.5),
            ],
        )
        distributions = Distributions([generic_dist])
        domain = ProductDomain(name="default")
        functions = Functions([])

        observables = {"x": (0.0, 10.0)}

        model = Model(
            parameterset=parameterset,
            distributions=distributions,
            domain=domain,
            functions=functions,
            progress=False,
            observables=observables,
        )

        # Get the compiled distribution
        dist_expr = model.distributions["test_dist"]

        # Create a function to evaluate it
        x_var = model.parameters["x"]
        c_var = model.parameters["c"]
        f = function([x_var, c_var], dist_expr)

        # x_var is the 1-D leaf — pass a plain 1-D array
        xs = np.linspace(0, 10, 10000)
        ys = f(xs, -0.5).squeeze()
        integral = np.trapezoid(ys, xs)

        # Should integrate to 1.0
        assert np.isclose(integral, 1.0, atol=1e-6)

    def test_model_with_two_observables(self):
        """Model with two observables normalizes distribution correctly."""
        # Create a simple GenericDist
        generic_dist = GenericDist(name="test_dist", expression="exp(c*x*y)")

        # Create Model with observables
        parameterset = ParameterSet(
            name="default",
            parameters=[
                ParameterPoint(name="c", value=-0.5),
            ],
        )
        distributions = Distributions([generic_dist])
        domain = ProductDomain(name="default")
        functions = Functions([])

        observables = {"x": (0.0, 10.0), "y": (-5.0, 5.0)}

        model = Model(
            parameterset=parameterset,
            distributions=distributions,
            domain=domain,
            functions=functions,
            progress=False,
            observables=observables,
        )

        # Get the compiled distribution
        dist_expr = model.distributions["test_dist"]

        # Both x and y are observables — model.parameters returns 1-D leaves
        x_var = model.parameters["x"]
        y_var = model.parameters["y"]
        c_var = model.parameters["c"]
        f = function([x_var, y_var, c_var], dist_expr)

        # Pass plain 1-D arrays
        xs = np.linspace(0, 10, 10000)
        ys = np.linspace(-5.0, 5.0, 10000)
        vals = f(xs, ys, -0.5).squeeze()
        integral = np.trapezoid(vals, xs)

        # Should integrate to 0.1 (integral over x gives 1.0, then over y gives
        # 1/(5-(-5)) = 0.1 for the joint normalization over [0,10] x [-5,5])
        assert np.isclose(integral, 0.1, atol=1e-6)

    def test_model_without_observables(self):
        """Model without observables doesn't normalize."""
        # Create a simple GenericDist
        generic_dist = GenericDist(name="test_dist", expression="exp(c*x)")

        # Create Model without observables
        parameterset = ParameterSet(
            name="default",
            parameters=[
                ParameterPoint(
                    name="x", value=0.0, kind=pt.vector
                ),  # for evaluating f(xs, -0.5) later
                ParameterPoint(name="c", value=-0.5),
            ],
        )
        distributions = Distributions([generic_dist])
        domain = ProductDomain(name="default")
        functions = Functions([])

        # Expect warning when overriding x to vector (not an observable)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            model = Model(
                parameterset=parameterset,
                distributions=distributions,
                domain=domain,
                functions=functions,
                progress=False,
                observables=None,
            )

        # Get the compiled distribution
        dist_expr = model.distributions["test_dist"]

        # Create a function to evaluate it
        x_var = model.parameters["x"]
        c_var = model.parameters["c"]
        f = function([x_var, c_var], dist_expr)

        # x is a non-observable vector override — pass a plain 1-D array
        xs = np.linspace(0, 10, 10000)
        ys = f(xs, -0.5).squeeze()
        integral = np.trapezoid(ys, xs)

        # Should NOT integrate to 1.0 (unnormalized)
        assert not np.isclose(integral, 1.0, atol=1e-6)


class TestWorkspaceNormalization:
    """Tests for Workspace normalization integration."""

    def test_workspace_computes_observables(self):
        """Workspace._compute_observables extracts observables from likelihoods+data."""
        # Create a minimal workspace with likelihoods and data
        workspace = Workspace(
            metadata=Metadata(hs3_version="0.3.0"),
            distributions=Distributions(
                [GenericDist(name="test_dist", expression="exp(c*x)")]
            ),
            data=Data(
                [
                    BinnedData(
                        name="test_data",
                        axes=[{"name": "x", "min": 0.0, "max": 10.0, "nbins": 10}],
                        contents=[1.0] * 10,
                    )
                ]
            ),
            likelihoods=Likelihoods(
                [
                    Likelihood(
                        name="test_likelihood",
                        distributions=["test_dist"],
                        data=["test_data"],
                    )
                ]
            ),
            domains=Domains([ProductDomain(name="default")]),
            parameter_points=ParameterPoints(
                [
                    ParameterSet(
                        name="default",
                        parameters=[ParameterPoint(name="c", value=-0.5)],
                    )
                ]
            ),
        )

        # Compute observables
        observables = workspace._compute_observables()

        # Should find x as an observable with bounds [0, 10]
        assert "x" in observables
        assert observables["x"] == (0.0, 10.0)

    def test_workspace_model_normalizes_generic_dist(self):
        """Workspace.model() creates Model with observables, normalizing GenericDist."""
        # Create a minimal workspace with likelihoods and data
        workspace = Workspace(
            metadata=Metadata(hs3_version="0.3.0"),
            distributions=Distributions(
                [GenericDist(name="test_dist", expression="exp(c*x)")]
            ),
            data=Data(
                [
                    BinnedData(
                        name="test_data",
                        axes=[{"name": "x", "min": 0.0, "max": 10.0, "nbins": 10}],
                        contents=[1.0] * 10,
                    )
                ]
            ),
            likelihoods=Likelihoods(
                [
                    Likelihood(
                        name="test_likelihood",
                        distributions=["test_dist"],
                        data=["test_data"],
                    )
                ]
            ),
            domains=Domains([ProductDomain(name="default")]),
            parameter_points=ParameterPoints(
                [
                    ParameterSet(
                        name="default",
                        parameters=[ParameterPoint(name="c", value=-0.5)],
                    )
                ]
            ),
        )

        # Create model
        model = workspace.model(0, progress=False)

        # Get the compiled distribution
        dist_expr = model.distributions["test_dist"]

        # Create a function to evaluate it
        x_var = model.parameters["x"]
        c_var = model.parameters["c"]
        f = function([x_var, c_var], dist_expr)

        # x is an observable — model.parameters["x"] is the 1-D leaf
        xs = np.linspace(0, 10, 10000)
        ys = f(xs, -0.5).squeeze()
        integral = np.trapezoid(ys, xs)

        # Should integrate to 1.0
        assert np.isclose(integral, 1.0, atol=1e-6)

    def test_workspace_without_likelihoods_no_normalization(self):
        """Workspace without likelihoods doesn't compute observables."""
        # Create a minimal workspace without likelihoods
        workspace = Workspace(
            metadata=Metadata(hs3_version="0.3.0"),
            distributions=Distributions(
                [GenericDist(name="test_dist", expression="exp(c*x)")]
            ),
            domains=Domains([ProductDomain(name="default")]),
            parameter_points=ParameterPoints(
                [
                    ParameterSet(
                        name="default",
                        parameters=[
                            ParameterPoint(
                                name="x", value=0.0, kind=pt.vector
                            ),  # for evaluating f(xs, -0.5) later
                            ParameterPoint(name="c", value=-0.5),
                        ],
                    )
                ]
            ),
        )

        # Compute observables
        observables = workspace._compute_observables()

        # Should be empty
        assert len(observables) == 0

        # Create model (suppress warning for kind override)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            model = workspace.model(0, progress=False)

        # Get the compiled distribution
        dist_expr = model.distributions["test_dist"]

        # Create a function to evaluate it
        x_var = model.parameters["x"]
        c_var = model.parameters["c"]
        f = function([x_var, c_var], dist_expr)

        # x is a non-observable vector override — pass a plain 1-D array
        xs = np.linspace(0, 10, 10000)
        ys = f(xs, -0.5).squeeze()
        integral = np.trapezoid(ys, xs)

        # Should NOT integrate to 1.0 (unnormalized)
        assert not np.isclose(integral, 1.0, atol=1e-6)

    def test_workspace_model_normalizes_gaussian_dist(self):
        """Workspace.model() normalizes GaussianDist over finite observable domain."""
        # Create a GaussianDist
        gaussian_dist = GaussianDist(name="gauss_dist", mean="mu", sigma="sigma", x="x")

        # Create a minimal workspace with likelihoods and data
        workspace = Workspace(
            metadata=Metadata(hs3_version="0.3.0"),
            distributions=Distributions([gaussian_dist]),
            data=Data(
                [
                    BinnedData(
                        name="test_data",
                        axes=[{"name": "x", "min": 100.0, "max": 160.0, "nbins": 60}],
                        contents=[1.0] * 60,
                    )
                ]
            ),
            likelihoods=Likelihoods(
                [
                    Likelihood(
                        name="test_likelihood",
                        distributions=["gauss_dist"],
                        data=["test_data"],
                    )
                ]
            ),
            domains=Domains([ProductDomain(name="default")]),
            parameter_points=ParameterPoints(
                [
                    ParameterSet(
                        name="default",
                        parameters=[
                            ParameterPoint(name="mu", value=130.0),
                            ParameterPoint(name="sigma", value=10.0),
                        ],
                    )
                ]
            ),
        )

        # Create model
        model = workspace.model(0, progress=False)

        # Get the compiled distribution
        dist_expr = model.distributions["gauss_dist"]

        # Create a function to evaluate it
        x_var = model.parameters["x"]
        mu_var = model.parameters["mu"]
        sigma_var = model.parameters["sigma"]
        f = function([x_var, mu_var, sigma_var], dist_expr)

        # x is an observable — model.parameters["x"] is the 1-D leaf
        xs = np.linspace(100, 160, 10000)
        ys = f(xs, 130.0, 10.0).squeeze()
        integral = np.trapezoid(ys, xs)

        # Should integrate to 1.0 (normalized over finite domain)
        assert np.isclose(integral, 1.0, atol=1e-6)


class TestNormalizationRegression:
    """Regression tests ensuring normalization correctness."""

    def test_normalization_substitutes_integration_variable(self):
        """The integration variable (leaf) must not leak into the normalized expression.

        After normalization, the denominator subgraph must be fully substituted
        to a constant — the observable leaf should not appear as a free input
        to the denominator.
        """
        generic_dist = GenericDist(name="test_dist", expression="exp(c*x)")

        parameterset = ParameterSet(
            name="default",
            parameters=[ParameterPoint(name="c", value=-0.5)],
        )
        model = Model(
            parameterset=parameterset,
            distributions=Distributions([generic_dist]),
            domain=ProductDomain(name="default"),
            functions=Functions([]),
            progress=False,
            observables={"x": (0.0, 10.0)},
        )

        dist_expr = model.distributions["test_dist"]

        # dist_expr is raw / integral (a True_div Apply node)
        assert dist_expr.owner is not None
        denominator = dist_expr.owner.inputs[1]

        # The integration variable must be fully substituted out of the denominator.
        # Only non-observable parameters (like "c") may remain as free inputs.
        denom_input_names = {
            v.name for v in explicit_graph_inputs([denominator]) if v.name
        }
        assert "x" not in denom_input_names
        assert denom_input_names <= {"c"}
