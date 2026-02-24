"""Integration tests for distribution normalization with Model and Workspace."""

from __future__ import annotations

import numpy as np
from pytensor.compile.function import function
from scipy.integrate import quad

from pyhs3.core import Model, Workspace
from pyhs3.data import BinnedData, Data
from pyhs3.distributions import Distributions
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

        # Integrate over the domain
        integral, _ = quad(lambda x: f(x, -0.5), 0, 10)

        # Should integrate to 1.0
        assert np.isclose(integral, 1.0, atol=1e-6)

    def test_model_without_observables(self):
        """Model without observables doesn't normalize."""
        # Create a simple GenericDist
        generic_dist = GenericDist(name="test_dist", expression="exp(c*x)")

        # Create Model without observables
        parameterset = ParameterSet(
            name="default",
            parameters=[
                ParameterPoint(name="c", value=-0.5),
            ],
        )
        distributions = Distributions([generic_dist])
        domain = ProductDomain(name="default")
        functions = Functions([])

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

        # Integrate over the domain
        integral, _ = quad(lambda x: f(x, -0.5), 0, 10)

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
        model = workspace.model(progress=False)

        # Get the compiled distribution
        dist_expr = model.distributions["test_dist"]

        # Create a function to evaluate it
        x_var = model.parameters["x"]
        c_var = model.parameters["c"]
        f = function([x_var, c_var], dist_expr)

        # Integrate over the domain
        integral, _ = quad(lambda x: f(x, -0.5), 0, 10)

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
                        parameters=[ParameterPoint(name="c", value=-0.5)],
                    )
                ]
            ),
        )

        # Compute observables
        observables = workspace._compute_observables()

        # Should be empty
        assert len(observables) == 0

        # Create model
        model = workspace.model(progress=False)

        # Get the compiled distribution
        dist_expr = model.distributions["test_dist"]

        # Create a function to evaluate it
        x_var = model.parameters["x"]
        c_var = model.parameters["c"]
        f = function([x_var, c_var], dist_expr)

        # Integrate over the domain
        integral, _ = quad(lambda x: f(x, -0.5), 0, 10)

        # Should NOT integrate to 1.0 (unnormalized)
        assert not np.isclose(integral, 1.0, atol=1e-6)
