from __future__ import annotations

import json
import math
from collections import OrderedDict

import networkx as nx
import numpy as np
import pytensor.tensor as pt
from pytensor import function as function

json_content = r"""
{
  "distributions": [
    {
      "coefficients": [
        "f"
      ],
      "extended": false,
      "name": "model",
      "summands": [
        "gx",
        "px"
      ],
      "type": "mixture_dist"
    },
    {
      "mean": "mean",
      "name": "gx",
      "sigma": "sigma",
      "type": "gaussian_dist",
      "x": "x"
    },
    {
      "mean": "mean2",
      "name": "px",
      "sigma": "sigma2",
      "type": "gaussian_dist",
      "x": "x"
    },
    {
      "coefficients": [
        "f_ctl"
      ],
      "extended": false,
      "name": "model_ctl",
      "summands": [
        "gx_ctl",
        "px_ctl"
      ],
      "type": "mixture_dist"
    },
    {
      "mean": "mean_ctl",
      "name": "gx_ctl",
      "sigma": "sigma",
      "type": "gaussian_dist",
      "x": "x"
    },
    {
      "mean": "mean2_ctl",
      "name": "px_ctl",
      "sigma": "sigma",
      "type": "gaussian_dist",
      "x": "x"
    }
  ],
  "domains": [
    {
      "axes": [
        {
          "max": 1.0,
          "min": 0.0,
          "name": "f"
        },
        {
          "max": 1.0,
          "min": 0.0,
          "name": "f_ctl"
        },
        {
          "max": 8.0,
          "min": -8.0,
          "name": "mean"
        },
        {
          "max": 3.0,
          "min": -3.0,
          "name": "mean2"
        },
        {
          "max": 3.0,
          "min": -3.0,
          "name": "mean2_ctl"
        },
        {
          "max": 8.0,
          "min": -8.0,
          "name": "mean_ctl"
        },
        {
          "max": 10.0,
          "min": 0.1,
          "name": "sigma"
        },
        {
          "max": 10.0,
          "min": 0.1,
          "name": "sigma2"
        },
        {
          "max": 8.0,
          "min": -8.0,
          "name": "x"
        }
      ],
      "name": "default_domain",
      "type": "product_domain"
    }
  ],
  "metadata": {
    "hs3_version": "0.2",
    "packages": [
      {
        "name": "ROOT",
        "version": "6.32.06"
      }
    ]
  },
  "misc": {
    "ROOT_internal": {
      "combined_distributions": {
        "simPdf": {
          "distributions": [
            "model_ctl",
            "model"
          ],
          "index_cat": "sample",
          "indices": [
            1,
            0
          ],
          "labels": [
            "control",
            "physics"
          ]
        }
      }
    }
  },
  "parameter_points": [
    {
      "name": "default_values",
      "parameters": [
        {
          "name": "f",
          "value": 0.2
        },
        {
          "name": "x",
          "value": 0.0
        },
        {
          "name": "mean",
          "value": 0.0
        },
        {
          "name": "sigma",
          "value": 0.3
        },
        {
          "name": "mean2",
          "value": 0.0
        },
        {
          "name": "sigma2",
          "value": 0.3
        },
        {
          "name": "f_ctl",
          "value": 0.5
        },
        {
          "name": "mean_ctl",
          "value": -3.0
        },
        {
          "name": "mean2_ctl",
          "value": -3.0
        }
      ]
    }
  ]
}
"""


class Workspace:
    """
    Manages the overall structure of the model including parameters, domains, and distributions.

    Args:
        data (dict): A dictionary containing model definitions including parameter points, distributions,
            and domains.

    Attributes:
        parameter_collection (ParameterCollection): Set of named parameter points.
        distribution_set (DistributionSet): All distributions used in the workspace.
        domain_collection (DomainCollection): Domain definitions for all parameters.
    """

    def __init__(self, data: dict):
        self.parameter_collection = ParameterCollection(data["parameter_points"])
        self.distribution_set = DistributionSet(data["distributions"])
        self.domain_collection = DomainCollection(data["domains"])

    def model(
        self,
        *,
        domain: int | str | DomainSet = 0,
        parameter_point: int | str | ParameterSet = 0,
    ) -> Model:
        """
        Constructs a `Model` object using the provided domain and parameter point.

        Args:
            domain (int | str | DomainSet): Identifier or object specifying the domain to use.
            parameter_point (int | str | ParameterSet): Identifier or object specifying the parameter values to use.

        Returns:
            Model: The constructed model object.
        """

        distlist = self.distribution_set
        assert set(parameter_point.points.keys()) == set(domain.domains.keys()), (
            "parameter and domain names do not match"
        )

        return Model(
            parameterset=parameter_point,
            distributions=self.distribution_set,
            domains=domain,
        )


class Model:
    """
    Represents a probabilistic model composed of parameters, domains, and distributions.

    Args:
        parameterset (ParameterSet): The parameter set used in the model.
        distributions (DistributionSet): Set of distributions to include.
        domains (DomainSet): Domain constraints for parameters.

    Attributes:
        parameters (dict[str, pt.TensorVariable]): Symbolic parameter variables.
        parameterset (ParameterSet): The original set of parameter values.
        distributions (dict[str, pt.TensorVariable]): Symbolic distribution expressions.
    """

    def __init__(
        self,
        *,
        parameterset: ParameterSet,
        distributions: DistributionSet,
        domains: DomainSet,
    ):
        self.parameters = {}
        self.parameterset = parameterset

        for parameter in parameterset:
            self.parameters[parameter.name] = boundedscalar(
                parameter.name, domains[parameter.name]
            )

        self.distributions = {}
        G = nx.DiGraph()
        for dist in distributions:
            G.add_node(dist.name, type="distribution")
            for parameter in dist.parameters:
                if not (
                    parameter in G and G.nodes[parameter]["type"] == "distribution"
                ):
                    G.add_node(parameter, type="parameter")
                G.add_edge(parameter, dist.name)

        for node in nx.topological_sort(G):
            if G.nodes[node]["type"] != "distribution":
                continue
            self.distributions[node] = distributions[node].expression(
                {**self.parameters, **self.distributions}
            )

    def pdf(self, name: str, **parametervalues: float):
        """
        Evaluates the probability density function of the specified distribution.

        Args:
            name (str): Name of the distribution to evaluate.
            **parametervalues (dict[str: float]): Values for each distribution parameter.

        Returns:
            float: The evaluated PDF value.
        """
        print(parametervalues)

        dist = self.distributions[name]
        # breakpoint()
        return dist.eval(self.parameters)

    def logpdf(self, name: str, **parametervalues: float):
        """
        Evaluates the natural logarithm of the PDF.

        Args:
            name (str): Name of the distribution to evaluate.
            **parametervalues (dict[str: float]): Values for each distribution parameter.

        Returns:
            float: The log of the PDF.
        """
        return np.log(self.pdf(name, **parametervalues))


class ParameterCollection:
    """
    A collection of named parameter sets.

    Args:
        parametersets (list): List of parameterset configurations.

    Attributes:
        sets (OrderedDict): Mapping from parameter set names to ParameterSet objects.
    """

    def __init__(self, parametersets: list):
        self.sets = OrderedDict()

        for parameterset_config in parametersets:
            parameterset = ParameterSet(
                parameterset_config["name"], parameterset_config["parameters"]
            )
            self.sets[parameterset.name] = parameterset

    def __getitem__(self, name: str) -> ParameterSet:
        if isinstance(name, str):
            return self.sets[name]
        return self.sets[self.sets.keys()[name]]


class ParameterSet:
    """
    Represents a single named set of parameter values.

    Args:
        name (str): Name of the parameter set.
        points (list): List of parameter point configurations.

    Attributes:
        name (str): Name of the parameter set.
        points (dict[str, ParameterPoint]): Mapping of parameter names to ParameterPoint objects.
    """

    def __init__(self, name: str, points: [ParameterPoint]):
        self.name = name

        self.points: dict[str, ParameterPoint] = OrderedDict()

        for points_config in points:
            point = ParameterPoint(points_config["name"], points_config["value"])
            self.points[point.name] = point

    def __getitem__(self, name: str) -> ParameterPoint:
        if isinstance(name, str):
            return self.points[name]
        return self.points[list(self.points.keys())[name]]

    def __iter__(self):
        return iter(self.points.values())


class ParameterPoint:
    """
    Represents a single parameter point.

    Attributes:
        name (str): Name of the parameter.
        value (float): Value of the parameter.
    """

    def __init__(self, name: str, value: float):
        self.name = name
        self.value = value


# @dataclass
# class ParameterPoint:
#     name: str
#     value: float
#     research python data classes


class DomainCollection:
    """
    Collection of named domain sets.

    Args:
        domainsets (list): List of domain set configurations.

    Attributes:
        domains (OrderedDict): Mapping of domain names to DomainSet objects.
    """

    def __init__(self, domainsets: [DomainSet]):
        self.domains = OrderedDict()

        for domain_config in domainsets:
            domain = DomainSet(
                domain_config["axes"], domain_config["name"], domain_config["type"]
            )
            self.domains[domain.name] = domain

    def __getitem__(self, item: str | int) -> DomainSet:
        if isinstance(item, int):
            return self.domains.keys()[item]
        return self.domains[item]


class DomainSet:
    """
    Represents a set of valid domains for parameters.

    Args:
        axes (list): List of domain configurations.
        name (str): Name of the domain set.
        type (str): Type of the domain.

    Attributes:
        domains (OrderedDict): Mapping of parameter names to allowed ranges.
    """

    def __init__(self, axes: [DomainPoint], name: str, type: str):
        self.name = name
        self.type = type
        self.domains = OrderedDict()

        for domain_config in axes:
            domain = DomainPoint(
                domain_config["name"], domain_config["min"], domain_config["max"]
            )
            self.domains[domain.name] = domain.range

    def __getitem__(self, item: int | str) -> (int, int):
        if isinstance(item, int):
            return list(self.domains.keys())[item]
        return self.domains[item]


class DomainPoint:
    """
    Represents a valid domain for a single parameter.

    Args:
        name (str): Name of the parameter.
        min (float): Minimum value.
        max (float): Maximum value.

    Attributes:
        range (tuple): Tuple containing (min, max).
        name (str): Name of the parameter.
        min (float): Minimum value.
        max (float): Maximum value.
    """

    def __init__(self, name: str, min: float, max: float):
        self.name = name
        self.min = min
        self.max = max
        self.range = (self.min, self.max)


class Distribution:
    """
    Base class for distributions.

    Args:
        name (str): Name of the distribution.
        type (str): Type identifier.

    Attributes:
        name (str): Name of the distribution.
        type (str): Type identifier.
        parameters (list[str]): initially empty list to be filled with parameter names.
    """

    def __init__(self, name: str, type: str):
        self.name = name
        self.type = type
        self.parameters = []


class GaussianDist(Distribution):
    """
    Subclass of Distribution representing a Gaussian distribution.

    Args:
        name (str): Name of the distribution.
        mean (str): Parameter name for the mean.
        sigma (str): Parameter name for the standard deviation.
        x (str): Input variable name.

    Attributes:
        name (str): Name of the distribution.
        mean (str): Parameter name for the mean.
        sigma (str): Parameter name for the standard deviation.
        x (str): Input variable name.
        parameters (list[str]): list containing mean, sigma, and x.
    """

    # need a way for the distribution to get the scalar function .parameter from parameterset
    def __init__(self, *, name: str, mean: str, sigma: str, x: str):
        super().__init__(name, "gaussian_dist")
        self.mean = mean
        self.sigma = sigma
        self.x = x
        self.parameters = [mean, sigma, x]

    def expression(self, distributionsandparameters: dict(str, pt.scalar)):
        """
        Builds a symbolic expression for the Gaussian PDF.

        Args:
            distributionsandparameters (dict): Mapping of names to pytensor variables.

        Returns:
            pt.TensorVariable: Symbolic representation of the Gaussian PDF.
        """
        # print("parameters: ", parameters)
        norm_const = 1.0 / (
            pt.sqrt(2 * math.pi) * distributionsandparameters[self.sigma]
        )
        exponent = pt.exp(
            -0.5
            * (
                (
                    distributionsandparameters[self.x]
                    - distributionsandparameters[self.mean]
                )
                / distributionsandparameters[self.sigma]
            )
            ** 2
        )
        return norm_const * exponent


class MixtureDist(Distribution):
    """
    Subclass of Distribution representing a mixture of distributions

    Args:
        name (str): Name of the distribution.
        coefficients (list): Coefficient parameter names.
        extended (bool): Whether the distribution is extended.
        summands (list): List of component distribution names.

    Attributes:
        name (str): Name of the distribution.
        coefficients (list[str]): Coefficient parameter names.
        extended (bool): Whether the distribution is extended.
        summands (list[str]): List of component distribution names.
        parameters (list[str]): List of coefficients and summands
    """

    def __init__(
        self, *, name: str, coefficients: [str], extended: bool, summands: [str]
    ):
        super().__init__(name, "mixture_dist")
        self.name = name
        self.coefficients = coefficients
        self.extended = extended
        self.summands = summands
        self.parameters = [*coefficients, *summands]

    def expression(self, distributionsandparameters: dict(str, pt.scalar)):
        """
        Builds a symbolic expression for the mixture distribution.

        Args:
            distributionsandparameters (dict): Mapping of names to pytensor variables.

        Returns:
            pt.TensorVariable: Symbolic representation of the mixture PDF.
        """
        mixturesum = 0
        coeffsum = 0
        i = 0
        for coeff in self.coefficients:
            coeffsum += distributionsandparameters[coeff]
            mixturesum += (
                distributionsandparameters[coeff]
                * distributionsandparameters[self.summands[i]]
            )
        mixturesum += (1 - coeffsum) * distributionsandparameters[self.summands[i]]
        return mixturesum


registered_distributions = {"gaussian_dist": GaussianDist, "mixture_dist": MixtureDist}


class DistributionSet:
    """
    Collection of distributions.

    Args:
        distributions (list[dict[str, str]]): List of distribution configurations.

    Attributes:
        dists (dict): Mapping of distribution names to Distribution objects.
    """

    def __init__(self, distributions: list[dict[str, str]]):
        self.dists = {}
        for dist_config in distributions:
            dist_type = dist_config.pop("type")
            the_dist = registered_distributions.get(dist_type, Distribution)
            dist = the_dist(**dist_config)

            self.dists[dist.name] = dist

    def __getitem__(self, item: str) -> Distribution:
        return self.dists[item]

    def __iter__(self):
        return iter(self.dists.values())


def boundedscalar(name: str, domain: tuple) -> pt.scalar:
    """
    Creates a pytensor scalar constrained within a given domain.

    Args:
        name (str): Name of the scalar.
        domain (tuple): Tuple specifying (min, max) range.

    Returns:
        pt.scalar: A pytensor scalar clipped to the domain range.
    """
    x = pt.scalar(name + "unconstrained")

    i = domain[0]
    f = domain[1]

    print(x, i, f)
    return pt.clip(x, i, f)


myworkspace = Workspace(json.loads(json_content))
mymodel = myworkspace.model(
    parameter_point=myworkspace.parameter_collection["default_values"],
    domain=myworkspace.domain_collection["default_domain"],
)


scalarranges = myworkspace.domain_collection["default_domain"]

f = boundedscalar("f", scalarranges["f"])
f_ctl = boundedscalar("f_ctl", scalarranges["f_ctl"])
mean = boundedscalar("mean", scalarranges["mean"])
mean2 = boundedscalar("mean2", scalarranges["mean2"])
sigma = boundedscalar("sigma", scalarranges["sigma"])
sigma2 = boundedscalar("sigma2", scalarranges["sigma2"])
mean_ctl = boundedscalar("mean_ctl", scalarranges["mean_ctl"])
mean2_ctl = boundedscalar("mean2_ctl", scalarranges["mean2_ctl"])
# breakpoint()
x = boundedscalar("x", scalarranges["x"])

print("f: ", mymodel.parameterset["f"].value)

physicspdfval = mymodel.pdf(
    "model",
    x=mymodel.parameterset["x"].value,
    f=mymodel.parameterset["f"].value,
    mean=mymodel.parameterset["mean"].value,
    sigma=mymodel.parameterset["sigma"].value,
    mean2=mymodel.parameterset["mean2"].value,
    sigma2=mymodel.parameterset["sigma2"].value,
)
physicspdfvalctl = mymodel.pdf(
    "model_ctl",
    x=mymodel.parameterset["x"].value,
    f_ctl=mymodel.parameterset["f_ctl"].value,
    mean_ctl=mymodel.parameterset["mean_ctl"].value,
    mean2_ctl=mymodel.parameterset["mean2_ctl"].value,
    sigma=mymodel.parameterset["sigma"].value,
)

print(physicspdfval)
print(physicspdfvalctl)


def gaussian_pdf(x, mu, sigma):
    norm_const = 1.0 / (pt.sqrt(2 * math.pi) * sigma)
    exponent = pt.exp(-0.5 * ((x - mu) / sigma) ** 2)
    return norm_const * exponent


def mixture_pdf(coeff, pdf1, pdf2):
    return coeff * pdf1 + (1.0 - coeff) * pdf2


print("distributions: ", mymodel.distributions)
gx = gaussian_pdf(x, mean, sigma)
px = gaussian_pdf(x, mean2, sigma2)
model = mixture_pdf(f, gx, px)

gx_ctl = gaussian_pdf(x, mean_ctl, sigma)
px_ctl = gaussian_pdf(x, mean2_ctl, sigma)
model_ctl = mixture_pdf(f_ctl, gx_ctl, px_ctl)

sample = pt.scalar("sample", dtype="int32")

simPdf = pt.switch(pt.eq(sample, 0), model, model_ctl)

pdf_physics = function([x, f, mean, sigma, mean2, sigma2], model, name="pdf_physics")

pdf_control = function(
    [x, f_ctl, mean_ctl, mean2_ctl, sigma], model_ctl, name="pdf_control"
)

pdf_combined = function(
    [sample, x, f, mean, sigma, mean2, sigma2, f_ctl, mean_ctl, mean2_ctl],
    simPdf,
    name="pdf_combined",
)

# default_params = {p["name"]: p["value"] for p in mymodel.startingpoints}
# default_params = mymodel.parameterset
# mymodel.parameters['x']
# breakpoint()
val_physics = pdf_physics(
    mymodel.parameterset["x"].value,
    mymodel.parameterset["f"].value,
    mymodel.parameterset["mean"].value,
    mymodel.parameterset["sigma"].value,
    mymodel.parameterset["mean2"].value,
    mymodel.parameterset["sigma2"].value,
)

val_control = pdf_control(
    mymodel.parameterset["x"].value,
    mymodel.parameterset["f_ctl"].value,
    mymodel.parameterset["mean_ctl"].value,
    mymodel.parameterset["mean2_ctl"].value,
    mymodel.parameterset["sigma"].value,
)

val_combined_physics = pdf_combined(
    0,  # sample
    mymodel.parameterset["x"].value,
    mymodel.parameterset["f"].value,
    mymodel.parameterset["mean"].value,
    mymodel.parameterset["sigma"].value,
    mymodel.parameterset["mean2"].value,
    mymodel.parameterset["sigma2"].value,
    mymodel.parameterset["f_ctl"].value,
    mymodel.parameterset["mean_ctl"].value,
    mymodel.parameterset["mean2_ctl"].value,
)

val_combined_control = pdf_combined(
    1,  # sample
    mymodel.parameterset["x"].value,
    mymodel.parameterset["f"].value,
    mymodel.parameterset["mean"].value,
    mymodel.parameterset["sigma"].value,
    mymodel.parameterset["mean2"].value,
    mymodel.parameterset["sigma2"].value,
    mymodel.parameterset["f_ctl"].value,
    mymodel.parameterset["mean_ctl"].value,
    mymodel.parameterset["mean2_ctl"].value,
)

print("Physics PDF:", val_physics)
print("Control PDF:", val_control)
print("Simultaneous PDF:", val_combined_physics)
print("Simultaneous PDF(control):", val_combined_control)

print(mymodel.parameterset)
