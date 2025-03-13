from __future__ import annotations
import json
import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import Union

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
    def __init__(self, data: dict):
        self.parameter_collection = ParameterCollection(data['parameter_points'])
        self.distribution_set = DistributionSet(data['distributions'])
        self.domain_collection = DomainCollection(data['domains'])
    def model(self, *, domain: int|str|DomainSet = 0, parameter_point: int|str|ParameterSet = 0) -> Model:
        ...
#         TODO: build list of parameters from parametercollection class
        distlist = self.distribution_set
#         TODO: assert that the set of parameter names is the same as the set of domain names
        assert(set(parameter_point.points.keys()) == set(domain.domains.keys())), "parameter and domain names do not match"

        return Model(parameterset= parameter_point, distributions= distlist, domains= domain)

class Model:
    def __init__(self, *, parameterset: ParameterSet, distributions: DistributionSet, domains: DomainSet):
        self.parameters = {}
        self.parameterset = parameterset
        for parameter in parameterset:
            self.parameters[parameter.name] = boundedscalar(parameter.name, domains[parameter.name])

        self.distributions = {}
        for dist in distributions:
            print(dist)
            self.distributions[dist.name] = dist.expression(self.parameters)


    def pdf(self):
        ...
    def logpdf(self):
        ...


class ParameterCollection:
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
        else:
            return self.sets[self.sets.keys()[name]]


class ParameterSet:
    def __init__(self, name: str, points: [ParameterPoint]):
        self.name = name

        self.points: dict[str, ParameterPoint] = OrderedDict()

        for points_config in points:
            point = ParameterPoint(points_config["name"], points_config["value"])
            self.points[point.name] = point

    def __getitem__(self, name: str) -> ParameterPoint:
        if isinstance(name, str):
            return self.points[name]
        else:
            return self.points[list(self.points.keys())[name]]

    def __iter__(self):
        return iter(self.points.values())


class ParameterPoint:
    def __init__(self, name: str, value: float):
        self.name = name
        self.value = value

@dataclass
class ParameterPoint:
    name: str
    value: float
#     research python data classes

class DomainCollection:
    def __init__(self, domainsets: [DomainSet]):
        self.domains = OrderedDict()

        for domain_config in domainsets:
            domain = DomainSet(domain_config['axes'], domain_config['name'], domain_config['type'])
            self.domains[domain.name] = domain

    def __getitem__(self, item: str|int) -> DomainSet:
        if isinstance(item, int):
            return self.domains.keys()[item]
        else:
            return self.domains[item]

class DomainSet:
    def __init__(self, axes: [DomainPoint], name: str, type: str):
        self.name = name
        self.type = type
        self.domains = OrderedDict()

        for domain_config in axes:
            domain = DomainPoint(domain_config['name'], domain_config['min'], domain_config['max'])
            self.domains[domain.name] = domain.range

    def __getitem__(self, item: int|str) -> (int, int):
        if isinstance(item, int):
            return list(self.domains.keys())[item]
        else:
            return self.domains[item]
class DomainPoint:
    def __init__(self, name: str, min: float, max: float):
        self.name = name
        self.min = min
        self.max = max
        self.range = (self.min, self.max)

class Distribution:
    def __init__(self, name: str, type: str):
        self.name = name
        self.type = type

class GaussianDist(Distribution):
    # need a way for the distribution to get the scalar function .parameter from parameterset
    def __init__(self, *, name: str, mean: str, sigma: str, x: str):
        super().__init__(name, 'gaussian_dist')
        self.mean = mean
        self.sigma = sigma
        self.x = x

    def expression(self, parameters: dict(str, pt.scalar)):
        norm_const = 1.0 / (pt.sqrt(2 * math.pi) * parameters[self.sigma])
        exponent = pt.exp(-0.5 * ((parameters[self.x] - parameters[self.mean]) / parameters[self.sigma]) ** 2)
        return norm_const * exponent

class MixtureDist(Distribution):
    def __init__(self, *, name: str, coefficients: list, extended: bool, summands: list):
        super().__init__(name, "mixture_dist")
        self.coefficients = coefficients
        self.extended = extended
        self.summands = summands

    @property
    def expression(self):
        ...



registered_distributions = {'gaussian_dist': GaussianDist, 'mixture_dist': MixtureDist}

class DistributionSet:
    def __init__(self, distributions: list[dict[str, str]]):
        self.dists = {}
        for dist_config in distributions:
            dist_type = dist_config.pop('type')
            the_dist = registered_distributions.get(dist_type, Distribution)
            dist = the_dist(**dist_config)

            self.dists[dist.name] = dist

    def __getitem__(self, item: str) -> Distribution:
        return self.dists[item]

    def __iter__(self):
        return iter(self.dists.values())

def boundedscalar(name: str, domain: tuple) -> pt.scalar:
    x = pt.scalar(name + "unconstrained")

    i = domain[0]
    f = domain[1]

    print(x, i, f)
    return pt.clip(x, i, f)

myworkspace = Workspace(json.loads(json_content))
mymodel = myworkspace.model(parameter_point= myworkspace.parameter_collection['default_values'], domain= myworkspace.domain_collection['default_domain'])

# breakpoint()

print(myworkspace.parameter_collection)


scalarranges = myworkspace.domain_collection['default_domain']

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


def gaussian_pdf(x, mu, sigma):
    norm_const = 1.0 / (pt.sqrt(2 * math.pi) * sigma)
    exponent = pt.exp(-0.5 * ((x - mu) / sigma) ** 2)
    return norm_const * exponent


def mixture_pdf(coeff, pdf1, pdf2):
    return coeff * pdf1 + (1.0 - coeff) * pdf2


gx = gaussian_pdf(x, mean, sigma)
#gx = mymodel.distributions[gx]
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
    mymodel.parameterset["sigma2"].value
)

val_control = pdf_control(
    mymodel.parameterset["x"].value,
    mymodel.parameterset["f_ctl"].value,
    mymodel.parameterset["mean_ctl"].value,
    mymodel.parameterset["mean2_ctl"].value,
    mymodel.parameterset["sigma"].value
)

val_combined_physics = pdf_combined(
    0, # sample
    mymodel.parameterset["x"].value,
    mymodel.parameterset["f"].value,
    mymodel.parameterset["mean"].value,
    mymodel.parameterset["sigma"].value,
    mymodel.parameterset["mean2"].value,
    mymodel.parameterset["sigma2"].value,
    mymodel.parameterset["f_ctl"].value,
    mymodel.parameterset["mean_ctl"].value,
    mymodel.parameterset["mean2_ctl"].value
)

val_combined_control = pdf_combined(
    1, # sample
    mymodel.parameterset["x"].value,
    mymodel.parameterset["f"].value,
    mymodel.parameterset["mean"].value,
    mymodel.parameterset["sigma"].value,
    mymodel.parameterset["mean2"].value,
    mymodel.parameterset["sigma2"].value,
    mymodel.parameterset["f_ctl"].value,
    mymodel.parameterset["mean_ctl"].value,
    mymodel.parameterset["mean2_ctl"].value
)

print("Physics PDF:", val_physics)
print("Control PDF:", val_control)
print("Simultaneous PDF:", val_combined_physics)
print("Simultaneous PDF(control):", val_combined_control)

print(mymodel.parameterset)
