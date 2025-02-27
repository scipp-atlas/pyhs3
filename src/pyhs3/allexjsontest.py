from __future__ import annotations

import json
import math
from collections import OrderedDict

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
    def __init__(self, data):
        self.name = data["domains"][0]["name"]
        self.startingpoints = data["parameter_points"][0]["parameters"]
        self.axes = data["domains"][0]["axes"]
        self.type = data["domains"][0]["type"]
        self.parameters = ParameterCollection(data["parameter_points"])
        self.distributions = DistributionSet(data["distributions"])
        self.domains = {
            a["name"]: (
                Domains(
                    data["domains"][0]["axes"],
                    data["domains"][0]["name"],
                    data["domains"][0]["type"],
                )
            )
            for a in data["domains"]
        }


class ParameterCollection:
    def __init__(self, parametersets):
        self.sets = OrderedDict()

        for parameterset_config in parametersets:
            parameterset = ParameterSet(
                parameterset_config["name"], parameterset_config["parameters"]
            )
            self.sets[parameterset.name] = parameterset

    def __getitem__(self, name):
        return self.sets[name]


class ParameterSet:
    def __init__(self, name, points):
        self.name = name

        self.points = OrderedDict()

        for points_config in points:
            point = ParameterPoint(points_config["name"], points_config["value"])
            self.points[point.name] = point.value

    def __getitem__(self, name):
        return self.points[name]


class ParameterPoint:
    def __init__(self, name, value):
        self.name = name
        self.value = value


class Domains:
    def __init__(self, axes, name, type):
        self.name = name
        self.type = type
        self.axesnames = [a["name"] for a in axes]
        self.ranges = {a["name"]: (a["min"], a["max"]) for a in axes}

    def __getitem__(self, name):
        if isinstance(name, int):
            # print(list(self.ranges)[name])
            return self.ranges[list(self.ranges)[name]]
            # name = list(self.points[name])
        return self.ranges[name]


class DistributionSet:
    def __init__(self, distributions):
        self.dists = OrderedDict()
        for dist_config in distributions:
            if dist_config["type"] == "gaussian_dist":
                dist = GaussianDist(
                    dist_config["name"],
                    dist_config["mean"],
                    dist_config["sigma"],
                    dist_config["x"],
                )
            elif dist_config["type"] == "mixture_dist":
                dist = MixtureDist(
                    dist_config["name"],
                    dist_config["coefficients"],
                    dist_config["extended"],
                    dist_config["summands"],
                )
            else:
                dist = Distribution(dist_config["name"], dist_config["type"])

            self.dists[dist.name] = dist

    def __getitem__(self, item):
        return self.dists[item]


class Distribution:
    def __init__(self, name, type):
        self.name = name
        self.type = type


class GaussianDist(Distribution):
    def __init__(self, name, mean, sigma, x):
        super().__init__(name, "gaussian_dist")
        self.mean = mean
        self.sigma = sigma
        self.x = x


class MixtureDist(Distribution):
    def __init__(self, name, coefficients, extended, summands):
        super().__init__(name, "mixture_dist")
        self.coefficients = coefficients
        self.extended = extended
        self.summands = summands


# for i in distributions:
#     if i[type] == gaussian:
#         GaussianDist(i)
#     elif i[type] == mixture:
#         ...
#     else:
#         return error
# class Distributions:
#     def __init__(self, dist):
#         self

# class ParameterPoints:
#     def __init__(self, name, points):
#         self.name = name
#         self.pnames = [p['name'] for p in points]
#         self.points = {p['name']: p['value'] for p in points}
#
#     def __getitem__(self, name):
#         if isinstance(name, int):
#             name = self.points[name]
#         else:
#             name = key
#         return self.points[name]
#
# foo = ParameterPoints("default", [{'name': 'x', 'value': 1.0}])
# foo['x'] -> 1.0
# foo[0] -> 1.0

# jsonmodel.parameters["default"]["x"] -> 1.0

# similar struct for domains


# def minimize(workspace, init_set='default', bounds="default_bounds"):
#     params = workspace.inits[init_set]
#     bounds = workspace.bounds[bounds]
#
#     compute(workspace.model(), bounds=bounds.dict(), inits=inits.dict())
#     compute(workspace.model(), x=1.0, y=2.0, z=3.0)
def boundedscalar(name, domain):
    x = pt.scalar(name + "unconstrained")
    # print(x)

    i = domain[0]
    f = domain[1]

    # boundedx = pt.math.sigmoid(x-i) - pt.math.sigmoid(f-x)

    return pt.clip(x, i, f)


mymodel = Workspace(json.loads(json_content))

print(mymodel.distributions["model"].summands)

# points = ParameterPoints(data['parameter_points'][0]['name'], data['parameter_points'][0]['parameters'])
# ranges = Domains(data['domains'][0]['axes'], data['domains'][0]['name'], data['domains'][0]['type'])

# print(points[0])
# print(points[1])
# print(points['f'])
# print(points['x'])
# print(mymodel.parameters['default_values'][0])

# print(ranges[0])
# print(ranges[2])
# print(ranges['f'])
# print(ranges['mean'])
# print(mymodel.domains['default_domain']['mean'])


# print(points.name, '\n\n')
# print(points.pnames, '\n\n')
# print(points.points)

# print(mymodel.name)
# print(mymodel.axes)
# print(mymodel.type)
# print(mymodel.startingpoints)
# print("domains:\n\n", data['domains'], "\n\n\n")
# print("axes:\n\n", data['domains'][0]['axes'], "\n\n\n")
# print("name:\n\n", data['domains'][0]['name'], "\n\n\n")
# print("type:\n\n", data['domains'][0]['type'], "\n\n\n")


scalarranges = mymodel.domains["default_domain"]

# print("\n\n\n",scalarranges,"\n\n\n")
# print(scalarranges["f"],"\n\n\n ")

f = boundedscalar("f", scalarranges["f"])
f_ctl = boundedscalar("f_ctl", scalarranges["f_ctl"])
mean = boundedscalar("mean", scalarranges["mean"])
mean2 = boundedscalar("mean2", scalarranges["mean2"])
sigma = boundedscalar("sigma", scalarranges["sigma"])
sigma2 = boundedscalar("sigma2", scalarranges["sigma2"])
mean_ctl = boundedscalar("mean_ctl", scalarranges["mean_ctl"])
mean2_ctl = boundedscalar("mean2_ctl", scalarranges["mean2_ctl"])
x = boundedscalar("x", scalarranges["x"])


def gaussian_pdf(x, mu, sigma):
    norm_const = 1.0 / (pt.sqrt(2 * math.pi) * sigma)
    exponent = pt.exp(-0.5 * ((x - mu) / sigma) ** 2)
    return norm_const * exponent


def mixture_pdf(coeff, pdf1, pdf2):
    return coeff * pdf1 + (1.0 - coeff) * pdf2


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
default_params = mymodel.parameters["default_values"]

val_physics = pdf_physics(
    default_params["x"],
    default_params["f"],
    default_params["mean"],
    default_params["sigma"],
    default_params["mean2"],
    default_params["sigma2"],
)

val_control = pdf_control(
    default_params["x"],
    default_params["f_ctl"],
    default_params["mean_ctl"],
    default_params["mean2_ctl"],
    default_params["sigma"],
)

val_combined_physics = pdf_combined(
    0,  # sample
    default_params["x"],
    default_params["f"],
    default_params["mean"],
    default_params["sigma"],
    default_params["mean2"],
    default_params["sigma2"],
    default_params["f_ctl"],
    default_params["mean_ctl"],
    default_params["mean2_ctl"],
)

val_combined_control = pdf_combined(
    1,  # sample
    default_params["x"],
    default_params["f"],
    default_params["mean"],
    default_params["sigma"],
    default_params["mean2"],
    default_params["sigma2"],
    default_params["f_ctl"],
    default_params["mean_ctl"],
    default_params["mean2_ctl"],
)

print("Physics PDF:", val_physics)
print("Control PDF:", val_control)
print("Simultaneous PDF:", val_combined_physics)
print("Simultaneous PDF(control):", val_combined_control)

print(
    default_params["x"],
    default_params["f"],
    default_params["mean"],
    default_params["sigma"],
    default_params["mean2"],
    default_params["sigma2"],
    default_params["f_ctl"],
    default_params["mean_ctl"],
    default_params["mean2_ctl"],
)
