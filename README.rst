..
  Comment: SPHINX-START

pure-python implementation of HS3
=================================

|GitHub Project| |GitHub Discussion|

|Docs from latest| |Docs from main|

|PyPI version| |Conda-forge version| |Supported Python versions| |PyPI platforms|

|Code Coverage| |CodeFactor| |pre-commit.ci Status| |Code style: black|

|Documentation Status| |GitHub Actions Status| |GitHub Actions Status: CI| |GitHub Actions Status: Docs| |GitHub Actions Status: Publish|

.. |GitHub Project| image:: https://img.shields.io/badge/GitHub--blue?style=social&logo=GitHub
   :target: https://github.com/scipp-atlas/pyhs3
.. |GitHub Discussion| image:: https://img.shields.io/static/v1?label=Discussions&message=Ask&color=blue&logo=github
   :target: https://github.com/scipp-atlas/pyhs3/discussions
.. |Docs from latest| image:: https://img.shields.io/badge/docs-v0.3.0-blue.svg
   :target: https://pyhs3.readthedocs.io/
.. |Docs from main| image:: https://img.shields.io/badge/docs-main-blue.svg
   :target: https://scipp-atlas.github.io/pyhs3
.. |PyPI version| image:: https://badge.fury.io/py/pyhs3.svg
   :target: https://badge.fury.io/py/pyhs3
.. |Conda-forge version| image:: https://img.shields.io/conda/vn/conda-forge/pyhs3.svg
   :target: https://prefix.dev/channels/conda-forge/packages/pyhs3
.. |Supported Python versions| image:: https://img.shields.io/pypi/pyversions/pyhs3.svg
   :target: https://pypi.org/project/pyhs3/
.. |PyPI platforms| image:: https://img.shields.io/pypi/pyversions/pyhs3
   :target: https://pypi.org/project/pyhs3/

.. |Code Coverage| image:: https://codecov.io/gh/scipp-atlas/pyhs3/graph/badge.svg?branch=main
   :target: https://codecov.io/gh/scipp-atlas/pyhs3?branch=main
.. |CodeFactor| image:: https://www.codefactor.io/repository/github/scipp-atlas/pyhs3/badge
   :target: https://www.codefactor.io/repository/github/scipp-atlas/pyhs3
.. |pre-commit.ci Status| image:: https://results.pre-commit.ci/badge/github/scipp-atlas/pyhs3/main.svg
   :target: https://results.pre-commit.ci/latest/github/scipp-atlas/pyhs3/main
   :alt: pre-commit.ci status
.. |Code style: black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black

.. |Documentation Status| image:: https://readthedocs.org/projects/pyhs3/badge/?version=latest
   :target: https://pyhs3.readthedocs.io/en/latest/?badge=latest
.. |GitHub Actions Status| image:: https://github.com/scipp-atlas/pyhs3/workflows/CI/badge.svg
   :target: https://github.com/scipp-atlas/pyhs3/actions
.. |GitHub Actions Status: CI| image:: https://github.com/scipp-atlas/pyhs3/actions/workflows/ci.yml/badge.svg
   :target: https://github.com/scipp-atlas/pyhs3/actions/workflows/ci.yml?query=branch%3Amain
.. |GitHub Actions Status: Docs| image:: https://github.com/scipp-atlas/pyhs3/actions/workflows/docs.yml/badge.svg
   :target: https://github.com/scipp-atlas/pyhs3/actions/workflows/docs.yml?query=branch%3Amain
.. |GitHub Actions Status: Publish| image:: https://github.com/scipp-atlas/pyhs3/actions/workflows/cd.yml/badge.svg
   :target: https://github.com/scipp-atlas/pyhs3/actions/workflows/cd.yml?query=branch%3Amain


Hello World
-----------

This section shows two ways to build the same statistical model and evaluate it: using PyHS3 Python objects directly, and using HS3 JSON-like dictionaries.

Hello World (Python)
~~~~~~~~~~~~~~~~~~~~~

This is how you use the ``pyhs3`` Python API to build a statistical model directly with objects:

.. code:: pycon

   >>> import pyhs3
   >>> import scipy
   >>> import math
   >>> import numpy as np
   >>> from pyhs3.distributions import GaussianDist
   >>> from pyhs3.parameter_points import ParameterPoint, ParameterSet
   >>> from pyhs3.domains import ProductDomain, Axis
   >>> from pyhs3.metadata import Metadata
   >>>
   >>> # Create metadata
   >>> metadata = Metadata(hs3_version="0.2")
   >>>
   >>> # Create a Gaussian distribution
   >>> gaussian_dist = GaussianDist(name="model", x="x", mean="mu", sigma="sigma")
   >>>
   >>> # Create parameter points with values
   >>> param_points = ParameterSet(
   ...     name="default_values",
   ...     parameters=[
   ...         ParameterPoint(name="x", value=0.0),
   ...         ParameterPoint(name="mu", value=0.0),
   ...         ParameterPoint(name="sigma", value=1.0),
   ...     ],
   ... )
   >>>
   >>> # Create domain with parameter bounds
   >>> domain = ProductDomain(
   ...     name="default_domain",
   ...     axes=[
   ...         Axis(name="x", min=-5.0, max=5.0),
   ...         Axis(name="mu", min=-2.0, max=2.0),
   ...         Axis(name="sigma", min=0.1, max=3.0),
   ...     ],
   ... )
   >>>
   >>> # Build workspace from objects
   >>> ws = pyhs3.Workspace(
   ...     metadata=metadata,
   ...     distributions=[gaussian_dist],
   ...     parameter_points=[param_points],
   ...     domains=[domain],
   ... )
   >>> model = ws.model()
   <BLANKLINE>
   >>> print(model)
   Model(
       mode: FAST_RUN
       parameters: 3 (sigma, mu, x)
       distributions: 1 (model)
       functions: 0 ()
   )
   >>> parameters = {par.name: np.array(par.value) for par in model.parameterset}
   >>> result = -2 * model.logpdf("model", **parameters)
   >>> print(f"parameters: {parameters}")
   parameters: {'x': array(0.), 'mu': array(0.), 'sigma': array(1.)}
   >>> print(f"nll: {result:.8f}")
   nll: 1.83787707
   >>>
   >>> # Serialize workspace back to dictionary for saving/sharing
   >>> workspace_dict = ws.model_dump()
   >>> print("Serialized workspace keys:", list(workspace_dict.keys()))
   Serialized workspace keys: ['metadata', 'distributions', 'functions', 'domains', 'parameter_points', 'data', 'likelihoods', 'analyses', 'misc']

Hello World (HS3)
~~~~~~~~~~~~~~~~~~

This is the same model built using HS3 JSON-like dictionary format:

.. code:: pycon

   >>> import pyhs3
   >>> import scipy
   >>> import math
   >>> import numpy as np
   >>> workspace_data = {
   ...     "metadata": {"hs3_version": "0.2"},
   ...     "distributions": [
   ...         {
   ...             "name": "model",
   ...             "type": "gaussian_dist",
   ...             "x": "x",
   ...             "mean": "mu",
   ...             "sigma": "sigma",
   ...         }
   ...     ],
   ...     "parameter_points": [
   ...         {
   ...             "name": "default_values",
   ...             "parameters": [
   ...                 {"name": "x", "value": 0.0},
   ...                 {"name": "mu", "value": 0.0},
   ...                 {"name": "sigma", "value": 1.0},
   ...             ],
   ...         }
   ...     ],
   ...     "domains": [
   ...         {
   ...             "name": "default_domain",
   ...             "type": "product_domain",
   ...             "axes": [
   ...                 {"name": "x", "min": -5.0, "max": 5.0},
   ...                 {"name": "mu", "min": -2.0, "max": 2.0},
   ...                 {"name": "sigma", "min": 0.1, "max": 3.0},
   ...             ],
   ...         }
   ...     ],
   ... }
   >>> ws = pyhs3.Workspace(**workspace_data)
   >>> model = ws.model()
   <BLANKLINE>
   >>> print(model)
   Model(
       mode: FAST_RUN
       parameters: 3 (sigma, mu, x)
       distributions: 1 (model)
       functions: 0 ()
   )
   >>> parameters = {par.name: np.array(par.value) for par in model.parameterset}
   >>> result = -2 * model.logpdf("model", **parameters)
   >>> print(f"parameters: {parameters}")
   parameters: {'x': array(0.), 'mu': array(0.), 'sigma': array(1.)}
   >>> print(f"nll: {result:.8f}")
   nll: 1.83787707
   >>> result_scipy = -2 * math.log(scipy.stats.norm.pdf(0, loc=0, scale=1))
   >>> print(f"nll: {result_scipy:.8f}")
   nll: 1.83787707
   >>>
   >>> # Round-trip: serialize workspace back to dictionary
   >>> serialized_dict = ws.model_dump()
   >>> print("Round-trip successful:", serialized_dict["metadata"]["hs3_version"])
   Round-trip successful: 0.2
   >>>
   >>> # Can recreate workspace from serialized dictionary
   >>> ws_roundtrip = pyhs3.Workspace(**serialized_dict)
   >>> model_roundtrip = ws_roundtrip.model()
   <BLANKLINE>
   >>> print("Round-trip model:", model_roundtrip.parameterset.name)
   Round-trip model: default_values
