.. diataxis: tutorial
.. status: implemented

Getting Started
================

You need pyhs3 installed. Nothing else.

In this tutorial you build a single-Gaussian statistical model using pyhs3's
Python objects, evaluate its negative log-likelihood at a point, and read off
the result. Follow it top to bottom; each step depends on the one before it.

Describe the model
-------------------

A pyhs3 model starts from four pieces: metadata, a distribution, a set of
parameter values, and a domain that bounds those parameters. Construct each
one directly as a pyhs3 object:

.. doctest:: getting-started

   >>> from pyhs3 import Workspace
   >>> from pyhs3.metadata import Metadata
   >>> from pyhs3.distributions import GaussianDist
   >>> from pyhs3.parameter_points import ParameterPoint, ParameterSet
   >>> from pyhs3.domains import ProductDomain
   >>>
   >>> metadata = Metadata(hs3_version="0.2")
   >>> gaussian = GaussianDist(name="gauss", x="x", mean="mu", sigma="sigma")
   >>> parameter_set = ParameterSet(
   ...     name="default_values",
   ...     parameters=[
   ...         ParameterPoint(name="x", value=0.0),
   ...         ParameterPoint(name="mu", value=0.0),
   ...         ParameterPoint(name="sigma", value=1.0),
   ...     ],
   ... )
   >>> domain = ProductDomain(
   ...     name="default_domain",
   ...     axes=[
   ...         dict(name="x", min=-5.0, max=5.0),
   ...         dict(name="mu", min=-2.0, max=2.0),
   ...         dict(name="sigma", min=0.1, max=3.0),
   ...     ],
   ... )

``x`` is the observable the Gaussian is defined over; ``mu`` and ``sigma`` are
its parameters. Every name here is a string reference, resolved when the
pieces are assembled into a workspace next.

Assemble the workspace
----------------------

A :class:`~pyhs3.Workspace` collects these pieces into one validated
container:

.. doctest:: getting-started

   >>> ws = Workspace(
   ...     metadata=metadata,
   ...     distributions=[gaussian],
   ...     parameter_points=[parameter_set],
   ...     domains=[domain],
   ... )

Build and inspect the model
----------------------------

Call ``ws.model(0)`` to build a :class:`~pyhs3.Model` from the workspace's
first domain and first parameter set:

.. doctest:: getting-started

   >>> model = ws.model(0)
   <BLANKLINE>
   >>> print(model)
   Model(
       mode: FAST_RUN
       parameters: 3 (sigma, mu, x)
       distributions: 1 (gauss)
       functions: 0 ()
   )

The printed summary confirms what got built: three parameters, one
distribution, no functions.

Evaluate the negative log-likelihood
-------------------------------------

Read the parameter values straight off the model's parameter set, and pass
them to ``logpdf`` as NumPy arrays:

.. doctest:: getting-started

   >>> import numpy as np
   >>> parameters = {par.name: np.array(par.value) for par in model.parameterset}
   >>> nll = -2 * model.logpdf("gauss", **parameters)
   >>> print(f"nll: {nll:.8f}")
   nll: 1.83787707

You now have a built model and a number derived from it.

What you built
----------------

The workspace you assembled is the same thing an HS3 JSON file describes:
:doc:`/howto/load_workspace` covers loading one from disk instead of
constructing it in Python. :doc:`/reference/api` catalogs every component and
method this tutorial touched (see :class:`~pyhs3.Workspace` and
:class:`~pyhs3.Model` directly); :doc:`/explanation/building_a_model` explains
what happens between assembling the workspace and getting a number out of it.
