.. diataxis: how-to
.. status: implemented

Evaluate a Model's PDF and Log-PDF
====================================

You have a ``Model`` built from an HS3 workspace and need probability
density (or log-density) values at one or more parameter points.

.. doctest:: evaluate-model

    >>> import pyhs3
    >>> import numpy as np
    >>> workspace_json = {
    ...     "metadata": {"hs3_version": "0.2"},
    ...     "distributions": [
    ...         {
    ...             "name": "gauss",
    ...             "type": "gaussian_dist",
    ...             "x": "x",
    ...             "mean": "mu",
    ...             "sigma": "sigma",
    ...         }
    ...     ],
    ... }
    >>> ws = pyhs3.Workspace(**workspace_json)
    >>> model = ws.model(0, progress=False)

Choose the type-safe or convenience API
------------------------------------------

pyhs3 gives you two pairs of evaluation methods:

- ``pdf`` / ``logpdf`` require every parameter value to be a numpy array.
  Use these in production code and performance-critical loops — no
  conversion happens on each call.
- ``pdf_unsafe`` / ``logpdf_unsafe`` accept plain Python floats and lists,
  converting them to numpy arrays for you. Use these in a notebook, a test,
  or anywhere the conversion overhead doesn't matter.

Evaluate with pre-converted arrays:

.. doctest::

    >>> pdf_value = model.pdf("gauss", x=np.array(0.0), mu=np.array(0.0), sigma=np.array(1.0))
    >>> pdf_value  # doctest: +ELLIPSIS
    array(0.398942...)

Or with the convenience API, passing floats directly:

.. doctest::

    >>> model.pdf_unsafe("gauss", x=0.0, mu=0.0, sigma=1.0)  # doctest: +ELLIPSIS
    array(0.398942...)

Get the log-density instead
------------------------------

``logpdf``/``logpdf_unsafe`` mirror ``pdf``/``pdf_unsafe`` exactly, but
return the natural logarithm of the density, evaluated directly in log
space rather than by taking ``log(pdf(...))`` yourself:

.. doctest::

    >>> model.logpdf(
    ...     "gauss", x=np.array(0.0), mu=np.array(0.0), sigma=np.array(1.0)
    ... )  # doctest: +ELLIPSIS
    array(-0.918938...)

See :doc:`/explanation/building_a_model` for why ``logpdf`` stays finite in cases where
``np.log(model.pdf(...))`` would underflow to ``-inf``.

Evaluate at many points at once
-----------------------------------

By default, a parameter compiles as a scalar tensor and only accepts a
single value per call. To evaluate at many points in one call, mark the
parameter as a vector *before* building the model:

.. doctest::

    >>> import pytensor.tensor as pt
    >>> workspace_json_with_params = dict(
    ...     workspace_json,
    ...     parameter_points=[
    ...         {
    ...             "name": "defaults",
    ...             "parameters": [
    ...                 {"name": "x", "value": 0.0},
    ...                 {"name": "mu", "value": 0.0},
    ...                 {"name": "sigma", "value": 1.0},
    ...             ],
    ...         }
    ...     ],
    ... )
    >>> ws_vec = pyhs3.Workspace(**workspace_json_with_params)
    >>> parameterset = ws_vec.parameter_points[0]
    >>> parameterset["x"].kind = pt.vector
    >>> import warnings
    >>> with warnings.catch_warnings(record=True) as w:
    ...     warnings.simplefilter("always")
    ...     vec_model = ws_vec.model(0, parameter_set=parameterset)
    ...     print(w[0].message)
    ...
    <BLANKLINE>
    Parameter 'x' has kind override vector (default would be scalar)
    >>> x_values = np.linspace(-3.0, 3.0, 5)
    >>> result = vec_model.pdf("gauss", x=x_values, mu=np.array(0.0), sigma=np.array(1.0))
    >>> result.shape
    (1, 5)

``x_values`` has shape ``(5,)``, one value per evaluation point. The result
carries a leading singleton axis, ``(1, 5)``, because overriding ``kind``
this way marks ``x`` as a plain vector parameter rather than an observable —
pyhs3 broadcasts a non-observable vector override differently from an
observable's data axis. Observables carried by a likelihood's data are
vectorized for you automatically, without needing this override at all; see
:doc:`/broadcasting` for the full rules on which parameters can be vectors,
which shape convention applies to each, and how their shapes combine.

Evaluate a joint likelihood for a fit
-----------------------------------------

``pdf``/``logpdf`` evaluate one distribution at a time. Fitting needs the
joint log-likelihood across every channel and constraint term at once — for
that, build the model from an ``Analysis`` or ``Likelihood`` instead of a
plain domain index, which exposes ``model.log_prob``, ``model.data``, and
``model.free_params``. Jaxifying the result requires the ``jax`` optional
extra (``pip install pyhs3[jax]``):

.. doctest:: fit-example

    >>> analysis_workspace = {
    ...     "metadata": {"hs3_version": "0.2"},
    ...     "distributions": [
    ...         {
    ...             "name": "gauss",
    ...             "type": "gaussian_dist",
    ...             "x": "x",
    ...             "mean": "mu",
    ...             "sigma": "sigma",
    ...         }
    ...     ],
    ...     "parameter_points": [
    ...         {
    ...             "name": "nominal",
    ...             "parameters": [
    ...                 {"name": "mu", "value": 0.0},
    ...                 {"name": "sigma", "value": 1.0},
    ...             ],
    ...         }
    ...     ],
    ...     "domains": [
    ...         {
    ...             "name": "fit_region",
    ...             "type": "product_domain",
    ...             "axes": [
    ...                 {"name": "mu", "min": -5.0, "max": 5.0},
    ...                 {"name": "sigma", "min": 0.01, "max": 10.0},
    ...             ],
    ...         }
    ...     ],
    ...     "data": [
    ...         {
    ...             "name": "observed_x",
    ...             "type": "unbinned",
    ...             "entries": [[0.4], [-0.2], [0.1]],
    ...             "axes": [{"name": "x", "min": -5.0, "max": 5.0}],
    ...         }
    ...     ],
    ...     "likelihoods": [
    ...         {"name": "gauss_likelihood", "distributions": ["gauss"], "data": ["observed_x"]}
    ...     ],
    ...     "analyses": [
    ...         {
    ...             "name": "gauss_fit",
    ...             "likelihood": "gauss_likelihood",
    ...             "domains": ["fit_region"],
    ...             "parameters_of_interest": ["mu"],
    ...             "init": "nominal",
    ...         }
    ...     ],
    ... }
    >>> analysis_ws = pyhs3.Workspace(**analysis_workspace)
    >>> fit_model = analysis_ws.model(analysis_ws.analyses["gauss_fit"])
    <BLANKLINE>
    >>> nll_graph = pyhs3.jaxify(-2 * fit_model.log_prob)
    >>> nll_value = nll_graph(**fit_model.data, **fit_model.free_params)[0]
    >>> nll_value.shape
    (1,)
    >>> print(f"-2 log L = {float(nll_value[0]):.4f}")  # doctest: +ELLIPSIS
    -2 log L = ...

``fit_model.free_params`` gives every non-constant parameter's starting value
from the workspace, and ``fit_model.data`` gives the observed arrays each
distribution needs, so the two dicts together are exactly what a jaxified
``log_prob`` expects as keyword arguments. See :attr:`~pyhs3.Model.log_prob`,
:attr:`~pyhs3.Model.data`, :attr:`~pyhs3.Model.free_params` for what each
requires, and :doc:`/explanation/building_a_model` for why the joint
likelihood is exposed in log space rather than probability space. pyhs3
doesn't ship a minimizer itself; this section stops at producing the
callable one needs. See :doc:`/howto/fit_model` for minimizing
``nll_graph`` over ``free_params`` with optimistix.
