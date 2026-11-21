.. diataxis: how-to
.. status: implemented

Minimize the NLL with SciPy or iminuit
======================================

You have an analysis in a workspace and want the best-fit parameters, using a
minimizer such as ``scipy.optimize.minimize`` or iminuit.
:meth:`~pyhs3.Model.compile_nll` compiles the negative log-likelihood and its
gradient once, as a function of a single vector of the free parameters, with
the observed data and every ``const`` parameter baked in.

The workspace below has one unbinned Gaussian with a free mean ``mu`` and
width ``sigma``:

.. doctest:: minimize-nll

    >>> import numpy as np
    >>> import pyhs3
    >>> ws = pyhs3.Workspace(
    ...     metadata={"hs3_version": "0.2"},
    ...     distributions=[
    ...         {
    ...             "name": "g",
    ...             "type": "gaussian_dist",
    ...             "x": "x_obs",
    ...             "mean": "mu",
    ...             "sigma": "sigma",
    ...         }
    ...     ],
    ...     domains=[
    ...         {
    ...             "name": "main",
    ...             "type": "product_domain",
    ...             "axes": [
    ...                 {"name": "mu", "min": -5.0, "max": 5.0},
    ...                 {"name": "sigma", "min": 0.1, "max": 10.0},
    ...             ],
    ...         }
    ...     ],
    ...     data=[
    ...         {
    ...             "name": "d",
    ...             "type": "unbinned",
    ...             "axes": [{"name": "x_obs", "min": -10.0, "max": 10.0}],
    ...             "entries": [[0.4], [1.1], [1.9], [2.3], [2.8], [3.6]],
    ...         }
    ...     ],
    ...     likelihoods=[{"name": "L", "distributions": ["g"], "data": ["d"]}],
    ...     analyses=[{"name": "A", "likelihood": "L", "domains": ["main"], "init": "init"}],
    ...     parameter_points=[
    ...         {
    ...             "name": "init",
    ...             "parameters": [
    ...                 {"name": "mu", "value": 1.0},
    ...                 {"name": "sigma", "value": 2.0},
    ...             ],
    ...         }
    ...     ],
    ... )
    >>> model = ws.model(ws.analyses["A"], progress=False)

Compile the likelihood
----------------------

.. doctest:: minimize-nll

    >>> nll = model.compile_nll()
    >>> sorted(nll.parameter_names)
    ['mu', 'sigma']

Every method of the result takes or returns a 1-D array with one entry per
free parameter, ordered as ``nll.parameter_names``. ``nll.initial`` is the
starting point from the analysis' parameter set and ``nll.bounds`` holds each
parameter's ``(lower, upper)`` domain bounds (``None`` where unbounded), in the
same order:

.. doctest:: minimize-nll

    >>> nll.to_dict(nll.initial) == {"mu": 1.0, "sigma": 2.0}
    True
    >>> dict(zip(nll.parameter_names, nll.bounds)) == {"mu": (-5.0, 5.0), "sigma": (0.1, 10.0)}
    True

Convert between a ``{name: value}`` dict and the vector with
``nll.to_vector`` and ``nll.to_dict``. ``nll.value(x)`` returns the NLL;
``nll.value_and_grad(x)`` returns the NLL and its gradient, shape ``(P,)`` for
``P`` free parameters, from one evaluation:

.. doctest:: minimize-nll

    >>> x = nll.to_vector({"mu": 2.0, "sigma": 1.0})
    >>> value, grad = nll.value_and_grad(x)
    >>> grad.shape
    (2,)

Minimize with SciPy
-------------------

Pass ``value_and_grad`` with ``jac=True`` so that SciPy uses the analytic
gradient instead of finite differences:

.. code-block:: python

    from scipy.optimize import minimize

    result = minimize(
        nll.value_and_grad, nll.initial, jac=True, bounds=nll.bounds, method="L-BFGS-B"
    )
    best_fit = nll.to_dict(result.x)

Minimize with iminuit
---------------------

iminuit asks for the value and the gradient in separate calls.
``nll.value`` and ``nll.grad`` share one cached evaluation per parameter
vector, so each point still costs one evaluation:

.. code-block:: python

    from iminuit import Minuit

    minuit = Minuit(nll.value, nll.initial, grad=nll.grad, name=nll.parameter_names)
    minuit.limits = nll.bounds
    minuit.errordef = Minuit.LIKELIHOOD
    minuit.migrad()
    best_fit = dict(zip(minuit.parameters, minuit.values))

Reuse the compiled function across sessions
-------------------------------------------

Compiling a large workspace takes minutes. Pass ``cache=True`` to store the
compiled function under ``$XDG_CACHE_HOME/pyhs3`` (``~/.cache/pyhs3``), or a
path to choose the directory. Compiling the same likelihood again with the same
options and the same pyhs3 and PyTensor versions then loads it in seconds;
``nll.from_cache`` tells you which happened. The cached file refers to
PyTensor's compiled C modules, so it is only valid on the machine that wrote
it.

.. code-block:: python

    nll = model.compile_nll(cache=True)

Choose the compile mode
-----------------------

By default ``compile_nll`` compiles without PyTensor's elemwise fusion, which
keeps the first compile of a large workspace short. Fused kernels evaluate
faster, so for a long fitting campaign (many fits or a fine profile scan),
pass ``fuse=True``, ideally together with ``cache=True`` so you pay the longer
compile once. On a 14-channel unbinned workspace with about 170 free
parameters, the first compile of the value and gradient took 279 s without
fusion and 700 s with it, and one evaluation inside a fit took about 16.6 ms
without fusion and 8.7 ms with it. Time your own workspace before choosing.

.. code-block:: python

    nll = model.compile_nll(fuse=True, cache=True)

Pass ``mode=`` with any PyTensor mode, for example ``mode="FAST_RUN"``, to
replace the default (see :func:`~pyhs3.compiled.default_nll_mode`); it cannot
be combined with ``fuse=True``. ``gradient=False`` compiles the value only, for
minimizers that do not use a gradient.
