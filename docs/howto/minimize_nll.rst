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

.. doctest:: minimize-nll

    >>> import tempfile
    >>> with tempfile.TemporaryDirectory() as cache_dir:
    ...     first = model.compile_nll(cache=cache_dir)
    ...     second = model.compile_nll(cache=cache_dir)
    ...
    >>> first.from_cache, second.from_cache
    (False, True)

In your own scripts, ``cache=True`` keeps the files in the default directory
across sessions:

.. code-block:: python

    nll = model.compile_nll(cache=True)

Compile with JAX for large workspaces
-------------------------------------

With the optional JAX dependency installed (``pip install pyhs3[jax]``),
``backend="jax"`` transpiles the likelihood to JAX and compiles it with XLA
instead of PyTensor's C backend. The result is the same ``CompiledNLL``, so the
SciPy and iminuit calls above do not change:

.. code-block:: python

    nll = model.compile_nll(backend="jax", cache=True)

``compile_nll`` returns once XLA has compiled the function, so the first
evaluation is not slower than the others. The backend switches JAX to float64
(``jax_enable_x64``), a process-wide JAX setting, because a likelihood needs
double precision. ``fuse`` and ``mode`` configure the C backend and cannot be
combined with ``backend="jax"``.

With ``cache=True`` the cache stores the exported JAX module, and JAX's
persistent compilation cache in its ``xla`` subdirectory stores the compiled
XLA executable, so a later run skips both the transpilation and the XLA
compile. JAX keeps one compilation cache directory per process: if you set
``jax_compilation_cache_dir`` yourself, ``compile_nll`` keeps your directory.
On a 14-channel unbinned workspace with about 170 free parameters, the value
and gradient took:

.. list-table::
   :header-rows: 1

   * - backend
     - first ``compile_nll``
     - cached ``compile_nll``
     - one ``value_and_grad``
   * - ``"c"`` (default)
     - 279 s
     - 20 s
     - 16.6 ms
   * - ``"jax"``
     - 95 s
     - 11 s
     - 1.4 ms

Evaluate many parameter points at once
--------------------------------------

For a grid scan, or a minimizer that advances many fits in lockstep, pass
``batched=True`` together with ``backend="jax"``. The methods then take a
``(K, P)`` array, one parameter vector per row with columns ordered as
``nll.parameter_names``, and return one value per row: ``value`` gives shape
``(K,)`` and ``grad`` gives ``(K, P)``. The example below evaluates nine values
of ``mu`` with every other parameter at its initial value:

.. code-block:: python

    nll = model.compile_nll(backend="jax", batched=True, cache=True)
    X = np.tile(nll.initial, (9, 1))  # (9, P): nine copies of the start point
    X[:, nll.parameter_names.index("mu")] = np.linspace(0.0, 4.0, 9)
    values, grads = nll.value_and_grad(X)  # (9,), (9, P)

``nll.initial``, ``nll.bounds``, ``nll.to_vector`` and ``nll.to_dict`` still
describe a single parameter vector; stack vectors as rows to build a batch.
One compiled likelihood accepts any batch size, but XLA compiles it once per
batch size ``K``, at the first call with that ``K``, so keep ``K`` fixed
across calls. With ``cache=True`` that XLA compile is also cached across
sessions. On a 14-channel unbinned workspace with about 170 free parameters,
one batched ``value_and_grad`` for 9 points took 3.6 ms against 14.5 ms for 9
one-point calls, and 20 ms against 102 ms for 64 points; the first call for
each batch size took about 20 s.

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

Pass ``mode=`` with any PyTensor mode, for example ``mode="CVM"``, to replace
the default (see :func:`~pyhs3.compiled.default_nll_mode`); it cannot be
combined with ``fuse=True``. PyTensor resolves ``mode="FAST_RUN"`` to its
Numba backend, which does not finish compiling large workspaces. ``gradient=False`` compiles the value only, for
minimizers that do not use a gradient.
