.. diataxis: how-to
.. status: implemented

Minimize the NLL with SciPy or iminuit
======================================

You have an analysis in a workspace and want the best-fit parameters, using a
minimizer such as ``scipy.optimize.minimize`` or iminuit.
:meth:`~pyhs3.Model.compile_nll` compiles the negative log-likelihood and its
gradient once, as a function of a single vector of the free parameters, with
every ``const`` parameter baked in and, unless you ask otherwise (see
`Fit other datasets without recompiling`_), the observed data too.

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
    ...     analyses=[
    ...         {
    ...             "name": "A",
    ...             "likelihood": "L",
    ...             "parameters_of_interest": ["mu"],
    ...             "domains": ["main"],
    ...             "init": "init",
    ...         }
    ...     ],
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
same order. The compiled function does not apply these bounds, so pass them to
your minimizer:

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

``nll.parameters_of_interest`` lists the analysis' parameters of interest that
are free, and ``nll.poi_indices`` their positions in the parameter vector:

.. doctest:: minimize-nll

    >>> nll.parameters_of_interest
    ('mu',)
    >>> nll.parameter_names[nll.poi_indices[0]]
    'mu'

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

Fit other datasets without recompiling
--------------------------------------

To fit toys or an Asimov dataset, pass ``data_inputs=True``. The binned
observed counts of each HistFactory channel then become inputs of the compiled
function: ``value``, ``grad`` and ``value_and_grad`` take a ``data`` mapping
with the same keys and shapes as ``model.observed_data``, one ``(B,)`` array
per channel for a channel of ``B`` bins, keyed by channel name. Without
``data``, they use the model's observed data. Unbinned data and the
constraint centres (auxiliary data) stay baked in.

.. doctest:: minimize-nll-data

    >>> import numpy as np
    >>> import pyhs3
    >>> from scipy.optimize import minimize
    >>> ws = pyhs3.from_hifa(
    ...     {
    ...         "channels": [
    ...             {
    ...                 "name": "SR",
    ...                 "samples": [
    ...                     {
    ...                         "name": "signal",
    ...                         "data": [5.0, 10.0],
    ...                         "modifiers": [
    ...                             {"name": "mu", "type": "normfactor", "data": None}
    ...                         ],
    ...                     },
    ...                     {"name": "background", "data": [50.0, 60.0], "modifiers": []},
    ...                 ],
    ...             }
    ...         ],
    ...         "observations": [{"name": "SR", "data": [52.0, 75.0]}],
    ...         "measurements": [{"name": "meas", "config": {"poi": "mu", "parameters": []}}],
    ...         "version": "1.0.0",
    ...     }
    ... )
    >>> model = ws.model(ws.analyses["meas"], progress=False)
    >>> model.observed_data  # shape (2,): one count per bin of SR
    {'model_SR': array([52., 75.])}
    >>> nll = model.compile_nll(data_inputs=True)
    >>> asimov = {"model_SR": np.array([60.0, 80.0])}  # expected counts at mu = 2
    >>> result = minimize(
    ...     nll.value_and_grad, nll.initial, args=(asimov,), jac=True, bounds=nll.bounds
    ... )
    >>> round(float(result.x[0]), 3)
    2.0

A wrong channel name raises ``KeyError`` and an array of the wrong shape raises
``ValueError``. With ``backend="jax", batched=True``, every row of the batch
uses the same ``data``. Leave ``data_inputs`` off when you fit only the
observed data: the compiled function then folds data-dependent terms such as
``log(n!)`` into constants once, instead of computing them on every call.

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
On large workspaces the JAX backend compiles faster than the C backend and
evaluates the value and gradient several times faster; time both on your own
workspace.

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
sessions. One batched call costs much less than the same number of one-point
calls, and the saving per point grows with the batch size.

Choose the compile mode
-----------------------

By default ``compile_nll`` compiles without PyTensor's elemwise fusion, which
keeps the first compile of a large workspace short. Fused kernels evaluate
faster, so for a long fitting campaign (many fits or a fine profile scan),
pass ``fuse=True``, ideally together with ``cache=True`` so you pay the longer
compile once. On large workspaces, fusion makes the first compile several
times slower and each evaluation faster. Time your own workspace
before choosing.

.. code-block:: python

    nll = model.compile_nll(fuse=True, cache=True)

Pass ``mode=`` with any PyTensor mode, for example ``mode="CVM"``, to replace
the default (see :func:`~pyhs3.compiled.default_nll_mode`); it cannot be
combined with ``fuse=True``. With PyTensor's ``config.linker`` set to
``"auto"`` (the default) or ``"numba"``, ``mode="FAST_RUN"`` compiles with
Numba, which does not finish compiling large workspaces; with ``"cvm"`` it is
the same as ``mode="CVM"``. ``gradient=False`` compiles the value only, for
minimizers that do not use a gradient.

Fix a C compile error with Apple's clang
----------------------------------------

On macOS, PyTensor's C backend uses whichever ``clang++`` it finds:
``pytensor.config.cxx`` shows which one. conda-forge's ``pytensor`` package
installs its own ``clang++``, which an activated conda or pixi environment
provides; otherwise, for example after ``pip install pyhs3``, it is Apple's
``/usr/bin/clang++``. For a large workspace, ``compile_nll`` with Apple's
compiler can stop with:

.. code-block:: text

    fatal error: bracket nesting level exceeded maximum of 256

The gradient of a parameter that many terms depend on sums all their
contributions in one operation, and the C code PyTensor generates for a very
wide sum nests deeper than Apple clang's default limit. Raise the limit before
the first compile, either in the environment:

.. code-block:: bash

    PYTENSOR_FLAGS="gcc__cxxflags=-fbracket-depth=4096" python your_fit.py

or in Python, before calling ``compile_nll``:

.. code-block:: python

    import pytensor

    pytensor.config.gcc__cxxflags = "-fbracket-depth=4096"

Alternatively, run in an activated conda or pixi environment with
conda-forge's ``pytensor``, whose ``clang++`` compiles these workspaces as is,
or use ``backend="jax"``, which does not go through the C compiler.
