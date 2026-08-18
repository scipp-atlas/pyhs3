.. diataxis: how-to
.. status: implemented

Debug a Model
==============

A model isn't evaluating the way you expect — a parameter is missing, a
distribution isn't there, or you need to see what pyhs3 actually compiled.

.. doctest:: debug-model

    >>> import pyhs3
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

Check what the model discovered
-----------------------------------

Print the model for a one-line overview of its mode and component counts:

.. doctest::

    >>> print(model)  # doctest: +ELLIPSIS
    Model(
        mode: ...
        parameters: ... (...)
        distributions: ... (...)
        functions: ... (...)
    )

List the exact parameters, distributions, and functions pyhs3 discovered by
name:

.. doctest::

    >>> sorted(model.parameters)
    ['mu', 'sigma', 'x']
    >>> sorted(model.distributions)
    ['gauss']
    >>> sorted(model.functions)
    []

If a parameter you expected is missing, it was never referenced by any
distribution or function in the workspace — pyhs3 only discovers parameters
that something actually depends on.

Inspect a distribution's computation graph
-----------------------------------------------

``graph_summary()`` reports how many inputs and operations went into
building one distribution, and whether it has been compiled yet:

.. doctest::

    >>> print(model.graph_summary("gauss"))  # doctest: +ELLIPSIS
    Distribution 'gauss':
        Input variables: ...
        Graph operations: ...
        Operation types: ...
        Mode: ...
        Compiled: ...

"Compiled: No" means pyhs3 hasn't built and cached the underlying PyTensor
function yet. That happens the first time you call ``.pdf()``, ``.logpdf()``,
or one of their ``_unsafe`` variants for that distribution.

Visualize the graph
-----------------------

``visualize_graph()`` renders the computation graph to an image file, using
the optional ``pydot`` dependency:

.. doctest::

    >>> import tempfile
    >>> with tempfile.TemporaryDirectory() as tmpdir:
    ...     path = model.visualize_graph("gauss", fmt="svg", path=tmpdir)
    ...     print(path.endswith("gauss_graph.svg"))  # doctest: +ELLIPSIS
    ...
    The output file is available at ...
    True

If ``pydot`` isn't installed, ``visualize_graph()`` raises ``ImportError``
with an install hint rather than failing with an unrelated traceback deeper
in PyTensor.

Recreate a model in a slower, more explicit compilation mode
------------------------------------------------------------------

The default compilation mode (``"FAST_RUN"``) optimizes for evaluation
speed at the cost of compile time and debuggability. Two stricter modes help
when you suspect a numerical or graph-construction bug rather than a
modeling mistake:

- ``mode="DebugMode"`` re-checks PyTensor's own optimizations against a
  reference implementation on every call, catching a rewrite that changes
  the result it shouldn't.
- ``mode="NanGuardMode"`` checks every intermediate value for ``NaN``,
  ``Inf``, and unusually large values as the graph executes, pinpointing the
  operation that first produced one rather than letting it propagate
  silently into the final result.

.. code-block:: python

    debug_model = ws.model(0, mode="DebugMode")
    nanguard_model = ws.model(0, mode="NanGuardMode")

See :class:`~pyhs3.Model` for every ``mode`` value pyhs3 accepts (documented
in its constructor docstring).
