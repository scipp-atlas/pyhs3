.. diataxis: reference
.. status: implemented

Model Reference
================

:class:`~pyhs3.Model` is the compiled, evaluable form of a
:class:`~pyhs3.Workspace`, built by calling ``workspace.model(target, ...)``.
Dispatch is based on the type of ``target``, which is required — there is no
no-argument form.

Building a model
-----------------

``target`` is an ``int``
~~~~~~~~~~~~~~~~~~~~~~~~~

- **Signature:** ``workspace.model(target: int, *, domain=None, parameter_set=None, progress=True, mode="FAST_RUN")``
- **Derives domain from:** ``target`` as an index into ``workspace.domains``,
  or ``domain`` if given.
- **Derives parameter set from:** ``parameter_set`` if given, else the first
  entry in ``workspace.parameter_points``.
- **Derives observables from:** every likelihood in the workspace, if their
  observables agree; raises ``ValueError`` if they don't — build the model
  from a specific likelihood instead when they conflict.
- **Builds:** the complete workspace graph — every distribution and
  function, whether or not any likelihood references it.

``target`` is a ``str``
~~~~~~~~~~~~~~~~~~~~~~~~

- **Signature:** ``workspace.model(target: str, *, domain=None, parameter_set=None, progress=True, mode="FAST_RUN")``
- **Resolution order:** searches ``workspace.analyses`` by name first, then
  ``workspace.likelihoods``, then falls back to the ``int`` path above,
  treating ``target`` as a domain name.
- **Constraints:** ``domain`` is rejected with ``ValueError`` if ``target``
  resolves to an analysis (an analysis carries its own domains).

``target`` is a :class:`~pyhs3.likelihoods.Likelihood`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Signature:** ``workspace.model(target: Likelihood, *, domain=None, parameter_set=None, progress=True, mode="FAST_RUN")``
- **Derives observables from:** ``target``'s own data axes.
- **Derives domain from:** ``domain`` if given, else a domain named
  ``"default_domain"``, else the first domain.
- **Builds:** only ``target``'s distributions and their transitive
  dependencies.
- **Grants access to:** :attr:`~pyhs3.Model.log_prob`,
  :attr:`~pyhs3.Model.data`, and :attr:`~pyhs3.Model.free_params` — see
  Likelihood-context properties below.

``target`` is an :class:`~pyhs3.analyses.Analysis`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Signature:** ``workspace.model(target: Analysis, *, parameter_set=None, progress=True, mode="FAST_RUN")`` (no ``domain`` kwarg — derived entirely from ``target``)
- **Derives domain, parameter set, and observables from:** ``target`` itself.
- **Builds:** only the analysis's likelihood's distributions and their
  transitive dependencies.
- **Grants access to:** :attr:`~pyhs3.Model.log_prob`,
  :attr:`~pyhs3.Model.data`, and :attr:`~pyhs3.Model.free_params` — see
  Likelihood-context properties below.

Both the ``Likelihood`` and ``Analysis`` paths grant the same access to
``log_prob``/``data`` — only the plain ``int``/domain-name path lacks a
likelihood context and cannot provide them. ``free_params``/``nominal_params``
are available regardless of how the model was built; see Likelihood-context
properties below.

``mode`` (all overloads)
~~~~~~~~~~~~~~~~~~~~~~~~~

- **Type:** ``str``
- **Default:** ``"FAST_RUN"``
- **Valid values:** ``"FAST_RUN"``, ``"FAST_COMPILE"``, ``"DebugMode"``,
  ``"NanGuardMode"``, or a PyTensor linker name such as ``"NUMBA"``,
  ``"JAX"``, or ``"PYTORCH"``. See :doc:`howto_debug_model` for when to
  reach for which.

Attributes
-----------

- **``model.domain``**: the :class:`~pyhs3.domains.Domain` the model was
  built with.
- **``model.parameterset``**: the :class:`~pyhs3.parameter_points.ParameterSet`
  the model was built with.
- **``model.parameters``**: ``dict[str, TensorVariable]``, every discovered
  parameter keyed by name. Iteration order follows the dependency graph's
  topological sort, an implementation detail, not a guaranteed stable
  order. For the order a specific distribution's compiled function actually
  expects, use ``pars()``/``parsort()`` below.
- **``model.distributions``**: ``dict[str, TensorVariable]``, every compiled
  distribution keyed by name.
- **``model.functions``**: ``dict[str, TensorVariable]``, every compiled
  function keyed by name.
- **``model.mode``**: ``str``, the compilation mode the model was built
  with.
- **``model.modifiers``**: ``dict[str, TensorVariable]``, every HistFactory
  modifier term (normalization factors, shape systematics, and similar)
  discovered while building the graph, keyed by name.
- **``model.log_distributions``**: ``dict[str, TensorVariable]``, the
  log-space expression for every distribution, keyed by name — what
  ``logpdf``/``logpdf_unsafe`` actually evaluate. See :doc:`building_a_model`.
- **``model.data``**, **``model.log_prob``**: only set when the model was
  built via ``workspace.model(analysis)`` or ``workspace.model(likelihood)``;
  each raises ``RuntimeError`` otherwise.
- **``model.nominal_params``**, **``model.free_params``**: read from
  ``model.parameterset`` and available on any model, regardless of how it was
  built. See Likelihood-context properties below for all four.

Methods
--------

``pdf(name, **parametervalues) -> NDArray[float64]``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Requires:** every value in ``parametervalues`` to already be an
  ``np.ndarray`` — including 0-d, e.g. ``np.array(0.0)``, not a numpy scalar
  type like ``np.float64``. A wrong type isn't caught by pyhs3 itself; it
  surfaces as whatever error the compiled backend raises.
- **Returns:** the evaluated probability density.

``logpdf(name, **parametervalues) -> NDArray[float64]``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Requires:** same as ``pdf``.
- **Returns:** the natural log of the probability density, evaluated via a
  separately compiled log-space expression — stays finite where the
  probability-space value would underflow to ``0.0``. See
  :doc:`building_a_model`.

``pdf_unsafe(name, **parametervalues) -> NDArray[float64]``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Accepts:** ``float``, ``list[float]``, or ``NDArray[float64]`` for each
  value; converts to arrays before delegating to ``pdf``.

``logpdf_unsafe(name, **parametervalues) -> NDArray[float64]``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Accepts:** same as ``pdf_unsafe``; delegates to ``logpdf``.

``pars(name: str) -> list[str]``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Returns:** the parameter names in the exact order the compiled function
  for ``name`` expects them, valid for both ``pdf``/``pdf_unsafe`` and
  ``logpdf``/``logpdf_unsafe`` (the two share identical free inputs by
  construction). Triggers compilation of ``name`` if it hasn't happened yet.
- This is the actual, sanctioned answer to "what order does pyhs3 want my
  parameters in" — not the iteration order of ``model.parameters`` above,
  which is an implementation detail of the dependency graph's topological
  sort.

``parsort(name: str, names: list[str]) -> list[int]``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Returns:** the indices that reorder ``names`` into ``pars(name)``'s
  order — the same relationship ``numpy.argsort`` has to a sort.

``graph_summary(name) -> str``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Returns:** input-variable count, graph-operation count, operation-type
  breakdown, compilation mode, and whether ``name`` has been compiled yet.
- **Raises:** ``ValueError`` if ``name`` is not a distribution in the model.

``visualize_graph(name, fmt="svg", outfile=None, path=None) -> str``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Returns:** the path to the rendered graph file.
- **Raises:** ``ImportError`` if the optional ``pydot`` dependency is not
  installed; ``ValueError`` if ``name`` is not a distribution in the model.

Likelihood-context properties
--------------------------------

``model.data`` and ``model.log_prob`` require the model to have been built via
``workspace.model(analysis)`` or ``workspace.model(likelihood)`` (both grant
the same access) and raise ``RuntimeError`` otherwise. ``model.nominal_params``
and ``model.free_params`` read from ``model.parameterset`` directly and work
on any model, but are documented here since they're normally used together
with the other two.

``model.data -> dict[str, np.ndarray]``
   Observed data arrays from the workspace, keyed by observable name.
   Requires a likelihood context; raises ``RuntimeError`` otherwise.

``model.nominal_params -> dict[str, float]``
   Every parameter's default value from the workspace parameter set,
   including parameters marked ``const=True``. Available on any model.

``model.free_params -> dict[str, float]``
   Same as ``nominal_params``, excluding ``const=True`` parameters: the
   correct dict to pass to a jaxified callable alongside ``model.data``.
   Available on any model.

``model.log_prob -> TensorVar``
   The symbolic joint log-probability for the full likelihood, as a 1-D
   PyTensor tensor of shape ``(M,)`` (``M`` = 1 for scalar parameters; ``M``
   matches the batch size for a vectorized profile scan). Parameters with
   ``const=True`` are baked in as constants and are not free inputs. Bounded
   parameters are clipped to their domain at construction time (see
   :doc:`building_a_model` — the gradient is exactly zero outside the bound,
   not an error). Normalization denominators are fixed at model-construction
   time; for **weighted** ``UnbinnedData``, a new ``Model`` must be built to
   use different weights. See :doc:`normalization` for the normalization
   convention itself, and :doc:`building_a_model` for why this is exposed in
   log space rather than probability space.

JAX transpilation
--------------------

Requires the ``jax`` optional extra (``pip install pyhs3[jax]``); raises
``ImportError`` with that install hint if it isn't present.

``pyhs3.jaxify(output: TensorVar, *, inputs=None) -> JaxifiedGraph``
   Compiles a PyTensor expression — typically ``model.log_prob`` — to a
   JAX-callable wrapper. ``inputs`` defaults to every symbolic graph input
   discovered automatically; pass an explicit sequence to fix the input set.

``JaxifiedGraph.__call__(**kwargs) -> tuple``
   Call by keyword — one value per input name. Returns a tuple (typically a
   1-tuple of JAX arrays); index ``[0]`` for the value itself.

``JaxifiedGraph.call_positional(*args) -> tuple``
   Call by position, in ``JaxifiedGraph.input_names`` order. Same tuple
   return as ``__call__``.

.. doctest:: jaxify-example

    >>> import math
    >>> import pytensor.tensor as pt
    >>> import pyhs3
    >>> x = pt.scalar("x")
    >>> mu = pt.scalar("mu")
    >>> sigma = pt.scalar("sigma")
    >>> pdf = pt.exp(-0.5 * ((x - mu) / sigma) ** 2) / (
    ...     sigma * pt.sqrt(pt.constant(2 * math.pi, dtype="float64"))
    ... )
    >>> jg = pyhs3.jaxify(pdf)
    >>> float(jg(x=0.0, mu=0.0, sigma=1.0)[0])
    0.3989422804014327

A minimizer integration jaxifies the log-space expression directly, rather
than jaxifying a probability-space ``pdf`` and taking its log; see
:doc:`building_a_model` for why. See :doc:`howto_evaluate_model` for
building a model from an ``Analysis``/``Likelihood`` and jaxifying its
``log_prob`` end to end.
