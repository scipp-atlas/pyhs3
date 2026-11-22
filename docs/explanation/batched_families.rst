.. diataxis: explanation
.. status: implemented

Building many functions of one type as a family
===============================================

Large workspaces repeat the same kind of function many times. A combined
measurement with fourteen channels can hold over four hundred
:class:`~pyhs3.functions.InterpolationFunction` instances, one per sample and
systematic, each with its own nominal value, variations, and nuisance
parameters. This page explains how pyhs3 builds such functions as one
vectorised computation per likelihood channel, and why.

Why one subgraph per function is slow
-------------------------------------

:class:`~pyhs3.Model` turns a workspace into a PyTensor graph by walking the
workspace's dependency graph in topological order and building each HS3 object
into a shared context. Built one at a time, four hundred interpolation
functions become four hundred subgraphs that differ only in their constants.
The cost does not come from evaluating them: PyTensor rewrites and compiles
every node of the graph, so a graph that grows with the number of functions
compiles for minutes.

PyTensor does not merge such subgraphs into a vectorised one itself: it has
no rewrite that turns many scalar computations of the same form into one
computation over a vector (see `pytensor discussion #1984
<https://github.com/pymc-devs/pytensor/discussions/1984>`_).

Families in the dependency graph
--------------------------------

The dependency graph already records every function's type and inputs before
anything is built, so pyhs3 groups functions there rather than rediscovering
the structure from the built graph. After building the dependency graph,
``pyhs3.networks.plan_function_families`` groups the instances of each
:class:`~pyhs3.functions.core.BatchableFunction` type by the first likelihood
distribution that depends on them. Each group of two or more becomes one
*family* node: its inputs are the union of its members' inputs, and every
member is rewired to depend on the family alone. When the model's walk reaches
the family node, one call to
:meth:`~pyhs3.functions.core.BatchableFunction.batch_expressions` builds all
members, and each member node takes its own expression from the result.

pyhf follows the same pattern for modifiers. Each modifier type pairs a builder,
which collects the data of every instance into arrays, with a combined applier
that evaluates all instances at once (for example ``histosys_combined``). In
pyhs3, the planner groups the instances and the batchable type supplies the
combined evaluation.

Families are scoped per likelihood distribution rather than shared across the
whole model, so that :meth:`~pyhs3.Model.pdf` of one distribution only
evaluates that distribution's functions. A function that depends on another
member of its group stays out of the family, since the family would otherwise
depend on itself.

How the interpolation family evaluates
--------------------------------------

For interpolation functions, every entry of every member becomes one element
of a flat entry axis:

- Each entry's parameter is gathered from one stacked vector. Most entries are
  a nuisance parameter times a constant, written as a
  :class:`~pyhs3.functions.ProductFunction`, so many entries share one gathered
  parameter.
- Entries are sorted by interpolation code, so each code's contributions are
  one contiguous slice evaluated once, and the slices are concatenated.
- Contributions are padded into a table of shape ``(F, E_max)``, with ``F``
  members and ``E_max`` the largest number of entries in one member, and
  reduced over the entry axis.
- The ``(F,)`` result is split into one scalar per member with a single
  ``Split``.

Constant nominal values and variations become numpy tables, decided from the
built tensors, so the family follows exactly the per-function semantics. A
member whose inputs are not all scalars, or whose entries interleave additive
and multiplicative codes, which do not commute, is built with its own
expression and logged at debug level.

Two graph patterns matter for compile time, both found by profiling PyTensor's
rewrites:

- Indexing the result once per member (``result[i]``) puts one scattered
  ``IncSubtensor`` per member into the gradient. PyTensor's
  ``pre_local_IncSubtensor_serialize`` rewrite then spends most of the rewrite
  time combining them. The gradient of a single ``Split`` is one ``Join``.
- Assembling the contributions with ``set_subtensor`` on a zero vector adds
  more ``IncSubtensor`` chains; sorting entries by code and concatenating
  avoids them.

Effect on a large workspace
---------------------------

On a fourteen-channel unbinned workspace with about 170 free parameters and
420 interpolation functions, the negative log-likelihood and its gradient gave
identical values with and without families, at the start point and at three
displaced points, and gradients that agree to a relative difference of about
1e-13:

.. list-table::
   :header-rows: 1

   * -
     - without families
     - with families
   * - ``log_prob`` nodes (after merging duplicates)
     - 16,593
     - 8,896
   * - model build
     - 7.3 s
     - 3.9 s
   * - ``compile_nll()`` (C, cold compile)
     - 266 s
     - 180 s
   * - one ``value_and_grad`` (C)
     - 15.7 ms
     - 9.3 ms
   * - ``compile_nll(backend="jax")``
     - 88 s
     - 40 s
   * - one ``value_and_grad`` (JAX)
     - 1.5 ms
     - 1.3 ms

A single family shared across all channels would save little more: 178 s for
the C compile and 36 s for the JAX compile, against 180 s and 40 s per channel.
