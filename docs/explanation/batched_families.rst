.. diataxis: explanation
.. status: implemented

Building many functions of one type as a family
===============================================

Large workspaces repeat the same kind of function many times. A combined
measurement over many channels can hold hundreds of
:class:`~pyhs3.functions.InterpolationFunction` instances, one per sample and
systematic, each with its own nominal value, variations, and nuisance
parameters. This page explains how pyhs3 builds such functions as one
vectorised computation per likelihood channel, and why. :class:`~pyhs3.Model`
does this for every batchable function type without any option to set or turn
off, and the likelihood it builds has the same value to rounding.

Why one subgraph per function is slow
-------------------------------------

:class:`~pyhs3.Model` turns a workspace into a PyTensor graph by walking the
workspace's dependency graph in topological order and building each HS3 object
into a shared context. Built one at a time, hundreds of interpolation
functions become hundreds of subgraphs that differ only in their constants.
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
:class:`~pyhs3.functions.BatchableFunction` type by the first *root*
distribution that depends on them: the likelihood's distributions and
auxiliary distributions, in order, or every distribution when the model has no
likelihood. Instances are grouped by their exact class, and each group of two
or more becomes one *family* node: its inputs are the union of its members'
inputs, and every member is rewired to depend on the family alone. When the
model's walk reaches the family node, one call to
:meth:`~pyhs3.functions.BatchableFunction.batch_expressions` builds all
members, and each member node takes its own expression from the result, named
after the member.

pyhf follows the same pattern for modifiers. Each modifier type pairs a builder,
which collects the data of every instance into arrays, with a combined applier
that evaluates all instances at once (for example ``histosys_combined``). In
pyhs3, the planner groups the instances and the batchable type supplies the
combined evaluation.

Families are scoped per root distribution rather than shared across the whole
model, so that :meth:`~pyhs3.Model.pdf` of one distribution evaluates its own
channel's family. A function that several channels share joins the family of
the first channel that uses it, which the other channels' graphs then include.
A function that depends on another member of its group stays out of the
family, since the family would otherwise depend on itself; for the same
reason each group is checked against the graph as already rewired by earlier
families, which keeps families of different types from depending on each other
in a cycle.

Families change the structure of the graph, not its inputs: the same
parameters stay free. :attr:`~pyhs3.compiled.CompiledNLL.parameter_names` orders
them by graph traversal, as always, so index a parameter vector through those
names rather than by position.

How the interpolation family evaluates
--------------------------------------

For interpolation functions, every entry of every member becomes one element
of a flat entry axis:

- Each entry's parameter is gathered from one stacked vector. An entry written
  as a :class:`~pyhs3.functions.ProductFunction` of one free parameter and
  constants shares that parameter's element, scaled by the product of the
  constants; any other scalar entry is its own element.
- Entries are sorted by interpolation code, so each code's contributions are
  one contiguous slice evaluated once, and the slices are concatenated. Every
  code from 0 to 6 is batched, using the same per-code formula as the
  per-function path.
- Contributions are padded into a table of shape ``(F, E_max)``, with ``F``
  members and ``E_max`` the largest number of entries in one member, and
  reduced over the entry axis: a product for multiplicative members (codes 1,
  5, 6) and a sum for additive ones (codes 0, 2, 3, 4), each reading only its
  own members' entries. ``positiveDefinite`` is applied per member.
- The ``(F,)`` result is split into one scalar per member with a single
  ``Split``.

Because the family uses the same formulas but reassociates sums and products,
it agrees with the per-function path to rounding rather than bit for bit.
Whether a nominal value or variation is a constant, and so becomes part of a
numpy table, is read from the built tensors. A ``ProductFunction`` entry, in
contrast, is re-expressed from its factors rather than read from its built
tensor, so the family has to change together with ``ProductFunction``.

A member is built with its own expression, and logged at debug level, if any
of its inputs is not a scalar or if its entries mix additive and
multiplicative codes. A model built with batched (``kind=pt.vector``)
parameters therefore gets no interpolation families.

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

On large workspaces dominated by interpolation functions, families shrink the
graph that PyTensor rewrites and compiles substantially. That shortens the
model build, the compile, and each evaluation, on both the C and the JAX
backend, while the likelihood and its gradient agree with the per-function
build to rounding. Scoping families per channel instead of sharing one family
across the whole model costs little extra compile time.
