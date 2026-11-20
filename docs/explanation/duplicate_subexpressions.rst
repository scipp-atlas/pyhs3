.. diataxis: explanation
.. status: implemented

Duplicate Subexpressions in the Model Graph
===========================================

A statistical model often contains the same computation many times. For
example, if a workspace has dozens of signal components that are separate HS3
distributions with identical parameters, they are the same distribution under
many names. pyhs3 builds one PyTensor graph for the whole likelihood, so every
copy of a computation adds nodes that PyTensor has to process before it can
compile the graph. pyhs3 removes these copies in two places, at two levels of
detail.

Identical HS3 objects
---------------------

Two distributions are identical as HS3 objects when they have the same type
and every field except ``name`` refers to the same parameters, functions and
constants. In one model they build exactly the same expression, so while the
:class:`~pyhs3.Model` is built, the second one reuses the first one's tensors
instead of building its own. The comparison key is
:meth:`~pyhs3.distributions.Distribution.structural_key`, whose docstring has
examples of equal and unequal keys.

Two distributions are not identical when any field differs, including a
constant in place of a parameter reference (``mean: 0.0`` against
``mean: "mu"``), even if the parameter's value happens to be ``0.0``.

Some distribution classes keep state on the object while their expression is
built, for example :class:`~pyhs3.distributions.MixtureDist`, which records
its coefficient and yield nodes for the extended likelihood. An alias would
never build and would miss that state, so such classes opt out with
``share_identical_expressions = False`` and are always built separately.

Identical subexpressions
------------------------

Most duplication sits below the level of whole objects: the same parameter
broadcast against the observable in several distributions, the same
``x - m0`` inside different functions, the same constants. Before
:attr:`~pyhs3.Model.log_prob` hands its graph to PyTensor, it passes it
through :func:`~pyhs3.tensorutils.merge_duplicate_nodes`. That function keeps
an operation only if no earlier operation applies the same ``Op`` to the same
inputs, visiting each node once from the inputs to the output, so whole
duplicate subgraphs collapse in one pass.

The merge is structural, not algebraic. These are kept apart:

- expressions that are only mathematically equal, such as ``x * y`` and
  ``y * x``, or ``exp(log(x))`` and ``x``. PyTensor's canonicalisation
  rewrites deal with those during compilation;
- constants that differ in dtype or shape, such as ``2.0`` as ``float64`` and
  as ``float32``, or ``[2.0]`` and ``2.0``;
- distinct graph inputs, even if they share a name or a value.

The docstring of :func:`~pyhs3.tensorutils.merge_duplicate_nodes` shows each
case as a runnable example.

Why pyhs3 does this instead of PyTensor
---------------------------------------

PyTensor removes duplicates itself: its ``MergeOptimizer`` is part of every
compilation. It runs inside PyTensor's graph rewriter, though, and replaces
one node at a time on a ``FunctionGraph`` that validates the graph and
notifies its attached features after each replacement. That cost grows with
the number of duplicates, and on a large workspace it dominates the time
before C code generation even starts.

pyhs3 avoids that cost in two ways. Sharing identical distributions never
builds the copies in the first place. The single sweep of
:func:`~pyhs3.tensorutils.merge_duplicate_nodes` visits each node once and
replaces nothing in place, so it is much cheaper than the rewrite on large
graphs.

Both steps leave the graph's inputs, their names and the computed values
unchanged; they only change how many nodes compute those values.
