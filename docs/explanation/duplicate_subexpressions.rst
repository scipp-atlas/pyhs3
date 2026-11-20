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

What sharing and merging compute
--------------------------------

Both steps are common-subexpression elimination: a computation is reused only
when it has exactly the same inputs, so it is guaranteed to produce the same
value. Nothing is approximated, reordered or batched.

**Identical HS3 objects.** Take two summands of a mixture that are separate
distributions with the same arguments:

.. math::

   f_A(x) = \mathrm{CB}(x \mid m_0, \sigma, \alpha, n), \qquad
   f_B(x) = \mathrm{CB}(x \mid m_0, \sigma, \alpha, n).

Without sharing, the graph evaluates the crystal ball, and its normalization
integral, twice. With sharing, it evaluates
:math:`t(x) = \mathrm{CB}(x \mid m_0, \sigma, \alpha, n)` once and uses it for
both, so the mixture term becomes

.. math::

   \log\big(c_A\, t(x) + c_B\, t(x)\big)

with a single copy of :math:`t`.

**Identical subexpressions.** Take two different distributions that share
part of their computation:

.. math::

   \log G(x \mid \mu, \sigma_1) = -\frac{(x-\mu)^2}{2\sigma_1^2} - \log Z_1,
   \qquad
   \log G(x \mid \mu, \sigma_2) = -\frac{(x-\mu)^2}{2\sigma_2^2} - \log Z_2.

The expressions differ, but both build :math:`u = x - \mu`, :math:`u^2`, and
the broadcasts of :math:`x` and :math:`\mu` to the event axis. After the merge,
:math:`u` and :math:`u^2` exist once and feed both; only the parts that
involve :math:`\sigma_1` or :math:`\sigma_2` stay separate.

**What neither step does.**

- **Batching.** :math:`G(x_1 \mid y_1)` and :math:`G(x_2 \mid y_2)` compute
  different values, so there is nothing to share. Rewriting them as one
  vectorized :math:`G([x_1, x_2] \mid [y_1, y_2])` is batching: the arithmetic
  stays the same, but one vector operation replaces :math:`K` scalar
  subgraphs, which shrinks the graph further.
  :class:`~pyhs3.functions.InterpolationFunction` batches its entries this
  way; batching across distributions is tracked in
  `issue #18 <https://github.com/scipp-atlas/pyhs3/issues/18>`_.
- **Algebra.** Expressions that are only mathematically equal stay separate:
  :math:`x\,y` and :math:`y\,x`, or :math:`c_A\, t + c_B\, t` and
  :math:`(c_A + c_B)\, t`. PyTensor's canonicalization rewrites handle some
  of these during compilation.

Sharing and merging remove repeated work and the nodes that carried it;
batching removes nodes but not work. The two are complementary.

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
