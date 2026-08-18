.. diataxis: explanation
.. status: implemented

How pyhs3 Builds a Model
==========================

.. mermaid::
   :config: {"theme": "forest", "darkMode": "true"}

   flowchart TD
       A["HS3 dict / JSON"] --> B["Workspace<br/>(validated data)"]
       B --> C["Dependency graph<br/>(parameters → functions → distributions)"]
       C --> D["Topological sort"]
       D --> E["PyTensor graph construction<br/>(graph built, not yet compiled)"]
       E --> F["First model.pdf / .logpdf call"]
       F --> G["Compiled function<br/>(cached for reuse)"]

A workspace holds validated data. It does not know how to evaluate
anything. Building a model is the process that turns that data into
something you can call. This page walks through the stages in the diagram
above: the dependency graph, the sort that orders it, the parameters those
nodes create, and what "compiled" actually means and when it happens.

Why a dependency graph
------------------------

A distribution or function can reference another function's output by name,
instead of a raw parameter. A function named ``signal_fraction`` with
expression ``signal_events / total_events`` depends on ``total_events``,
which might itself be a sum of two other names. pyhs3 cannot compile
``signal_fraction`` before it knows what ``total_events`` evaluates to, so it
first builds a graph of every such reference:

.. code-block:: text

   signal_events ─┐
                  ├─▶ total_events ─▶ signal_fraction ─▶ measured_fraction
   background_events ─┘

This is not a graph you construct: pyhs3 builds it by scanning every
distribution and function's declared inputs for names that match another
function's output.

Topological sort: why order matters
--------------------------------------

Once the graph exists, pyhs3 sorts it topologically: every node appears only
after every node it depends on. Compiling ``signal_fraction`` before
``total_events`` exists would leave its formula referencing an undefined
name. The sort is what guarantees that never happens, regardless of the
order distributions and functions were listed in the workspace.

Parameter discovery and bounds
---------------------------------

Any name that appears in a distribution or function but is never itself the
output of a function is a leaf of the graph — a parameter. pyhs3 does not
require you to declare these in ``parameter_points``; it creates one for
every leaf it finds. If a ``domains`` entry names that parameter, pyhs3
clips the resulting tensor to the stated ``min``/``max`` at construction
time (via ``pt.clip``), so pyhs3 itself never proposes a value outside it;
if no domain names it, the parameter is created unbounded.

This clip has a consequence worth knowing if you hand ``model.log_prob`` to
an external gradient-based minimizer: ``pt.clip``'s gradient is exactly zero
outside the bound. A parameter the optimizer pushes past its bound during a
line search sees no gradient there at all — indistinguishable from
convergence, with no error or warning raised.

This is also why a minimal workspace with only a ``distributions`` list
still builds a working model: the parameters it references did not need to
be declared anywhere else, only used.

Observables and vectorization
--------------------------------

A parameter that is one of a likelihood's data axes is treated differently
from an ordinary free parameter: pyhs3 creates it as a 1D vector rather than
a scalar, so a model built from a likelihood can be evaluated against every
event or bin at once instead of one at a time. The mechanics of vector
parameters — how to opt an ordinary parameter into the same treatment, and
what changes about evaluation once you do — are covered in
:doc:`/broadcasting`, not here.

Why the joint likelihood is exposed in log space
----------------------------------------------------

Building a model from an :class:`~pyhs3.analyses.Analysis` or a
:class:`~pyhs3.likelihoods.Likelihood` exposes ``model.log_prob``: the joint
log-probability across every channel and constraint term, summed once in
log space rather than multiplied in probability space and logged
afterward. Per-distribution evaluation follows the same logic:
:doc:`/howto/evaluate_model`'s ``logpdf``/``logpdf_unsafe`` evaluate
directly in log space rather than computing ``log(pdf(...))`` afterward.

The reason in both cases is the same: for a multi-channel likelihood built
from many per-bin or per-channel probability factors (a HistFactory model is
the common case), the probability-space product can underflow to exactly
zero in floating point well before the log of that product would. Summing
log-probabilities avoids the underflow rather than working around it after
the fact. This is also why ``model.log_prob`` is what gets passed to
:func:`pyhs3.jaxify` for gradient-based minimization, instead of jaxifying a
probability-space ``pdf`` and taking a log of the result.

See :doc:`/normalization` for how pyhs3 normalizes a distribution over its
observable domain, a separate concern from the log-space summation
described here.

What "compiled" means for a distribution
--------------------------------------------

A distribution's graph is built (topologically sorted, tensors created) as
soon as the model is constructed, but not compiled into an executable
function until the first call to ``pdf``/``logpdf`` for that distribution —
:meth:`~pyhs3.Model.graph_summary` reports this as ``Compiled: No`` or
``Compiled: Yes``. Building the graph is cheap and happens for every
distribution up front; compiling is the expensive step PyTensor performs once
per distribution, on first use, under whichever ``mode`` the model was
constructed with, and the result is cached for reuse.
