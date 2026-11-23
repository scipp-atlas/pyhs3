.. diataxis: explanation
.. status: implemented

Comparing Absolute NLL Values with RooFit
=========================================

The negative log-likelihood ``-model.log_prob`` (see
:attr:`~pyhs3.Model.log_prob`, which returns the log-likelihood) can differ from
the value RooFit or quickFit reports for the same workspace and dataset. The
difference is a constant that does not depend on any parameter. pyhs3 keeps
its convention in both cases below, and this page explains where each constant
comes from so you can reconcile absolute values.

Fits, profile scans, and :math:`\Delta\mathrm{NLL}` intervals are unaffected:
a constant added to the NLL moves the minimum's value, not its location or
the difference between two points. Only a comparison of absolute NLL values
sees the offset.

Simultaneous distributions over several channels
------------------------------------------------

RooFit's ``RooSimultaneous`` treats the index category as an observable and
normalizes over the observable ``x`` and the category ``idx`` together. With
channel densities :math:`f_c(x)` that each integrate to one over ``x``, the
normalization is

.. math::

   Z = \sum_{c} \int f_c(x)\,\mathrm{d}x = n_\text{channels},

so RooFit's density is :math:`p(x, c) = f_c(x) / n_\text{channels}`. Each
event contributes an extra :math:`\ln n_\text{channels}` to RooFit's NLL:

.. math::

   \mathrm{NLL}_\text{RooFit} = \mathrm{NLL}_\text{pyhs3} + W \ln n_\text{channels},

where :math:`W` is the sum of the event weights of the dataset. :math:`W` is
the weighted sum, not the number of entries: an entry with weight
:math:`10^{-9}` adds almost nothing, so RooFit padding or "ghost" entries
barely change :math:`W`. For example, :math:`W = 642` across 14 channels gives
:math:`642 \ln 14 \approx 1694.3`.

pyhs3 does not apply this normalization because the combined distribution of
the HS3 specification is a plain product over its channels. Its category keys
"introduce no variable or observable", so nothing in the likelihood encodes a
channel count to divide by. See the combined distribution in the
`HS3 composite distributions chapter
<https://github.com/hep-statistics-serialization-standard/hep-statistics-serialization-standard/blob/master/docs/chapters/2.1.2_composite_distributions.md>`_.

Extended unbinned data
----------------------

For an extended unbinned distribution with :math:`N` events and expected yield
:math:`\nu`, the HS3 specification writes the event-count term as
:math:`\mathrm{Poisson}(N \mid \nu)`, which contains :math:`-\ln N!` in the
log-likelihood. pyhs3 omits that term, as RooFit does. An extended mixture
contributes :math:`\sum_j \ln \sum_i c_i f_i(x_j) - \nu` (see
:meth:`pyhs3.distributions.MixtureDist.log_prob_terms`), so the
pyhs3 NLL is smaller than the specification's literal form by
:math:`\ln N!` per extended unbinned dataset. This constant is the same one
RooFit omits, so it does not add to the RooFit offset above.

The specification does not yet state whether omitting the term is allowed. The
proposal to allow it is
`hep-statistics-serialization-standard PR #164
<https://github.com/hep-statistics-serialization-standard/hep-statistics-serialization-standard/pull/164>`_,
which follows
`issue #91
<https://github.com/hep-statistics-serialization-standard/hep-statistics-serialization-standard/issues/91>`_.

Weighted data enters as :math:`\sum_k w_k \ln f(x_k)`, without dividing by
:math:`\sum_k w_k`, which matches RooFit's weighted NLL.

To compute an NLL to compare against a RooFit value, see
:doc:`/howto/evaluate_model`.
