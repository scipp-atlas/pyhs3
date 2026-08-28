# Provisional HS3 0.3 compatibility

This branch implements the compatibility slice needed to load, round-trip,
build, and evaluate the current ROOT Higgs-discovery workspaces. It is based on
the structured interpolation output of ROOT commit
`eac146db1e842e1881c3167cbc3b656a9516a591` and the HS3 0.3 proposals available
during development, in particular HS3 pull request 103. It is not a claim of
complete conformance with a finalized HS3 0.3 specification.

Existing workspaces that declare `hs3_version: "0.2"` are feature-detected and
retain their metadata when serialized. pyhs3 does not silently rewrite the
declared version. A coordinated version bump can update new examples and
producer metadata after the specification is finalized.

## Product-domain axes

Product-domain axes are a strict structural union of coordinate, constant,
regular, and irregular axes. Regular axes preserve `nbins`; irregular axes
preserve `edges` and optional matching redundant bounds. Domain lookups return
the effective first/last bounds for binned axes.

Validation rejects boolean or non-integral bin counts, non-positive `nbins`,
non-finite or unordered bounds and edges, conflicting binning discriminators,
and fields belonging to another axis form. Integer-valued JSON floats remain
accepted for `nbins` to match ROOT's producer behavior.

## Structured interpolation

Interpolation is represented by the required object `{type, in, out}`. The
explicit `out: null` field is retained even when callers serialize with
`exclude_none=True`. The supported ROOT forms are:

| Form | PiecewiseInterpolation | FlexibleInterpVar |
| --- | --- | --- |
| `{add, poly1, null}` | yes | yes |
| `{mult, exp, null}` | yes | yes |
| `{add, poly2, poly1}` | yes | yes |
| `{add, poly6, poly1}` | yes | no |
| `{mult, poly6, exp}` | yes | yes |
| `{mult, poly6, poly1}` | yes | no |

HistFactory channels accept `default_interpolation`, while `normsys` and
`histosys` modifiers may override it. The override is resolved before checking
whether the form is representable by FlexibleInterpVar or
PiecewiseInterpolation. The serialized placement is preserved; an inherited
channel default is not copied into every modifier.

Each sample evaluates all `histosys` entries as one ordered piecewise group
starting at the nominal template, and all `normsys` entries as one ordered
flexible group starting at one. Additive forms contribute a nominal-relative
delta; multiplicative forms scale the running result. Positivity is applied
once after the complete group. HistFactory shape results are clamped at zero,
flexible results use a dtype-matched positive minimum, and only the ROOT
polynomial/exponential `normsys` form clamps non-positive anchors to epsilon.

Standalone `interpolation` and `interpolation0d` functions use the same strict
descriptor model. One descriptor can be broadcast across all variables, or
one can be supplied for every variable. Mismatched payload lengths, legacy
integer codes, unsupported combinations, missing descriptors, and additional
descriptor fields are rejected.

The existing interpolation performance work remains in use. In particular,
`{mult, poly6, exp}` dispatches directly to the cached, stacked `tensordot`
implementation. Immutable coefficient/exponent caches, dtype-matched
constants, materialized anchor tensors, bin-vectorized graphs, channel context
caches, and homogeneous sample batching are retained. Descriptor dispatch
happens while the graph is constructed, with no per-bin runtime Python branch.
The low-level code-named kernels remain available for callers even though
serialized models no longer select them with integer codes.

## Named HistFactory constraints

The `constraint` field on `normfactor`, `normsys`, and `histosys` is an optional
distribution-name reference. The referenced distribution is an ordinary graph
dependency and its serialized analytic log density is evaluated exactly;
pyhs3 does not replace it with a generated unit Gaussian, Poisson, or lognormal
term.

References are validated across the workspace. Missing targets, HistFactory
targets, targets unrelated to the modifier parameter, event-observable
dependencies, indirect HistFactory dependencies, and cycles are rejected.
Constraint factors are registered once per likelihood by distribution name and
deduplicated across modifiers, channels, product factors, and explicit
auxiliary distributions. Internal `staterror` and `shapesys` constraints remain
supported independently.

Historical values such as `Gauss`, `Poisson`, and `LogNormal` are treated as
ordinary names. If no distribution with that name exists, validation fails
rather than changing the model silently.

## Current ROOT wire forms

The canonical generic-function discriminator is `generic`; the historical
pyhs3 `generic_function` spelling remains an import alias and serializes
canonically. Sum operands may be numeric constants or references. Numeric
CrystalBall exponents are accepted, and omission of an unset mixture
`ref_coef_norm` is safe when using exclude-none serialization.

## Parameter snapshots

`ParameterSet.overlay` explicitly composes a partial snapshot over a complete
base set without mutating either input. `Workspace.model` exposes this through
`base_parameter_set` on every dispatch path. Matching records are replaced as
complete records, so fields such as `const`, `nbins`, and `kind` follow the
selected snapshot instead of being merged field by field.

This is an evaluation convenience, not implicit inheritance added to the HS3
wire representation.

## Likelihood data binding and graph lifetime

Repeated observable names are supported when their serialized bounds agree.
Each distribution/data pair receives a deterministic input binding, namespaced
by the data object where needed. Pair-local tensors are substituted lazily into
the distribution graph, so two data sets with the same axis name no longer
overwrite one another.

Observable initializers are excluded from `free_params` and remain event
vectors even if stale parameter metadata requests a scalar kind. JAX calls
preserve pyhs3's original argument order despite PyTensor name sanitization.
Captured HistFactory expressions release their per-context construction cache,
which prevents one built analysis from retaining another complete model graph.

Gaussian normalization now uses an exact antiderivative, and normalization
integrals remove the synthetic one-point event axis. This keeps scalar and
batched parameter shapes consistent in PyTensor and JAX.

## ROOT numerical semantics

Extended mixtures use a stable signed, max-shifted log sum. Signed coefficients
therefore remain evaluable without underflow, while a non-positive final
density or expected yield produces `-inf` instead of `NaN`.

Bernstein distributions map a physical observable range to the unit interval
before evaluating the polynomial, matching ROOT's coordinate convention and
normalization behavior.

## Scope boundary

This work targets the Higgs-discovery compatibility corpus and current ROOT
producer output. It deliberately does not add categorical domains, per-bin
named constraint arrays, future explicit auxiliary-data syntax, full RooFit
fit/scan parity infrastructure, or unrelated HS3 proposals. Those should be
reviewed and committed separately once their wire contracts are stable.
