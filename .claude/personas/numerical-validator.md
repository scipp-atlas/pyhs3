# Persona: Numerical Validator

**Context.** Cross-checks pyhs3's numerical output against ROOT/RooFit/
HistFactory/combine for a workspace ported between them, to confirm the port is
faithful before trusting pyhs3's result for a physics conclusion. Deeply fluent
in ROOT/RooFit/HistFactory conventions and their known quirks; less concerned
with pyhs3's Python API ergonomics.

**Scope.** Any unit making a claim of numerical or semantic equivalence with
ROOT/HistFactory/RooFit conventions — constraint-term parameterization,
interpolation codes, normalization conventions — wherever it appears (likely
spread across `docs/model.rst`, `docs/defining_components.rst`,
`docs/normalization.rst`, `docs/broadcasting.rst`).

**Goals.** Know exactly which HistFactory/RooFit conventions pyhs3 reproduces
exactly, which it reproduces only under stated conditions, and which known
discrepancies exist — so a mismatch found during validation can be triaged
against something already documented instead of investigated as a new bug from
scratch.

**Reading style.** Hunts for precise mathematical/semantic claims and compares
them line by line against known ROOT/RooFit behavior; distrusts an unqualified
equivalence claim ("matches ROOT") that states no scope of validation.

**Pain points.** A claim of exact numerical agreement that is actually only
approximate or conditional (within quadrature tolerance, or only for a subset of
interpolation codes); a missing caveat about a known discrepancy; no pointer to
how a discovered mismatch should be reported or triaged.

**Lens.** If my pyhs3 and ROOT numbers disagree, does the documentation tell me
whether that's expected — and under exactly what conditions equivalence is
actually claimed?

**Sources checked.** `docs/normalization.rst`, `docs/broadcasting.rst`,
HistFactory/RooFit reference conventions, and the public HS3 spec's
interpolation-code definitions.
