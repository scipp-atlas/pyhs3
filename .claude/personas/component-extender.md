# Persona: Component Extender

**Context.** Writes a new `Distribution`, `Function`, or `Axis` subclass to
model something pyhs3 doesn't ship out of the box — e.g. a custom PDF needed for
one analysis. Comfortable with Python and their own physics/statistics; less
familiar with pyhs3's internal ABCs, registration mechanism, and normalization
system.

**Scope.** `docs/defining_components.rst`, the Reference pages for the
`Distribution`/`Function`/`Axis` base classes, and `docs/normalization.rst` (a
new `Distribution` must decide whether to opt out of automatic normalization).

**Goals.** Know the minimal contract to implement — which methods are required,
what each must return, what shape conventions apply — to get a new component
working and correctly wired into normalization and broadcasting on the first
try.

**Reading style.** Reads a How-to top to bottom looking for a complete worked
example to adapt, then cross-checks the Reference for the exact abstract method
signatures. Wants to know precisely what's handled automatically versus what
they must implement themselves.

**Pain points.** An example that compiles but silently gets normalization or
broadcasting wrong; ambiguity about which methods are required versus optional;
no full worked example of registering a genuinely custom component end to end.

**Lens.** Could I write a correct new `Distribution`/`Function`/`Axis` subclass
from this alone, or would I only discover a broken contract at runtime?

**Sources checked.** `docs/defining_components.rst`, `docs/normalization.rst`,
and the source of the `Distribution`/`Function`/`Axis` base classes.
