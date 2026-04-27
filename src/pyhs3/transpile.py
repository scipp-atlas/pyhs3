"""
JAX transpilation helpers for pyhs3.

Provides :func:`jaxify` and :class:`JaxifiedGraph` for converting any PyTensor
expression into a callable JAX function suitable for use with JAX-based
optimizers (e.g. ``optimistix``).

Requires the ``jax`` optional extra::

    pip install pyhs3[jax]

which pulls in ``pytensor[jax]`` and, transitively, JAX itself.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from pyhs3.typing.aliases import TensorVar

# ---------------------------------------------------------------------------
# Optional JAX imports — graceful at import time; error raised on first use.
# ---------------------------------------------------------------------------

_IMPORT_ERROR: ImportError | None

try:
    import jax  # noqa: F401
    import jax.numpy as jnp
    from pytensor.compile import mode as _ptmode
    from pytensor.graph.fg import FunctionGraph
    from pytensor.graph.traversal import explicit_graph_inputs
    from pytensor.link.jax.dispatch import jax_funcify  # type: ignore[attr-defined]

    _IMPORT_ERROR = None
except ImportError as _exc:
    _IMPORT_ERROR = _exc


def _require_jax() -> None:
    if _IMPORT_ERROR is not None:
        msg = (
            "pyhs3.transpile requires JAX. "
            "Install with `pip install pyhs3[jax]` "
            "(which adds `pytensor[jax]`)."
        )
        raise ImportError(msg) from _IMPORT_ERROR


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class JaxifiedGraph:
    """A JAX-callable wrapper around a compiled PyTensor expression.

    Produced by :func:`jaxify`. Supports keyword-argument calls, positional
    calls, and partitioning into free/fixed parameter vectors for use with
    optimistix or ``jax.grad``.

    Attributes:
        inputs: Tuple of PyTensor input variables, in evaluation order.
        input_names: Names of those variables (same order as ``inputs``).
        fn: The raw JAX callable returned by ``jax_funcify``.
    """

    inputs: tuple[TensorVar, ...]
    input_names: tuple[str, ...]
    fn: Callable[..., tuple]  # type: ignore[type-arg]

    def __call__(self, **kwargs: object) -> object:
        """Call by keyword argument.

        Parameters
        ----------
        **kwargs:
            One value per input name, as JAX arrays or Python scalars.

        Returns
        -------
        The first output of the underlying JAX function (scalar or array).
        """
        ordered = [kwargs[n] for n in self.input_names]
        return self.fn(*ordered)[0]

    def call_positional(self, *args: object) -> object:
        """Call with positional arguments in ``input_names`` order.

        Parameters
        ----------
        *args:
            Values in the same order as ``self.input_names``.

        Returns
        -------
        The first output of the underlying JAX function.
        """
        return self.fn(*args)[0]

    def with_partition(
        self,
        free: Sequence[str],
        fixed: Sequence[str],
    ) -> Callable[[object, object], object]:
        """Return an optimistix-ready ``f(free_vec, fixed_vec) -> scalar``.

        Parameters
        ----------
        free:
            Names of parameters that will be optimized (packed into a 1-D
            vector in this order).
        fixed:
            Names of parameters that are held constant (packed into a 1-D
            vector in this order).

        Returns
        -------
        Callable
            ``f(free_vec, fixed_vec)`` that unpacks the two vectors, reassembles
            the full parameter dict, and evaluates the expression.

        Raises
        ------
        KeyError
            If any name in ``free`` or ``fixed`` is not in ``self.input_names``.
        """
        name_set = set(self.input_names)
        for name in list(free) + list(fixed):
            if name not in name_set:
                msg = f"Parameter {name!r} not in graph inputs {sorted(name_set)}"
                raise KeyError(msg)

        free_names = tuple(free)
        fixed_names = tuple(fixed)
        fn = self.fn
        input_names = self.input_names

        def _f(free_vec: object, fixed_vec: object) -> object:
            param_dict: dict[str, object] = {}
            for i, name in enumerate(free_names):
                param_dict[name] = free_vec[i]  # type: ignore[index]
            for i, name in enumerate(fixed_names):
                param_dict[name] = fixed_vec[i]  # type: ignore[index]
            ordered = jnp.array([param_dict[n] for n in input_names])
            return fn(*ordered)[0]

        return _f


def jaxify(
    output: TensorVar,
    *,
    inputs: Sequence[TensorVar] | None = None,
    optimize: bool = True,
) -> JaxifiedGraph:
    """Convert a PyTensor expression into a JAX-callable :class:`JaxifiedGraph`.

    Parameters
    ----------
    output:
        The PyTensor output variable to compile.
    inputs:
        Explicit list of input variables.  If ``None``, the full set of
        graph inputs (variables with no owner, i.e. symbolic parameters) is
        discovered automatically via ``explicit_graph_inputs``.
    optimize:
        Whether to run the JAX optimizer rewrites before funcifying.
        Default ``True``; set to ``False`` for debugging.

    Returns
    -------
    JaxifiedGraph
        Wrapper exposing the compiled JAX function plus input metadata.

    Examples
    --------
    >>> import math
    >>> import pytensor.tensor as pt
    >>> x = pt.scalar("x")
    >>> mu = pt.scalar("mu")
    >>> sigma = pt.scalar("sigma")
    >>> pdf = pt.exp(-0.5 * ((x - mu) / sigma) ** 2) / (
    ...     sigma * pt.sqrt(pt.constant(2 * math.pi, dtype="float64"))
    ... )
    >>> from pyhs3.transpile import jaxify
    >>> jg = jaxify(pdf)          # doctest: +SKIP
    >>> float(jg(x=0.0, mu=0.0, sigma=1.0))  # doctest: +SKIP
    0.3989422804014327
    """
    _require_jax()

    if inputs is None:
        # Filter out unnamed nodes (constants, shared vars without explicit names)
        # so that every entry in input_names is a non-None string.
        inputs = [
            v  # type: ignore[misc]
            for v in explicit_graph_inputs([output])
            if v.name is not None
        ]

    fgraph = FunctionGraph(inputs=list(inputs), outputs=[output], clone=True)
    if optimize:
        _ptmode.JAX.optimizer.rewrite(fgraph)
    fn = jax_funcify(fgraph)

    named_inputs: tuple[TensorVar, ...] = tuple(inputs)
    names: tuple[str, ...] = tuple(v.name for v in named_inputs)  # type: ignore[misc]
    return JaxifiedGraph(named_inputs, names, fn)
