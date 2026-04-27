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
from typing import cast

from pytensor.compile import mode as _ptmode
from pytensor.graph.fg import FunctionGraph
from pytensor.graph.traversal import explicit_graph_inputs

from pyhs3.typing.aliases import TensorVar

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class JaxifiedGraph:
    """A JAX-callable wrapper around a compiled PyTensor expression.

    Produced by :func:`jaxify`. Supports keyword-argument calls (the primary
    interface for dict-pytree-based optimizers like optimistix or everwillow)
    and positional calls via :meth:`call_positional`.

    Attributes:
        inputs: Tuple of PyTensor input variables, in evaluation order.
        input_names: Names of those variables (same order as ``inputs``).
        fn: The raw JAX callable returned by ``jax_funcify``.
    """

    inputs: tuple[TensorVar, ...]
    input_names: tuple[str, ...]
    fn: Callable[..., tuple]  # type: ignore[type-arg]

    def __call__(self, **kwargs: object) -> object:
        """Call by keyword argument — pure passthrough to :attr:`fn`.

        ``jax_funcify`` generates a function whose parameter names match the
        original PyTensor variable names, so kwargs are forwarded directly.
        Python itself raises ``TypeError`` for missing or unexpected names.

        The typical usage pattern with optimistix or everwillow is::

            @jax.jit
            def nll(free_params):          # free_params is a dict pytree
                all_params = {**free_params, **fixed_params}
                return -2 * jnp.log(jg(**all_params)[0])

        Parameters
        ----------
        **kwargs:
            One value per input name, as JAX arrays or Python scalars.

        Returns
        -------
        Whatever :attr:`fn` returns (typically a 1-tuple of JAX arrays).
        """
        return self.fn(**kwargs)

    def call_positional(self, *args: object) -> object:
        """Call with positional arguments — pure passthrough to :attr:`fn`.

        Parameters
        ----------
        *args:
            Values in the same order as ``self.input_names``.

        Returns
        -------
        Whatever :attr:`fn` returns (typically a 1-tuple of JAX arrays).
        """
        return self.fn(*args)


def jaxify(
    output: TensorVar,
    *,
    inputs: Sequence[TensorVar] | None = None,
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
    >>> jg = jaxify(pdf)
    >>> float(jg(x=0.0, mu=0.0, sigma=1.0)[0])
    0.3989422804014327
    """
    try:
        from pytensor.link.jax.dispatch.basic import jax_funcify  # noqa: PLC0415
    except ImportError as exc:
        msg = "pyhs3.transpile requires JAX. Install with `pip install pyhs3[jax]`."
        raise ImportError(msg) from exc

    if inputs is None:
        # Filter out unnamed nodes (constants, shared vars without explicit names)
        # so that every entry in input_names is a non-None string.
        inputs = cast(
            list[TensorVar],
            [v for v in explicit_graph_inputs([output]) if v.name is not None],
        )

    fgraph = FunctionGraph(inputs=list(inputs), outputs=[output], clone=True)
    _ptmode.JAX.optimizer.rewrite(fgraph)
    fn = jax_funcify(fgraph)

    named_inputs: tuple[TensorVar, ...] = tuple(inputs)
    raw_names = tuple(v.name for v in named_inputs)
    if any(name is None for name in raw_names):
        msg = (
            "All inputs must be named for kwargs-based dispatch. "
            "Provide named TensorVariables or use call_positional()."
        )
        raise ValueError(msg)
    names: tuple[str, ...] = tuple(cast(str, name) for name in raw_names)
    if len(set(names)) != len(names):
        duplicates = sorted(name for name in set(names) if names.count(name) > 1)
        msg = f"Input names must be unique; duplicates found: {duplicates}"
        raise ValueError(msg)
    return JaxifiedGraph(named_inputs, names, fn)
