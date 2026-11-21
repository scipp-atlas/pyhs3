"""
Compiled negative log-likelihood for minimizers.

:class:`CompiledNLL` is what :meth:`pyhs3.Model.compile_nll` returns: the
model's negative log-likelihood (and its gradient) compiled once as a function
of a single vector of free parameters, with the data and every constant
parameter baked in.
"""

from __future__ import annotations

import hashlib
import logging
import os
import pickle
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pytensor
import pytensor.tensor as pt
from pytensor.compile.maker import function
from pytensor.compile.mode import Mode, get_mode
from pytensor.configdefaults import config
from pytensor.gradient import grad
from pytensor.graph.basic import Constant, Variable
from pytensor.graph.traversal import io_toposort
from pytensor.link.c.basic import OpWiseCLinker

from pyhs3._version import version as pyhs3_version
from pyhs3.typing.aliases import TensorVar

log = logging.getLogger(__name__)

Bounds = tuple[float | None, float | None]


def default_nll_mode(*, fuse: bool = False) -> Mode:
    """PyTensor mode :meth:`pyhs3.Model.compile_nll` uses by default.

    The op-wise C linker with the ``FAST_RUN`` rewrites, excluding the
    in-place rewrites and, unless ``fuse=True``, the elemwise-fusion rewrites.

    - Fusion merges elementwise operations into ``Composite`` ops, each a
      distinct kernel that the C compiler builds separately; on large
      workspaces there are thousands of them, and compiling them dominates
      the cold compile time.  Without fusion, elementwise operations are
      generic, so one compiled module serves many nodes.  Fused kernels
      evaluate faster, though: on a 14-channel unbinned workspace (about 170
      free parameters) the value and gradient compiled in 279 s without
      fusion and 700 s with it, while one evaluation inside a fit took about
      16.6 ms without fusion and 8.7 ms with it.  ``fuse=False`` therefore
      favours the time to a first result, ``fuse=True`` long fitting
      campaigns (with the compiled function cached on disk).
    - The in-place rewrites add compile time without speeding up evaluation
      of a likelihood.

    Args:
        fuse: Keep the elemwise-fusion rewrites.

    Returns:
        Mode: The PyTensor compile mode.
    """
    excluded = ["inplace"] if fuse else ["inplace", "fusion"]
    return Mode(
        linker=OpWiseCLinker(),  # type: ignore[no-untyped-call]
        optimizer=get_mode("FAST_RUN")  # type: ignore[no-untyped-call]
        .excluding(*excluded)
        .provided_optimizer,
    )


def _graph_fingerprint(outputs: Sequence[Variable[Any, Any]]) -> str:
    """Structural hash of a graph, independent of object identity.

    Each variable is identified by how it is computed: graph inputs by name
    and type, constants by type, shape and value, and apply nodes by their
    ``Op`` (class and properties) and the identities of their inputs.  Two
    graphs with the same fingerprint compute the same outputs from inputs of
    the same names and types.
    """
    ids: dict[Variable[Any, Any], str] = {}

    def ident(var: Variable[Any, Any]) -> str:
        if var not in ids:
            if isinstance(var, Constant):
                data = np.asarray(var.data)
                digest = hashlib.sha256(data.tobytes()).hexdigest()
                ids[var] = f"const:{var.type}:{data.shape}:{digest}"
            else:
                ids[var] = f"input:{var.name}:{var.type}"
        return ids[var]

    hasher = hashlib.sha256()
    for node in io_toposort([], list(outputs)):
        op = node.op
        # _props_dict is how PyTensor itself compares and hashes Ops by value.
        props = op._props_dict() if hasattr(op, "__props__") else {}  # pylint: disable=protected-access
        op_id = (
            f"{type(op).__module__}.{type(op).__qualname__}:{sorted(props.items())!r}"
        )
        inputs = ",".join(ident(var) for var in node.inputs)
        node_id = hashlib.sha256(f"{op_id}({inputs})".encode()).hexdigest()
        for index, out in enumerate(node.outputs):
            ids[out] = f"{node_id}:{index}"
        hasher.update(node_id.encode())
    hasher.update(",".join(ident(out) for out in outputs).encode())
    return hasher.hexdigest()


def _mode_key(mode: Mode | str | None, *, fuse: bool) -> str:
    """Stable description of a compile mode for the cache key."""
    if mode is None:
        return f"default_nll_mode(fuse={fuse})"
    if isinstance(mode, str):
        return mode
    query = mode.provided_optimizer
    parts = [
        f"{name}={sorted(getattr(query, name, ()) or ())}"
        for name in ("include", "require", "exclude")
    ]
    return (
        f"{type(mode.linker).__qualname__}|{type(query).__qualname__}|{'|'.join(parts)}"
    )


def _default_cache_dir() -> Path:
    """``$XDG_CACHE_HOME/pyhs3`` (``~/.cache/pyhs3`` if unset)."""
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "pyhs3"


class CompiledNLL:
    """Negative log-likelihood compiled as a function of one parameter vector.

    Build it with :meth:`pyhs3.Model.compile_nll`.  The free parameters are
    ordered as :attr:`parameter_names`; every method takes or returns a 1-D
    ``float`` array of that length.  Data and constant parameters are baked
    into the compiled function.

    :meth:`value`, :meth:`grad` and :meth:`value_and_grad` share one cached
    evaluation per parameter vector, so a minimizer that asks for the value
    and then the gradient at the same point (SciPy with separate ``fun`` and
    ``jac``, iminuit with ``grad``) pays for one evaluation.

    Examples:
        With SciPy:

        .. code-block:: python

            nll = model.compile_nll()
            result = scipy.optimize.minimize(
                nll.value_and_grad, nll.initial, jac=True, bounds=nll.bounds
            )
            best_fit = nll.to_dict(result.x)

        With iminuit:

        .. code-block:: python

            minuit = iminuit.Minuit(nll.value, nll.initial, grad=nll.grad, name=nll.parameter_names)
            minuit.limits = nll.bounds
            minuit.migrad()
    """

    def __init__(
        self,
        fn: Callable[[npt.NDArray[np.float64]], list[npt.NDArray[np.float64]]],
        parameter_names: Sequence[str],
        initial: Sequence[float],
        bounds: Sequence[Bounds],
        dtype: str,
        *,
        has_gradient: bool,
        from_cache: bool = False,
    ) -> None:
        self._fn = fn
        #: Free parameter names, in the order of every parameter vector.
        self.parameter_names: tuple[str, ...] = tuple(parameter_names)
        self._dtype = dtype
        #: Starting point from the model's parameter set, shape ``(P,)``.
        self.initial: npt.NDArray[np.float64] = np.asarray(initial, dtype=dtype)
        #: ``(lower, upper)`` per parameter from the domain; ``None`` if unbounded.
        self.bounds: list[Bounds] = list(bounds)
        self.has_gradient = has_gradient
        #: Whether the compiled function was loaded from the on-disk cache.
        self.from_cache = from_cache
        self._last_x: bytes | None = None
        self._last_out: list[npt.NDArray[np.float64]] = []

    def __len__(self) -> int:
        return len(self.parameter_names)

    def to_vector(self, values: Mapping[str, float]) -> npt.NDArray[np.float64]:
        """Parameter vector from a ``{name: value}`` mapping of every free parameter."""
        unknown = set(values) - set(self.parameter_names)
        if unknown:
            msg = f"Not free parameters of this likelihood: {sorted(unknown)}"
            raise KeyError(msg)
        missing = [name for name in self.parameter_names if name not in values]
        if missing:
            msg = f"Missing values for free parameters: {missing}"
            raise KeyError(msg)
        return np.array(
            [values[name] for name in self.parameter_names], dtype=self._dtype
        )

    def to_dict(self, x: npt.ArrayLike) -> dict[str, float]:
        """``{name: value}`` mapping from a parameter vector."""
        return {
            name: float(value)
            for name, value in zip(self.parameter_names, self._check(x), strict=True)
        }

    def _check(self, x: npt.ArrayLike) -> npt.NDArray[np.float64]:
        array = np.asarray(x, dtype=self._dtype)
        if array.shape != (len(self.parameter_names),):
            msg = (
                f"Parameter vector has shape {array.shape}; expected "
                f"{len(self.parameter_names)} values ordered as parameter_names."
            )
            raise ValueError(msg)
        return array

    def _evaluate(self, x: npt.ArrayLike) -> list[npt.NDArray[np.float64]]:
        array = self._check(x)
        key = array.tobytes()
        if key != self._last_x:
            self._last_out = self._fn(array)
            self._last_x = key
        return self._last_out

    def value(self, x: npt.ArrayLike) -> float:
        """Negative log-likelihood at ``x``."""
        return float(self._evaluate(x)[0])

    def grad(self, x: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Gradient of the negative log-likelihood at ``x``, shape ``(P,)``."""
        return self.value_and_grad(x)[1]

    def value_and_grad(self, x: npt.ArrayLike) -> tuple[float, npt.NDArray[np.float64]]:
        """Negative log-likelihood and its gradient at ``x`` from one evaluation."""
        if not self.has_gradient:
            msg = "This likelihood was compiled with gradient=False."
            raise RuntimeError(msg)
        out = self._evaluate(x)
        return float(out[0]), np.array(out[1])


def compile_nll(
    log_prob: TensorVar,
    free: Sequence[TensorVar],
    baked: Mapping[TensorVar, npt.ArrayLike],
    initial: Sequence[float],
    bounds: Sequence[Bounds],
    *,
    gradient: bool = True,
    fuse: bool = False,
    mode: Mode | str | None = None,
    cache: bool | str | os.PathLike[str] | None = None,
) -> CompiledNLL:
    """Compile ``-sum(log_prob)`` as a function of one vector of ``free`` scalars.

    This is the implementation behind :meth:`pyhs3.Model.compile_nll`; see
    that method for the user-facing description.

    Args:
        log_prob: Joint log-likelihood; summed over its (parameter-batch) axis.
        free: Scalar input variables to expose, in vector order.
        baked: Remaining inputs (e.g. observed data) and the values baked in.
        initial: Starting value per free parameter.
        bounds: ``(lower, upper)`` per free parameter.
        gradient: Also compile the gradient with respect to ``free``.
        fuse: Keep the elemwise-fusion rewrites in the default mode (see
            :func:`default_nll_mode`).
        mode: PyTensor mode; ``default_nll_mode(fuse=fuse)`` if ``None``.
        cache: ``True`` for the default cache directory, a path for a
            specific one, ``None``/``False`` for no on-disk cache.

    Returns:
        CompiledNLL: The compiled likelihood.
    """
    nll = -pt.sum(log_prob)  # type: ignore[no-untyped-call]
    outputs: list[TensorVar] = [nll]
    if gradient:
        # Differentiate with respect to the original scalars and stack, rather
        # than with respect to x[i], which would scatter each component back
        # into x with an IncSubtensor.
        outputs.append(pt.stack(cast(list[TensorVar], grad(nll, list(free)))))

    dtype = str(config.floatX)
    x = cast(TensorVar, pt.vector("x", dtype=dtype))
    givens: list[tuple[Variable[Any, Any], Variable[Any, Any]]] = [
        (var, x[index])  # pylint: disable=unsubscriptable-object
        for index, var in enumerate(free)
    ]
    givens.extend(
        (var, pt.constant(np.asarray(value, dtype=var.dtype)))
        for var, value in baked.items()
    )
    names = [var.name or f"x[{index}]" for index, var in enumerate(free)]
    if mode is not None and fuse:
        msg = "fuse=True selects the default mode with fusion; do not combine it with mode=."
        raise ValueError(msg)
    compile_mode = default_nll_mode(fuse=fuse) if mode is None else mode

    def build() -> Callable[[npt.NDArray[np.float64]], list[npt.NDArray[np.float64]]]:
        return cast(
            Callable[[npt.NDArray[np.float64]], list[npt.NDArray[np.float64]]],
            function(
                [x],
                outputs,
                givens=givens,
                mode=compile_mode,
                on_unused_input="ignore",
                trust_input=True,
            ),
        )

    if not cache:
        return CompiledNLL(
            build(), names, initial, bounds, dtype, has_gradient=gradient
        )

    cache_dir = _default_cache_dir() if cache is True else Path(cache)
    key = hashlib.sha256(
        "|".join(
            [
                _graph_fingerprint([*outputs, *(new for _, new in givens)]),
                repr(names),
                _mode_key(mode, fuse=fuse),
                str(gradient),
                pyhs3_version,
                pytensor.__version__,
            ]
        ).encode()
    ).hexdigest()
    path = cache_dir / f"nll-{key}.pkl"
    if path.exists():
        with path.open("rb") as handle:
            fn = pickle.load(handle)
        log.info("Loaded compiled likelihood from %s", path)
        return CompiledNLL(
            fn, names, initial, bounds, dtype, has_gradient=gradient, from_cache=True
        )

    fn = build()
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as sink:
        pickle.dump(fn, sink)
    tmp.replace(path)
    log.info("Cached compiled likelihood at %s", path)
    return CompiledNLL(fn, names, initial, bounds, dtype, has_gradient=gradient)
