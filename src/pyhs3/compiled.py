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
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import IO, Any, Literal, cast

import numpy as np
import numpy.typing as npt
import pytensor
import pytensor.tensor as pt
from pytensor.compile.maker import function
from pytensor.compile.mode import Mode, get_mode
from pytensor.configdefaults import config
from pytensor.gradient import grad
from pytensor.graph.basic import Constant, Variable
from pytensor.graph.replace import clone_replace
from pytensor.graph.traversal import io_toposort
from pytensor.link.c.basic import OpWiseCLinker

from pyhs3._version import version as pyhs3_version
from pyhs3.transpile import jaxify
from pyhs3.typing.aliases import TensorVar

try:  # JAX is an optional dependency, needed only for backend="jax".
    import jax
    from jax import export as jax_export
except ImportError:  # pragma: no cover - exercised by the non-JAX CI variants
    jax = None
    jax_export = None

log = logging.getLogger(__name__)

Bounds = tuple[float | None, float | None]
Backend = Literal["c", "jax"]
EvalFn = Callable[[npt.NDArray[np.float64]], list[npt.NDArray[np.float64]]]


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
        """Wrap a compiled function of the parameter vector.

        Args:
            fn: Compiled function taking the ``(P,)`` parameter vector and
                returning ``[nll]``, or ``[nll, grad]`` if ``has_gradient``.
            parameter_names: Names of the ``P`` free parameters, in vector
                order.
            initial: Starting value per parameter.
            bounds: ``(lower, upper)`` per parameter; ``None`` if unbounded.
            dtype: Dtype of the parameter vector expected by ``fn``.
            has_gradient: Whether ``fn`` also returns the gradient.
            from_cache: Whether ``fn`` was loaded from the on-disk cache.
        """
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
    backend: Backend = "c",
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
        backend: ``"c"`` for PyTensor's C linker, ``"jax"`` to transpile to
            JAX and compile with XLA.
        fuse: Keep the elemwise-fusion rewrites in the default mode (see
            :func:`default_nll_mode`).
        mode: PyTensor mode; ``default_nll_mode(fuse=fuse)`` if ``None``.
        cache: ``True`` for the default cache directory, a path for a
            specific one, ``None``/``False`` for no on-disk cache.

    Returns:
        CompiledNLL: The compiled likelihood.
    """
    if backend not in ("c", "jax"):
        msg = f"Unknown backend {backend!r}; use 'c' or 'jax'."
        raise ValueError(msg)
    if backend == "jax" and (fuse or mode is not None):
        msg = "fuse= and mode= configure the C backend; they cannot be used with backend='jax'."
        raise ValueError(msg)
    if mode is not None and fuse:
        msg = "fuse=True selects the default mode with fusion; do not combine it with mode=."
        raise ValueError(msg)
    if backend == "jax" and jax is None:
        msg = "compile_nll(backend='jax') requires JAX. Install with `pip install pyhs3[jax]`."
        raise ImportError(msg)

    dtype = str(config.floatX)
    nll = -pt.sum(log_prob)  # type: ignore[no-untyped-call]
    outputs: list[TensorVar] = [nll]
    if gradient:
        # Differentiate with respect to the original scalars and stack, rather
        # than with respect to x[i], which would scatter each component back
        # into x with an IncSubtensor.  With no free parameters the gradient
        # is an empty (0,) vector.
        outputs.append(
            pt.stack(cast(list[TensorVar], grad(nll, list(free))))
            if free
            else cast(TensorVar, pt.zeros((0,), dtype=dtype))
        )

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

    builder: _CBuilder | _JaxBuilder
    if backend == "jax":
        builder = _JaxBuilder(outputs, givens, x, len(free), gradient=gradient)
    else:
        builder = _CBuilder(
            outputs,
            givens,
            x,
            default_nll_mode(fuse=fuse) if mode is None else mode,
        )

    def make(fn: EvalFn, *, from_cache: bool = False) -> CompiledNLL:
        return CompiledNLL(
            fn,
            names,
            initial,
            bounds,
            dtype,
            has_gradient=gradient,
            from_cache=from_cache,
        )

    if not cache:
        return make(builder.build())

    cache_dir = _default_cache_dir() if cache is True else Path(cache)
    key = hashlib.sha256(
        "|".join(
            [
                backend,
                _graph_fingerprint([*outputs, *(new for _, new in givens)]),
                repr(names),
                _mode_key(mode, fuse=fuse),
                str(gradient),
                pyhs3_version,
                pytensor.__version__,
                jax.__version__ if backend == "jax" and jax is not None else "",
            ]
        ).encode()
    ).hexdigest()
    if backend == "jax":
        _use_xla_cache(cache_dir / "xla")
    path = cache_dir / f"nll-{key}.{builder.suffix}"
    if path.exists():
        fn = builder.load(path.read_bytes())
        log.debug("Loaded compiled likelihood from %s", path)
        return make(fn, from_cache=True)

    fn = builder.build()
    _publish(path, builder.dump)
    log.debug("Cached compiled likelihood at %s", path)
    return make(fn)


def _use_xla_cache(directory: Path) -> None:
    """Point JAX's persistent compilation cache at ``directory`` unless one is set.

    JAX keeps one persistent compilation cache per process, set up at the
    first compile after ``jax_compilation_cache_dir`` is set; changing the
    option afterwards has no effect.  pyhs3 therefore never replaces a
    directory that is already set, whether by the user or by an earlier
    :func:`compile_nll` call.
    """
    assert jax is not None
    if jax.config.jax_compilation_cache_dir is None:
        jax.config.update("jax_compilation_cache_dir", str(directory))


def _publish(path: Path, write: Callable[[IO[bytes]], None]) -> None:
    """Write a cache file atomically.

    Each writer writes into its own temporary file in the cache directory and
    publishes it with an atomic rename, so concurrent writers never share or
    remove each other's temporary files and readers never see a partial
    file.  The temporary file is removed if writing fails.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f"{path.stem}.", suffix=".tmp", delete=False
    ) as sink:
        tmp = Path(sink.name)
        try:
            write(sink)
        except BaseException:
            sink.close()
            tmp.unlink(missing_ok=True)
            raise
    tmp.replace(path)


class _CBuilder:
    """Compile with a PyTensor C mode; cache entries are pickled functions."""

    suffix = "pkl"

    def __init__(
        self,
        outputs: list[TensorVar],
        givens: list[tuple[Variable[Any, Any], Variable[Any, Any]]],
        x: TensorVar,
        mode: Mode | str,
    ) -> None:
        self._outputs = outputs
        self._givens = givens
        self._x = x
        self._mode = mode
        self._fn: EvalFn | None = None

    def build(self) -> EvalFn:
        self._fn = cast(
            EvalFn,
            function(
                [self._x],
                self._outputs,
                givens=self._givens,
                mode=self._mode,
                on_unused_input="ignore",
                trust_input=True,
            ),
        )
        return self._fn

    def dump(self, sink: IO[bytes]) -> None:
        pickle.dump(self._fn, sink)

    def load(self, data: bytes) -> EvalFn:
        return cast(EvalFn, pickle.loads(data))


class _JaxBuilder:
    """Transpile to JAX and compile with XLA; cache entries are ``jax.export`` modules.

    The transpiled function is exported (``jax.export``) and compiled ahead
    of time from the exported module, both when it is freshly built and when
    it is loaded from the cache, so the XLA compilation happens in
    :func:`compile_nll`, not on the first evaluation.  Loading a cached
    module skips the PyTensor-to-JAX transpilation and tracing; XLA still
    compiles it, and :func:`compile_nll` points JAX's persistent compilation
    cache at the pyhs3 cache (see :func:`_use_xla_cache`) so that later
    sessions load the XLA executable instead.  Both paths compile the same
    exported module, so they share that executable.
    """

    suffix = "jaxexport"

    def __init__(
        self,
        outputs: list[TensorVar],
        givens: list[tuple[Variable[Any, Any], Variable[Any, Any]]],
        x: TensorVar,
        n_free: int,
        *,
        gradient: bool,
    ) -> None:
        # Likelihoods need double precision; JAX defaults to float32.  This
        # is a process-wide JAX setting.
        jax.config.update("jax_enable_x64", True)
        self._outputs = outputs
        self._givens = givens
        self._x = x
        self._spec = jax.ShapeDtypeStruct((n_free,), np.float64)
        self._gradient = gradient
        self._exported: Any = None

    def _compile(self, exported: Any) -> EvalFn:
        """Compile an exported module ahead of time and wrap it for CompiledNLL."""
        self._exported = exported
        compiled = jax.jit(exported.call).lower(self._spec).compile()
        gradient = self._gradient

        def evaluate(x: npt.NDArray[np.float64]) -> list[npt.NDArray[np.float64]]:
            # One output vector: [nll] or [nll, *grad], shape (1,) or (1 + P,).
            out = np.asarray(compiled(x))
            return [out[0], out[1:]] if gradient else [out[0]]

        return evaluate

    def build(self) -> EvalFn:
        nll, *rest = cast(
            list[TensorVar], clone_replace(self._outputs, replace=self._givens)
        )
        joint = (
            pt.concatenate([nll[None], rest[0]])  # type: ignore[no-untyped-call]
            if self._gradient
            else nll[None]
        )
        graph = jaxify(cast(TensorVar, joint), inputs=[self._x])
        traced = jax.jit(lambda xv: cast(Sequence[Any], graph.call_positional(xv))[0])
        return self._compile(jax_export.export(traced)(self._spec))

    def dump(self, sink: IO[bytes]) -> None:
        sink.write(self._exported.serialize())

    def load(self, data: bytes) -> EvalFn:
        return self._compile(jax_export.deserialize(bytearray(data)))
