"""
Compiled negative log-likelihood for minimizers.

:class:`CompiledNLL` is what :meth:`pyhs3.Model.compile_nll` returns: the
model's negative log-likelihood (and its gradient) compiled once as a function
of a single vector of free parameters, with every constant parameter baked in
and the observed data either baked in or passed per call.  Its Hessian is
compiled on the first call to :meth:`CompiledNLL.hessian`.
"""

from __future__ import annotations

import hashlib
import logging
import os
import pickle
import tempfile
from collections.abc import Callable, Collection, Mapping, Sequence
from pathlib import Path
from typing import IO, Any, Literal, cast

import numpy as np
import numpy.typing as npt
import pytensor
import pytensor.tensor as pt
from pytensor.compile.maker import function
from pytensor.compile.mode import Mode, get_mode
from pytensor.configdefaults import config
from pytensor.gradient import grad, jacobian
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

Bounds = tuple[float, float]
Backend = Literal["c", "jax"]
# Takes the parameter vector, then one array per data input, in order.
EvalFn = Callable[..., list[npt.NDArray[np.float64]]]


def default_nll_mode(*, fuse: bool = False) -> Mode:
    """PyTensor mode :meth:`pyhs3.Model.compile_nll` uses by default.

    The op-wise C linker with the rewrites of PyTensor's C mode (``"CVM"``),
    excluding the in-place rewrites and, unless ``fuse=True``, the
    elemwise-fusion rewrites.  ``"FAST_RUN"`` is not used: with PyTensor's
    ``config.linker`` set to ``"auto"`` (the default) or ``"numba"`` it
    resolves to the Numba mode, whose rewrite set targets the Numba linker;
    only with ``"cvm"`` does it resolve to ``"CVM"``.

    - Fusion merges elementwise operations into ``Composite`` ops, each a
      distinct kernel that the C compiler builds separately; on large
      workspaces there are thousands of them, and compiling them dominates
      the cold compile time.  Without fusion, elementwise operations are
      generic, so one compiled module serves many nodes.  Fused kernels
      evaluate faster, though: on large workspaces fusion makes the first
      compile several times slower and each evaluation faster.
      ``fuse=False`` therefore favours the time to a first result,
      ``fuse=True`` long fitting campaigns (with the compiled function cached
      on disk).
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
        optimizer=get_mode("CVM")  # type: ignore[no-untyped-call]
        .excluding(*excluded)
        .provided_optimizer,
    )


def _graph_fingerprint(
    outputs: Sequence[Variable[Any, Any]],
    as_inputs: Collection[Variable[Any, Any]] = (),
) -> str:
    """Structural hash of a graph, independent of object identity.

    Each variable is identified by how it is computed: graph inputs, and the
    variables in ``as_inputs`` (constants that the compiled function takes
    as inputs instead), by name and type, other constants by type, shape and
    value, and apply nodes by their ``Op`` (class and properties) and the
    identities of their inputs.  Two graphs with the same fingerprint
    compute the same outputs from inputs of the same names and types.
    """
    ids: dict[Variable[Any, Any], str] = {}
    replaced = set(as_inputs)

    def ident(var: Variable[Any, Any]) -> str:
        if var not in ids:
            if isinstance(var, Constant) and var not in replaced:
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
    ``float`` array of that length.  Constant parameters are baked into the
    compiled function.  The observed data are baked in too, unless the
    likelihood was compiled with ``data_inputs=True``: then each method takes
    an optional ``data`` mapping that replaces the binned observed counts for
    that call (see :meth:`value`).

    :meth:`value`, :meth:`grad` and :meth:`value_and_grad` share one cached
    evaluation per parameter vector and data, so a minimizer that asks for
    the value and then the gradient at the same point (SciPy with separate
    ``fun`` and ``jac``, iminuit with ``grad``) pays for one evaluation.
    :meth:`hessian` is a separate function, compiled at its first call and
    cached the same way.

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
        fn: EvalFn,
        parameter_names: Sequence[str],
        initial: Sequence[float],
        bounds: Sequence[Bounds],
        dtype: str,
        *,
        has_gradient: bool,
        build_hessian: Callable[[], EvalFn],
        from_cache: bool = False,
        parameters_of_interest: Sequence[str] = (),
        data: Mapping[str, npt.ArrayLike] | None = None,
    ) -> None:
        """Wrap a compiled function of the parameter vector.

        Args:
            fn: Compiled function taking the ``(P,)`` parameter vector, then
                one array per entry of ``data`` in its order, and returning
                ``[nll]``, or ``[nll, grad]`` if ``has_gradient``.
            parameter_names: Names of the ``P`` free parameters, in vector
                order.
            initial: Starting value per parameter.
            bounds: ``(lower, upper)`` per parameter; ``-inf``/``inf`` for an
                open side.
            dtype: Dtype of the parameter vector expected by ``fn``.
            has_gradient: Whether ``fn`` also returns the gradient.
            build_hessian: Compiles the Hessian, a function with the
                arguments of ``fn`` returning ``[hessian]``; called at the
                first call to :meth:`hessian`.
            from_cache: Whether ``fn`` was loaded from the on-disk cache.
            parameters_of_interest: Names of the parameters of interest; those
                that are not free parameters are left out.
            data: Default value of each data input of ``fn``, keyed by the
                name the ``data`` argument of :meth:`value` uses; empty or
                ``None`` if the data are baked into ``fn``.
        """
        self._fn = fn
        #: Free parameter names, in the order of every parameter vector.
        self.parameter_names: tuple[str, ...] = tuple(parameter_names)
        self._dtype = dtype
        #: Starting point from the model's parameter set, shape ``(P,)``.
        self.initial: npt.NDArray[np.float64] = np.asarray(initial, dtype=dtype)
        #: ``(lower, upper)`` per parameter from the domain; ``-inf``/``inf`` for
        #: an open side or a parameter outside the domain.
        self.bounds: list[Bounds] = list(bounds)
        self.has_gradient = has_gradient
        #: Whether the compiled function was loaded from the on-disk cache.
        self.from_cache = from_cache
        #: Parameters of interest that are free parameters, in the order given
        #: (for a model, the order the analysis lists them).  A parameter of
        #: interest held ``const`` is baked into the function and left out.
        self.parameters_of_interest: tuple[str, ...] = tuple(
            name for name in parameters_of_interest if name in self.parameter_names
        )
        #: Index in :attr:`parameter_names` (and in every parameter vector) of
        #: each of :attr:`parameters_of_interest`.
        self.poi_indices: tuple[int, ...] = tuple(
            self.parameter_names.index(name) for name in self.parameters_of_interest
        )
        self._data_names: tuple[str, ...] = tuple(data or {})
        self._default_data: tuple[npt.NDArray[np.float64], ...] = tuple(
            np.asarray(value, dtype=np.float64) for value in (data or {}).values()
        )
        self._last_x: bytes | None = None
        self._last_out: list[npt.NDArray[np.float64]] = []
        self._build_hessian = build_hessian
        self._hessian_fn: EvalFn | None = None
        self._last_hessian_x: bytes | None = None
        self._last_hessian: npt.NDArray[np.float64] = np.empty((0, 0))

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

    def _check_argument(self, x: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Validate the argument of :meth:`value` and :meth:`value_and_grad`."""
        return self._check(x)

    def _check_data(
        self, data: Mapping[str, npt.ArrayLike] | None
    ) -> tuple[npt.NDArray[np.float64], ...]:
        """Arrays to pass for the data inputs, in order; the defaults if ``data`` is ``None``."""
        if data is None:
            return self._default_data
        if not self._data_names:
            msg = (
                "This likelihood has its observed data baked in; compile it "
                "with data_inputs=True to pass data=."
            )
            raise ValueError(msg)
        unknown = sorted(set(data) - set(self._data_names))
        if unknown:
            msg = f"No observed data input for: {unknown}; the inputs are {list(self._data_names)}"
            raise KeyError(msg)
        missing = [name for name in self._data_names if name not in data]
        if missing:
            msg = f"Missing observed data for: {missing}"
            raise KeyError(msg)
        arrays = []
        for name, default in zip(self._data_names, self._default_data, strict=True):
            array = np.asarray(data[name], dtype=np.float64)
            if array.shape != default.shape:
                msg = (
                    f"Observed data for '{name}' has shape {array.shape}; "
                    f"expected {default.shape}, one count per bin."
                )
                raise ValueError(msg)
            arrays.append(array)
        return tuple(arrays)

    def _arguments(
        self, x: npt.ArrayLike, data: Mapping[str, npt.ArrayLike] | None
    ) -> tuple[bytes, tuple[npt.NDArray[np.float64], ...]]:
        """Results-cache key and the arguments of the compiled functions."""
        arrays = (self._check_argument(x), *self._check_data(data))
        # The data arrays have fixed shapes, so the concatenated bytes
        # identify the parameter vector (or batch) and the data together.
        return b"".join(item.tobytes() for item in arrays), arrays

    def _evaluate(
        self, x: npt.ArrayLike, data: Mapping[str, npt.ArrayLike] | None
    ) -> list[npt.NDArray[np.float64]]:
        key, (array, *arrays) = self._arguments(x, data)
        if key != self._last_x:
            self._last_out = self._fn(array, *arrays)
            self._last_x = key
        return self._last_out

    def value(
        self, x: npt.ArrayLike, data: Mapping[str, npt.ArrayLike] | None = None
    ) -> float:
        """Negative log-likelihood at ``x``.

        Args:
            x: Parameter vector, shape ``(P,)``, ordered as
                :attr:`parameter_names`.
            data: Binned observed counts to evaluate on instead of the
                model's own (``data_inputs=True`` only), keyed like
                :attr:`pyhs3.Model.observed_data`: one ``(B,)`` array per
                HistFactory channel, ``B`` the channel's total number of
                bins, for every channel.  ``None`` uses the model's observed
                data.

        Raises:
            ValueError: If ``x`` or an array of ``data`` has the wrong shape,
                or if ``data`` is given to a likelihood compiled without
                ``data_inputs=True``.
            KeyError: If ``data`` names a channel without observed data, or
                misses one.
        """
        return float(self._evaluate(x, data)[0])

    def grad(
        self, x: npt.ArrayLike, data: Mapping[str, npt.ArrayLike] | None = None
    ) -> npt.NDArray[np.float64]:
        """Gradient of the negative log-likelihood at ``x``, shape ``(P,)``.

        ``data`` is as for :meth:`value`.
        """
        return self.value_and_grad(x, data)[1]

    def value_and_grad(
        self, x: npt.ArrayLike, data: Mapping[str, npt.ArrayLike] | None = None
    ) -> tuple[float, npt.NDArray[np.float64]]:
        """Negative log-likelihood and its gradient at ``x`` from one evaluation.

        ``data`` is as for :meth:`value`.
        """
        if not self.has_gradient:
            msg = "This likelihood was compiled with gradient=False."
            raise RuntimeError(msg)
        out = self._evaluate(x, data)
        return float(out[0]), np.array(out[1])

    def hessian(
        self, x: npt.ArrayLike, data: Mapping[str, npt.ArrayLike] | None = None
    ) -> npt.NDArray[np.float64]:
        """Hessian of the negative log-likelihood at ``x``, shape ``(P, P)``.

        Entry ``[i, j]`` is the second derivative with respect to parameters
        ``i`` and ``j`` of :attr:`parameter_names`.  The matrix is float64
        and exactly symmetric.  At a minimum, its inverse is the covariance
        estimate of the fitted parameters.  The Hessian is compiled at the
        first call, so a likelihood that never needs it pays nothing.
        ``data`` is as for :meth:`value`.
        """
        key, arrays = self._arguments(x, data)
        if key != self._last_hessian_x:
            if self._hessian_fn is None:
                self._hessian_fn = self._build_hessian()
            out = np.asarray(self._hessian_fn(*arrays)[0], dtype=np.float64)
            # The two derivative orders agree only up to rounding; averaging
            # with the transpose (of the last two axes, for a batch) makes the
            # result exactly symmetric.
            self._last_hessian = 0.5 * (out + np.swapaxes(out, -1, -2))
            self._last_hessian_x = key
        return self._last_hessian.copy()


class BatchedCompiledNLL(CompiledNLL):
    """Negative log-likelihood evaluated at a batch of parameter vectors.

    Build it with ``Model.compile_nll(backend="jax", batched=True)``.
    :meth:`value`, :meth:`grad` and :meth:`value_and_grad` take a ``(K, P)``
    array, one parameter vector per row with columns ordered as
    :attr:`parameter_names`, for any batch size ``K``, and return one result
    per row: :meth:`value` gives shape ``(K,)`` and :meth:`grad` ``(K, P)``.
    :attr:`initial`, :attr:`bounds`, :meth:`to_vector` and :meth:`to_dict`
    describe one parameter vector, as on :class:`CompiledNLL`; stack such
    vectors as rows to build a batch.

    XLA compiles the function once per batch size, at the first call with
    that ``K``; later calls with the same ``K`` reuse it.

    With ``data_inputs=True``, the ``data`` argument holds one ``(B,)`` array
    per channel, as for :class:`CompiledNLL`, shared by every row of the
    batch.

    Example:
        A grid of 9 values of one parameter, the others at their initial
        values:

        .. code-block:: python

            nll = model.compile_nll(backend="jax", batched=True)
            X = np.tile(nll.initial, (9, 1))  # (K, P)
            X[:, nll.parameter_names.index("mu")] = np.linspace(0.0, 4.0, 9)
            values, grads = nll.value_and_grad(X)  # (9,), (9, P)
    """

    def _check_argument(self, x: npt.ArrayLike) -> npt.NDArray[np.float64]:
        array = np.asarray(x, dtype=self._dtype)
        if array.ndim != 2 or array.shape[1] != len(self.parameter_names):
            msg = (
                f"Parameter batch has shape {array.shape}; expected "
                f"(K, {len(self.parameter_names)}), one parameter vector per row "
                "ordered as parameter_names."
            )
            raise ValueError(msg)
        return array

    def value(  # type: ignore[override]
        self, x: npt.ArrayLike, data: Mapping[str, npt.ArrayLike] | None = None
    ) -> npt.NDArray[np.float64]:
        """Negative log-likelihood per row of ``x``, shape ``(K,)``."""
        return np.array(self._evaluate(x, data)[0])

    def value_and_grad(  # type: ignore[override]
        self, x: npt.ArrayLike, data: Mapping[str, npt.ArrayLike] | None = None
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Negative log-likelihood ``(K,)`` and gradient ``(K, P)`` per row of ``x``."""
        if not self.has_gradient:
            msg = "This likelihood was compiled with gradient=False."
            raise RuntimeError(msg)
        out = self._evaluate(x, data)
        return np.array(out[0]), np.array(out[1])


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
    batched: bool = False,
    parameters_of_interest: Sequence[str] = (),
    data: Mapping[str, tuple[TensorVar, npt.ArrayLike]] | None = None,
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
        batched: Evaluate a ``(K, P)`` batch of parameter vectors per call
            (``backend="jax"`` only); returns a :class:`BatchedCompiledNLL`.
        parameters_of_interest: Names of the parameters of interest, exposed
            as :attr:`CompiledNLL.parameters_of_interest` with their
            :attr:`CompiledNLL.poi_indices`; names of variables not in
            ``free`` are left out.  They do not change the compiled function.
        data: Variables of ``log_prob`` (e.g. the constants holding observed
            bin counts) to replace by per-call inputs, each with its default
            value, keyed by the name the ``data`` argument of
            :meth:`CompiledNLL.value` uses.  The compiled function takes them
            after the parameter vector, in this order.

    Returns:
        CompiledNLL: The compiled likelihood.
    """
    if backend not in ("c", "jax"):
        msg = f"Unknown backend {backend!r}; use 'c' or 'jax'."
        raise ValueError(msg)
    if backend == "jax" and (fuse or mode is not None):
        msg = "fuse= and mode= configure the C backend; they cannot be used with backend='jax'."
        raise ValueError(msg)
    if batched and backend != "jax":
        msg = "batched=True requires backend='jax'."
        raise ValueError(msg)
    if mode is not None and fuse:
        msg = "fuse=True selects the default mode with fusion; do not combine it with mode=."
        raise ValueError(msg)
    if backend == "jax" and jax is None:
        msg = "compile_nll(backend='jax') requires JAX. Install with `pip install pyhs3[jax]`."
        raise ImportError(msg)

    dtype = str(config.floatX)
    nll = -pt.sum(log_prob)  # type: ignore[no-untyped-call]
    # Differentiate with respect to the original scalars and stack, rather
    # than with respect to x[i], which would scatter each component back
    # into x with an IncSubtensor.  With no free parameters the gradient
    # is an empty (0,) vector.
    gradient_graph = (
        pt.stack(cast(list[TensorVar], grad(nll, list(free))))
        if free
        else cast(TensorVar, pt.zeros((0,), dtype=dtype))
    )
    outputs: list[TensorVar] = [nll, gradient_graph] if gradient else [nll]

    x = cast(TensorVar, pt.vector("x", dtype=dtype))
    givens: list[tuple[Variable[Any, Any], Variable[Any, Any]]] = [
        (var, x[index])  # pylint: disable=unsubscriptable-object
        for index, var in enumerate(free)
    ]
    givens.extend(
        (var, pt.constant(np.asarray(value, dtype=var.dtype)))
        for var, value in baked.items()
    )
    # Each data variable is replaced by a fresh variable of its type (and so
    # of its static shape), an input of the compiled function.
    data = data or {}
    data_inputs: list[TensorVar] = []
    for var, _ in data.values():
        data_inputs.append(cast(TensorVar, var.type(name=var.name)))
        givens.append((var, data_inputs[-1]))
    defaults = {name: value for name, (_, value) in data.items()}
    names = [var.name or f"x[{index}]" for index, var in enumerate(free)]

    inputs = [x, *data_inputs]
    c_mode = default_nll_mode(fuse=fuse) if mode is None else mode
    builder: _CBuilder | _JaxBuilder
    if backend == "jax":
        builder = _JaxBuilder(
            outputs, givens, inputs, len(free), gradient=gradient, batched=batched
        )
    else:
        builder = _CBuilder(outputs, givens, inputs, c_mode)

    cache_dir: Path | None = None
    if cache:
        cache_dir = _default_cache_dir() if cache is True else Path(cache)
        if backend == "jax":
            _use_xla_cache(cache_dir / "xla")

    def cache_path(kind: str, graph: Sequence[TensorVar], variant: str) -> Path | None:
        """Cache file of the function of ``graph``; ``None`` without a cache."""
        if cache_dir is None:
            return None
        key = hashlib.sha256(
            "|".join(
                [
                    backend,
                    _graph_fingerprint(
                        [*graph, *(new for _, new in givens)],
                        as_inputs=[var for var, _ in data.values()],
                    ),
                    repr(names),
                    _mode_key(mode, fuse=fuse),
                    variant,
                    str(batched),
                    pyhs3_version,
                    pytensor.__version__,
                    jax.__version__ if backend == "jax" and jax is not None else "",
                ]
            ).encode()
        ).hexdigest()
        return cache_dir / f"{kind}-{key}.{builder.suffix}"

    def build_hessian() -> EvalFn:
        # JAX differentiates the transpiled NLL twice; the C backend takes
        # the Jacobian of the symbolic gradient.
        hessian_builder: _CBuilder | _JaxBuilder
        if backend == "jax":
            hessian_builder = _JaxBuilder(
                [nll],
                givens,
                inputs,
                len(free),
                gradient=False,
                batched=batched,
                hessian=True,
            )
            graph = nll
        else:
            hessian_builder = _CBuilder(
                [gradient_graph], givens, inputs, c_mode, hessian_wrt=free
            )
            graph = gradient_graph
        return _load_or_build(
            hessian_builder, cache_path("hessian", [graph], "hessian")
        )[0]

    fn, from_cache = _load_or_build(builder, cache_path("nll", outputs, str(gradient)))
    return (BatchedCompiledNLL if batched else CompiledNLL)(
        fn,
        names,
        initial,
        bounds,
        dtype,
        has_gradient=gradient,
        build_hessian=build_hessian,
        from_cache=from_cache,
        parameters_of_interest=parameters_of_interest,
        data=defaults,
    )


def _load_or_build(
    builder: _CBuilder | _JaxBuilder, path: Path | None
) -> tuple[EvalFn, bool]:
    """Load the compiled function from ``path``, or build it and write it there.

    Returns the function and whether it was loaded.  Without a ``path``
    (no cache) the function is built and not written.
    """
    if path is None:
        return builder.build(), False
    if path.exists():
        fn = builder.load(path.read_bytes())
        log.debug("Loaded compiled function from %s", path)
        return fn, True
    fn = builder.build()
    _publish(path, builder.dump)
    log.debug("Cached compiled function at %s", path)
    return fn, False


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
        inputs: list[TensorVar],
        mode: Mode | str,
        *,
        hessian_wrt: Sequence[TensorVar] | None = None,
    ) -> None:
        """Hold the graph to compile.

        With ``hessian_wrt``, ``outputs`` is the ``(P,)`` gradient with
        respect to those ``P`` scalars, and the compiled function returns
        its ``(P, P)`` Jacobian, the Hessian, instead.
        """
        self._outputs = outputs
        self._givens = givens
        self._inputs = inputs
        self._mode = mode
        self._hessian_wrt = hessian_wrt
        self._fn: EvalFn | None = None

    def build(self) -> EvalFn:
        outputs = self._outputs
        mode = self._mode
        if self._hessian_wrt is not None:
            # Vectorizing the backward pass turns the gradient's
            # IncSubtensors into Blockwise ones.  Rewriting those back into
            # plain IncSubtensors infers static shapes over the whole graph
            # each time, which on a large workspace makes the compile an
            # order of magnitude slower than the gradient's.  Kept as
            # Blockwise they loop over the P columns in Python instead: a
            # slower evaluation, but a Hessian is evaluated a few times per
            # fit, not once per minimizer step.
            mode = get_mode(mode).excluding("local_blockwise_inc_subtensor")  # type: ignore[no-untyped-call]
            wrt = list(self._hessian_wrt)
            hessian = cast(TensorVar, pt.zeros((0, 0), dtype=outputs[0].dtype))
            if wrt:
                # Column j of the Hessian is the (P,) derivative of the
                # gradient with respect to scalar j; vectorize=True computes
                # all P columns in one batched backward pass rather than a
                # scan over rows.
                columns = jacobian(outputs[0], wrt, vectorize=True)  # type: ignore[no-untyped-call]
                hessian = pt.stack(cast(list[TensorVar], columns), axis=1)
            outputs = [hessian]
        self._fn = cast(
            EvalFn,
            function(
                self._inputs,
                outputs,
                givens=self._givens,
                mode=mode,
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
        inputs: list[TensorVar],
        n_free: int,
        *,
        gradient: bool,
        batched: bool,
        hessian: bool = False,
    ) -> None:
        """Hold the graph to transpile.

        With ``hessian=True``, ``outputs`` is ``[nll]`` and the compiled
        function returns its ``(P, P)`` Hessian (``(K, P, P)`` for a batch)
        from ``jax.hessian``.
        """
        # Likelihoods need double precision; JAX defaults to float32.  This
        # is a process-wide JAX setting.
        jax.config.update("jax_enable_x64", True)
        self._outputs = outputs
        self._givens = givens
        # The parameter vector x, then the data inputs.
        self._inputs = inputs
        # Batched input is (K, P) with a symbolic batch size, so one exported
        # module serves every K.  Data inputs have their static (B,) shapes
        # and are shared by every row of a batch.
        shape = jax_export.symbolic_shape(f"K, {n_free}") if batched else (n_free,)
        self._specs = [
            jax.ShapeDtypeStruct(shape, np.float64),
            *(jax.ShapeDtypeStruct(var.type.shape, np.float64) for var in inputs[1:]),
        ]
        self._gradient = gradient
        self._batched = batched
        self._hessian = hessian
        self._exported: Any = None

    def _compile(self, exported: Any) -> EvalFn:
        """Compile an exported module and wrap it for CompiledNLL.

        A one-point module is compiled ahead of time.  XLA needs concrete
        shapes, so a batched module is compiled by ``jax.jit`` once per batch
        size, at the first call with that size.
        """
        self._exported = exported
        call = jax.jit(exported.call)
        compiled = call if self._batched else call.lower(*self._specs).compile()
        gradient = self._gradient
        hessian = self._hessian

        def evaluate(
            x: npt.NDArray[np.float64], *data: npt.NDArray[np.float64]
        ) -> list[npt.NDArray[np.float64]]:
            # [nll] or [nll, *grad] on the last axis: shape (1 + P,), or
            # (K, 1 + P) for a (K, P) batch.  A Hessian is (P, P), or
            # (K, P, P) for a batch.
            out = np.asarray(compiled(x, *data))
            if hessian:
                return [out]
            return [out[..., 0], out[..., 1:]] if gradient else [out[..., 0]]

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
        graph = jaxify(cast(TensorVar, joint), inputs=self._inputs)

        def evaluate_graph(xv: Any, *data: Any) -> Any:
            return cast(Sequence[Any], graph.call_positional(xv, *data))[0]

        # jax.hessian differentiates with respect to the parameter vector,
        # the first argument: forward-over-reverse on the (1,) NLL, then the
        # leading axis of length 1 is dropped.
        def hessian_at(xv: Any, *data: Any) -> Any:
            return jax.hessian(evaluate_graph)(xv, *data)[0]

        one_point = hessian_at if self._hessian else evaluate_graph

        # A batch maps the one-point function over the rows of x, with the
        # data shared by every row; this has to happen before export, since
        # JAX cannot vmap an exported module.
        in_axes = (0, *(None for _ in self._inputs[1:]))
        traced = jax.jit(
            jax.vmap(one_point, in_axes=in_axes) if self._batched else one_point
        )
        return self._compile(jax_export.export(traced)(*self._specs))

    def dump(self, sink: IO[bytes]) -> None:
        sink.write(self._exported.serialize())

    def load(self, data: bytes) -> EvalFn:
        return self._compile(jax_export.deserialize(bytearray(data)))
