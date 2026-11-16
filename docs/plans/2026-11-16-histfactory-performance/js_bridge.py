"""Python client for js_serve.mjs: call the histfactory_js likelihood (2NLL and gradient) from Python."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import numpy.typing as npt

HERE = Path(__file__).resolve().parent

FloatArray = npt.NDArray[np.float64]


class JSLikelihood:
    """The JS likelihood behind a pipe, with the model summary the server sends on start-up.

    Calling the object with a (N,) parameter vector returns ``(twice_nll, grad)`` with ``grad`` of shape (N,).
    ``calls`` counts every call. A point where the expected yields are not positive gives
    ``twice_nll = inf``, exactly as the JS returns it.
    """

    def __init__(self, engine_dir: Path, workspace: Path) -> None:
        self._process = subprocess.Popen(
            ["node", str(HERE / "js_serve.mjs"), str(engine_dir), str(workspace)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
        )
        assert self._process.stdout is not None
        meta = json.loads(self._process.stdout.readline())
        self.n_pars: int = meta["N"]
        self.init: FloatArray = np.asarray(meta["init"], dtype=np.float64)
        self.lower: FloatArray = np.asarray(meta["lower"], dtype=np.float64)
        self.upper: FloatArray = np.asarray(meta["upper"], dtype=np.float64)
        self.poi_index: int = meta["poiIndex"]
        prefit = np.asarray(meta["prefitUnc"], dtype=np.float64)
        # same fallback width as the engine: a tenth of the parameter's range where it has no constraint
        self.prefit_width: FloatArray = np.where(
            prefit > 0, prefit, np.maximum((self.upper - self.lower) / 10, 1e-3)
        )
        self.calls = 0

    def __call__(self, x: FloatArray) -> tuple[float, FloatArray]:
        assert self._process.stdin is not None
        assert self._process.stdout is not None
        self.calls += 1
        self._process.stdin.write(np.ascontiguousarray(x, dtype=np.float64).tobytes())
        self._process.stdin.flush()
        reply = np.frombuffer(
            self._process.stdout.read(8 * (self.n_pars + 1)), dtype=np.float64
        ).copy()
        return float(reply[0]), reply[1:]

    def close(self) -> None:
        self._process.terminate()
        self._process.wait()
