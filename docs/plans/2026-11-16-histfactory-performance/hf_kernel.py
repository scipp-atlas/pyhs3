"""Vectorised HistFactory 2NLL as a fixed-size PyTensor graph over sparse index arrays.

The model enters as data (index arrays and precomputed code4 coefficients), not as graph structure,
so the graph has the same few hundred nodes however many modifiers the workspace has. The arrays
come from js_dump_model.mjs, so the model is exactly the one histfactory_js evaluates.

Shapes used throughout:
    P  parameters             E  normsys entries (one per (block, normsys modifier))
    H  histosys entries       C  cells (one per (sample, bin) present in a channel)
    K  blocks ((channel, sample) pairs)     B  bins (all channels concatenated)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import pytensor.tensor as pt
from scipy.special import gammaln

IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]

ProdMode = Literal["prod", "unrolled"]


def padded_index(rows: Any, cols: Any, nrows: int) -> tuple[IntArray, BoolArray]:
    """Group ``cols`` by ``rows`` into a padded (nrows, kmax) index matrix and its (nrows, kmax) fill mask.

    Row r of the index matrix lists, in input order, every col whose row is r; the remaining slots
    hold index 0 and are False in the mask.
    """
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    counts = np.bincount(rows, minlength=nrows)
    kmax = max(int(counts.max()) if counts.size else 0, 1)
    index = np.zeros((nrows, kmax), dtype=np.int64)
    mask = np.zeros((nrows, kmax), dtype=bool)
    order = np.argsort(rows, kind="stable")
    sorted_rows = rows[order]
    # position of each entry within its row: its rank minus the number of entries in earlier rows
    slot = np.arange(len(sorted_rows)) - np.repeat(np.cumsum(counts) - counts, counts)
    index[sorted_rows, slot] = cols[order]
    mask[sorted_rows, slot] = True
    return index, mask


def padded_prod(
    table: pt.TensorVariable, index: IntArray, mask: BoolArray, mode: ProdMode
) -> pt.TensorVariable:
    """Row-wise product of ``table[index]`` over the filled slots; padded slots contribute 1.

    ``table`` is (M,), ``index``/``mask`` are (rows, kmax), and the result is (rows,).

    ``"prod"`` uses one ``pt.prod`` over the padded axis: fewest nodes, but PyTensor implements neither
    its JAX gradient nor its second derivative. ``"unrolled"`` multiplies the kmax columns one by one,
    which every backend can differentiate twice.
    """
    if mode == "prod":
        return pt.prod(pt.switch(mask, table[index], 1.0), axis=1)
    product = None
    for j in range(index.shape[1]):
        column = (
            table[index[:, j]]
            if mask[:, j].all()
            else pt.switch(mask[:, j], table[index[:, j]], 1.0)
        )
        product = column if product is None else product * column
    return product


def load(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build(
    m: dict[str, Any], prod_mode: ProdMode = "prod"
) -> tuple[pt.TensorVariable, pt.TensorVariable]:
    """Return ``(theta, twice_nll)`` for the dumped model ``m``: theta is a (P,) input, twice_nll a scalar."""
    n_bins, n_blocks, n_cells = m["nBins"], m["nBlocks"], m["nCells"]
    theta = pt.dvector("theta")  # (P,)

    # normsys: code4 polynomial for |a| < 1, exponential extrapolation outside.
    ns_par = np.asarray(m["nsPar"], dtype=np.int64)  # (E,)
    coef = np.asarray(m["nsCoef"]).reshape(
        -1, 6
    )  # (E, 6): the 6 polynomial coefficients, precomputed once
    ln_hi = np.asarray(m["nsLnHi"])  # (E,)
    ln_lo = np.asarray(m["nsLnLo"])  # (E,)
    a = theta[ns_par]  # (E,)
    poly = 1 + a * (
        coef[:, 0]
        + a
        * (
            coef[:, 1]
            + a * (coef[:, 2] + a * (coef[:, 3] + a * (coef[:, 4] + a * coef[:, 5])))
        )
    )
    # Each exponential's argument is clipped into the branch where it is used, so the branch the switch
    # discards can neither overflow nor leak a nonzero gradient.
    normsys = pt.switch(
        a >= 1,
        pt.exp(pt.maximum(a, 1) * ln_hi),
        pt.switch(a <= -1, pt.exp(-pt.minimum(a, -1) * ln_lo), poly),
    )  # (E,)

    # Per-block overall factor: product of the block's normfactors/lumi times product of its normsys factors.
    scal_index, scal_mask = padded_index(
        m["scalBlock"], m["scalPar"], n_blocks
    )  # (K, s_max) into theta
    ns_index, ns_mask = padded_index(
        m["nsBlock"], np.arange(len(ns_par)), n_blocks
    )  # (K, e_max) into normsys
    block_factor = padded_prod(theta, scal_index, scal_mask, prod_mode) * padded_prod(
        normsys, ns_index, ns_mask, prod_mode
    )  # (K,)

    # histosys: additive per-cell shifts, linear outside |a| <= 1 and a 6th-order polynomial inside.
    hs_par = np.asarray(m["hsPar"], dtype=np.int64)  # (H,)
    hs_cell = np.asarray(m["hsCell"], dtype=np.int64)  # (H,)
    up = np.asarray(m["hsUp"])  # (H,): hi - nominal
    dn = np.asarray(m["hsDn"])  # (H,): nominal - lo
    ah = theta[hs_par]
    ah2 = ah * ah
    inner = 0.5 * (up + dn) * ah + 0.0625 * (up - dn) * ah2 * (
        15 + ah2 * (-10 + ah2 * 3)
    )
    shift = pt.switch(ah > 1, up * ah, pt.switch(ah < -1, dn * ah, inner))  # (H,)
    # scatter-add the H shifts into their C cells
    cell_base = pt.inc_subtensor(
        pt.as_tensor_variable(np.asarray(m["nominal"]))[hs_cell], shift
    )  # (C,)

    # Per-cell product of bin-wise factors (staterror, shapesys, shapefactor).
    gam_index, gam_mask = padded_index(
        m["gamCell"], m["gamPar"], n_cells
    )  # (C, g_max) into theta
    cell_gamma = padded_prod(theta, gam_index, gam_mask, prod_mode)  # (C,)

    cell_block = np.asarray(m["cellBlock"], dtype=np.int64)  # (C,)
    cell_bin = np.asarray(m["cellBin"], dtype=np.int64)  # (C,)
    cell_nu = cell_base * block_factor[cell_block] * cell_gamma  # (C,)
    nu = pt.inc_subtensor(
        pt.zeros((n_bins,), dtype="float64")[cell_bin], cell_nu
    )  # (B,): sum of the cells per bin

    observed = np.asarray(m["observed"])  # (B,)
    aux = np.asarray(m["auxdata"])
    aux_index = np.asarray(m["auxIndex"])
    gauss_index = np.asarray(m["gaussIdx"], dtype=np.int64)
    gauss_sigma = np.asarray(m["gaussSigma"])
    pois_index = np.asarray(m["poisIdx"], dtype=np.int64)
    pois_tau = np.asarray(m["poisTau"])
    gauss_aux = aux[aux_index[gauss_index]]
    pois_aux = aux[aux_index[pois_index]]
    # Normalisation constants of every Poisson and Gaussian term, computed once outside the graph.
    constant = (
        2 * gammaln(observed + 1).sum()
        + (2 * np.log(gauss_sigma) + np.log(2 * np.pi)).sum()
        + 2 * gammaln(pois_aux + 1).sum()
    )
    # Bins with n = 0 contribute 2*nu and no log term, even where nu = 0. Elsewhere nu <= 0 gives a
    # non-finite 2NLL, which the minimiser treats as outside the model's domain (as the JS does).
    main = 2 * pt.sum(nu - pt.switch(observed > 0, observed * pt.log(nu), 0.0))
    gauss = pt.sum(((gauss_aux - theta[gauss_index]) / gauss_sigma) ** 2)
    rate = theta[pois_index] * pois_tau
    pois = 2 * pt.sum(rate - pt.switch(pois_aux > 0, pois_aux * pt.log(rate), 0.0))
    return theta, constant + main + gauss + pois
