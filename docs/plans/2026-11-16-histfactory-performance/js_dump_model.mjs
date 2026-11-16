// Dumps the histfactory_js model's flattened structure (sparse, COO-style index arrays) plus reference
// 2NLL values and gradients at four points, so hf_kernel.py can build the identical model and be checked.
//
// Usage: node js_dump_model.mjs ENGINE_DIR WORKSPACE.json OUT.json
import fs from "node:fs";
import { loadEngine, loadModel } from "./js_engine.mjs";

const [engineDir, workspacePath, outPath] = process.argv.slice(2);
const HF = loadEngine(engineDir);
const { model: m, likelihood } = loadModel(HF, workspacePath);

// One row per (block, modifier) or (cell, modifier) entry. A block is a (channel, sample) pair;
// a cell is a (sample, bin) pair.
const cellBin = [];
const cellBlock = [];
const nominal = [];
const scalBlock = [];
const scalPar = [];
const nsBlock = [];
const nsPar = [];
const nsLnHi = [];
const nsLnLo = [];
const nsCoef = [];
const hsCell = [];
const hsPar = [];
const hsUp = [];
const hsDn = [];
const gamCell = [];
const gamPar = [];
for (const b of m.blocks) {
  for (let i = 0; i < b.n; i++) {
    cellBin.push(b.offset + i);
    cellBlock.push(b.id);
    nominal.push(b.nominal[i]);
  }
  for (const p of b.scal) {
    scalBlock.push(b.id);
    scalPar.push(p);
  }
  for (let k = 0; k < b.nsPar.length; k++) {
    nsBlock.push(b.id);
    nsPar.push(b.nsPar[k]);
    nsLnHi.push(b.nsLnHi[k]);
    nsLnLo.push(b.nsLnLo[k]);
    nsCoef.push(...b.nsCoef.slice(6 * k, 6 * k + 6));
  }
  for (let k = 0; k < b.hsPar.length; k++) {
    for (let i = 0; i < b.n; i++) {
      hsCell.push(b.cell + i);
      hsPar.push(b.hsPar[k]);
      hsUp.push(b.hsUp[k * b.n + i]);
      hsDn.push(b.hsDn[k * b.n + i]);
    }
  }
  for (const list of b.gam) {
    for (let i = 0; i < b.n; i++) {
      if (list[i] >= 0) {
        gamCell.push(b.cell + i);
        gamPar.push(list[i]);
      }
    }
  }
}

// Reference points: the initial values, then three seeded perturbations of +-0.3 prefit widths.
let seed = 12345;
const uniform = () => (seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648;
const refs = [];
for (let r = 0; r < 4; r++) {
  const x = Float64Array.from(m.init, (v, p) => {
    const shift = r === 0 ? 0 : (uniform() - 0.5) * 0.6 * (m.prefitUnc[p] || 0.1);
    return Math.min(Math.max(v + shift, m.lower[p]), m.upper[p]);
  });
  const g = new Float64Array(m.nPars);
  const f = likelihood.value(x, g);
  refs.push({ x: [...x], f, g: [...g] });
}

fs.writeFileSync(
  outPath,
  JSON.stringify({
    nPars: m.nPars,
    nBins: m.nBins,
    nBlocks: m.blocks.length,
    nCells: m.nCells,
    names: m.names,
    poiIndex: m.poiIndex,
    init: [...m.init],
    lower: [...m.lower],
    upper: [...m.upper],
    fixed: [...m.fixed],
    prefitUnc: [...m.prefitUnc],
    observed: [...m.observed.subarray(0, m.nBins)],
    auxdata: m.auxdata,
    auxIndex: [...m.auxIndex],
    gaussIdx: [...m.gaussIdx],
    gaussSigma: [...m.gaussSigma],
    poisIdx: [...m.poisIdx],
    poisTau: [...m.poisTau],
    cellBin,
    cellBlock,
    nominal,
    scalBlock,
    scalPar,
    nsBlock,
    nsPar,
    nsLnHi,
    nsLnLo,
    nsCoef,
    hsCell,
    hsPar,
    hsUp,
    hsDn,
    gamCell,
    gamPar,
    refs,
  }),
);
console.log({ cells: cellBin.length, scal: scalPar.length, normsys: nsPar.length, histosysEntries: hsPar.length, gammaEntries: gamPar.length });
