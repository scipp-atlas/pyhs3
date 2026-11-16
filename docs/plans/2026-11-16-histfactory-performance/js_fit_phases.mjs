// Times one full histfactory_js fit and splits its likelihood calls by phase, then runs a
// warm-started 21-point profile scan of the parameter of interest.
//
// Usage: node js_fit_phases.mjs ENGINE_DIR WORKSPACE.json
import { loadEngine, loadModel } from "./js_engine.mjs";

const [engineDir, workspacePath] = process.argv.slice(2);
const HF = loadEngine(engineDir);
const tBuild = performance.now();
const { model, likelihood } = loadModel(HF, workspacePath);
const buildMs = performance.now() - tBuild;

let calls = 0;
const value = likelihood.value;
likelihood.value = (x, g) => {
  calls++;
  return value(x, g);
};

// Per-call cost of the likelihood alone, without and with the gradient.
const x0 = Float64Array.from(model.init);
const grad = new Float64Array(model.nPars);
const reps = 2000;
let t = performance.now();
for (let i = 0; i < reps; i++) value(x0);
const valueMs = (performance.now() - t) / reps;
t = performance.now();
for (let i = 0; i < reps; i++) value(x0, grad);
const valueGradMs = (performance.now() - t) / reps;

// Wrap the two call sites inside infer.fit so each records its own calls and wall time.
const phases = [];
const wrap = (name, fn) => (...args) => {
  const c0 = calls;
  const t0 = performance.now();
  const result = fn(...args);
  phases.push({ phase: name, calls: calls - c0, ms: Math.round(performance.now() - t0), iterations: result.iterations });
  return result;
};
HF.hessian = wrap("hessian", HF.hessian);
HF.minimize = wrap("bfgs", HF.minimize);

const c0 = calls;
t = performance.now();
const fit = await HF.infer.fit(model, likelihood, {});
const fitMs = performance.now() - t;
const accounted = phases.reduce((sum, p) => sum + p.calls, 0);
const poi = model.poiIndex;
console.log(
  JSON.stringify(
    {
      nChannels: model.channels.length,
      nBins: model.nBins,
      nPars: model.nPars,
      nBlocks: model.blocks.length,
      nCells: model.nCells,
      buildMs: Math.round(buildMs),
      valueMs,
      valueGradMs,
      fit: { ms: Math.round(fitMs), calls: calls - c0, phases, newtonPolishCalls: calls - c0 - accounted },
      twiceNll: fit.twiceNll,
      poi: model.poiName,
      poiHat: fit.pars[poi],
      poiUnc: fit.unc[poi],
    },
    null,
    1,
  ),
);

phases.length = 0;
const c1 = calls;
t = performance.now();
const lo = fit.pars[poi] - 2 * fit.unc[poi];
const hi = fit.pars[poi] + 2 * fit.unc[poi];
const scan = await HF.infer.scan(model, likelihood, fit, poi, HF.num.linspace(lo, hi, 21));
console.log(
  JSON.stringify({
    scan21: {
      ms: Math.round(performance.now() - t),
      calls: calls - c1,
      bfgsIterationsPerPoint: phases.map((p) => p.iterations),
      interval: scan.interval,
    },
  }),
);
