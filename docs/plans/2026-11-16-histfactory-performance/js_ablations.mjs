// Ablations of the histfactory_js minimiser on one workspace:
//   1. is the seed Hessian at the starting point positive definite (i.e. does fit() actually use it)?
//   2. how much does the starting inverse-Hessian diagonal matter?
//   3. what does a finite-difference gradient (2n value calls per gradient) cost instead of the analytic one?
//
// Usage: node js_ablations.mjs ENGINE_DIR WORKSPACE.json
import { freeIndices, hessianStep, loadEngine, loadModel, prefitWidth } from "./js_engine.mjs";

const [engineDir, workspacePath] = process.argv.slice(2);
const HF = loadEngine(engineDir);
const { model, likelihood } = loadModel(HF, workspacePath);
const free = freeIndices(model);
const n = free.length;
const x0 = Float64Array.from(model.init);

// 1. Seed Hessian at the starting point, exactly as infer.fit computes it (forward differences).
const H0 = HF.hessian(likelihood.value, x0, free, free.map((p) => hessianStep(model, x0, p)), false);
console.log(JSON.stringify({ seedHessianPositiveDefinite: !!HF.num.cholesky(H0, n), n }));

// Counts every call of fg, and the value calls made inside a finite-difference gradient.
function counted(fg) {
  const counter = { calls: 0 };
  return [(x, g) => (counter.calls++, fg(x, g)), counter];
}

// Central-difference gradient from value calls only: what an optimiser does without a user gradient.
function finiteDifferenceGradient(valueOnly) {
  return (x, g) => {
    const f = valueOnly(x);
    if (!g) return f;
    const xp = Float64Array.from(x);
    for (const p of free) {
      const h = hessianStep(model, x, p);
      xp[p] = x[p] + h;
      const fp = valueOnly(xp);
      xp[p] = x[p] - h;
      const fm = valueOnly(xp);
      xp[p] = x[p];
      g[p] = (fp - fm) / (2 * h);
    }
    return f;
  };
}

function run(label, fgBase, diag) {
  let valueCalls = 0;
  const valueOnly = (x) => (valueCalls++, likelihood.value(x));
  const [fg, counter] = counted(fgBase === "fd" ? finiteDifferenceGradient(valueOnly) : fgBase);
  const t = performance.now();
  const result = HF.minimize(fg, x0, model.lower, model.upper, free, { diag: Float64Array.from(diag), tol: 1e-6, maxIter: 5000 });
  console.log(
    JSON.stringify({
      label,
      outerCalls: counter.calls,
      fdValueCalls: valueCalls,
      iterations: result.iterations,
      converged: result.converged,
      twiceNll: +result.f.toFixed(6),
      poiHat: +result.x[model.poiIndex].toFixed(5),
      ms: Math.round(performance.now() - t),
    }),
  );
}

const sigmaDiag = free.map((p) => 0.5 * prefitWidth(model, p) ** 2);
// 2. Starting inverse-Hessian diagonal. 2NLL of a unit-width Gaussian constraint has curvature 2/sigma^2.
run("analytic gradient, diag 0.5*sigma_prefit^2 (engine default)", likelihood.value, sigmaDiag);
run("analytic gradient, identity", likelihood.value, free.map(() => 1));
run("analytic gradient, identity * 1e-3", likelihood.value, free.map(() => 1e-3));
// 3. Same minimiser, gradient from 2n value calls. Slow: about two minutes on the HWW workspace.
run("finite-difference gradient, diag 0.5*sigma_prefit^2", "fd", sigmaDiag);
