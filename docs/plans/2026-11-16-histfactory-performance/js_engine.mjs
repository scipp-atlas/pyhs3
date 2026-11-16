// Shared loader for the histfactory_js engine files, so every harness builds the model the same way.
//
// ENGINE_DIR must contain numerics.js, workspace.js, model.js, minimize.js and infer.js from
// histfactory_js (not redistributed here; see README.md).
import fs from "node:fs";
import vm from "node:vm";

const ENGINE_FILES = ["numerics", "workspace", "model", "minimize", "infer"];

// Evaluates the engine scripts in this context; they attach themselves to globalThis.HF.
export function loadEngine(engineDir) {
  for (const name of ENGINE_FILES) {
    const path = `${engineDir}/${name}.js`;
    vm.runInThisContext(fs.readFileSync(path, "utf8"), { filename: path });
  }
  return globalThis.HF;
}

// Builds the model for the first measurement and its likelihood on the observed data.
export function loadModel(HF, workspacePath) {
  const ws = HF.parseWorkspace(fs.readFileSync(workspacePath, "utf8"));
  const model = new HF.Model(ws, 0);
  return { model, likelihood: model.likelihood(model.observed) };
}

// Indices of the parameters that are not held constant.
export function freeIndices(model) {
  const free = [];
  for (let p = 0; p < model.nPars; p++) if (!model.fixed[p]) free.push(p);
  return free;
}

// Prefit width the engine uses for a parameter: its constraint width, else a tenth of its range.
export function prefitWidth(model, p) {
  return model.prefitUnc[p] > 0 ? model.prefitUnc[p] : Math.max((model.upper[p] - model.lower[p]) / 10, 1e-3);
}

// Finite-difference step the engine uses for second derivatives (infer.js stepSize).
export function hessianStep(model, x, p) {
  return 1e-4 * (model.prefitUnc[p] > 0 ? model.prefitUnc[p] : Math.max(Math.abs(x[p]), 1));
}
