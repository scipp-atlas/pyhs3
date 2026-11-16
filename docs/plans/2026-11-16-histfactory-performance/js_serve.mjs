// Serves the histfactory_js likelihood over stdin/stdout so Python optimisers can drive the identical function.
//
// Protocol: the first stdout line is a JSON model summary. Then, per request, the client writes
// 8*N bytes (float64 parameter vector) and reads back 8*(N+1) bytes (2NLL followed by its gradient).
//
// Usage: node js_serve.mjs ENGINE_DIR WORKSPACE.json   (normally started by js_bridge.py)
import { loadEngine, loadModel } from "./js_engine.mjs";

const [engineDir, workspacePath] = process.argv.slice(2);
const HF = loadEngine(engineDir);
const { model, likelihood } = loadModel(HF, workspacePath);
const N = model.nPars;
process.stdout.write(
  JSON.stringify({
    N,
    init: [...model.init],
    lower: [...model.lower],
    upper: [...model.upper],
    fixed: [...model.fixed],
    prefitUnc: [...model.prefitUnc],
    poiIndex: model.poiIndex,
    names: model.names,
  }) + "\n",
);

const requestBytes = 8 * N;
const grad = new Float64Array(N);
const reply = new Float64Array(N + 1);
let pending = Buffer.alloc(0);
process.stdin.on("data", (chunk) => {
  pending = Buffer.concat([pending, chunk]);
  while (pending.length >= requestBytes) {
    const x = new Float64Array(pending.buffer.slice(pending.byteOffset, pending.byteOffset + requestBytes));
    pending = pending.subarray(requestBytes);
    reply[0] = likelihood.value(x, grad);
    reply.set(grad, 1);
    process.stdout.write(Buffer.from(reply.buffer));
  }
});
