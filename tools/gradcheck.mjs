// Finite-difference check of every tensor op used by the mini GPT.
// Run: node tools/gradcheck.mjs
import { createModel, forward, namedParams } from '../assets/js/core/gpt.js';
import { backward } from '../assets/js/core/tensor.js';
import { mulberry32 } from '../assets/js/core/rng.js';

const cfg = { vocabSize: 11, blockSize: 5, nLayer: 2, nHead: 2, nEmbd: 8 };
const model = createModel(cfg, 3);
// make LayerNorm params non-trivial
const r = mulberry32(5);
for (const [, t] of namedParams(model)) for (let i = 0; i < t.size; i++) t.data[i] += (r() - 0.5) * 0.2;
const B = 2, T = 5;
const ids = Int32Array.from({ length: B * T }, () => Math.floor(r() * 11));
const tg = Int32Array.from({ length: B * T }, () => Math.floor(r() * 11));
const lossOf = () => forward(model, ids, B, T, tg).loss.data[0];
const { loss } = forward(model, ids, B, T, tg);
backward(loss);
let worst = 0;
for (const [name, t] of namedParams(model)) {
  for (let k = 0; k < 4; k++) {
    const i = Math.floor(r() * t.size);
    const old = t.data[i];
    const h = 1e-2;
    t.data[i] = old + h; const lp = lossOf();
    t.data[i] = old - h; const lm = lossOf();
    t.data[i] = old;
    const num = (lp - lm) / (2 * h);
    const an = t.grad[i];
    const rel = Math.abs(num - an) / Math.max(1e-4, Math.abs(num) + Math.abs(an));
    worst = Math.max(worst, rel);
    if (rel > 0.05) console.log('MISMATCH', name, i, num, an);
  }
}
console.log('worst relative error', worst.toFixed(4));
