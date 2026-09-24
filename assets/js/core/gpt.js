// Mini GPT: a character-level decoder-only transformer built on tensor.js.
// Same architecture as GPT-2, only tiny:
//   token embedding + position embedding
//   → [LayerNorm → causal self-attention → +residual → LayerNorm → MLP(4×, GELU) → +residual] × nLayer
//   → LayerNorm → linear head → next-token logits

import { Tensor, noGrad, backward, embedding, add, linear, layerNorm, gelu, causalAttention, crossEntropy, AdamW } from './tensor.js';
import { mulberry32 } from './rng.js';
import { decodeStep } from './sampling.js';

export const DEFAULT_CONFIG = { blockSize: 32, nLayer: 2, nHead: 4, nEmbd: 48 };

/**
 * @param {{ vocabSize: number, blockSize: number, nLayer: number, nHead: number, nEmbd: number }} config
 * @param {number} seed
 */
export function createModel(config, seed = 1337) {
  const { vocabSize: V, blockSize: T, nLayer: L, nEmbd: D } = config;
  if (D % config.nHead) throw new Error('nEmbd는 nHead로 나누어떨어져야 한다');
  const rand = mulberry32(seed);
  const std = 0.02;
  const projStd = std / Math.sqrt(2 * L); // GPT-2: scale residual projections by depth
  const p = {
    wte: Tensor.randn([V, D], std, rand),
    wpe: Tensor.randn([T, D], std, rand),
    layers: [],
    lnfG: Tensor.fill([D], 1, true),
    lnfB: Tensor.zeros([D], true),
    headW: Tensor.randn([D, V], std, rand),
    headB: Tensor.zeros([V], true),
  };
  for (let l = 0; l < L; l++) {
    p.layers.push({
      ln1G: Tensor.fill([D], 1, true),
      ln1B: Tensor.zeros([D], true),
      qkvW: Tensor.randn([D, 3 * D], std, rand),
      qkvB: Tensor.zeros([3 * D], true),
      projW: Tensor.randn([D, D], projStd, rand),
      projB: Tensor.zeros([D], true),
      ln2G: Tensor.fill([D], 1, true),
      ln2B: Tensor.zeros([D], true),
      fcW: Tensor.randn([D, 4 * D], std, rand),
      fcB: Tensor.zeros([4 * D], true),
      fc2W: Tensor.randn([4 * D, D], projStd, rand),
      fc2B: Tensor.zeros([D], true),
    });
  }
  return { config: { ...config }, params: p };
}

/** Flat list of [name, tensor] in a fixed order (optimizer, serialisation). */
export function namedParams(model) {
  const p = model.params;
  const out = [['wte', p.wte], ['wpe', p.wpe]];
  p.layers.forEach((layer, l) => Object.entries(layer).forEach(([k, t]) => out.push([`h${l}.${k}`, t])));
  out.push(['lnfG', p.lnfG], ['lnfB', p.lnfB], ['headW', p.headW], ['headB', p.headB]);
  return out;
}

export function paramCount(model) {
  return namedParams(model).reduce((s, [, t]) => s + t.size, 0);
}

/**
 * Forward pass.
 * @param {Int32Array|number[]} ids  B*T token ids (row-major)
 * @param {Int32Array|number[]|null} targets  B*T next-token ids, or null
 * @param {{ keepAttention?: boolean }} opts  keep attention maps [layer] → Float32Array [B][H][T][T]
 */
export function forward(model, ids, B, T, targets = null, { keepAttention = false } = {}) {
  const { nHead: H } = model.config;
  const p = model.params;
  const pos = new Int32Array(B * T);
  for (let i = 0; i < B * T; i++) pos[i] = i % T;
  let x = add(embedding(p.wte, ids), embedding(p.wpe, pos));
  const attention = [];
  for (const layer of p.layers) {
    const keep = keepAttention ? {} : null;
    const a = causalAttention(linear(layerNorm(x, layer.ln1G, layer.ln1B), layer.qkvW, layer.qkvB), B, T, H, keep);
    x = add(x, linear(a, layer.projW, layer.projB));
    const m = linear(gelu(linear(layerNorm(x, layer.ln2G, layer.ln2B), layer.fcW, layer.fcB)), layer.fc2W, layer.fc2B);
    x = add(x, m);
    if (keep) attention.push(keep.att);
  }
  const logits = linear(layerNorm(x, p.lnfG, p.lnfB), p.headW, p.headB);
  const loss = targets ? crossEntropy(logits, targets) : null;
  return { logits, loss, attention };
}

/** Random training windows from a token array. y is x shifted by one. */
export function getBatch(data, B, T, rand) {
  const x = new Int32Array(B * T);
  const y = new Int32Array(B * T);
  for (let b = 0; b < B; b++) {
    const start = Math.floor(rand() * (data.length - T - 1));
    for (let t = 0; t < T; t++) {
      x[b * T + t] = data[start + t];
      y[b * T + t] = data[start + t + 1];
    }
  }
  return { x, y };
}

/** A trainer that owns the optimizer. step() runs one update and returns { loss, gradNorm, lr }; `steps` counts them. */
export function createTrainer(model, data, { batchSize = 16, lr = 3e-3, warmup = 50, totalSteps = 2000, minLrRatio = 0.1, seed = 7 } = {}) {
  const params = namedParams(model).map(([, t]) => t);
  const opt = new AdamW(params, { lr });
  const rand = mulberry32(seed);
  const T = model.config.blockSize;
  let stepNo = 0;
  const schedule = (s) => {
    if (s < warmup) return (lr * (s + 1)) / warmup;
    const progress = Math.min(1, (s - warmup) / Math.max(1, totalSteps - warmup));
    return lr * (minLrRatio + (1 - minLrRatio) * 0.5 * (1 + Math.cos(Math.PI * progress)));
  };
  return {
    /** Number of optimisation steps taken so far. */
    get steps() {
      return stepNo;
    },
    step() {
      const { x, y } = getBatch(data, batchSize, T, rand);
      opt.zeroGrad();
      const { loss } = forward(model, x, batchSize, T, y);
      backward(loss);
      const gradNorm = opt.clip(1);
      const stepLr = schedule(stepNo);
      opt.step(stepLr);
      stepNo++;
      return { loss: loss.data[0], gradNorm, lr: stepLr };
    },
  };
}

/** Average loss on fixed windows of `data` (no gradient). */
export function evaluate(model, data, { windows = 16, seed = 99 } = {}) {
  const T = model.config.blockSize;
  const rand = mulberry32(seed);
  const { x, y } = getBatch(data, windows, T, rand);
  return noGrad(() => forward(model, x, windows, T, y).loss.data[0]);
}

/** Logits for the token after `context` (uses the last blockSize tokens). */
export function nextLogits(model, context) {
  const T = Math.min(model.config.blockSize, Math.max(1, context.length));
  const ids = Int32Array.from(context.slice(-T));
  return noGrad(() => {
    const { logits } = forward(model, ids, 1, ids.length);
    const V = logits.shape[1];
    return Array.from(logits.data.subarray((ids.length - 1) * V, ids.length * V));
  });
}

/**
 * Generate `length` new tokens after `context`.
 * @param {object} decode  options for sampling.decodeStep (temperature, k, p, penalty, greedy)
 */
export function generate(model, context, length, decode = {}) {
  const rand = decode.rand ?? mulberry32(decode.seed ?? 1);
  const out = [...context];
  const steps = [];
  for (let i = 0; i < length; i++) {
    const logits = nextLogits(model, out);
    const { index, probs } = decodeStep(logits, { ...decode, history: out.slice(-model.config.blockSize), rand });
    steps.push({ index, p: probs[index] });
    out.push(index);
  }
  return { ids: out, steps };
}

// ---------------------------------------------------------------- (de)serialisation

function toBase64(f32) {
  const bytes = new Uint8Array(f32.buffer, f32.byteOffset, f32.byteLength);
  if (typeof Buffer !== 'undefined') return Buffer.from(bytes).toString('base64');
  let s = '';
  for (let i = 0; i < bytes.length; i += 0x8000) s += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  return btoa(s);
}

function fromBase64(b64) {
  let bytes;
  if (typeof Buffer !== 'undefined') bytes = Uint8Array.from(Buffer.from(b64, 'base64'));
  else bytes = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
  return new Float32Array(bytes.buffer);
}

/** Checkpoint = config + vocabulary + weights (+ anything in meta). */
export function serialize(model, itos, meta = {}) {
  return {
    format: 'llmlab-minigpt-1',
    config: model.config,
    itos,
    meta,
    weights: Object.fromEntries(namedParams(model).map(([k, t]) => [k, toBase64(t.data)])),
  };
}

export function deserialize(ckpt) {
  const model = createModel(ckpt.config, 0);
  for (const [name, t] of namedParams(model)) {
    const data = fromBase64(ckpt.weights[name]);
    if (data.length !== t.size) throw new Error(`체크포인트 크기가 맞지 않는다: ${name}`);
    t.data.set(data);
  }
  const stoi = new Map(ckpt.itos.map((c, i) => [c, i]));
  return {
    model,
    meta: ckpt.meta ?? {},
    vocab: {
      size: ckpt.itos.length,
      itos: ckpt.itos,
      stoi,
      encode: (s) => [...s].map((c) => stoi.get(c) ?? -1).filter((i) => i >= 0),
      decode: (ids) => ids.map((i) => ckpt.itos[i] ?? '').join(''),
    },
  };
}
