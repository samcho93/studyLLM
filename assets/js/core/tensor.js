// Minimal tensor autograd for the mini GPT (week 7).
// Same idea as value.js, but each node is a whole Float32Array so a forward pass
// is a few dozen nodes instead of millions. Only the ops a GPT needs exist, each
// with a hand-written backward. 2-D tensors are row-major [rows, cols].

let gradEnabled = true;

/** Run fn without building the graph (generation, evaluation). */
export function noGrad(fn) {
  const prev = gradEnabled;
  gradEnabled = false;
  try {
    return fn();
  } finally {
    gradEnabled = prev;
  }
}

export class Tensor {
  constructor(data, shape, requiresGrad = false) {
    this.data = data instanceof Float32Array ? data : Float32Array.from(data);
    this.shape = shape;
    this.requiresGrad = requiresGrad;
    this.grad = requiresGrad ? new Float32Array(this.data.length) : null;
    this.parents = [];
    this.backwardFn = null;
  }

  get size() {
    return this.data.length;
  }

  static zeros(shape, requiresGrad = false) {
    return new Tensor(new Float32Array(shape.reduce((a, b) => a * b, 1)), shape, requiresGrad);
  }

  static fill(shape, value, requiresGrad = false) {
    const t = Tensor.zeros(shape, requiresGrad);
    t.data.fill(value);
    return t;
  }

  /** Normal(0, std) initialisation. */
  static randn(shape, std, rand, requiresGrad = true) {
    const t = Tensor.zeros(shape, requiresGrad);
    for (let i = 0; i < t.data.length; i += 2) {
      let u = 0;
      while (u === 0) u = rand();
      const v = rand();
      const r = Math.sqrt(-2 * Math.log(u)) * std;
      t.data[i] = r * Math.cos(2 * Math.PI * v);
      if (i + 1 < t.data.length) t.data[i + 1] = r * Math.sin(2 * Math.PI * v);
    }
    return t;
  }
}

/** Create an op output; wires the graph only when some input needs a gradient. */
function result(data, shape, parents, backwardFn) {
  const needs = gradEnabled && parents.some((p) => p.requiresGrad);
  const t = new Tensor(data, shape, needs);
  if (needs) {
    t.parents = parents;
    t.backwardFn = backwardFn;
  }
  return t;
}

/** Back-propagate from a scalar loss through the whole graph. */
export function backward(loss) {
  const order = [];
  const seen = new Set();
  const visit = (t) => {
    if (seen.has(t)) return;
    seen.add(t);
    t.parents.forEach(visit);
    order.push(t);
  };
  visit(loss);
  loss.grad.fill(0);
  loss.grad[0] = 1;
  for (let i = order.length - 1; i >= 0; i--) order[i].backwardFn?.();
}

// ---------------------------------------------------------------- ops

/** Row lookup: W [V, D], ids (length N) → [N, D]. */
export function embedding(W, ids) {
  const [, D] = W.shape;
  const N = ids.length;
  const out = new Float32Array(N * D);
  for (let n = 0; n < N; n++) out.set(W.data.subarray(ids[n] * D, ids[n] * D + D), n * D);
  const t = result(out, [N, D], [W], () => {
    if (!W.requiresGrad) return;
    for (let n = 0; n < N; n++) {
      const g = t.grad;
      const o = ids[n] * D;
      for (let j = 0; j < D; j++) W.grad[o + j] += g[n * D + j];
    }
  });
  return t;
}

/** Elementwise a + b (same shape). */
export function add(a, b) {
  const out = new Float32Array(a.size);
  for (let i = 0; i < out.length; i++) out[i] = a.data[i] + b.data[i];
  const t = result(out, a.shape, [a, b], () => {
    for (let i = 0; i < out.length; i++) {
      if (a.requiresGrad) a.grad[i] += t.grad[i];
      if (b.requiresGrad) b.grad[i] += t.grad[i];
    }
  });
  return t;
}

/** x [N, K] · W [K, M] + b [M] → [N, M]. */
export function linear(x, W, b = null) {
  const [N, K] = x.shape;
  const M = W.shape[1];
  const X = x.data;
  const Wd = W.data;
  const out = new Float32Array(N * M);
  for (let n = 0; n < N; n++) {
    const o = n * M;
    if (b) out.set(b.data, o);
    for (let k = 0; k < K; k++) {
      const xv = X[n * K + k];
      if (xv === 0) continue;
      const w = k * M;
      for (let m = 0; m < M; m++) out[o + m] += xv * Wd[w + m];
    }
  }
  const parents = b ? [x, W, b] : [x, W];
  const t = result(out, [N, M], parents, () => {
    const G = t.grad;
    // dx = G · Wᵀ  (both rows contiguous)
    if (x.requiresGrad) {
      const dX = x.grad;
      for (let n = 0; n < N; n++) {
        const g = n * M;
        for (let k = 0; k < K; k++) {
          const w = k * M;
          let s = 0;
          for (let m = 0; m < M; m++) s += G[g + m] * Wd[w + m];
          dX[n * K + k] += s;
        }
      }
    }
    // dW = xᵀ · G
    if (W.requiresGrad) {
      const dW = W.grad;
      for (let n = 0; n < N; n++) {
        const g = n * M;
        for (let k = 0; k < K; k++) {
          const xv = X[n * K + k];
          if (xv === 0) continue;
          const w = k * M;
          for (let m = 0; m < M; m++) dW[w + m] += xv * G[g + m];
        }
      }
    }
    if (b?.requiresGrad) {
      const dB = b.grad;
      for (let n = 0; n < N; n++) for (let m = 0; m < M; m++) dB[m] += G[n * M + m];
    }
  });
  return t;
}

/** Layer normalisation over the last dimension. */
export function layerNorm(x, gamma, beta, eps = 1e-5) {
  const [N, D] = x.shape;
  const X = x.data;
  const out = new Float32Array(N * D);
  const xhat = new Float32Array(N * D);
  const rstd = new Float32Array(N);
  for (let n = 0; n < N; n++) {
    const o = n * D;
    let mean = 0;
    for (let j = 0; j < D; j++) mean += X[o + j];
    mean /= D;
    let v = 0;
    for (let j = 0; j < D; j++) v += (X[o + j] - mean) ** 2;
    const r = 1 / Math.sqrt(v / D + eps);
    rstd[n] = r;
    for (let j = 0; j < D; j++) {
      const h = (X[o + j] - mean) * r;
      xhat[o + j] = h;
      out[o + j] = h * gamma.data[j] + beta.data[j];
    }
  }
  const t = result(out, [N, D], [x, gamma, beta], () => {
    const G = t.grad;
    for (let n = 0; n < N; n++) {
      const o = n * D;
      let mdh = 0;
      let mdhx = 0;
      for (let j = 0; j < D; j++) {
        const dh = G[o + j] * gamma.data[j];
        mdh += dh;
        mdhx += dh * xhat[o + j];
        if (gamma.requiresGrad) gamma.grad[j] += G[o + j] * xhat[o + j];
        if (beta.requiresGrad) beta.grad[j] += G[o + j];
      }
      mdh /= D;
      mdhx /= D;
      if (x.requiresGrad) {
        for (let j = 0; j < D; j++) {
          const dh = G[o + j] * gamma.data[j];
          x.grad[o + j] += rstd[n] * (dh - mdh - xhat[o + j] * mdhx);
        }
      }
    }
  });
  return t;
}

const GELU_C = Math.sqrt(2 / Math.PI);

/** GELU (tanh approximation), as in GPT-2. */
export function gelu(x) {
  const X = x.data;
  const out = new Float32Array(X.length);
  for (let i = 0; i < X.length; i++) {
    const v = X[i];
    out[i] = 0.5 * v * (1 + Math.tanh(GELU_C * (v + 0.044715 * v * v * v)));
  }
  const t = result(out, x.shape, [x], () => {
    for (let i = 0; i < X.length; i++) {
      const v = X[i];
      const u = GELU_C * (v + 0.044715 * v * v * v);
      const th = Math.tanh(u);
      const du = GELU_C * (1 + 3 * 0.044715 * v * v);
      x.grad[i] += t.grad[i] * (0.5 * (1 + th) + 0.5 * v * (1 - th * th) * du);
    }
  });
  return t;
}

/**
 * Causal multi-head self-attention.
 * qkv [B*T, 3D] holds q | k | v side by side; returns [B*T, D].
 * If `keep` is given, attention weights are stored in keep.att as [B][H][T][T].
 */
export function causalAttention(qkv, B, T, H, keep = null) {
  const D = qkv.shape[1] / 3;
  const hs = D / H;
  const scale = 1 / Math.sqrt(hs);
  const Q = qkv.data;
  const out = new Float32Array(B * T * D);
  const att = new Float32Array(B * H * T * T);
  const row = 3 * D;
  for (let b = 0; b < B; b++) {
    for (let h = 0; h < H; h++) {
      const qo = h * hs;
      const ko = D + h * hs;
      const vo = 2 * D + h * hs;
      const a0 = (b * H + h) * T * T;
      for (let t = 0; t < T; t++) {
        const qi = (b * T + t) * row + qo;
        let max = -Infinity;
        for (let s = 0; s <= t; s++) {
          const ki = (b * T + s) * row + ko;
          let dot = 0;
          for (let j = 0; j < hs; j++) dot += Q[qi + j] * Q[ki + j];
          dot *= scale;
          att[a0 + t * T + s] = dot;
          if (dot > max) max = dot;
        }
        let sum = 0;
        for (let s = 0; s <= t; s++) {
          const e = Math.exp(att[a0 + t * T + s] - max);
          att[a0 + t * T + s] = e;
          sum += e;
        }
        const oi = (b * T + t) * D + h * hs;
        for (let s = 0; s <= t; s++) {
          const w = (att[a0 + t * T + s] /= sum);
          const vi = (b * T + s) * row + vo;
          for (let j = 0; j < hs; j++) out[oi + j] += w * Q[vi + j];
        }
      }
    }
  }
  if (keep) keep.att = att;
  const t = result(out, [B * T, D], [qkv], () => {
    const G = t.grad;
    const dQ = qkv.grad;
    const datt = new Float32Array(T);
    for (let b = 0; b < B; b++) {
      for (let h = 0; h < H; h++) {
        const qo = h * hs;
        const ko = D + h * hs;
        const vo = 2 * D + h * hs;
        const a0 = (b * H + h) * T * T;
        for (let tt = 0; tt < T; tt++) {
          const gi = (b * T + tt) * D + h * hs;
          let dot = 0;
          for (let s = 0; s <= tt; s++) {
            const vi = (b * T + s) * row + vo;
            const w = att[a0 + tt * T + s];
            let d = 0;
            for (let j = 0; j < hs; j++) {
              d += G[gi + j] * Q[vi + j];
              dQ[vi + j] += w * G[gi + j]; // dV
            }
            datt[s] = d;
            dot += w * d;
          }
          const qi = (b * T + tt) * row + qo;
          for (let s = 0; s <= tt; s++) {
            const ds = att[a0 + tt * T + s] * (datt[s] - dot) * scale;
            if (ds === 0) continue;
            const ki = (b * T + s) * row + ko;
            for (let j = 0; j < hs; j++) {
              dQ[qi + j] += ds * Q[ki + j]; // dQ
              dQ[ki + j] += ds * Q[qi + j]; // dK
            }
          }
        }
      }
    }
  });
  return t;
}

/** Mean cross-entropy of logits [N, V] against integer targets; returns a [1] tensor. */
export function crossEntropy(logits, targets) {
  const [N, V] = logits.shape;
  const L = logits.data;
  const probs = new Float32Array(N * V);
  let loss = 0;
  for (let n = 0; n < N; n++) {
    const o = n * V;
    let max = -Infinity;
    for (let v = 0; v < V; v++) if (L[o + v] > max) max = L[o + v];
    let sum = 0;
    for (let v = 0; v < V; v++) {
      const e = Math.exp(L[o + v] - max);
      probs[o + v] = e;
      sum += e;
    }
    for (let v = 0; v < V; v++) probs[o + v] /= sum;
    loss -= Math.log(Math.max(probs[o + targets[n]], 1e-12));
  }
  const t = result(new Float32Array([loss / N]), [1], [logits], () => {
    const g = t.grad[0] / N;
    const dL = logits.grad;
    for (let n = 0; n < N; n++) {
      const o = n * V;
      for (let v = 0; v < V; v++) dL[o + v] += g * probs[o + v];
      dL[o + targets[n]] -= g;
    }
  });
  return t;
}

// ---------------------------------------------------------------- optimizer

/** AdamW over a list of parameter tensors. */
export class AdamW {
  constructor(params, { lr = 3e-3, beta1 = 0.9, beta2 = 0.99, eps = 1e-8, weightDecay = 0.01 } = {}) {
    Object.assign(this, { params, lr, beta1, beta2, eps, weightDecay });
    this.m = params.map((p) => new Float32Array(p.size));
    this.v = params.map((p) => new Float32Array(p.size));
    this.t = 0;
  }

  zeroGrad() {
    this.params.forEach((p) => p.grad.fill(0));
  }

  /** Clip the global gradient norm; returns the norm before clipping. */
  clip(maxNorm = 1) {
    let sq = 0;
    for (const p of this.params) for (const g of p.grad) sq += g * g;
    const norm = Math.sqrt(sq);
    if (norm > maxNorm) {
      const s = maxNorm / norm;
      for (const p of this.params) for (let i = 0; i < p.grad.length; i++) p.grad[i] *= s;
    }
    return norm;
  }

  step(lr = this.lr) {
    this.t++;
    const { beta1: b1, beta2: b2, eps } = this;
    const c1 = 1 - b1 ** this.t;
    const c2 = 1 - b2 ** this.t;
    this.params.forEach((p, k) => {
      const m = this.m[k];
      const v = this.v[k];
      const decay = p.shape.length > 1 ? this.weightDecay : 0; // no decay on biases / norms
      for (let i = 0; i < p.size; i++) {
        const g = p.grad[i];
        m[i] = b1 * m[i] + (1 - b1) * g;
        v[i] = b2 * v[i] + (1 - b2) * g * g;
        p.data[i] -= lr * ((m[i] / c1) / (Math.sqrt(v[i] / c2) + eps) + decay * p.data[i]);
      }
    });
  }
}
