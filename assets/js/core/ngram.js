// n-gram language model over a list of token strings (characters by default).
// The simplest language model: count what follows each context, divide to get
// probabilities, sample one token at a time.

import { sampleIndex } from './rng.js';

export const BOS = '⟨s⟩'; // start marker: "what can begin a text?"

/**
 * Count n-grams.
 * @param {string[]} tokens
 * @param {number} n  context length + 1 (2 = bigram)
 */
export function countNgrams(tokens, n = 2) {
  const counts = new Map(); // context (joined) -> Map(next -> count)
  const padded = [...Array(n - 1).fill(BOS), ...tokens];
  for (let i = n - 1; i < padded.length; i++) {
    const ctx = padded.slice(i - n + 1, i).join('\u0001');
    const next = padded[i];
    let row = counts.get(ctx);
    if (!row) counts.set(ctx, (row = new Map()));
    row.set(next, (row.get(next) ?? 0) + 1);
  }
  const vocab = [...new Set(tokens)].sort();
  return { n, counts, vocab };
}

const ctxKey = (context, n) => {
  const pad = [...Array(n - 1).fill(BOS), ...context];
  return pad.slice(pad.length - (n - 1)).join('\u0001');
};

/**
 * Next-token distribution after `context` with add-k (Laplace) smoothing.
 * Returns [{ token, count, p }] sorted by probability.
 */
export function nextDist(model, context, { k = 0 } = {}) {
  const row = model.counts.get(ctxKey(context, model.n)) ?? new Map();
  let total = 0;
  row.forEach((c) => (total += c));
  const V = model.vocab.length;
  const denom = total + k * V;
  if (denom === 0) return [];
  const toks = k > 0 ? model.vocab : [...row.keys()];
  return toks
    .map((token) => {
      const count = row.get(token) ?? 0;
      return { token, count, p: (count + k) / denom };
    })
    .filter((d) => d.p > 0)
    .sort((a, b) => b.p - a.p);
}

/** Probability of one token after a context (0 if unseen and k = 0). */
export function prob(model, context, token, { k = 0 } = {}) {
  const row = model.counts.get(ctxKey(context, model.n)) ?? new Map();
  let total = 0;
  row.forEach((c) => (total += c));
  const V = model.vocab.length;
  const denom = total + k * V;
  return denom === 0 ? 0 : ((row.get(token) ?? 0) + k) / denom;
}

/**
 * Generate tokens one at a time.
 * @returns {{ tokens: string[], steps: { context: string[], chosen: string, p: number, options: number }[] }}
 */
export function generate(model, { start = [], length = 40, rand = Math.random, k = 0, temperature = 1 } = {}) {
  const out = [...start];
  const steps = [];
  for (let i = 0; i < length; i++) {
    const dist = nextDist(model, out, { k });
    if (!dist.length) break; // dead end: this context never appeared
    const w = dist.map((d) => Math.pow(d.p, 1 / Math.max(temperature, 1e-3)));
    const pick = dist[sampleIndex(w, rand)];
    steps.push({ context: out.slice(-(model.n - 1)), chosen: pick.token, p: pick.p, options: dist.length });
    out.push(pick.token);
  }
  return { tokens: out, steps };
}

/**
 * Average negative log-likelihood (nats/token) of `tokens` under the model.
 * Unseen transitions have probability 0 → Infinity unless smoothed (k > 0).
 */
export function nll(model, tokens, { k = 0 } = {}) {
  let sum = 0;
  let zeros = 0;
  const ctx = [];
  for (const t of tokens) {
    const p = prob(model, ctx, t, { k });
    if (p === 0) zeros++;
    else sum -= Math.log(p);
    ctx.push(t);
  }
  return { nll: zeros ? Infinity : sum / tokens.length, zeros, n: tokens.length };
}
