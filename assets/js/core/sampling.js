// Turning scores into a chosen token: softmax, temperature, top-k, top-p.
// Every function takes plain arrays of numbers so it can be shown step by step.

import { sampleIndex } from './rng.js';

/** Numerically stable softmax with temperature. */
export function softmax(logits, temperature = 1) {
  const t = Math.max(temperature, 1e-6);
  let max = -Infinity;
  for (const x of logits) if (x > max) max = x;
  const exps = logits.map((x) => (x === -Infinity ? 0 : Math.exp((x - max) / t)));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / sum);
}

/** Keep the k most probable entries, renormalized. k <= 0 means "no limit". */
export function topK(probs, k) {
  if (!k || k <= 0 || k >= probs.length) return probs.slice();
  const cutoff = [...probs].sort((a, b) => b - a)[k - 1];
  let kept = 0;
  const out = probs.map((p) => {
    // ties at the cutoff: keep only as many as needed
    if (p > cutoff || (p === cutoff && kept < k)) {
      kept++;
      return p;
    }
    return 0;
  });
  const s = out.reduce((a, b) => a + b, 0);
  return out.map((p) => p / s);
}

/** Nucleus sampling: smallest set whose cumulative probability reaches p. */
export function topP(probs, p) {
  if (p >= 1) return probs.slice();
  const order = probs.map((q, i) => [q, i]).sort((a, b) => b[0] - a[0]);
  const keep = new Set();
  let cum = 0;
  for (const [q, i] of order) {
    keep.add(i);
    cum += q;
    if (cum >= p) break;
  }
  const out = probs.map((q, i) => (keep.has(i) ? q : 0));
  const s = out.reduce((a, b) => a + b, 0);
  return out.map((q) => q / s);
}

/** Lower the score of tokens that already appeared (CTRL-style repetition penalty). */
export function repetitionPenalty(logits, history, penalty = 1) {
  if (penalty === 1) return logits.slice();
  const seen = new Set(history);
  return logits.map((x, i) => (seen.has(i) ? (x > 0 ? x / penalty : x * penalty) : x));
}

/** Full decoding pipeline. Returns the chosen index and the final distribution. */
export function decodeStep(logits, { temperature = 1, k = 0, p = 1, penalty = 1, history = [], greedy = false, rand = Math.random } = {}) {
  const adjusted = repetitionPenalty(logits, history, penalty);
  if (greedy || temperature === 0) {
    let best = 0;
    for (let i = 1; i < adjusted.length; i++) if (adjusted[i] > adjusted[best]) best = i;
    const probs = adjusted.map((_, i) => (i === best ? 1 : 0));
    return { index: best, probs };
  }
  let probs = softmax(adjusted, temperature);
  probs = topK(probs, k);
  probs = topP(probs, p);
  return { index: sampleIndex(probs, rand), probs };
}

/** Shannon entropy in bits: how "spread out" a distribution is. */
export function entropyBits(probs) {
  let h = 0;
  for (const q of probs) if (q > 0) h -= q * Math.log2(q);
  return h;
}
