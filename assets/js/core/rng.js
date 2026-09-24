// Reproducible randomness. Every widget and Challenge that samples takes a seed,
// so a class sees the same "random" text on every screen.

/** mulberry32: tiny 32-bit PRNG. Returns a function giving floats in [0, 1). */
export function mulberry32(seed = 42) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Pick an index with probability proportional to weights (need not sum to 1). */
export function sampleIndex(weights, rand = Math.random) {
  let total = 0;
  for (let i = 0; i < weights.length; i++) total += weights[i];
  let r = rand() * total;
  for (let i = 0; i < weights.length; i++) {
    r -= weights[i];
    if (r < 0) return i;
  }
  // floating point leftovers: return the last non-zero weight
  for (let i = weights.length - 1; i >= 0; i--) if (weights[i] > 0) return i;
  return 0;
}

/** Standard normal via Box–Muller. */
export function gaussian(rand = Math.random) {
  let u = 0;
  while (u === 0) u = rand();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rand());
}
