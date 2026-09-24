// Count-based word vectors (week 3). No DOM — works in pages, workers and Node.
//
//   words → co-occurrence counts (window) → PPMI → truncated SVD (k dims)
//   → cosine similarity · nearest neighbours · 2D projection
//
// Matrices are plain objects { rows, cols, data: Float64Array } in row-major order.

// ------------------------------------------------------------------ words

/**
 * Common Korean particles / endings stripped from the end of a word (eojeol).
 * Deliberately small: this is a teaching tokenizer, not a morphological analyser.
 * Longer suffixes are tried first. One suffix at most is removed.
 */
export const PARTICLES = ['에서', '으로', '이다', '한다', '은', '는', '이', '가', '을', '를', '에', '의', '로', '와', '과', '도'];
const SUFFIXES = [...PARTICLES].sort((a, b) => b.length - a.length);
const PARTICLE_SET = new Set(PARTICLES);

/**
 * Strip one particle from a word.
 * Rule: a 1-char suffix is removed only if at least 2 chars remain
 * (so 온도 · 강의 · 차이 stay whole); a 2-char suffix only needs 1 char left (것이다 → 것).
 */
export function stripParticle(word) {
  for (const suf of SUFFIXES) {
    if (!word.endsWith(suf)) continue;
    const stem = word.slice(0, -suf.length);
    if (stem.length >= (suf.length === 1 ? 2 : 1)) return stem;
  }
  return word;
}

/**
 * Split text into words: anything that is not Hangul / Latin / digit is a separator,
 * Latin is lower-cased, then one particle is stripped from each word.
 * A word that is only a particle (left over from "모델(LLM)은" → "은") is dropped.
 * "모델은 토큰을 벡터로 바꾼다." → ['모델', '토큰', '벡터', '바꾼다']
 */
export function tokenizeWords(text) {
  return (text.toLowerCase().match(/[가-힣a-z0-9]+/g) ?? []).filter((w) => !PARTICLE_SET.has(w)).map(stripParticle);
}

/** One word list per document (windows never cross documents). */
export function corpusWords(corpus) {
  return corpus.documents.map((d) => tokenizeWords(d.text));
}

/**
 * Vocabulary of words that occur at least `minCount` times, most frequent first
 * (ties broken alphabetically so ids are stable).
 */
export function buildVocab(docs, minCount = 2) {
  const freq = new Map();
  for (const doc of docs) for (const w of doc) freq.set(w, (freq.get(w) ?? 0) + 1);
  const entries = [...freq].filter(([, c]) => c >= minCount).sort((a, b) => b[1] - a[1] || (a[0] < b[0] ? -1 : 1));
  const words = entries.map(([w]) => w);
  return {
    words,
    counts: entries.map(([, c]) => c),
    index: new Map(words.map((w, i) => [w, i])),
    size: words.length,
    totalTokens: docs.reduce((a, d) => a + d.length, 0),
  };
}

// ------------------------------------------------------------------ matrices

export const matrix = (rows, cols) => ({ rows, cols, data: new Float64Array(rows * cols) });
export const row = (M, i) => M.data.subarray(i * M.cols, (i + 1) * M.cols);

/** One-hot vectors: the identity matrix. Every pair of different words has cosine 0. */
export function oneHot(n) {
  const M = matrix(n, n);
  for (let i = 0; i < n; i++) M.data[i * n + i] = 1;
  return M;
}

/**
 * Symmetric co-occurrence counts: for every word, count vocabulary words within
 * `window` positions to the left and right (same document). Positions of
 * out-of-vocabulary words still count toward the distance.
 */
export function cooccurrence(docs, vocab, window = 2) {
  const n = vocab.size;
  const M = matrix(n, n);
  for (const doc of docs) {
    const ids = doc.map((w) => vocab.index.get(w) ?? -1);
    for (let i = 0; i < ids.length; i++) {
      const a = ids[i];
      if (a < 0) continue;
      const hi = Math.min(ids.length - 1, i + window);
      for (let j = i + 1; j <= hi; j++) {
        const b = ids[j];
        if (b < 0) continue;
        M.data[a * n + b] += 1;
        M.data[b * n + a] += 1;
      }
    }
  }
  return M;
}

/**
 * Positive PMI: log( P(w,c) / (P(w) P(c)) ), negative values clipped to 0.
 * High when two words meet more often than their frequencies alone predict,
 * so the most frequent words stop dominating every vector.
 * alpha < 1 (e.g. 0.75) smooths the context distribution and tames the
 * known bias of PMI toward very rare words; the course uses plain alpha = 1.
 */
export function ppmi(C, { alpha = 1 } = {}) {
  const n = C.rows;
  const rowSum = new Float64Array(n);
  const ctx = new Float64Array(n); // context counts, optionally smoothed: count^alpha
  let total = 0;
  let ctxTotal = 0;
  for (let i = 0; i < n; i++) {
    let s = 0;
    for (let j = 0; j < n; j++) s += C.data[i * n + j];
    rowSum[i] = s;
    total += s;
    ctx[i] = s ** alpha;
    ctxTotal += ctx[i];
  }
  const M = matrix(n, n);
  if (total === 0) return M;
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      const c = C.data[i * n + j];
      if (c <= 0) continue;
      // P(w,c) / (P(w) · Pα(c)),  Pα(c) = count(c)^α / Σ count^α
      const pmi = Math.log((c / total) / ((rowSum[i] / total) * (ctx[j] / ctxTotal)));
      if (pmi > 0) M.data[i * n + j] = pmi;
    }
  }
  return M;
}

// ------------------------------------------------------------------ linear algebra

function dot(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

const norm = (a) => Math.sqrt(dot(a, a));

/** Cosine similarity. Returns 0 when either vector is all zeros. */
export function cosine(a, b) {
  const na = norm(a);
  const nb = norm(b);
  return na === 0 || nb === 0 ? 0 : dot(a, b) / (na * nb);
}

/** Orthonormalise the columns of Q (rows × k, row-major) in place (modified Gram–Schmidt). */
export function gramSchmidt(Q) {
  const { rows, cols: k, data } = Q;
  for (let c = 0; c < k; c++) {
    for (let p = 0; p < c; p++) {
      let d = 0;
      for (let r = 0; r < rows; r++) d += data[r * k + c] * data[r * k + p];
      for (let r = 0; r < rows; r++) data[r * k + c] -= d * data[r * k + p];
    }
    let s = 0;
    for (let r = 0; r < rows; r++) s += data[r * k + c] ** 2;
    s = Math.sqrt(s) || 1;
    for (let r = 0; r < rows; r++) data[r * k + c] /= s;
  }
  return Q;
}

// A · B for row-major matrices
function matmul(A, B) {
  const C = matrix(A.rows, B.cols);
  const { cols: n } = A;
  const m = B.cols;
  for (let i = 0; i < A.rows; i++) {
    for (let t = 0; t < n; t++) {
      const a = A.data[i * n + t];
      if (a === 0) continue;
      const bo = t * m;
      const co = i * m;
      for (let j = 0; j < m; j++) C.data[co + j] += a * B.data[bo + j];
    }
  }
  return C;
}

// Aᵀ · B
function matmulT(A, B) {
  const C = matrix(A.cols, B.cols);
  const n = A.cols;
  const m = B.cols;
  for (let t = 0; t < A.rows; t++) {
    for (let i = 0; i < n; i++) {
      const a = A.data[t * n + i];
      if (a === 0) continue;
      for (let j = 0; j < m; j++) C.data[i * m + j] += a * B.data[t * m + j];
    }
  }
  return C;
}

// deterministic pseudo-random start (mulberry32) so results are reproducible
function seeded(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296 - 0.5;
  };
}

/**
 * Truncated SVD by subspace (block power) iteration:
 *   Q ← orth(Mᵀ M Q) repeated `iters` times → top-k right singular vectors.
 * Returns word vectors U · Σ^power (rows × k) and the singular values.
 * power = 0.5 (default) softens the largest directions, as in Levy et al. (2015).
 */
export function truncatedSVD(M, k, { iters = 40, seed = 7, power = 0.5 } = {}) {
  k = Math.min(k, M.cols);
  const rand = seeded(seed);
  let Q = matrix(M.cols, k);
  for (let i = 0; i < Q.data.length; i++) Q.data[i] = rand();
  gramSchmidt(Q);
  for (let it = 0; it < iters; it++) {
    Q = gramSchmidt(matmulT(M, matmul(M, Q)));
  }
  const MQ = matmul(M, Q); // = U Σ
  const sigma = new Float64Array(k);
  for (let c = 0; c < k; c++) {
    let s = 0;
    for (let r = 0; r < MQ.rows; r++) s += MQ.data[r * k + c] ** 2;
    sigma[c] = Math.sqrt(s);
  }
  // sort components by singular value (subspace iteration does not guarantee order)
  const order = [...sigma.keys()].sort((a, b) => sigma[b] - sigma[a]);
  const V = matrix(M.rows, k);
  order.forEach((c, dst) => {
    const scale = sigma[c] > 0 ? sigma[c] ** power / sigma[c] : 0; // U Σ^p = (U Σ) Σ^(p-1)
    for (let r = 0; r < M.rows; r++) V.data[r * k + dst] = MQ.data[r * k + c] * scale;
  });
  return { vectors: V, sigma: Float64Array.from(order.map((c) => sigma[c])) };
}

/** Rows scaled to length 1 (all-zero rows stay zero). */
export function normalizeRows(M) {
  const out = matrix(M.rows, M.cols);
  for (let i = 0; i < M.rows; i++) {
    const r = row(M, i);
    const n = norm(r);
    if (n > 0) for (let j = 0; j < M.cols; j++) out.data[i * M.cols + j] = r[j] / n;
  }
  return out;
}

/**
 * Cosine similarity between every pair of rows (rows × rows), computed once as
 * N · Nᵀ with N = normalizeRows(M). Handy when many neighbour lists are needed.
 */
export function cosineMatrix(M) {
  const N = normalizeRows(M);
  const n = N.rows;
  const d = N.cols;
  const S = matrix(n, n);
  for (let i = 0; i < n; i++) {
    const a = N.data.subarray(i * d, (i + 1) * d);
    for (let j = i; j < n; j++) {
      const b = N.data.subarray(j * d, (j + 1) * d);
      const s = dot(a, b);
      S.data[i * n + j] = s;
      S.data[j * n + i] = s;
    }
  }
  return S;
}

/** Full cosine-similarity table for the given row ids. */
export function similarityMatrix(M, ids) {
  return ids.map((a) => ids.map((b) => cosine(row(M, a), row(M, b))));
}

/**
 * The `k` rows most similar (cosine) to row `i`, excluding `i` itself.
 * Ties (e.g. all zeros for one-hot) keep vocabulary order = frequency order.
 */
export function nearest(M, i, k = 8, words = null) {
  const q = row(M, i);
  const out = [];
  for (let j = 0; j < M.rows; j++) {
    if (j === i) continue;
    out.push({ id: j, word: words?.[j], sim: cosine(q, row(M, j)) });
  }
  out.sort((a, b) => b.sim - a.sim || a.id - b.id);
  return out.slice(0, k);
}

/** Nearest rows to an arbitrary query vector (e.g. an analogy a − b + c). */
export function nearestToVector(M, q, k = 8, words = null, exclude = []) {
  const skip = new Set(exclude);
  const out = [];
  for (let j = 0; j < M.rows; j++) {
    if (skip.has(j)) continue;
    out.push({ id: j, word: words?.[j], sim: cosine(q, row(M, j)) });
  }
  out.sort((a, b) => b.sim - a.sim || a.id - b.id);
  return out.slice(0, k);
}

/** Analogy "a is to b as c is to ?" → nearest to (b − a + c), excluding a, b, c. */
export function analogy(M, a, b, c, k = 5, words = null) {
  const q = new Float64Array(M.cols);
  const na = normalizeRow(row(M, a));
  const nb = normalizeRow(row(M, b));
  const nc = normalizeRow(row(M, c));
  for (let j = 0; j < M.cols; j++) q[j] = nb[j] - na[j] + nc[j];
  return nearestToVector(M, q, k, words, [a, b, c]);
}

function normalizeRow(r) {
  const n = norm(r);
  return n === 0 ? r : r.map((x) => x / n);
}

/**
 * 2D projection for plotting: rows are length-normalised (so the picture reflects
 * cosine, not frequency), centred, then projected on the top-2 principal axes (PCA
 * by the same power iteration). Returns [[x, y], …] and the share of variance kept.
 */
export function project2D(M, { iters = 60, seed = 3 } = {}) {
  const X = normalizeRows(M);
  const { rows: n, cols: d } = X;
  const mean = new Float64Array(d);
  for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) mean[j] += X.data[i * d + j] / n;
  let totalVar = 0;
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < d; j++) {
      X.data[i * d + j] -= mean[j];
      totalVar += X.data[i * d + j] ** 2;
    }
  }
  const k = Math.min(2, d);
  const rand = seeded(seed);
  let Q = matrix(d, k);
  for (let i = 0; i < Q.data.length; i++) Q.data[i] = rand();
  gramSchmidt(Q);
  for (let it = 0; it < iters; it++) Q = gramSchmidt(matmulT(X, matmul(X, Q)));
  const P = matmul(X, Q);
  let kept = 0;
  for (let i = 0; i < P.data.length; i++) kept += P.data[i] ** 2;
  const pts = [];
  for (let i = 0; i < n; i++) pts.push([P.data[i * k], k > 1 ? P.data[i * k + 1] : 0]);
  return { points: pts, explained: totalVar > 0 ? kept / totalVar : 0 };
}
