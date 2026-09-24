// Byte Pair Encoding, written to be read.
// Pre-tokenize on whitespace (a word starts with "▁" like SentencePiece), start
// from single characters (or UTF-8 bytes), then repeatedly merge the most
// frequent adjacent pair. The ordered merge list *is* the tokenizer.

export const WORD_START = '▁';

/** Split text into words; each word keeps a leading ▁ marking the space before it. */
export function pretokenize(text) {
  return text
    .split(/\s+/)
    .filter(Boolean)
    .map((w) => WORD_START + w);
}

const enc = new TextEncoder();
const dec = new TextDecoder('utf-8', { fatal: false });

/** Initial symbols of a word: characters, or UTF-8 bytes stored as latin1 chars. */
export function initialSymbols(word, unit = 'char') {
  if (unit === 'byte') {
    // keep ▁ as one symbol so the display stays readable
    const body = word.startsWith(WORD_START) ? word.slice(1) : word;
    const bytes = [...enc.encode(body)].map((b) => String.fromCharCode(b));
    return word.startsWith(WORD_START) ? [WORD_START, ...bytes] : bytes;
  }
  return [...word];
}

/** Human-readable form of a symbol. Byte symbols decode to text when they form whole characters. */
export function showSymbol(sym, unit = 'char') {
  if (unit !== 'byte') return sym;
  const lead = sym.startsWith(WORD_START) ? WORD_START : '';
  const body = lead ? sym.slice(1) : sym;
  if (!body) return lead;
  const bytes = Uint8Array.from([...body].map((c) => c.charCodeAt(0)));
  const text = dec.decode(bytes);
  if (!text.includes('�')) return lead + text;
  return lead + [...bytes].map((b) => (b < 128 ? String.fromCharCode(b) : `<${b.toString(16).toUpperCase()}>`)).join('');
}

function wordFreqs(text, unit) {
  const freq = new Map();
  for (const w of pretokenize(text)) freq.set(w, (freq.get(w) ?? 0) + 1);
  return [...freq].map(([word, count]) => ({ word, count, syms: initialSymbols(word, unit) }));
}

function pairCounts(words) {
  const pairs = new Map();
  for (const { syms, count } of words) {
    for (let i = 0; i < syms.length - 1; i++) {
      const key = syms[i] + '\u0000' + syms[i + 1];
      pairs.set(key, (pairs.get(key) ?? 0) + count);
    }
  }
  return pairs;
}

function mergeWord(syms, a, b) {
  const out = [];
  for (let i = 0; i < syms.length; i++) {
    if (i < syms.length - 1 && syms[i] === a && syms[i + 1] === b) {
      out.push(a + b);
      i++;
    } else out.push(syms[i]);
  }
  return out;
}

/** Total tokens the whole text currently takes (sum over words of symbols × count). */
function tokenTotal(words) {
  return words.reduce((s, w) => s + w.syms.length * w.count, 0);
}

/**
 * Train BPE step by step. Yields after every merge so a widget can animate it.
 * @param {string} text
 * @param {{ merges?: number, unit?: 'char'|'byte', minCount?: number }} opts
 */
export function* trainSteps(text, { merges = 200, unit = 'char', minCount = 2 } = {}) {
  const words = wordFreqs(text, unit);
  const base = new Set(words.flatMap((w) => w.syms));
  yield { step: 0, merge: null, vocabSize: base.size, tokens: tokenTotal(words), words };
  let vocabSize = base.size;
  for (let step = 1; step <= merges; step++) {
    const pairs = pairCounts(words);
    let best = null;
    let bestCount = 0;
    // ties broken by key order so training is deterministic
    for (const [key, c] of pairs) if (c > bestCount || (c === bestCount && best !== null && key < best)) [best, bestCount] = [key, c];
    if (!best || bestCount < minCount) return;
    const [a, b] = best.split('\u0000');
    for (const w of words) w.syms = mergeWord(w.syms, a, b);
    vocabSize++;
    yield { step, merge: { a, b, merged: a + b, count: bestCount }, vocabSize, tokens: tokenTotal(words), words };
  }
}

/** Train to completion. Returns { merges, vocab, unit }. */
export function train(text, opts = {}) {
  const unit = opts.unit ?? 'char';
  const merges = [];
  let base = null;
  for (const s of trainSteps(text, opts)) {
    if (s.step === 0) base = new Set(s.words.flatMap((w) => w.syms));
    else merges.push(s.merge);
  }
  const vocab = [...base, ...merges.map((m) => m.merged)];
  return { merges, vocab, unit };
}

/** Encode one word with a trained tokenizer: apply merges in the order they were learned. */
export function encodeWord(tok, word) {
  let syms = initialSymbols(word, tok.unit);
  const rank = tok._rank ?? (tok._rank = new Map(tok.merges.map((m, i) => [m.a + '\u0000' + m.b, i])));
  for (;;) {
    let bestI = -1;
    let bestR = Infinity;
    for (let i = 0; i < syms.length - 1; i++) {
      const r = rank.get(syms[i] + '\u0000' + syms[i + 1]);
      if (r !== undefined && r < bestR) [bestR, bestI] = [r, i];
    }
    if (bestI < 0) return syms;
    syms = [...syms.slice(0, bestI), syms[bestI] + syms[bestI + 1], ...syms.slice(bestI + 2)];
  }
}

/** Encode text into token strings. */
export function encode(tok, text) {
  return pretokenize(text).flatMap((w) => encodeWord(tok, w));
}

/** Decode token strings back to text. */
export function decode(tok, tokens) {
  const joined = tokens.join('');
  const words = joined.split(WORD_START).filter(Boolean);
  if (tok.unit === 'byte') {
    return words.map((w) => dec.decode(Uint8Array.from([...w].map((c) => c.charCodeAt(0))))).join(' ');
  }
  return words.join(' ');
}
