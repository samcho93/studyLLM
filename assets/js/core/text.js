// Corpus loading and character vocabulary. No DOM — works in pages, workers and Node.

const CORPUS_URL = new URL('../../data/corpus/corpus.json', import.meta.url);

let cache = null;

/** Load the shared corpus (same department documents as studyRAG + LLM course notes). */
export async function loadCorpus() {
  if (!cache) {
    cache = (async () => {
      if (typeof process !== 'undefined' && process.versions?.node && CORPUS_URL.protocol === 'file:') {
        const { readFile } = await import('node:fs/promises');
        return JSON.parse(await readFile(CORPUS_URL, 'utf8'));
      }
      const res = await fetch(CORPUS_URL);
      if (!res.ok) throw new Error(`코퍼스를 불러오지 못했다 (${res.status})`);
      return res.json();
    })();
    cache.catch(() => (cache = null));
  }
  return cache;
}

/** All document texts joined into one training string (documents separated by a blank line). */
export function corpusText(corpus, ids = null) {
  return corpus.documents
    .filter((d) => !ids || ids.includes(d.id))
    .map((d) => d.text)
    .join('\n\n');
}

/**
 * Character-level vocabulary: every distinct character becomes one token.
 * Characters are sorted so the ids are stable across runs.
 */
export function charVocab(text) {
  const chars = [...new Set(text)].sort();
  const stoi = new Map(chars.map((c, i) => [c, i]));
  return {
    size: chars.length,
    itos: chars,
    stoi,
    encode: (s) => [...s].map((c) => stoi.get(c) ?? -1).filter((i) => i >= 0),
    decode: (ids) => ids.map((i) => chars[i] ?? '').join(''),
  };
}

/** Split Korean prose into sentences ("…다." / "…." / newline). */
export function sentences(text) {
  return text
    .split(/(?<=[.!?])\s+|\n+/)
    .map((s) => s.trim())
    .filter(Boolean);
}

/** Visible form of whitespace for token displays. */
export function showWs(s) {
  return s.replace(/ /g, '␣').replace(/\n/g, '↵');
}
