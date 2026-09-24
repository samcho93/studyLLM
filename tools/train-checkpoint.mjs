// Offline: pre-train the mini GPT checkpoint used by weeks 7–9 and 12.
// Run: node tools/train-checkpoint.mjs [steps] [out]
// Writes assets/data/models/mini-gpt.json (config · vocabulary · weights · loss history · samples).
import { writeFile } from 'node:fs/promises';
import { loadCorpus, corpusText, charVocab } from '../assets/js/core/text.js';
import { createModel, createTrainer, evaluate, generate, paramCount, serialize } from '../assets/js/core/gpt.js';

const STEPS = Number(process.argv[2] ?? 4000);
const OUT = process.argv[3] ?? new URL('../assets/data/models/mini-gpt.json', import.meta.url);
const VAL_DOCS = ['rag-vectordb']; // held out: the model never sees this document

const corpus = await loadCorpus();
const all = corpusText(corpus);
const vocab = charVocab(all); // vocabulary from everything so validation has no unknown chars
const trainIds = corpus.documents.filter((d) => !VAL_DOCS.includes(d.id)).map((d) => d.id);
const train = Int32Array.from(vocab.encode(corpusText(corpus, trainIds)));
const val = Int32Array.from(vocab.encode(corpusText(corpus, VAL_DOCS)));

const config = { vocabSize: vocab.size, blockSize: 48, nLayer: 2, nHead: 4, nEmbd: 48 };
const model = createModel(config, 1337);
const trainer = createTrainer(model, train, { batchSize: 16, lr: 3e-3, warmup: 100, totalSteps: STEPS });
console.log('params', paramCount(model), 'train chars', train.length, 'val chars', val.length);

const history = [];
const samples = [];
const prompt = vocab.encode('실습실은');
const t0 = Date.now();
let avg = null;
for (let s = 1; s <= STEPS; s++) {
  const { loss } = trainer.step();
  avg = avg === null ? loss : 0.95 * avg + 0.05 * loss;
  if (s % 50 === 0 || s === 1) {
    const valLoss = evaluate(model, val, { windows: 24 });
    const trainLoss = evaluate(model, train, { windows: 24 });
    history.push({ step: s, train: +trainLoss.toFixed(4), val: +valLoss.toFixed(4) });
    if ([1, 100, 300, 1000, 2000, STEPS].includes(s) || s % 1000 === 0) {
      const text = vocab.decode(generate(model, prompt, 60, { temperature: 0.8, seed: 3 }).ids);
      samples.push({ step: s, text });
      console.log(`step ${s} train ${trainLoss.toFixed(3)} val ${valLoss.toFixed(3)} ${((Date.now() - t0) / 1000).toFixed(0)}s | ${text.replace(/\n/g, '↵')}`);
    }
  }
}
const ckpt = serialize(model, vocab.itos, { steps: STEPS, valDocs: VAL_DOCS, history, samples, trainedWith: 'tools/train-checkpoint.mjs', batchSize: 16 });
await writeFile(OUT, JSON.stringify(ckpt));
console.log('saved', OUT.toString(), 'in', ((Date.now() - t0) / 60000).toFixed(1), 'min');
