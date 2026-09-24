// Offline: pre-train the mini GPT checkpoint used by weeks 7–9 and 12.
// Run: node tools/train-checkpoint.mjs [steps] [out]
// Writes snapshots assets/data/models/mini-gpt-<step>.json (config · vocabulary · weights · loss history · samples).
// Snapshots: an under-trained, a balanced and a memorised model — week 8 uses the balanced one, weeks 9/12 compare them.
import { writeFile } from 'node:fs/promises';
import { loadCorpus, corpusText, charVocab } from '../assets/js/core/text.js';
import { createModel, createTrainer, evaluate, generate, paramCount, serialize } from '../assets/js/core/gpt.js';

const STEPS = Number(process.argv[2] ?? 3000);
const SNAPSHOTS = (process.argv[3] ?? '300,800,3000').split(',').map(Number);
const outUrl = (s) => new URL(`../assets/data/models/mini-gpt-${s}.json`, import.meta.url);
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
    if ([1, 100, 300, 800, 1000, 2000, STEPS].includes(s) || s % 1000 === 0) {
      const text = vocab.decode(generate(model, prompt, 60, { temperature: 0.8, seed: 3 }).ids);
      samples.push({ step: s, text });
      console.log(`step ${s} train ${trainLoss.toFixed(3)} val ${valLoss.toFixed(3)} ${((Date.now() - t0) / 1000).toFixed(0)}s | ${text.replace(/\n/g, '↵')}`);
    }
    if (SNAPSHOTS.includes(s)) {
      const ckpt = serialize(model, vocab.itos, { steps: s, valDocs: VAL_DOCS, history: [...history], samples: [...samples], trainedWith: 'tools/train-checkpoint.mjs', batchSize: 16 });
      await writeFile(outUrl(s), JSON.stringify(ckpt));
      console.log('saved snapshot', s);
    }
  }
}
console.log('done in', ((Date.now() - t0) / 60000).toFixed(1), 'min');
