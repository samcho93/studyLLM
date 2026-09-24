// Trains the mini GPT off the main thread so the page stays responsive (week 7).
// Protocol (page → worker):
//   { type: 'init', config, train: { batchSize, lr, totalSteps, warmup, seed }, valDocs?, prompt? }
//   { type: 'run' } · { type: 'pause' } · { type: 'step', n } (run n steps then pause)
//   { type: 'sample', prompt, length, temperature, seed } · { type: 'export' }
// Worker → page:
//   { type: 'ready', params, vocabSize, trainChars, valChars }
//   { type: 'progress', step, loss, trainLoss?, valLoss?, lr, gradNorm, msPerStep, running }
//   { type: 'sample', step, prompt, text } · { type: 'export', ckpt } · { type: 'error', message }

import { loadCorpus, corpusText, charVocab } from './text.js';
import { createModel, createTrainer, evaluate, generate, paramCount, serialize } from './gpt.js';

let S = null; // { model, trainer, vocab, train, val, running, cfg }

const post = (m) => self.postMessage(m);

async function init(msg) {
  const corpus = await loadCorpus();
  const all = corpusText(corpus);
  const vocab = charVocab(all);
  const valDocs = msg.valDocs ?? ['rag-vectordb'];
  const trainIds = corpus.documents.filter((d) => !valDocs.includes(d.id)).map((d) => d.id);
  const train = Int32Array.from(vocab.encode(corpusText(corpus, trainIds)));
  const val = Int32Array.from(vocab.encode(corpusText(corpus, valDocs)));
  const config = { ...msg.config, vocabSize: vocab.size };
  const model = createModel(config, msg.train?.seed ?? 1337);
  const trainer = createTrainer(model, train, msg.train ?? {});
  S = { model, trainer, vocab, train, val, running: false, cfg: msg, evalEvery: msg.evalEvery ?? 25, sampleEvery: msg.sampleEvery ?? 100, prompt: msg.prompt ?? '실습실은', stopAt: Infinity };
  post({ type: 'ready', params: paramCount(model), vocabSize: vocab.size, trainChars: train.length, valChars: val.length, config });
  sample(S.prompt, 60, 0.8, 3);
}

function sample(prompt, length = 60, temperature = 0.8, seed = 3) {
  const ids = S.vocab.encode(prompt);
  const out = generate(S.model, ids.length ? ids : [0], length, { temperature, seed });
  post({ type: 'sample', step: S.trainer.steps, prompt, text: S.vocab.decode(out.ids) });
}

function loop() {
  if (!S?.running) return;
  const t0 = performance.now();
  let last = null;
  let n = 0;
  // run steps for ~60 ms, then yield so messages (pause, sample) get through
  while (performance.now() - t0 < 60 && S.trainer.steps < S.stopAt) {
    last = S.trainer.step();
    n++;
    const step = S.trainer.steps;
    if (step % S.evalEvery === 0 || step === 1) {
      last.trainLoss = evaluate(S.model, S.train, { windows: 12 });
      last.valLoss = evaluate(S.model, S.val, { windows: 12 });
      last.evalAt = step;
      break;
    }
  }
  const ms = n ? (performance.now() - t0) / n : 0;
  if (last) post({ type: 'progress', step: S.trainer.steps, ...last, msPerStep: ms, running: true });
  if (S.trainer.steps % S.sampleEvery === 0 && last) sample(S.prompt);
  if (S.trainer.steps >= S.stopAt) {
    S.running = false;
    S.stopAt = Infinity;
    post({ type: 'progress', step: S.trainer.steps, running: false });
    sample(S.prompt);
    return;
  }
  setTimeout(loop, 0);
}

self.onmessage = async ({ data: msg }) => {
  try {
    if (msg.type === 'init') return await init(msg);
    if (!S) return;
    if (msg.type === 'run' || msg.type === 'step') {
      if (msg.type === 'step') S.stopAt = S.trainer.steps + (msg.n ?? 1);
      if (!S.running) {
        S.running = true;
        loop();
      }
    } else if (msg.type === 'pause') {
      S.running = false;
      post({ type: 'progress', step: S.trainer.steps, running: false });
    } else if (msg.type === 'sample') {
      if (msg.prompt !== undefined) S.prompt = msg.prompt;
      sample(S.prompt, msg.length, msg.temperature, msg.seed);
    } else if (msg.type === 'export') {
      post({ type: 'export', ckpt: serialize(S.model, S.vocab.itos, { steps: S.trainer.steps, trainedWith: 'browser' }) });
    }
  } catch (err) {
    post({ type: 'error', message: String(err?.message ?? err) });
  }
};
