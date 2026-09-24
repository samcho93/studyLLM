// Web Worker for the week 7 trainer widget (w07-trainer.js).
// Same message protocol as core/gpt-worker.js, plus the widget-specific eval /
// sample cadence. It keeps its own step counter (core/gpt.js exposes `steps`).
//
// Page → worker:
//   { type: 'init', config, train: { batchSize, lr, totalSteps, warmup }, valDocs?, prompt?, evalEvery?, sampleEvery? }
//   { type: 'run' } · { type: 'pause' } · { type: 'step', n } (run n steps then pause)
//   { type: 'sample', prompt, length, temperature, seed }
// Worker → page:
//   { type: 'ready', params, vocabSize, trainChars, valChars, config }
//   { type: 'progress', step, loss?, trainLoss?, valLoss?, evalAt?, lr?, gradNorm?, msPerStep?, running }
//   { type: 'sample', step, prompt, text } · { type: 'error', message }

import { loadCorpus, corpusText, charVocab } from '../core/text.js';
import { createModel, createTrainer, evaluate, generate, paramCount } from '../core/gpt.js';

let S = null;

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
  const model = createModel(config, msg.seed ?? 1337);
  const trainer = createTrainer(model, train, msg.train ?? {}); // batch sampler seed: engine default (7)
  S = {
    model,
    trainer,
    vocab,
    train,
    val,
    stepNo: 0,
    running: false,
    stopAt: Infinity,
    evalEvery: msg.evalEvery ?? 25,
    sampleEvery: msg.sampleEvery ?? 50,
    prompt: msg.prompt ?? '실습실은',
  };
  post({ type: 'ready', params: paramCount(model), vocabSize: vocab.size, trainChars: train.length, valChars: val.length, config });
  sample(S.prompt);
}

function sample(prompt, length = 60, temperature = 0.8, seed = 3) {
  const ids = S.vocab.encode(prompt);
  const out = generate(S.model, ids.length ? ids : [0], length, { temperature, seed });
  post({ type: 'sample', step: S.stepNo, prompt, text: S.vocab.decode(out.ids) });
}

function stop() {
  S.running = false;
  S.stopAt = Infinity;
  post({ type: 'progress', step: S.stepNo, running: false });
}

function loop() {
  if (!S?.running) return;
  const t0 = performance.now();
  let last = null;
  let n = 0;
  // run steps for ~60 ms, then yield so 'pause' / 'sample' messages get through
  while (performance.now() - t0 < 60 && S.stepNo < S.stopAt) {
    last = S.trainer.step();
    S.stepNo++;
    n++;
    if (!Number.isFinite(last.loss)) break;
    if (S.stepNo === 1 || S.stepNo % S.evalEvery === 0) {
      last.trainLoss = evaluate(S.model, S.train, { windows: 12 });
      last.valLoss = evaluate(S.model, S.val, { windows: 12 });
      last.evalAt = S.stepNo;
      break;
    }
  }
  const ms = n ? (performance.now() - t0) / n : 0;
  if (last) post({ type: 'progress', step: S.stepNo, ...last, msPerStep: ms, running: true });
  if (last && !Number.isFinite(last.loss)) return stop(); // exploded: stop, the page explains
  if (last && S.stepNo % S.sampleEvery === 0) sample(S.prompt);
  if (S.stepNo >= S.stopAt) {
    const sampled = S.stepNo % S.sampleEvery === 0;
    stop();
    if (!sampled) sample(S.prompt); // e.g. after “1스텝”: show what one update changed
    return;
  }
  setTimeout(loop, 0);
}

self.onmessage = async ({ data: msg }) => {
  try {
    if (msg.type === 'init') return await init(msg);
    if (!S) return;
    if (msg.type === 'run' || msg.type === 'step') {
      S.stopAt = msg.type === 'step' ? S.stepNo + (msg.n ?? 1) : Infinity;
      if (!S.running) {
        S.running = true;
        loop();
      }
    } else if (msg.type === 'pause') {
      if (S.running) stop();
    } else if (msg.type === 'sample') {
      if (msg.prompt !== undefined) S.prompt = msg.prompt;
      sample(S.prompt, msg.length, msg.temperature, msg.seed);
    }
  } catch (err) {
    post({ type: 'error', message: String(err?.message ?? err) });
  }
};
