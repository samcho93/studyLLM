// Runs a small instruction-tuned LLM entirely in the browser (week 16).
// Transformers.js (ONNX Runtime Web, WASM backend) in a Web Worker so the page stays responsive.
// Page → worker: { type: 'load', model, dtype } · { type: 'generate', messages, maxNewTokens, temperature } · { type: 'stop' }
// Worker → page: { type: 'progress', file, loaded, total } · { type: 'ready', model, dtype, loadMs }
//                { type: 'token', text } · { type: 'done', text, tokens, firstTokenMs, totalMs } · { type: 'error', message }

// Same pinned build as studyRAG's embedding widgets
const TRANSFORMERS_URL = 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1';

let lib = null;
let generator = null;
let loadedKey = '';
let stopper = null;

const post = (m) => self.postMessage(m);

async function load({ model, dtype }) {
  const key = `${model}|${dtype}`;
  if (generator && loadedKey === key) return post({ type: 'ready', model, dtype, loadMs: 0, cached: true });
  if (!lib) {
    lib = await import(TRANSFORMERS_URL);
    lib.env.allowLocalModels = false;
  }
  generator = null;
  const t0 = performance.now();
  generator = await lib.pipeline('text-generation', model, {
    dtype,
    device: 'wasm',
    progress_callback: (e) => {
      if (e.status === 'progress') post({ type: 'progress', file: e.file, loaded: e.loaded ?? 0, total: e.total ?? 0 });
    },
  });
  loadedKey = key;
  post({ type: 'ready', model, dtype, loadMs: performance.now() - t0 });
}

async function generate({ messages, maxNewTokens = 96, temperature = 0 }) {
  if (!generator) throw new Error('모델을 먼저 불러온다');
  const t0 = performance.now();
  let firstTokenMs = null;
  let tokens = 0;
  stopper = new lib.InterruptableStoppingCriteria();
  const streamer = new lib.TextStreamer(generator.tokenizer, {
    skip_prompt: true,
    skip_special_tokens: true,
    callback_function: (text) => {
      if (firstTokenMs === null) firstTokenMs = performance.now() - t0;
      post({ type: 'token', text });
    },
    token_callback_function: () => {
      tokens++;
    },
  });
  const sampling = temperature > 0 ? { do_sample: true, temperature, top_p: 0.9 } : { do_sample: false };
  const out = await generator(messages, { max_new_tokens: maxNewTokens, streamer, stopping_criteria: stopper, ...sampling });
  const last = out[0].generated_text.at(-1);
  post({ type: 'done', text: typeof last === 'string' ? last : last.content, tokens, firstTokenMs: firstTokenMs ?? 0, totalMs: performance.now() - t0 });
  stopper = null;
}

self.onmessage = async ({ data }) => {
  try {
    if (data.type === 'load') await load(data);
    else if (data.type === 'generate') await generate(data);
    else if (data.type === 'stop') stopper?.interrupt();
  } catch (err) {
    post({ type: 'error', message: String(err?.message ?? err) });
  }
};
