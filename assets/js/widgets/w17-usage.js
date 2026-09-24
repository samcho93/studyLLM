// w17 sLLM 사용법 스니펫
// One concept: the same open model and the same messages can be run five different ways
// (browser, Ollama, Python transformers, llama.cpp, an OpenAI-compatible server). What changes
// is where it runs, which file format it needs (ONNX · GGUF · safetensors) and how much memory
// the weights take (params × bits ÷ 8). Failure modes: a runtime with no published build for
// that model, a license that forbids the use, a quantization that does not exist.

import { registerOutput, unregisterOutput } from '../site/result.js';

const DATA_URL = new URL('../../data/w17/models.json', import.meta.url);
const WORKER_URL = new URL('../core/local-llm-worker.js', import.meta.url);
const BROWSER_DEMO = { id: 'HuggingFaceTB/SmolLM2-135M-Instruct', dtype: 'q8', mb: 137 };

export const USAGE_MODELS = ['smollm2-135m', 'qwen2.5-0.5b', 'qwen3.5-0.8b', 'exaone-4.0-1.2b', 'kanana-1.5-2.1b', 'gemma-4-e2b', 'qwen3-4b-2507', 'phi-4-mini'];

export const RUNTIMES = [
  { id: 'tjs', label: '브라우저 · Transformers.js', where: '브라우저 (이 사이트의 16주차 방식)', format: 'ONNX' },
  { id: 'ollama', label: 'Ollama', where: '내 PC 터미널 (ollama.com에서 설치)', format: 'GGUF (Ollama가 관리)' },
  { id: 'hf', label: 'Python · transformers', where: 'Python · Colab (GPU 권장)', format: 'safetensors' },
  { id: 'llamacpp', label: 'llama.cpp · GGUF', where: '내 PC 터미널 (CPU만으로도 동작)', format: 'GGUF' },
  { id: 'openai', label: 'OpenAI 호환 서버 · vLLM', where: 'GPU 서버 · Colab (NVIDIA GPU)', format: 'safetensors' },
];

export const TASKS = [
  { id: 'chat', label: '채팅' },
  { id: 'summary', label: '요약' },
  { id: 'json', label: 'JSON 추출' },
];

export const QUANTS = [
  { id: '16', label: '16비트 (fp16 · bf16)', bits: 16, tjs: 'fp16', gguf: ['F16', 'BF16'] },
  { id: '8', label: '8비트 (int8 · Q8_0)', bits: 8, tjs: 'q8', gguf: ['Q8_0'] },
  { id: '4', label: '4비트 (q4 · Q4_K_M)', bits: 4, tjs: 'q4', gguf: ['Q4_K_M', 'Q4_0'] },
];

// The course corpus: facility-rules (same document as week 12 · studyRAG week 1)
export const DOC =
  '실습실은 수업 시간 외에도 평일 오후 9시까지 개방된다. 주말과 공휴일에는 사전 신청한 학생만 이용할 수 있으며, 신청은 이용일 이틀 전까지 학과 행정실에 한다. 실습실 안에서는 음료를 포함한 모든 음식물 섭취가 금지된다. GPU 서버에 장시간 학습 작업을 실행할 때는 공용 캘린더에 사용 시간을 등록해야 하며, 한 사람이 연속으로 사용할 수 있는 시간은 최대 12시간이다.';

export const JSON_KEYS = ['open_until', 'food_allowed', 'gpu_max_hours', 'weekend_apply_days_before'];

export function messagesFor(task) {
  if (task === 'summary')
    return [
      { role: 'system', content: '주어진 규정을 한국어 세 문장으로 요약한다.' },
      { role: 'user', content: DOC },
    ];
  if (task === 'json')
    return [
      { role: 'system', content: `규정에서 정보를 뽑아 JSON 객체 하나만 출력한다. 키: ${JSON_KEYS.join(', ')}` },
      { role: 'user', content: DOC },
    ];
  return [
    { role: 'system', content: '너는 친절한 조교다. 한국어로 두 문장 이내로 답한다.' },
    { role: 'user', content: '대규모 언어 모델(LLM)이 무엇인지 설명해 줘.' },
  ];
}

/** Weight memory in GB = params(B) × bits ÷ 8. Runtime memory is larger (KV cache, activations). */
export const weightGB = (paramsB, bits) => (paramsB * bits) / 8;

function pickGguf(m, quant) {
  const g = m.runtimes?.gguf;
  if (!g) return null;
  const name = quant.gguf.find((q) => g.files[q]);
  return name ? { repo: g.repo, quant: name, bytes: g.files[name] } : { repo: g.repo, quant: null, available: Object.keys(g.files) };
}

const isMultimodal = (m) => m.modality.some((x) => x !== 'text');
const indent = (s, n) => s.replace(/\n/g, `\n${' '.repeat(n)}`);
const jsonMsgs = (msgs) => JSON.stringify(msgs, null, 2);

/**
 * Build the snippet for (model, runtime, task, quant).
 * @returns {{ ok: boolean, lang: string, code: string, notes: string[] }}
 */
export function buildSnippet(m, runtimeId, taskId, quantId, licenses = {}) {
  const quant = QUANTS.find((q) => q.id === quantId) ?? QUANTS[1];
  const msgs = messagesFor(taskId);
  const isJson = taskId === 'json';
  const notes = [];
  const lic = licenses[m.license];
  if (lic?.commercial === 'no') notes.push(`⛔ ${lic.label}: 수업 · 연구용으로만 쓴다. 이 코드로 상업 서비스를 만들면 라이선스 위반이다.`);
  else if (lic?.commercial === 'conditional') notes.push(`⚠ ${lic.label}: ${lic.note}`);

  if (runtimeId === 'tjs') {
    const onnx = m.runtimes?.onnx;
    if (!onnx)
      return {
        ok: false,
        lang: 'text',
        code: `// ${m.name}: 이 사이트가 쓰는 Transformers.js 3.8.1에서 동작을 확인한 ONNX 변환본이 없다.\n// 허브에서 "${m.name} ONNX"를 검색해 onnx-community 변환본이 있는지 보고,\n// 없으면 Ollama · llama.cpp로 내 PC에서 돌린다.`,
        notes: [...notes, '브라우저 실행은 ONNX 파일이 있어야 한다. 모든 모델이 변환되어 있지는 않다.', `${m.name}의 가중치만 ${quant.bits}비트 기준 ${weightGB(m.paramsB, quant.bits).toFixed(2)}GB — 브라우저 탭 메모리 한도(보통 수 GB)를 먼저 따진다.`],
      };
    const code = `import { pipeline } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1';

const generator = await pipeline('text-generation', '${onnx}', {
  dtype: '${quant.tjs}',   // ${quant.label}
  device: 'wasm',
});
const messages = ${indent(jsonMsgs(msgs), 0)};
const out = await generator(messages, { max_new_tokens: ${isJson ? 128 : 96}, do_sample: false });
const text = out[0].generated_text.at(-1).content;
console.log(text);${isJson ? `
try { console.log(JSON.parse(text)); } catch { console.log('JSON 파싱 실패: 작은 모델은 형식을 자주 어긴다'); }` : ''}`;
    notes.push('처음 한 번 모델 파일을 내려받고 이후에는 브라우저 캐시를 쓴다. 16주차 실행기와 같은 코드다.');
    if (quant.id === '4') notes.push('16주차에서 본 것처럼 q4 ONNX는 임베딩 표를 fp32로 두어 q8보다 파일이 클 수 있다.');
    return { ok: true, lang: 'javascript', code, notes };
  }

  if (runtimeId === 'ollama') {
    let tag = m.runtimes?.ollama;
    const g = pickGguf(m, QUANTS[2]);
    if (!tag && g?.repo) {
      tag = `hf.co/${g.repo}${g.quant ? `:${g.quant}` : ''}`;
      notes.push('Ollama 공식 라이브러리에 없는 모델이다. 허깅페이스의 GGUF 저장소를 hf.co/… 형식으로 바로 받는다.');
    }
    if (!tag)
      return {
        ok: false,
        lang: 'text',
        code: `# ${m.name}: Ollama 공식 라이브러리 태그도, 허깅페이스 GGUF 파일도 확인하지 못했다.\n# 직접 GGUF로 변환(llama.cpp의 convert_hf_to_gguf.py)하거나 Python transformers로 돌린다.`,
        notes: [...notes, '모든 모델이 모든 런타임에 배포되어 있지는 않다. 고르기 전에 배포 형식을 확인한다.'],
      };
    const body = { model: tag, messages: msgs, stream: false };
    if (isJson) body.format = 'json';
    if (taskId !== 'chat') body.options = { temperature: 0 };
    const code = `# 1) 모델 받기 + 대화형 실행 (터미널)
ollama pull ${tag}
ollama run ${tag}

# 2) REST API: Ollama는 http://localhost:11434 에서 대기한다
curl http://localhost:11434/api/chat -d '${JSON.stringify(body)}'

# 3) Python 파일 (pip install ollama) — 여기부터는 .py 파일에 넣는다
from ollama import chat
response = chat(model="${tag}", messages=${indent(jsonMsgs(msgs), 0)}${isJson ? ', format="json"' : ''}${taskId !== 'chat' ? ', options={"temperature": 0}' : ''})
print(response.message.content)

# 4) OpenAI 호환 주소도 있다: base_url="http://localhost:11434/v1", api_key="ollama"(아무 값)`;
    notes.push('/api/chat은 기본이 스트리밍이다. 한 번에 받으려면 "stream": false를 넣는다.');
    notes.push('Windows PowerShell에서는 작은따옴표 JSON이 깨진다. curl 대신 3) Python 예시를 쓴다.');
    notes.push(`기본 태그의 양자화 방식은 모델마다 다르다. ollama.com/library 태그 페이지에서 파일 크기로 확인한다 (선택한 ${quant.bits}비트와 다를 수 있다).`);
    return { ok: true, lang: 'bash', code, notes };
  }

  if (runtimeId === 'hf') {
    const q =
      quant.id === '16'
        ? ''
        : `\nfrom transformers import BitsAndBytesConfig\nbnb = BitsAndBytesConfig(${quant.id === '8' ? 'load_in_8bit=True' : 'load_in_4bit=True'})  # pip install bitsandbytes · NVIDIA GPU 필요\n`;
    const qArg = quant.id === '16' ? '' : ', quantization_config=bnb';
    let code;
    if (isMultimodal(m)) {
      code = `# pip install -U transformers torch accelerate${q}
from transformers import AutoProcessor, AutoModelForMultimodalLM

MODEL_ID = "${m.hf}"
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForMultimodalLM.from_pretrained(MODEL_ID, dtype="auto", device_map="auto"${qArg})

messages = ${indent(jsonMsgs(msgs), 0)}
inputs = processor.apply_chat_template(
    messages, tokenize=True, return_dict=True, return_tensors="pt", add_generation_prompt=True
).to(model.device)
input_len = inputs["input_ids"].shape[-1]
outputs = model.generate(**inputs, max_new_tokens=256, do_sample=False)
print(processor.decode(outputs[0][input_len:], skip_special_tokens=True))`;
      notes.push('이미지 · 음성도 받는 멀티모달 모델이라 텍스트 생성 파이프라인 대신 AutoProcessor + AutoModelForMultimodalLM을 쓴다 (모델 카드 방식).');
    } else {
      code = `# pip install -U transformers torch accelerate${q}
from transformers import pipeline

pipe = pipeline("text-generation", model="${m.hf}", dtype="auto", device_map="auto"${quant.id === '16' ? '' : ', model_kwargs={"quantization_config": bnb}'})
messages = ${indent(jsonMsgs(msgs), 0)}
out = pipe(messages, max_new_tokens=256, do_sample=False)
print(out[0]["generated_text"][-1]["content"])  # 파이프라인이 채팅 템플릿(10주차)을 자동 적용한다`;
    }
    if (isJson) code += `\n\nimport json\ntext = ${isMultimodal(m) ? 'processor.decode(outputs[0][input_len:], skip_special_tokens=True)' : 'out[0]["generated_text"][-1]["content"]'}\ntry:\n    print(json.loads(text))\nexcept json.JSONDecodeError:\n    print("JSON 파싱 실패 → 형식 강제(Ollama format · response_format)나 재시도가 필요하다")`;
    if (m.id.startsWith('exaone')) notes.push('EXAONE 4.0은 transformers 4.54.0 이상이 필요하다 (모델 카드).');
    if (m.runtimes?.gated || m.hf?.startsWith('meta-llama') || m.hf?.startsWith('google/gemma-3')) notes.push('허브에서 약관 동의가 필요한(gated) 모델이다. 먼저 hf auth login.');
    notes.push('Colab 무료 GPU에서 그대로 실행된다. 이 코드는 LoRA 미세조정(10주차)의 출발점이기도 하다.');
    return { ok: true, lang: 'python', code, notes };
  }

  if (runtimeId === 'llamacpp') {
    const g = pickGguf(m, quant);
    if (!g)
      return {
        ok: false,
        lang: 'text',
        code: `# ${m.name}: 확인된 GGUF 저장소가 없다.\n# 변환: python convert_hf_to_gguf.py <모델 폴더> → llama-quantize로 ${quant.gguf[0]} 생성`,
        notes: [...notes, 'GGUF는 llama.cpp · Ollama가 쓰는 한 파일짜리 형식이다. 없으면 직접 변환해야 한다.'],
      };
    if (!g.quant)
      return {
        ok: false,
        lang: 'text',
        code: `# ${g.repo}에는 ${quant.gguf.join(' / ')} 파일이 없다.\n# 있는 파일: ${g.available.join(', ')}`,
        notes: [...notes, '양자화 파일은 올린 사람이 고른 것만 있다. 다른 비트 수를 고르거나 직접 llama-quantize로 만든다.'],
      };
    const body = { messages: msgs, temperature: taskId === 'chat' ? 0.7 : 0 };
    if (isJson) body.response_format = { type: 'json_object' };
    const code = `# 설치: https://github.com/ggml-org/llama.cpp (릴리스 바이너리 · brew install llama.cpp 등)
# 1) 대화형 실행: -hf <저장소>:<양자화> 로 파일을 받아 온다 (${(g.bytes / 1e9).toFixed(2)}GB)
llama-cli -hf ${g.repo}:${g.quant}

# 2) OpenAI 호환 서버 (기본 포트 8080) + 내장 웹 UI
llama-server -hf ${g.repo}:${g.quant} --port 8080

curl http://localhost:8080/v1/chat/completions -H "Content-Type: application/json" -d '${JSON.stringify(body)}'`;
    notes.push('최신 llama.cpp README는 같은 기능을 llama cli · llama serve 형태로도 소개한다. 설치한 버전의 명령 이름을 확인한다.');
    notes.push('CPU만으로도 돈다. 4비트 GGUF + 노트북이 “GPU 없는 로컬 LLM”의 가장 흔한 조합이다.');
    return { ok: true, lang: 'bash', code, notes };
  }

  // OpenAI-compatible server (vLLM)
  const extra = isJson
    ? `,
    response_format={
        "type": "json_schema",
        "json_schema": {"name": "rules", "schema": {"type": "object", "properties": {${JSON_KEYS.map((k) => `"${k}": {}`).join(', ')}}, "required": ${JSON.stringify(JSON_KEYS)}}},
    }`
    : '';
  const code = `# GPU 서버 · Colab 터미널
pip install vllm
vllm serve ${m.hf} --max-model-len 8192   # http://localhost:8000/v1

# 클라이언트: 11주차의 OpenAI SDK 코드에서 base_url만 바꾼다
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
resp = client.chat.completions.create(
    model="${m.hf}",
    messages=${indent(jsonMsgs(msgs), 4)},
    temperature=${taskId === 'chat' ? 0.7 : 0}${extra}
)
print(resp.choices[0].message.content)`;
  notes.push('같은 클라이언트 코드가 Ollama(:11434/v1) · llama-server(:8080/v1)에도 그대로 붙는다. 이것이 “OpenAI 호환”의 뜻이다.');
  notes.push('--max-model-len을 줄이면 KV 캐시 메모리가 줄어든다 (9주차 메모리 계산기). 모델 카드의 최대 컨텍스트를 그대로 쓰면 작은 GPU에서 메모리가 부족할 수 있다.');
  if (quant.id !== '16') notes.push(`vLLM에서 ${quant.bits}비트로 돌리려면 미리 양자화된 체크포인트(AWQ · GPTQ · FP8 등)를 골라 serve한다.`);
  return { ok: true, lang: 'python', code, notes };
}

// ---------- widget ----------

const template = document.createElement('template');
template.innerHTML = `
  <div class="widget w17-usage">
    <h3 class="widget__title">sLLM 사용법 스니펫</h3>
    <div class="widget__controls">
      <label class="field"><span class="field__label">실행 방식</span><select data-in="runtime"></select></label>
      <label class="field"><span class="field__label">모델</span><select data-in="model"></select></label>
      <label class="field"><span class="field__label">작업</span><select data-in="task"></select></label>
      <label class="field"><span class="field__label">양자화</span><select data-in="quant"></select></label>
    </div>
    <div class="widget__status" data-slot="status" role="status" aria-live="polite"></div>
    <div class="w17-snippet">
      <div class="w17-snippet-head"><span data-slot="where"></span><button type="button" class="btn small ghost" data-act="copy">📋 복사</button></div>
      <pre class="w17-code"><code data-slot="code"></code></pre>
    </div>
    <ul class="w17-notes" data-slot="notes"></ul>
    <div class="btn-row">
      <button type="button" class="btn small primary" data-act="run">▶ 브라우저에서 지금 실행 (SmolLM2-135M · ${BROWSER_DEMO.mb}MB)</button>
      <button type="button" class="btn small ghost" data-act="stop" disabled>■ 중지</button>
    </div>
    <div data-slot="out-inline"></div>
  </div>`;

const outTemplate = document.createElement('template');
outTemplate.innerHTML = `
  <div class="w17-uout">
    <div class="stat-row" data-slot="mem"></div>
    <div class="w17-bars" data-slot="bars"></div>
    <div class="w17-run" data-slot="run"></div>
  </div>`;

const state = new WeakMap();
let seq = 0;

/**
 * @param {HTMLElement} el
 * @param {{ outputTitle?: string, runtime?: string, model?: string, task?: string }} options
 */
export async function mount(el, options = {}) {
  const root = template.content.firstElementChild.cloneNode(true);
  const out = outTemplate.content.firstElementChild.cloneNode(true);
  el.replaceChildren(root);
  const $ = (sel) => root.querySelector(sel) ?? out.querySelector(sel);
  const ctrl = new AbortController();
  const outputId = `w17-usage:${++seq}`;
  const st = { ctrl, outputId, worker: null };
  state.set(el, st);

  const s = { runtime: options.runtime ?? 'ollama', model: options.model ?? 'qwen3.5-0.8b', task: options.task ?? 'json', quant: '4', busy: false, loaded: false, answer: '' };
  let data = null;
  let models = [];

  $('[data-in=runtime]').innerHTML = RUNTIMES.map((r) => `<option value="${r.id}">${esc(r.label)}</option>`).join('');
  $('[data-in=task]').innerHTML = TASKS.map((t) => `<option value="${t.id}">${esc(t.label)}</option>`).join('');
  $('[data-in=quant]').innerHTML = QUANTS.map((q) => `<option value="${q.id}">${esc(q.label)}</option>`).join('');
  registerOutput(outputId, { title: options.outputTitle ?? 'sLLM 사용법 · 메모리 어림', node: out, inlineHost: $('[data-slot=out-inline]') });
  $('[data-slot=status]').innerHTML = '<span class="spinner" aria-hidden="true"></span> models.json을 불러오는 중…';
  $('[data-slot=run]').innerHTML = `<p class="w17-sub">▶ 버튼은 선택과 관계없이 <b>SmolLM2-135M</b>(영어 전용, ${BROWSER_DEMO.mb}MB)으로 지금 고른 작업을 이 브라우저에서 실행한다. 다른 모델은 위 코드를 내 PC · Colab에서 실행한다.</p>`;

  const model = () => models.find((m) => m.id === s.model) ?? models[0];

  function render() {
    const m = model();
    const snip = buildSnippet(m, s.runtime, s.task, s.quant, data.licenses);
    const rt = RUNTIMES.find((r) => r.id === s.runtime);
    $('[data-slot=where]').innerHTML = `${snip.ok ? '✓' : '✗'} <b>${esc(rt.where)}</b> · 파일 형식 ${esc(rt.format)}`;
    $('[data-slot=code]').textContent = snip.code;
    $('[data-slot=code]').className = `language-${snip.lang}`;
    $('[data-slot=notes]').innerHTML = snip.notes.map((n) => `<li>${esc(n)}</li>`).join('');
    $('[data-slot=status]').innerHTML = snip.ok
      ? `${esc(m.name)} · ${esc(data.licenses[m.license]?.label ?? m.license)} · <a href="${esc(m.source)}" target="_blank" rel="noopener">모델 카드</a>`
      : `<span class="widget__error">이 조합은 확인된 배포본이 없다 — 실패도 정보다. 다른 실행 방식이나 모델을 고른다.</span>`;
    renderMemory(m);
  }

  function renderMemory(m) {
    const q = QUANTS.find((x) => x.id === s.quant);
    const gb = weightGB(m.paramsB, q.bits);
    const g = pickGguf(m, q);
    $('[data-slot=mem]').innerHTML = [
      stat('파라미터', `${m.paramsB >= 1 ? m.paramsB.toFixed(2) + 'B' : Math.round(m.paramsB * 1000) + 'M'}`),
      stat(`가중치 (${q.bits}비트)`, gb >= 1 ? `${gb.toFixed(2)}GB` : `${Math.round(gb * 1000)}MB`),
      stat('실제 GGUF', g?.bytes ? `${(g.bytes / 1e9).toFixed(2)}GB (${g.quant})` : '—'),
    ].join('');
    const max = weightGB(m.paramsB, 16);
    $('[data-slot=bars]').innerHTML =
      QUANTS.map((x) => {
        const v = weightGB(m.paramsB, x.bits);
        return `<div class="w17-bar-row${x.id === s.quant ? ' on' : ''}"><span>${x.bits}비트</span><span class="w17-bar"><i style="width:${((v / max) * 100).toFixed(1)}%"></i></span><span class="num">${v >= 1 ? v.toFixed(2) + 'GB' : Math.round(v * 1000) + 'MB'}</span></div>`;
      }).join('') +
      `<p class="w17-sub">가중치 = 파라미터 × 비트 ÷ 8 (16주차 공식). 실제 실행에는 KV 캐시(컨텍스트 길이에 비례, 9주차)와 런타임 여유가 더 든다.${m.paramsNote ? ` · ${esc(m.paramsNote)}` : ''}</p>`;
  }

  function ensureWorker() {
    if (st.worker) return st.worker;
    const w = new Worker(WORKER_URL, { type: 'module' });
    const files = new Map();
    const runEl = () => $('[data-slot=run]');
    w.onmessage = ({ data: msg }) => {
      if (msg.type === 'progress') {
        files.set(msg.file, msg.loaded);
        let loaded = 0;
        files.forEach((v) => (loaded += v));
        runEl().innerHTML = `<p><span class="spinner" aria-hidden="true"></span> SmolLM2-135M 내려받는 중 · ${(loaded / 1e6).toFixed(0)} / 약 ${BROWSER_DEMO.mb}MB (최초 1회)</p>`;
      } else if (msg.type === 'ready') {
        s.loaded = true;
        generate();
      } else if (msg.type === 'token') {
        s.answer += msg.text;
        const a = runEl().querySelector('.w17-answer');
        if (a) a.textContent = s.answer;
      } else if (msg.type === 'done') {
        s.busy = false;
        syncButtons();
        let check = '';
        if (s.task === 'json') {
          try {
            JSON.parse(msg.text.trim().replace(/^```(?:json)?\s*|\s*```$/g, ''));
            check = '<p class="w17-ok">✓ JSON.parse 성공</p>';
          } catch {
            check = '<p class="w17-bad">✗ JSON.parse 실패 — 135M 영어 모델은 한국어 규정에서 형식을 지키지 못한다. 그래서 형식 강제(Ollama format · response_format)와 더 큰 한국어 모델이 필요하다.</p>';
          }
        }
        const tps = msg.tokens / Math.max(0.001, msg.totalMs / 1000);
        runEl().innerHTML = `<p class="w17-sub">💻 이 브라우저에서 실행 · SmolLM2-135M · ${msg.tokens}토큰 · ${tps.toFixed(1)} tok/s</p><div class="w17-answer"></div>${check}`;
        runEl().querySelector('.w17-answer').textContent = msg.text;
      } else if (msg.type === 'error') {
        s.busy = false;
        syncButtons();
        runEl().innerHTML = `<p class="widget__error">실행 오류: ${esc(msg.message)} — huggingface.co · cdn.jsdelivr.net 접속과 브라우저 메모리를 확인한다.</p>`;
      }
    };
    w.onerror = (e) => {
      s.busy = false;
      syncButtons();
      runEl().innerHTML = `<p class="widget__error">워커를 시작하지 못했다: ${esc(e.message ?? '')}</p>`;
    };
    st.worker = w;
    return w;
  }

  function generate() {
    s.answer = '';
    $('[data-slot=run]').innerHTML = `<p class="w17-sub"><span class="spinner" aria-hidden="true"></span> 생성 중 · 작업: ${esc(TASKS.find((t) => t.id === s.task).label)}</p><div class="w17-answer"></div>`;
    ensureWorker().postMessage({ type: 'generate', messages: messagesFor(s.task), maxNewTokens: 128, temperature: 0 });
  }

  function runInBrowser() {
    s.busy = true;
    syncButtons();
    if (s.loaded) return generate();
    $('[data-slot=run]').innerHTML = '<p><span class="spinner" aria-hidden="true"></span> 라이브러리와 모델을 준비하는 중…</p>';
    ensureWorker().postMessage({ type: 'load', model: BROWSER_DEMO.id, dtype: BROWSER_DEMO.dtype });
  }

  function syncButtons() {
    $('[data-act=run]').disabled = s.busy || !data;
    $('[data-act=stop]').disabled = !s.busy;
  }

  const on = (sel, type, fn) => $(sel).addEventListener(type, fn, { signal: ctrl.signal });
  for (const key of ['runtime', 'model', 'task', 'quant'])
    on(`[data-in=${key}]`, 'change', (e) => {
      s[key] = e.target.value;
      if (data) render();
    });
  root.addEventListener(
    'click',
    async (e) => {
      const act = e.target.closest('[data-act]')?.dataset.act;
      if (act === 'copy') {
        const btn = e.target.closest('[data-act]');
        try {
          await navigator.clipboard.writeText($('[data-slot=code]').textContent);
          btn.textContent = '✓ 복사됨';
        } catch {
          btn.textContent = '복사 실패 — 직접 선택';
        }
        setTimeout(() => (btn.textContent = '📋 복사'), 1500);
      }
      if (act === 'run') runInBrowser();
      if (act === 'stop') st.worker?.postMessage({ type: 'stop' });
    },
    { signal: ctrl.signal },
  );
  syncButtons();

  try {
    const res = await fetch(DATA_URL, { signal: ctrl.signal });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    data = await res.json();
  } catch (err) {
    if (ctrl.signal.aborted) return;
    $('[data-slot=status]').innerHTML = `<span class="widget__error">models.json을 불러오지 못했다: ${esc(err.message)}</span>`;
    return;
  }
  if (ctrl.signal.aborted) return;
  models = USAGE_MODELS.map((id) => data.models.find((m) => m.id === id)).filter(Boolean);
  $('[data-in=model]').innerHTML = models.map((m) => `<option value="${m.id}">${esc(m.name)} · ${esc(data.licenses[m.license]?.label ?? '')}</option>`).join('');
  for (const key of ['runtime', 'model', 'task', 'quant']) $(`[data-in=${key}]`).value = s[key];
  syncButtons();
  render();
}

export function unmount(el) {
  const st = state.get(el);
  if (!st) return;
  st.ctrl.abort();
  st.worker?.terminate();
  unregisterOutput(st.outputId);
  state.delete(el);
  el.replaceChildren();
}

const stat = (label, value) => `<div class="stat"><span class="stat__label">${label}</span><span class="stat__value">${value}</span></div>`;

function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
}
