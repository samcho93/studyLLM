// w16 온디바이스 LLM 실행기
// One concept: an LLM can run on your own machine — no server, no key, nothing leaves the
// browser — but you trade quality and speed for it. The download size is set by
// parameter count × bytes per weight (quantization).

import { registerOutput, unregisterOutput } from '../site/result.js';

export const MODELS = [
  { id: 'HuggingFaceTB/SmolLM2-135M-Instruct', dtype: 'q8', label: 'SmolLM2-135M · int8 (137MB · 영어 전용)', short: 'SmolLM2-135M', params: 135e6, mb: 137 },
  { id: 'onnx-community/Qwen2.5-0.5B-Instruct', dtype: 'q8', label: 'Qwen2.5-0.5B · int8 (512MB · 다국어)', short: 'Qwen2.5-0.5B', params: 494e6, mb: 512 },
];

const PRESETS = [
  { id: 'en', label: '영어 질문', system: '', user: 'What is a language model? Answer in one sentence.' },
  { id: 'ko', label: '한국어 질문', system: '', user: '언어 모델이란 무엇인가? 한 문장으로 답하라.' },
  { id: 'fact', label: '우리 학과 사실 (환각)', system: '', user: 'In the AI Application Software department lab, how many hours can one person use the GPU server continuously?' },
  {
    id: 'ctx',
    label: '문서를 넣어 주면',
    system: 'Answer only from the document. If it is not in the document, say "I don\'t know".',
    user: 'Document: "When running long training jobs on the GPU server, one person may use it continuously for up to 12 hours."\n\nQuestion: How many hours can one person use the GPU server continuously?',
  },
];

const RECORDED_URL = new URL('../../data/w16/recorded.json', import.meta.url);
const WORKER_URL = new URL('../core/local-llm-worker.js', import.meta.url);

const template = document.createElement('template');
template.innerHTML = `
  <div class="widget w16-llm">
    <h3 class="widget__title">온디바이스 LLM 실행기</h3>
    <div class="widget__controls">
      <label class="field">
        <span class="field__label">모델</span>
        <select data-in="model"></select>
      </label>
      <label class="field">
        <span class="field__label">최대 생성 토큰 <output data-out="max"></output></span>
        <input type="range" data-in="max" min="16" max="256" step="16">
      </label>
      <label class="field">
        <span class="field__label">온도 (0 = 그리디) <output data-out="temp"></output></span>
        <input type="range" data-in="temp" min="0" max="1.5" step="0.1">
      </label>
    </div>
    <div class="btn-row w16-presets" data-slot="presets"></div>
    <label class="field w16-prompt">
      <span class="field__label">시스템 프롬프트 (선택)</span>
      <input type="text" data-in="system" spellcheck="false">
    </label>
    <label class="field w16-prompt">
      <span class="field__label">사용자 메시지</span>
      <textarea data-in="user" spellcheck="false"></textarea>
    </label>
    <div class="btn-row">
      <button type="button" class="btn small primary" data-act="load">⬇ 모델 불러오기</button>
      <button type="button" class="btn small primary" data-act="run" disabled>▶ 생성</button>
      <button type="button" class="btn small ghost" data-act="stop" disabled>■ 중지</button>
    </div>
    <div class="widget__status" data-slot="status" role="status" aria-live="polite"></div>
    <div class="bar-progress" data-slot="bar" hidden><div class="bar-progress__fill"></div></div>
    <div data-slot="out-inline"></div>
  </div>`;

const outTemplate = document.createElement('template');
outTemplate.innerHTML = `
  <div class="w16-out">
    <div data-slot="badge"></div>
    <div class="w16-answer" data-slot="answer" aria-live="polite"></div>
    <div class="stat-row" data-slot="stats"></div>
    <details>
      <summary>녹화된 실행 기록 (수업 PC 실측)</summary>
      <div class="w16-table-wrap"><table class="w16-table" data-slot="recorded"></table></div>
    </details>
  </div>`;

const state = new WeakMap();
let seq = 0;

/**
 * @param {HTMLElement} el
 * @param {{ outputTitle?: string, preset?: string }} options
 */
export async function mount(el, options = {}) {
  const root = template.content.firstElementChild.cloneNode(true);
  const out = outTemplate.content.firstElementChild.cloneNode(true);
  el.replaceChildren(root);
  const $ = (sel) => root.querySelector(sel) ?? out.querySelector(sel);
  const ctrl = new AbortController();
  const outputId = `w16-local-llm:${++seq}`;
  const st = { ctrl, outputId, worker: null };
  state.set(el, st);

  const s = { model: 0, max: 64, temp: 0, preset: options.preset ?? 'en', loaded: '', busy: false, recorded: [], answer: '', loadMs: 0 };

  $('[data-in=model]').innerHTML = MODELS.map((m, i) => `<option value="${i}">${esc(m.label)}</option>`).join('');
  $('[data-in=max]').value = String(s.max);
  $('[data-in=temp]').value = String(s.temp);
  $('[data-slot=presets]').innerHTML = PRESETS.map((p) => `<button type="button" class="btn ghost small" data-preset="${p.id}" aria-pressed="false">${esc(p.label)}</button>`).join('');
  registerOutput(outputId, { title: options.outputTitle ?? '온디바이스 LLM 결과', node: out, inlineHost: $('[data-slot=out-inline]') });

  const model = () => MODELS[s.model];
  const status = (html) => ($('[data-slot=status]').innerHTML = html);

  function applyPreset(id) {
    const p = PRESETS.find((x) => x.id === id) ?? PRESETS[0];
    s.preset = p.id;
    $('[data-in=system]').value = p.system;
    $('[data-in=user]').value = p.user;
    root.querySelectorAll('[data-preset]').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.preset === p.id)));
  }

  function syncControls() {
    $('[data-out=max]').textContent = `${s.max}`;
    $('[data-out=temp]').textContent = s.temp.toFixed(1);
    const isLoaded = s.loaded === model().id;
    const lb = $('[data-act=load]');
    lb.textContent = isLoaded ? '✓ 불러옴' : `⬇ 모델 불러오기 (${model().mb}MB)`;
    lb.disabled = s.busy || isLoaded;
    $('[data-act=run]').disabled = s.busy || !isLoaded;
    $('[data-act=stop]').disabled = !s.busy || !isLoaded;
  }

  /** Before anything is downloaded, show a recorded run for the current model + preset. */
  function showRecorded() {
    const rec = s.recorded.find((r) => r.model === model().id && r.preset === s.preset);
    $('[data-slot=badge]').innerHTML = `<span class="chip">📼 녹화된 결과 · ${esc(model().short)}</span> <span class="w16-muted">모델을 불러오면 이 PC에서 직접 실행한다</span>`;
    $('[data-slot=answer]').textContent = rec ? rec.text : '(이 조합의 녹화 기록이 없다. 모델을 불러와 직접 실행한다.)';
    $('[data-slot=stats]').innerHTML = rec
      ? [stat('불러오기', `${(rec.loadMs / 1000).toFixed(1)}초`), stat('첫 토큰', `${(rec.firstTokenMs / 1000).toFixed(2)}초`), stat('생성 토큰', rec.tokens), stat('속도', `${(rec.tokens / (rec.totalMs / 1000)).toFixed(1)} tok/s`)].join('')
      : '';
  }

  function renderRecordedTable() {
    const rows = s.recorded
      .map((r) => {
        const m = MODELS.find((x) => x.id === r.model);
        const p = PRESETS.find((x) => x.id === r.preset);
        return `<tr><td>${esc(m?.short ?? r.model)}</td><td>${esc(p?.label ?? r.preset)}</td><td>${(r.tokens / (r.totalMs / 1000)).toFixed(1)}</td><td class="w16-rec-text">${esc(r.text)}</td></tr>`;
      })
      .join('');
    $('[data-slot=recorded]').innerHTML = `<thead><tr><th>모델</th><th>질문</th><th>tok/s</th><th>답</th></tr></thead><tbody>${rows}</tbody>`;
  }

  function ensureWorker() {
    if (st.worker) return st.worker;
    const w = new Worker(WORKER_URL, { type: 'module' });
    const files = new Map();
    w.onmessage = ({ data }) => {
      if (data.type === 'progress') {
        files.set(data.file, data);
        let loaded = 0;
        files.forEach((f) => (loaded += f.loaded));
        const pct = Math.min(99, (loaded / (model().mb * 1e6)) * 100);
        $('[data-slot=bar]').hidden = false;
        $('[data-slot=bar] .bar-progress__fill').style.width = `${pct}%`;
        status(`<span class="spinner" aria-hidden="true"></span> 내려받는 중 · ${(loaded / 1e6).toFixed(0)} / 약 ${model().mb}MB · 최초 1회만 (이후 브라우저 캐시)`);
      } else if (data.type === 'ready') {
        s.loaded = model().id;
        s.busy = false;
        s.loadMs = data.loadMs;
        $('[data-slot=bar]').hidden = true;
        status(`✓ ${esc(model().short)} 준비 완료 (${(data.loadMs / 1000).toFixed(1)}초). 이제 네트워크를 끊어도 동작한다.`);
        syncControls();
        run();
      } else if (data.type === 'token') {
        s.answer += data.text;
        $('[data-slot=answer]').textContent = s.answer;
      } else if (data.type === 'done') {
        s.busy = false;
        $('[data-slot=answer]').textContent = data.text;
        const tps = data.tokens / Math.max(0.001, data.totalMs / 1000);
        $('[data-slot=stats]').innerHTML = [
          stat('첫 토큰', `${(data.firstTokenMs / 1000).toFixed(2)}초`),
          stat('생성 토큰', data.tokens),
          stat('속도', `${tps.toFixed(1)} tok/s`),
          stat('전체', `${(data.totalMs / 1000).toFixed(1)}초`),
        ].join('');
        status(`✓ 생성 완료 · 서버로 보낸 데이터 0바이트`);
        syncControls();
      } else if (data.type === 'error') {
        s.busy = false;
        $('[data-slot=bar]').hidden = true;
        status(`<span class="widget__error">실행 오류: ${esc(data.message)} — 네트워크(huggingface.co 차단 여부)와 브라우저 메모리를 확인한다.</span>`);
        syncControls();
      }
    };
    w.onerror = (e) => {
      s.busy = false;
      status(`<span class="widget__error">워커를 시작하지 못했다: ${esc(e.message ?? '')}</span>`);
      syncControls();
    };
    st.worker = w;
    return w;
  }

  function load() {
    s.busy = true;
    syncControls();
    status('<span class="spinner" aria-hidden="true"></span> 라이브러리와 모델을 준비하는 중…');
    ensureWorker().postMessage({ type: 'load', model: model().id, dtype: model().dtype });
  }

  function run() {
    if (s.loaded !== model().id) return;
    const system = $('[data-in=system]').value.trim();
    const user = $('[data-in=user]').value.trim();
    if (!user) return;
    const messages = system ? [{ role: 'system', content: system }, { role: 'user', content: user }] : [{ role: 'user', content: user }];
    s.busy = true;
    s.answer = '';
    $('[data-slot=badge]').innerHTML = `<span class="chip ok">💻 이 PC에서 실행 · ${esc(model().short)}</span>`;
    $('[data-slot=answer]').textContent = '';
    $('[data-slot=stats]').innerHTML = '';
    status('<span class="spinner" aria-hidden="true"></span> 생성 중… (토큰이 나오는 대로 표시)');
    syncControls();
    ensureWorker().postMessage({ type: 'generate', messages, maxNewTokens: s.max, temperature: s.temp });
  }

  const on = (sel, type, fn) => $(sel).addEventListener(type, fn, { signal: ctrl.signal });
  on('[data-in=model]', 'change', (e) => {
    s.model = Number(e.target.value);
    syncControls();
    if (s.loaded !== model().id) showRecorded();
  });
  on('[data-in=max]', 'input', (e) => {
    s.max = Number(e.target.value);
    syncControls();
  });
  on('[data-in=temp]', 'input', (e) => {
    s.temp = Number(e.target.value);
    syncControls();
  });
  on('[data-slot=presets]', 'click', (e) => {
    const id = e.target.closest('[data-preset]')?.dataset.preset;
    if (!id) return;
    applyPreset(id);
    if (s.loaded === model().id) run();
    else showRecorded();
  });
  root.addEventListener(
    'click',
    (e) => {
      const act = e.target.closest('[data-act]')?.dataset.act;
      if (act === 'load') load();
      if (act === 'run') run();
      if (act === 'stop') st.worker?.postMessage({ type: 'stop' });
    },
    { signal: ctrl.signal },
  );

  applyPreset(s.preset);
  syncControls();
  status('모델은 아직 내려받지 않았다. 아래는 수업 PC에서 녹화한 결과다.');
  try {
    const res = await fetch(RECORDED_URL, { signal: ctrl.signal });
    s.recorded = (await res.json()).runs;
  } catch {
    s.recorded = [];
  }
  if (ctrl.signal.aborted) return;
  renderRecordedTable();
  showRecorded();
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
