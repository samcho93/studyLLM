// w07 브라우저 GPT 학습기
// One concept: training = repeat (batch → forward → loss → backward → update)
// and watch what the loss curve and the samples do. The mini GPT trains in a
// Web Worker (w07-train-worker.js) so the page stays responsive.
// Failure modes on purpose: a learning rate that is too large stalls high,
// long training overfits (val loss rises, samples copy the corpus verbatim).

import { registerOutput, unregisterOutput } from '../site/result.js';
import { loadCorpus, corpusText, charVocab } from '../core/text.js';
import { createModel, evaluate, deserialize, generate, paramCount } from '../core/gpt.js';
import { mulberry32 } from '../core/rng.js';

// Own worker (same protocol as core/gpt-worker.js): see the note at the top of w07-train-worker.js.
const WORKER_URL = new URL('./w07-train-worker.js', import.meta.url);
const MODEL_URL = (step) => new URL(`../../data/models/mini-gpt-${step}.json`, import.meta.url);
const PRETRAINED = [3000, 800]; // try the fully trained snapshot first
const VAL_DOCS = ['rag-vectordb'];
const EVAL_EVERY = 25;
const SAMPLE_EVERY = 50; // must be a multiple of EVAL_EVERY (the worker yields at eval steps)
const SAMPLE_LEN = 60;
const WARMUP = 30;
const TOTAL = 2000; // cosine schedule length = automatic stop
const COPY_MIN = 10;
const EMA = 0.1;

const LRS = [
  { v: 1e-3, label: '0.001 (작다)' },
  { v: 3e-3, label: '0.003' },
  { v: 5e-3, label: '0.005 (기본)' },
  { v: 3e-2, label: '0.03 (크다)' },
  { v: 1e-1, label: '0.1 (너무 크다)' },
];
const DEFAULT_CFG = { nEmbd: 32, nLayer: 2, blockSize: 32, batch: 8, lr: 5e-3 };
const PRESETS = {
  base: { ...DEFAULT_CFG },
  hot: { ...DEFAULT_CFG, lr: 1e-1 },
  tiny: { nEmbd: 32, nLayer: 1, blockSize: 16, batch: 8, lr: 5e-3 },
};

const template = document.createElement('template');
template.innerHTML = `
  <div class="widget w07-tr">
    <h3 class="widget__title">브라우저 GPT 학습기</h3>
    <div data-slot="status" class="widget__status" role="status" aria-live="polite">
      <span class="spinner" aria-hidden="true"></span> <span>문서셋과 학습 엔진을 준비하는 중…</span>
    </div>
    <div class="widget__controls">
      <label class="field">
        <span class="field__label">모델 폭 (임베딩 차원 D)</span>
        <select data-in="nEmbd">
          <option value="32">작게 · D32</option>
          <option value="48">보통 · D48 (완성 모델과 같음, 느림)</option>
        </select>
      </label>
      <label class="field">
        <span class="field__label">트랜스포머 블록 수</span>
        <select data-in="nLayer">
          <option value="1">1층</option>
          <option value="2">2층</option>
          <option value="3">3층</option>
        </select>
      </label>
      <label class="field">
        <span class="field__label">문맥 길이 (blockSize)</span>
        <select data-in="blockSize">
          <option value="16">16글자</option>
          <option value="32">32글자</option>
          <option value="48">48글자</option>
        </select>
      </label>
      <label class="field">
        <span class="field__label">학습률 (최대값)</span>
        <select data-in="lr">${LRS.map((o) => `<option value="${o.v}">${o.label}</option>`).join('')}</select>
      </label>
      <label class="field">
        <span class="field__label">배치 크기 (한 스텝의 창 수)</span>
        <select data-in="batch">
          <option value="4">4</option>
          <option value="8">8</option>
          <option value="16">16</option>
        </select>
      </label>
      <label class="field">
        <span class="field__label">샘플 시작 글자</span>
        <span class="w07-prompt-row">
          <input type="text" data-in="prompt" maxlength="24" spellcheck="false">
          <button type="button" class="btn small ghost" data-act="sample" title="지금 모델로 샘플 뽑기">✍ 샘플</button>
        </span>
      </label>
    </div>
    <p class="w07-cfg" data-out="cfg"></p>
    <div class="btn-row">
      <button type="button" class="btn small primary" data-act="run" disabled>▶ 학습</button>
      <button type="button" class="btn small" data-act="step" disabled>1스텝</button>
      <button type="button" class="btn small" data-act="reset" disabled>⟲ 처음부터</button>
      <button type="button" class="btn small ghost" data-act="pre">📦 완성 모델 불러오기 (3,000스텝 사전학습)</button>
    </div>
    <div class="btn-row w07-presets">
      <span class="w07-muted">비교 실험:</span>
      <button type="button" class="btn small ghost" data-preset="hot">학습률 너무 큼 (0.1)</button>
      <button type="button" class="btn small ghost" data-preset="tiny">작은 모델 (1층 · 문맥 16)</button>
      <button type="button" class="btn small ghost" data-preset="base">기본 설정으로</button>
    </div>
    <p class="w07-pending" data-slot="pending" hidden>설정이 바뀌었다. <b>⟲ 처음부터</b>를 누르면 새 설정의 모델을 무작위 가중치에서 다시 학습한다. 지금 곡선은 회색으로 남겨 비교한다.</p>
    <div data-slot="out-inline"></div>
  </div>`;

const outTemplate = document.createElement('template');
outTemplate.innerHTML = `
  <div class="w07-out">
    <div data-slot="verdict" aria-live="polite"></div>
    <div class="stat-row" data-slot="stats"></div>
    <h4>손실 곡선 <small class="w07-muted">— 낮을수록 다음 글자를 잘 맞힌다</small></h4>
    <div class="w07-chart" data-slot="chart"></div>
    <div class="legend" data-slot="legend"></div>
    <h4>샘플 변천사 <small class="w07-muted">— 같은 시작 글자 · 같은 seed · 온도 0.8. 바뀌는 것은 가중치뿐이다</small></h4>
    <ol class="w07-samples" data-slot="samples" tabindex="0" aria-label="학습 스텝별 생성 샘플"></ol>
    <div class="legend" aria-hidden="true">
      <span><i class="w07-lg-prompt"></i>시작 글자</span>
      <span><i class="w07-lg-copy"></i>코퍼스 원문 그대로 ${COPY_MIN}자 이상 (외운 부분)</span>
    </div>
    <div data-slot="pre"></div>
  </div>`;

const state = new WeakMap();
let seq = 0;

/**
 * @param {HTMLElement} el
 * @param {{ outputTitle?: string, prompt?: string }} options
 */
export async function mount(el, options = {}) {
  const root = template.content.firstElementChild.cloneNode(true);
  const out = outTemplate.content.firstElementChild.cloneNode(true);
  el.replaceChildren(root);
  const $ = (sel) => root.querySelector(sel) ?? out.querySelector(sel);
  const ctrl = new AbortController();
  const outputId = `w07-trainer:${++seq}`;

  const s = {
    form: { ...DEFAULT_CFG },
    cfg: { ...DEFAULT_CFG },
    prompt: options.prompt ?? '실습실은',
    worker: null,
    gen: 0, // worker generation (ignore messages from a terminated worker)
    phase: 'loading', // loading | booting | ready | running | paused | done | nan | error
    errMsg: '',
    params: 0,
    trainChars: 0,
    step: 0,
    ema: null,
    lastLoss: null,
    lr: 0,
    gradNorm: 0,
    ms: 0,
    points: [], // [step, ema]
    evals: [], // { step, train, val }
    samples: [], // { step, text, promptLen, mask }
    ghosts: [], // earlier runs for comparison
    pending: false,
    corpus: null,
    vocab: null,
    trainIds: null,
    trainText: '',
    pre: null, // pretrained model panel
    preToken: 0,
    raf: 0,
    drawn: { samples: '', pre: '' },
  };
  const st = { ctrl, outputId, s };
  state.set(el, st);

  for (const k of ['nEmbd', 'nLayer', 'blockSize', 'lr', 'batch']) $(`[data-in=${k}]`).value = String(s.form[k]);
  $('[data-in=prompt]').value = s.prompt;

  registerOutput(outputId, { title: options.outputTitle ?? '브라우저 GPT 학습기 결과', node: out, inlineHost: $('[data-slot=out-inline]') });

  // ------------------------------------------------------------ worker

  function boot() {
    if (s.worker) s.worker.terminate();
    if (s.points.length > 3) {
      s.ghosts.push({ label: cfgLabel(s.cfg), pts: s.points.slice(), vals: s.evals.slice() });
      if (s.ghosts.length > 3) s.ghosts.shift();
    }
    s.cfg = { ...s.form };
    Object.assign(s, { phase: 'booting', step: 0, ema: null, lastLoss: null, lr: 0, gradNorm: 0, ms: 0, points: [], evals: [], samples: [], pending: false, errMsg: '' });
    const gen = ++s.gen;
    let w;
    try {
      w = new Worker(WORKER_URL, { type: 'module' });
    } catch (err) {
      s.phase = 'error';
      s.errMsg = `이 브라우저는 모듈 워커를 지원하지 않는다 (${err.message}). 최신 Chrome·Edge·Firefox·Safari를 쓴다.`;
      schedule();
      return;
    }
    s.worker = w;
    w.onmessage = ({ data }) => {
      if (gen === s.gen) onWorker(data);
    };
    w.onerror = (e) => {
      e.preventDefault?.();
      if (gen !== s.gen) return;
      s.phase = 'error';
      s.errMsg = `학습 워커를 시작하지 못했다 (${e.message || '모듈 로딩 실패'}). file://로 열었다면 python -m http.server로 연다.`;
      schedule();
    };
    const { nEmbd, nLayer, blockSize, batch, lr } = s.cfg;
    w.postMessage({
      type: 'init',
      config: { blockSize, nLayer, nHead: 4, nEmbd },
      train: { batchSize: batch, lr, warmup: WARMUP, totalSteps: TOTAL }, // model seed 1337, batch seed 7 (engine defaults)
      valDocs: VAL_DOCS,
      evalEvery: EVAL_EVERY,
      sampleEvery: SAMPLE_EVERY,
      prompt: s.prompt,
    });
    computeInitialLoss(gen);
    schedule();
  }

  /** Step-0 loss of the very same untrained model (same seed as the worker). Cheap: one forward pass. */
  function computeInitialLoss(gen) {
    if (!s.vocab) return;
    setTimeout(() => {
      if (gen !== s.gen || ctrl.signal.aborted) return;
      const { nEmbd, nLayer, blockSize } = s.cfg;
      const model = createModel({ vocabSize: s.vocab.size, blockSize, nLayer, nHead: 4, nEmbd }, 1337);
      const train = evaluate(model, s.trainIds, { windows: 12 });
      const val = evaluate(model, s.valIds, { windows: 12 });
      if (!s.evals.some((e) => e.step === 0)) s.evals.unshift({ step: 0, train, val });
      schedule();
    }, 30);
  }

  function onWorker(m) {
    if (m.type === 'ready') {
      s.params = m.params;
      s.trainChars = m.trainChars;
      s.phase = 'ready';
    } else if (m.type === 'progress') {
      if (m.loss !== undefined) {
        s.step = m.step;
        s.lastLoss = m.loss;
        if (!Number.isFinite(m.loss)) {
          s.phase = 'nan';
          s.worker.postMessage({ type: 'pause' });
        } else {
          s.ema = s.ema === null ? m.loss : (1 - EMA) * s.ema + EMA * m.loss;
          s.points.push([m.step, s.ema]);
        }
        s.lr = m.lr;
        s.gradNorm = m.gradNorm;
        if (m.msPerStep) s.ms = s.ms ? 0.8 * s.ms + 0.2 * m.msPerStep : m.msPerStep;
        if (m.trainLoss !== undefined) s.evals.push({ step: m.evalAt ?? m.step, train: m.trainLoss, val: m.valLoss });
      }
      if (s.phase !== 'nan') s.phase = m.running ? 'running' : s.step >= TOTAL ? 'done' : s.step > 0 ? 'paused' : 'ready';
    } else if (m.type === 'sample') {
      addSample(m.step, m.prompt, m.text);
    } else if (m.type === 'error') {
      s.phase = 'error';
      s.errMsg = m.message;
    }
    schedule();
  }

  function addSample(step, prompt, text) {
    const promptLen = Math.max(1, s.vocab ? s.vocab.encode(prompt).length : [...prompt].length);
    const gen = [...text].slice(promptLen).join('');
    const item = { step, text, promptLen, mask: s.trainText ? copyMask(gen, s.trainText) : [], manual: step % SAMPLE_EVERY !== 0 };
    const i = s.samples.findIndex((x) => x.step === step);
    if (i >= 0) s.samples[i] = item;
    else if (s.samples.at(-1)?.manual) s.samples[s.samples.length - 1] = item; // keep one off-grid sample (1스텝 clicks)
    else s.samples.push(item);
  }

  const send = (msg) => s.worker?.postMessage(msg);

  // ------------------------------------------------------------ pretrained

  async function loadPretrained() {
    if (s.pre?.phase === 'loading') return;
    if (s.pre?.phase === 'ready') return samplePretrained();
    s.pre = { phase: 'loading' };
    schedule();
    for (const step of PRETRAINED) {
      try {
        const res = await fetch(MODEL_URL(step));
        if (!res.ok) continue;
        const ckpt = await res.json();
        if (ctrl.signal.aborted) return;
        const { model, vocab, meta } = deserialize(ckpt);
        s.pre = { phase: 'ready', step, model, vocab, meta, params: paramCount(model), config: ckpt.config, text: '', mask: [], promptLen: 0 };
        break;
      } catch {
        /* try the next snapshot */
      }
    }
    if (s.pre.phase !== 'ready') {
      s.pre = { phase: 'error' };
      schedule();
      return;
    }
    await samplePretrained();
  }

  /** Generate on the main thread, a few tokens per task so the page never freezes. */
  async function samplePretrained() {
    const p = s.pre;
    if (p?.phase !== 'ready') return;
    const token = ++s.preToken;
    const rand = mulberry32(3);
    let ids = p.vocab.encode(s.prompt);
    if (!ids.length) ids = [0];
    const promptLen = ids.length;
    p.busy = true;
    schedule();
    for (let i = 0; i < SAMPLE_LEN; i++) {
      ids = generate(p.model, ids, 1, { temperature: 0.8, rand }).ids;
      if (i % 4 === 3) {
        await new Promise((r) => setTimeout(r, 0));
        if (token !== s.preToken || ctrl.signal.aborted) return;
      }
    }
    const text = p.vocab.decode(ids);
    p.text = text;
    p.promptLen = promptLen;
    p.mask = copyMask([...text].slice(promptLen).join(''), s.trainText);
    p.busy = false;
    schedule();
  }

  // ------------------------------------------------------------ render

  function schedule() {
    if (s.raf || ctrl.signal.aborted) return;
    s.raf = requestAnimationFrame(() => {
      s.raf = 0;
      render();
    });
  }

  function render() {
    const running = s.phase === 'running';
    const live = ['ready', 'paused', 'running', 'done'].includes(s.phase);
    const runBtn = $('[data-act=run]');
    runBtn.textContent = running ? '⏸ 멈춤' : s.step > 0 && s.phase !== 'done' ? '▶ 이어서 학습' : '▶ 학습';
    runBtn.disabled = !live || s.phase === 'done';
    $('[data-act=step]').disabled = !live || running || s.phase === 'done';
    $('[data-act=reset]').disabled = s.phase === 'loading' || s.phase === 'booting';
    $('[data-act=reset]').classList.toggle('primary', s.pending || s.phase === 'nan' || s.phase === 'done');
    $('[data-slot=pending]').hidden = !s.pending;
    const status = $('[data-slot=status]');
    if (s.phase === 'error') {
      status.hidden = false;
      status.innerHTML = `<div class="widget__error">${esc(s.errMsg)}</div>`;
    } else if (s.phase === 'loading' || s.phase === 'booting') {
      status.hidden = false;
      status.innerHTML = `<span class="spinner" aria-hidden="true"></span> <span>${s.phase === 'loading' ? '문서셋과 학습 엔진을 준비하는 중…' : '무작위 가중치로 모델을 만드는 중…'}</span>`;
    } else status.hidden = true;

    const { nEmbd, nLayer, blockSize, batch } = s.cfg;
    const perStep = batch * blockSize;
    $('[data-out=cfg]').innerHTML = s.params
      ? `파라미터 <b>${s.params.toLocaleString()}</b>개 · 한 스텝 = 창 ${batch}개 × ${blockSize}글자 = <b>${perStep}</b>개의 “다음 글자” 문제 · 학습 코퍼스 ${s.trainChars.toLocaleString()}자 → 약 <b>${Math.round(s.trainChars / perStep)}</b>스텝이면 코퍼스를 한 번 훑는다(1에폭)`
      : `D${nEmbd} · ${nLayer}층 · 문맥 ${blockSize} · 배치 ${batch}`;

    renderStats();
    $('[data-slot=verdict]').innerHTML = verdict();
    $('[data-slot=chart]').innerHTML = chartSvg();
    $('[data-slot=legend]').innerHTML = [
      '<span><i class="w07-lg-ema"></i>학습 손실 (이동평균)</span>',
      '<span><i class="w07-lg-train"></i>학습 손실 (고정 창 평가)</span>',
      '<span><i class="w07-lg-val"></i>검증 손실 (안 본 문서)</span>',
      s.ghosts.length ? '<span><i class="w07-lg-ghost"></i>이전 실행</span>' : '',
    ].join('');
    // lists are rebuilt only when they change, so text stays selectable while training runs
    const last = s.samples.at(-1);
    const sKey = `${s.gen}|${s.samples.length}|${last?.step}|${last?.text}`;
    if (sKey !== s.drawn.samples) {
      s.drawn.samples = sKey;
      renderSamples();
    }
    const p = s.pre;
    const pKey = `${p?.phase}|${p?.busy}|${p?.text}|${s.trainChars}`;
    if (pKey !== s.drawn.pre) {
      s.drawn.pre = pKey;
      renderPretrained();
    }
  }

  function renderStats() {
    const lastEval = s.evals.at(-1);
    const epochs = s.trainChars ? (s.step * s.cfg.batch * s.cfg.blockSize) / s.trainChars : 0;
    const tps = s.ms ? (s.cfg.batch * s.cfg.blockSize * 1000) / s.ms : 0;
    const remain = s.ms && s.step < TOTAL ? ((TOTAL - s.step) * s.ms) / 1000 : 0;
    $('[data-slot=stats]').innerHTML = [
      stat('스텝', `${s.step.toLocaleString()} / ${TOTAL.toLocaleString()}`),
      stat('손실 (이동평균)', s.phase === 'nan' ? 'NaN' : s.ema === null ? (s.evals[0] ? s.evals[0].train.toFixed(2) : '—') : s.ema.toFixed(3)),
      stat('검증 손실', lastEval ? lastEval.val.toFixed(3) : '—'),
      stat('ms / 스텝', s.ms ? s.ms.toFixed(0) : '—'),
      stat('글자 / 초', tps ? Math.round(tps).toLocaleString() : '—'),
      stat('학습률', s.lr ? s.lr.toExponential(1) : '—'),
      stat('기울기 노름', s.gradNorm ? s.gradNorm.toFixed(2) : '—'),
      stat('에폭', epochs.toFixed(1)),
      stat('파라미터', s.params ? s.params.toLocaleString() : '—'),
      stat('남은 시간', remain ? fmtTime(remain) : '—'),
    ].join('');
  }

  function renderSamples() {
    const box = $('[data-slot=samples]');
    const atBottom = box.scrollHeight - box.scrollTop - box.clientHeight < 30;
    box.innerHTML = s.samples.length
      ? s.samples
          .map((x) => {
            const copied = x.mask.filter(Boolean).length;
            const rate = x.mask.length ? copied / x.mask.length : 0;
            return `<li><span class="w07-step">${x.step.toLocaleString()}</span><span class="w07-text">${textHtml(x.text, x.promptLen, x.mask)}</span>${rate > 0 ? `<span class="w07-copy-badge" title="생성 부분 중 코퍼스에 그대로 있는 ${COPY_MIN}자 이상 구간의 비율">복사 ${Math.round(rate * 100)}%</span>` : ''}</li>`;
          })
          .join('')
      : '<li class="w07-muted">샘플을 기다리는 중…</li>';
    if (atBottom) box.scrollTop = box.scrollHeight;
  }

  function renderPretrained() {
    const host = $('[data-slot=pre]');
    const p = s.pre;
    const btn = $('[data-act=pre]');
    btn.disabled = p?.phase === 'loading';
    if (!p) {
      host.innerHTML = '';
      return;
    }
    if (p.phase === 'loading') {
      host.innerHTML = '<h4>완성 모델</h4><p class="w07-muted"><span class="spinner" aria-hidden="true"></span> 체크포인트(약 0.6MB)를 받는 중…</p>';
      return;
    }
    if (p.phase === 'error') {
      host.innerHTML = '<div class="widget__error">완성 모델 체크포인트를 불러오지 못했다. 네트워크를 확인한다.</div>';
      return;
    }
    const h = p.meta.history ?? [];
    const last = h.at(-1);
    const best = h.reduce((a, b) => (b.val < a.val ? b : a), h[0] ?? { val: NaN, step: 0 });
    const c = p.config;
    const epochs = (p.step * (p.meta.batchSize ?? 16) * c.blockSize) / (s.trainChars || 6470);
    const copied = p.mask.filter(Boolean).length;
    const rate = p.mask.length ? copied / p.mask.length : 0;
    const fallback = p.step !== PRETRAINED[0] ? ` <span class="w07-muted">(3,000스텝 파일이 없어 ${p.step.toLocaleString()}스텝 스냅샷을 불러왔다)</span>` : '';
    host.innerHTML = `
      <h4>완성 모델 · ${p.step.toLocaleString()}스텝 사전학습${fallback}</h4>
      <p class="w07-muted">D${c.nEmbd} · ${c.nLayer}층 · 문맥 ${c.blockSize} · 배치 ${p.meta.batchSize ?? 16} · 파라미터 ${p.params.toLocaleString()}개 · 코퍼스를 약 ${Math.round(epochs)}번 반복해 봤다</p>
      <div class="stat-row">
        ${stat('학습 손실', last ? last.train.toFixed(2) : '—')}
        ${stat('검증 손실', last ? last.val.toFixed(2) : '—')}
        ${stat('검증 최저', best ? `${best.val.toFixed(2)} @${best.step}` : '—')}
        ${stat('원문 복사율', p.busy ? '…' : `${Math.round(rate * 100)}%`)}
      </div>
      <div class="w07-chart">${historySvg(h)}</div>
      <p class="w07-pre-sample">${p.busy ? '<span class="w07-muted">생성 중…</span>' : textHtml(p.text, p.promptLen, p.mask)}</p>
      ${
        p.busy
          ? ''
          : rate >= 0.5
            ? `<div class="callout callout--more"><span class="callout__title">잘 쓴 것이 아니라 외운 것이다</span><p>생성한 ${p.mask.length}자 중 ${copied}자(${Math.round(rate * 100)}%)가 코퍼스에 그대로 있는 문장이다. 학습 손실은 ${last?.train.toFixed(2)}까지 내려갔지만 검증 손실은 step ${best.step}의 ${best.val.toFixed(2)}에서 ${last?.val.toFixed(2)}까지 올랐다. 6,470자짜리 코퍼스를 수백 번 반복해 본 모델은 문서를 통째로 외운다.</p></div>`
            : `<div class="callout"><span class="callout__title">내 모델과 비교한다</span><p>같은 시작 글자·같은 seed로 뽑은 글이다. 원문 복사율 ${Math.round(rate * 100)}%. 학습 곡선의 검증 손실 최저점(step ${best.step})과 지금 위치를 비교한다.</p></div>`
      }`;
  }

  // ------------------------------------------------------------ verdict

  function verdict() {
    const lnV = s.vocab ? Math.log(s.vocab.size) : Math.log(495);
    const V = s.vocab?.size ?? 495;
    if (s.phase === 'loading' || s.phase === 'booting' || s.phase === 'error') return '';
    if (s.phase === 'nan') {
      return box('danger', '💥 손실이 NaN이 되었다 — 학습이 폭발했다', `step ${s.step}에서 손실이 숫자가 아니게 되었다(NaN). 가중치가 한 번에 너무 멀리 움직여 소프트맥스의 지수가 넘쳤다. 학습을 자동으로 멈췄다. 학습률을 낮추고 <b>⟲ 처음부터</b>를 누른다.`);
    }
    const init = s.evals.find((e) => e.step === 0);
    if (s.step === 0) {
      return box(
        '',
        '아직 한 번도 학습하지 않았다',
        `가중치 ${s.params ? s.params.toLocaleString() : ''}개가 무작위라 모델은 글자 ${V}종에 거의 같은 확률(1/${V})을 준다. 그래서 손실은 −ln(1/${V}) = ln ${V} ≈ <b>${lnV.toFixed(2)}</b>${init ? `이고, 실제로 잰 값도 <b>${init.train.toFixed(2)}</b>이다` : '에서 시작한다'}. 아래 샘플은 글자 수프다. <b>▶ 학습</b>을 누른다.`,
      );
    }
    const ema = s.ema ?? lnV;
    if (s.step > 20 && ema > lnV + 0.5) {
      return box('danger', '📈 발산: 손실이 처음보다 커졌다', `이동평균 손실 ${ema.toFixed(2)} — 찍기 수준(ln ${V} = ${lnV.toFixed(2)})보다 높다. 학습률이 너무 커서 한 번 움직일 때마다 골짜기를 넘어간다. 학습률을 낮춘다.`);
    }
    // stalled: no progress over the last 100 steps while still high
    const back = s.points.find(([st]) => st >= s.step - 100);
    const stalled = s.step >= 200 && ema > 3.6 && back && back[1] - ema < 0.1;
    if (stalled) {
      const big = s.cfg.lr >= 3e-2;
      return box(
        'danger',
        '⏸ 정체: 손실이 내려가지 않는다',
        big
          ? `학습률 ${s.cfg.lr}은 너무 크다. 최근 100스텝 동안 손실이 ${back[1].toFixed(2)} → ${ema.toFixed(2)}, 거의 그대로다. 한 걸음이 너무 커서 골짜기 바닥으로 내려가지 못하고 벽 사이를 튕겨 다닌다. 샘플을 보면 자주 나오는 글자(공백·“다”·“.”)만 늘어 있다. 0.005로 낮춰 다시 학습하면 같은 스텝에서 손실이 2점대로 내려간다.`
          : `최근 100스텝 동안 손실이 ${back[1].toFixed(2)} → ${ema.toFixed(2)}, 거의 그대로다. 학습률이나 모델 크기를 바꿔 본다.`,
      );
    }
    const vals = s.evals.filter((e) => e.step > 0);
    const best = vals.reduce((a, b) => (b.val < a.val ? b : a), vals[0]);
    const recent = vals.slice(-3); // eval on 12 windows is noisy: average the last three
    const recentVal = recent.length ? recent.reduce((a, e) => a + e.val, 0) / recent.length : Infinity;
    const lastEval = vals.at(-1);
    if (best && recent.length === 3 && recentVal >= best.val + 0.3 && lastEval.step >= best.step + 100 && lastEval.train < best.train - 0.3) {
      const copyN = s.samples.filter((x) => x.mask.some(Boolean)).length;
      return box(
        'more',
        '🧠 과대적합: 모델이 코퍼스를 외우기 시작했다',
        `학습 손실은 ${lastEval.train.toFixed(2)}까지 계속 내려가는데, 검증 손실(학습에 쓰지 않은 “벡터 데이터베이스” 문서)은 step ${best.step}의 최저 ${best.val.toFixed(2)}에서 ${lastEval.val.toFixed(2)}까지 올라갔다. 새 글에 쓸 규칙이 아니라 학습 문서 자체를 외우고 있다는 신호다${copyN ? ` — 샘플의 물결 밑줄이 원문을 그대로 베낀 부분이다(${copyN}개 샘플)` : ''}. 지금 ${((s.step * s.cfg.batch * s.cfg.blockSize) / s.trainChars).toFixed(0)}에폭째다. 9주차에 이 현상을 수치로 다룬다.`,
      );
    }
    if (s.phase === 'done') {
      return box('ok', `${TOTAL.toLocaleString()}스텝 완료`, `학습률 스케줄이 끝났다. 손실 ${ema.toFixed(2)}. 샘플 변천사를 위에서부터 읽어 본다. 📦 완성 모델과 비교하거나 설정을 바꿔 ⟲ 처음부터 다시 한다.`);
    }
    if (s.cfg.lr <= 1e-3 && s.step >= 100 && ema > 3.3) {
      return box('', '🐢 천천히 내려간다', `학습률 ${s.cfg.lr}은 안전하지만 한 걸음이 작다. 기본값(0.005)이라면 이 스텝에서 손실이 이미 더 낮다. 회색 이전 곡선과 비교한다.`);
    }
    let msg;
    if (ema > 4.5) msg = ['글자 빈도부터 배운다', '가장 먼저 배우는 것은 “어떤 글자가 자주 나오는가”다. 공백, “다”, “.”, “이”가 샘플에 늘어난다. 1주차 유니그램 수준이다.'];
    else if (ema > 3.2) msg = ['자주 붙는 글자 쌍을 배운다', '“토큰”, “모델”, “벡터”처럼 짧은 조각이 보이기 시작한다. 1주차 바이그램이 빈도표로 하던 일을 신경망이 가중치로 하고 있다.'];
    else if (ema > 2.2) msg = ['단어와 띄어쓰기 모양이 나온다', '어절 길이, “~다.”로 끝나는 문장, 숫자와 %가 제자리를 찾는다. 뜻은 아직 없다.'];
    else msg = ['문장처럼 보인다 — 원문 복사율을 확인한다', '손실이 2 아래로 내려가면 문장이 그럴듯해진다. 그러나 검증 손실도 같이 내려가는지, 샘플에 물결 밑줄(원문 복사)이 생기는지 확인한다.'];
    return box(s.phase === 'running' ? '' : 'ok', `${s.phase === 'running' ? '학습 중' : '멈춤'} · ${msg[0]}`, `${msg[1]} 손실 ${ema.toFixed(2)} ≈ 매번 약 ${Math.exp(ema).toFixed(0)}개 후보 중에서 고르는 정도의 불확실성이다.`);
  }

  // ------------------------------------------------------------ charts

  function chartSvg() {
    const lnV = s.vocab ? Math.log(s.vocab.size) : Math.log(495);
    const maxStep = Math.max(s.step, ...s.ghosts.map((g) => g.pts.at(-1)?.[0] ?? 0));
    const xMax = niceMax(Math.max(200, maxStep * 1.08));
    const series = [];
    s.ghosts.forEach((g, i) => series.push({ pts: g.pts, cls: 'w07-c-ghost', label: i === s.ghosts.length - 1 ? g.label : '' }));
    series.push({ pts: s.points, cls: 'w07-c-ema' });
    const vals = s.evals.filter((e) => Number.isFinite(e.val));
    series.push({ pts: vals.map((e) => [e.step, e.train]), cls: 'w07-c-train', dots: true, noLine: true });
    series.push({ pts: vals.map((e) => [e.step, e.val]), cls: 'w07-c-val', dots: true });
    const marks = [{ y: lnV, label: `찍기 수준 ln ${s.vocab?.size ?? 495} = ${lnV.toFixed(2)}` }];
    const best = vals.filter((e) => e.step > 0).reduce((a, b) => (!a || b.val < a.val ? b : a), null);
    const vmarks = best && s.step >= best.step + 100 ? [{ x: best.step, label: `검증 최저 ${best.val.toFixed(2)}` }] : [];
    const yMax = Math.max(7, Math.ceil(Math.max(0, ...vals.map((e) => e.val), ...s.points.slice(-50).map(([, v]) => v))));
    return lineChart({ xMax, yMax, series, marks, vmarks, nan: s.phase === 'nan' });
  }

  function historySvg(h) {
    if (!h.length) return '';
    const yMax = Math.ceil(Math.max(7, ...h.map((e) => e.val)));
    const best = h.reduce((a, b) => (b.val < a.val ? b : a));
    return lineChart({
      xMax: niceMax(h.at(-1).step),
      yMax,
      series: [
        { pts: h.map((e) => [e.step, e.train]), cls: 'w07-c-train' },
        { pts: h.map((e) => [e.step, e.val]), cls: 'w07-c-val' },
      ],
      marks: [],
      vmarks: [{ x: best.step, label: `검증 최저 ${best.val.toFixed(2)}` }],
      height: 150,
    });
  }

  // ------------------------------------------------------------ events

  const on = (sel, type, fn) => root.querySelectorAll(sel).forEach((n) => n.addEventListener(type, fn, { signal: ctrl.signal }));
  on('select[data-in]', 'change', (e) => {
    const k = e.target.dataset.in;
    s.form[k] = Number(e.target.value);
    applyForm();
  });
  on('[data-in=prompt]', 'change', (e) => {
    s.prompt = e.target.value;
    requestSample();
  });
  on('[data-in=prompt]', 'keydown', (e) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      s.prompt = e.target.value;
      requestSample();
    }
  });
  root.addEventListener(
    'click',
    (e) => {
      const presetKey = e.target.closest('[data-preset]')?.dataset.preset;
      if (presetKey) {
        s.form = { ...PRESETS[presetKey] };
        for (const k of Object.keys(s.form)) $(`[data-in=${k}]`).value = String(s.form[k]);
        if (s.worker) boot();
        return;
      }
      const act = e.target.closest('[data-act]')?.dataset.act;
      if (act === 'run') {
        if (s.phase === 'running') send({ type: 'pause' });
        else if (s.phase !== 'nan' && s.step < TOTAL) {
          send({ type: 'step', n: TOTAL - s.step });
          s.phase = 'running';
        }
      } else if (act === 'step') {
        send({ type: 'step', n: 1 });
      } else if (act === 'reset') {
        boot();
      } else if (act === 'sample') {
        s.prompt = $('[data-in=prompt]').value;
        requestSample();
      } else if (act === 'pre') {
        loadPretrained();
      }
      schedule();
    },
    { signal: ctrl.signal },
  );

  function applyForm() {
    const same = Object.keys(s.form).every((k) => s.form[k] === s.cfg[k]);
    if (same) s.pending = false;
    else if (s.step === 0 && s.phase !== 'running' && s.worker) boot();
    else s.pending = true;
    schedule();
  }

  function requestSample() {
    send({ type: 'sample', prompt: s.prompt, length: SAMPLE_LEN, temperature: 0.8, seed: 3 });
    if (s.pre?.phase === 'ready') samplePretrained();
  }

  // ------------------------------------------------------------ start

  try {
    s.corpus = await loadCorpus();
    if (ctrl.signal.aborted) return;
    s.vocab = charVocab(corpusText(s.corpus));
    const trainDocIds = s.corpus.documents.filter((d) => !VAL_DOCS.includes(d.id)).map((d) => d.id);
    s.trainText = corpusText(s.corpus, trainDocIds);
    s.trainIds = Int32Array.from(s.vocab.encode(s.trainText));
    s.valIds = Int32Array.from(s.vocab.encode(corpusText(s.corpus, VAL_DOCS)));
    boot();
  } catch (err) {
    s.phase = 'error';
    s.errMsg = `문서셋을 불러오지 못했다 (${err.message}). file://로 열었다면 python -m http.server로 연다.`;
    render();
  }
}

export function unmount(el) {
  const st = state.get(el);
  if (!st) return;
  st.ctrl.abort();
  st.s.worker?.terminate();
  st.s.worker = null;
  st.s.preToken++;
  if (st.s.raf) cancelAnimationFrame(st.s.raf);
  unregisterOutput(st.outputId);
  state.delete(el);
  el.replaceChildren();
}

// ---------------------------------------------------------------- helpers

const cfgLabel = (c) => `D${c.nEmbd}·${c.nLayer}층·문맥${c.blockSize}·배치${c.batch}·lr ${c.lr}`;

/** For each generated character: is it inside a run of ≥ COPY_MIN chars found verbatim in the source? */
export function copyMask(gen, source) {
  const g = [...gen];
  const mask = new Array(g.length).fill(false);
  let i = 0;
  while (i < g.length) {
    let L = 0;
    while (i + L < g.length && source.includes(g.slice(i, i + L + 1).join(''))) L++;
    if (L >= COPY_MIN) {
      for (let j = i; j < i + L; j++) mask[j] = true;
      i += L;
    } else i++;
  }
  return mask;
}

function textHtml(text, promptLen, mask) {
  const chars = [...text];
  const show = (c) => (c === '\n' ? '↵' : c);
  let html = `<span class="w07-p">${esc(chars.slice(0, promptLen).map(show).join(''))}</span>`;
  const rest = chars.slice(promptLen);
  let i = 0;
  while (i < rest.length) {
    const c = !!mask[i];
    let j = i;
    while (j < rest.length && !!mask[j] === c) j++;
    const piece = esc(rest.slice(i, j).map(show).join(''));
    html += c ? `<mark class="w07-copy">${piece}</mark>` : piece;
    i = j;
  }
  return html;
}

function niceMax(v) {
  const steps = [200, 300, 400, 500, 600, 800, 1000, 1200, 1500, 2000, 2500, 3000, 4000, 5000];
  return steps.find((x) => x >= v) ?? Math.ceil(v / 1000) * 1000;
}

/** Inline SVG line chart; colours come from CSS classes (tokens only). */
function lineChart({ xMax, yMax, series, marks = [], vmarks = [], height = 190, nan = false }) {
  const W = 340;
  const H = height;
  const L = 30;
  const R = 8;
  const T = 10;
  const B = 22;
  const x = (v) => L + (v / xMax) * (W - L - R);
  const y = (v) => T + (1 - Math.min(v, yMax) / yMax) * (H - T - B);
  const parts = [];
  const yStep = yMax > 8 ? 2 : 1;
  for (let v = 0; v <= yMax; v += yStep) {
    parts.push(`<line class="w07-grid" x1="${L}" x2="${W - R}" y1="${y(v)}" y2="${y(v)}"/><text class="w07-ax" x="${L - 5}" y="${y(v) + 3.5}" text-anchor="end">${v}</text>`);
  }
  const xStep = [50, 100, 200, 250, 500, 1000].find((d) => xMax / d <= 6) ?? 1000;
  for (let v = 0; v <= xMax; v += xStep) {
    parts.push(`<text class="w07-ax" x="${x(v)}" y="${H - 6}" text-anchor="middle">${v.toLocaleString()}</text>`);
  }
  parts.push(`<text class="w07-ax" x="${W - R}" y="${H - 16}" text-anchor="end">step</text>`);
  for (const m of marks) {
    parts.push(`<line class="w07-mark" x1="${L}" x2="${W - R}" y1="${y(m.y)}" y2="${y(m.y)}"/><text class="w07-mark-t" x="${W - R}" y="${y(m.y) - 4}" text-anchor="end">${esc(m.label)}</text>`);
  }
  for (const m of vmarks) {
    const xx = x(m.x);
    parts.push(`<line class="w07-vmark" x1="${xx}" x2="${xx}" y1="${T}" y2="${H - B}"/><text class="w07-vmark-t" x="${Math.min(xx + 4, W - 90)}" y="${T + 10}">${esc(m.label)}</text>`);
  }
  for (const sr of series) {
    const pts = thin(sr.pts.filter(([, v]) => Number.isFinite(v)), 400);
    if (!pts.length) continue;
    if (!sr.noLine && pts.length > 1) parts.push(`<polyline class="${sr.cls}" fill="none" points="${pts.map(([a, b]) => `${x(a).toFixed(1)},${y(b).toFixed(1)}`).join(' ')}"/>`);
    if (sr.dots) for (const [a, b] of pts) parts.push(`<circle class="${sr.cls} dot" cx="${x(a).toFixed(1)}" cy="${y(b).toFixed(1)}" r="2.2"/>`);
    if (sr.label) {
      const [a, b] = pts.at(-1);
      parts.push(`<text class="w07-ghost-t" x="${Math.min(x(a), W - R)}" y="${y(b) - 5}" text-anchor="end">${esc(sr.label)}</text>`);
    }
  }
  if (nan) parts.push(`<text class="w07-nan" x="${(L + W) / 2}" y="${H / 2}" text-anchor="middle">NaN 💥</text>`);
  return `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="스텝에 따른 손실 곡선" style="width:100%;height:auto">${parts.join('')}</svg>`;
}

function thin(pts, max) {
  if (pts.length <= max) return pts;
  const k = Math.ceil(pts.length / max);
  const out = pts.filter((_, i) => i % k === 0);
  if (out.at(-1) !== pts.at(-1)) out.push(pts.at(-1));
  return out;
}

function box(kind, title, body) {
  const cls = kind ? ` callout--${kind}` : '';
  return `<div class="callout${cls}"><span class="callout__title">${title}</span><p>${body}</p></div>`;
}

const stat = (label, value) => `<div class="stat"><span class="stat__label">${label}</span><span class="stat__value">${value}</span></div>`;

function fmtTime(sec) {
  if (sec < 60) return `${Math.round(sec)}초`;
  return `${Math.floor(sec / 60)}분 ${Math.round(sec % 60)}초`;
}

function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
}
