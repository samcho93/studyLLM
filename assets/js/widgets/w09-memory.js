// w09 LLM 메모리 계산기
// One concept: model size decides what fits where. Inference needs the weights
// (params × bytes) plus the KV cache (2 · layers · kv heads · head_dim · context ·
// bytes · batch); training with Adam needs ~16 bytes per parameter; training
// compute is ≈ 6 · N · D FLOPs. All numbers are estimates (activations,
// framework overhead and quantisation scales are ignored).
// Failure modes: a 70B model does not fit one 80 GB GPU even in fp16; a long
// context makes the KV cache larger than the weights; training time in years.

import { registerOutput, unregisterOutput } from '../site/result.js';

/** Model presets. mini = our checkpoint (exact); gpt2 = published structure (≈124M); 1B/7B/70B = representative configurations (대표 설정 예시). */
export const PRESETS = {
  mini: { label: '우리 미니 GPT (0.1M)', params: 106959, layers: 2, kvHeads: 4, headDim: 12, ctx: 48, tokens: 3000 * 16 * 48, note: '7~9주차 모델 · 실제 값 · 학습 토큰 = 3000스텝 × 배치 16 × 48글자' },
  gpt2: { label: 'GPT-2 small (124M)', params: 124e6, layers: 12, kvHeads: 12, headDim: 64, ctx: 1024, tokens: null, note: '공개된 구조 · 12층 · 헤드 12 × 64차원' },
  b1: { label: '1B (대표 설정 예시)', params: 1e9, layers: 16, kvHeads: 8, headDim: 64, ctx: 4096, tokens: null, note: '대표 설정 예시 · 16층 · KV 헤드 8(GQA) × 64차원' },
  b7: { label: '7B (대표 설정 예시)', params: 7e9, layers: 32, kvHeads: 32, headDim: 128, ctx: 4096, tokens: null, note: '대표 설정 예시 · 32층 · 헤드 32 × 128차원' },
  b70: { label: '70B (대표 설정 예시)', params: 70e9, layers: 80, kvHeads: 8, headDim: 128, ctx: 4096, tokens: null, note: '대표 설정 예시 · 80층 · KV 헤드 8(GQA) × 128차원' },
};

export const PRECISIONS = {
  fp32: { label: 'fp32 (4바이트)', bytes: 4 },
  fp16: { label: 'fp16 · bf16 (2바이트)', bytes: 2 },
  int8: { label: 'int8 (1바이트)', bytes: 1 },
  int4: { label: 'int4 (0.5바이트)', bytes: 0.5 },
};

/** Approximate dense 16-bit matrix throughput (rough public figures — estimates). */
export const GPUS_TFLOPS = [
  { id: 'consumer', label: 'RTX 4090급 · 약 165 TFLOPS', tflops: 165 },
  { id: 'a100', label: 'A100급 · 약 312 TFLOPS', tflops: 312 },
  { id: 'h100', label: 'H100급 · 약 990 TFLOPS', tflops: 990 },
];

export const GPU_SIZES = [8, 24, 80];
export const TRAIN_BYTES_PER_PARAM = 16; // fp16 weight 2 + grad 2 + fp32 master 4 + Adam m 4 + v 4
export const CHINCHILLA = 20; // tokens per parameter (rule of thumb)
export const GB = 1e9;

/** Memory estimate in bytes. */
export function memory({ params, bytes, layers, kvHeads, headDim, ctx, batch, kvBytes }) {
  const weights = params * bytes;
  const kvPerToken = 2 * layers * kvHeads * headDim * kvBytes; // K and V for every layer
  const kv = kvPerToken * ctx * batch;
  return { weights, kvPerToken, kv, infer: weights + kv, train: params * TRAIN_BYTES_PER_PARAM };
}

/** Training compute ≈ 6 · N · D FLOPs (forward 2ND + backward 4ND). */
export const trainFlops = (params, tokens) => 6 * params * tokens;

/** Wall-clock seconds for `flops` on `gpus` GPUs of `tflops` at utilisation `util`. */
export const trainSeconds = (flops, tflops, util, gpus = 1) => flops / (tflops * 1e12 * util * gpus);

// ---------------------------------------------------------------- widget

const P_MIN = 5; // log10 params slider range: 100K … 1T
const P_MAX = 12;
const D_MIN = 6; // log10 tokens: 1M … 100T
const D_MAX = 14;

const template = document.createElement('template');
template.innerHTML = `
  <div class="widget w09-mem">
    <h3 class="widget__title">LLM 메모리 계산기</h3>
    <div class="widget__controls">
      <label class="field">
        <span class="field__label">모델 프리셋</span>
        <select data-in="preset"></select>
      </label>
      <label class="field">
        <span class="field__label">파라미터 수 N <output data-out="params"></output></span>
        <input type="range" data-in="params" min="${P_MIN}" max="${P_MAX}" step="0.01">
      </label>
      <label class="field">
        <span class="field__label">가중치 정밀도</span>
        <select data-in="prec"></select>
      </label>
      <label class="field">
        <span class="field__label">문맥 길이 (토큰) <output data-out="ctx"></output></span>
        <input type="range" data-in="ctx" min="4" max="17" step="1">
      </label>
      <label class="field">
        <span class="field__label">배치 (동시 요청 수) <output data-out="batch"></output></span>
        <input type="range" data-in="batch" min="0" max="7" step="1">
      </label>
      <label class="field">
        <span class="field__label">KV 캐시 정밀도</span>
        <select data-in="kvprec"></select>
      </label>
    </div>
    <fieldset class="w09-kv">
      <legend>KV 캐시 구조 (프리셋이 채운다 · 직접 바꿔도 된다)</legend>
      <label class="field"><span class="field__label">층 수</span><input type="number" data-in="layers" min="1" max="256" step="1"></label>
      <label class="field"><span class="field__label">KV 헤드 수</span><input type="number" data-in="kvHeads" min="1" max="256" step="1"></label>
      <label class="field"><span class="field__label">헤드 차원</span><input type="number" data-in="headDim" min="1" max="1024" step="1"></label>
    </fieldset>
    <fieldset class="w09-kv">
      <legend>학습 계산량 (추정)</legend>
      <label class="field w09-wide"><span class="field__label">학습 토큰 D <output data-out="tokens"></output></span><input type="range" data-in="tokens" min="${D_MIN}" max="${D_MAX}" step="0.01"></label>
      <label class="field"><span class="field__label">GPU</span><select data-in="gpu"></select></label>
      <label class="field"><span class="field__label">GPU 개수</span><input type="number" data-in="gpus" min="1" max="100000" step="1"></label>
    </fieldset>
    <div class="btn-row">
      <button type="button" class="btn small ghost" data-act="chin">D = 20N (Chinchilla 어림)</button>
      <button type="button" class="btn small ghost" data-act="70b">70B fp16 · 한 장에 올려 보기</button>
      <button type="button" class="btn small ghost" data-act="longctx">7B · 문맥 128K</button>
      <button type="button" class="btn small ghost" data-act="int4">7B int4 · 8GB에 넣기</button>
    </div>
    <div data-slot="out-inline"></div>
  </div>`;

const outTemplate = document.createElement('template');
outTemplate.innerHTML = `
  <div class="w09-mem-out">
    <div data-slot="verdict" aria-live="polite"></div>
    <div class="stat-row" data-slot="stats"></div>
    <h4>메모리 <small class="w09-muted">— 로그 눈금 · 세로선 = GPU 8 / 24 / 80GB · 1GB = 10⁹바이트 · 활성값 제외</small></h4>
    <div class="w09-bars" data-slot="bars"></div>
    <h4>학습 계산량 <small class="w09-muted">— 추정치</small></h4>
    <div class="stat-row" data-slot="compute"></div>
    <details class="w09-formula">
      <summary>계산식 보기</summary>
      <pre data-slot="formula"></pre>
    </details>
  </div>`;

const state = new WeakMap();
let seq = 0;
const DEFAULTS = { preset: 'b7', prec: 'fp16', kvprec: 'fp16', batch: 1, gpu: 'a100', gpus: 1, util: 0.4 };

/**
 * @param {HTMLElement} el
 * @param {{ preset?: string, outputTitle?: string }} options
 */
export function mount(el, options = {}) {
  const root = template.content.firstElementChild.cloneNode(true);
  const out = outTemplate.content.firstElementChild.cloneNode(true);
  el.replaceChildren(root);
  const $ = (sel) => root.querySelector(sel) ?? out.querySelector(sel);
  const ctrl = new AbortController();
  const outputId = `w09-memory:${++seq}`;
  state.set(el, { ctrl, outputId });

  const s = {};
  function applyPreset(id) {
    const p = PRESETS[id];
    Object.assign(s, { preset: id, params: p.params, layers: p.layers, kvHeads: p.kvHeads, headDim: p.headDim, ctx: p.ctx, tokens: p.tokens ?? CHINCHILLA * p.params });
  }
  Object.assign(s, DEFAULTS);
  applyPreset(options.preset && PRESETS[options.preset] ? options.preset : DEFAULTS.preset);

  $('[data-in=preset]').innerHTML = Object.entries(PRESETS).map(([id, p]) => `<option value="${id}">${esc(p.label)}</option>`).join('') + '<option value="custom" disabled>직접 설정</option>';
  const precOpts = Object.entries(PRECISIONS).map(([id, p]) => `<option value="${id}">${esc(p.label)}</option>`).join('');
  $('[data-in=prec]').innerHTML = precOpts;
  $('[data-in=kvprec]').innerHTML = precOpts;
  $('[data-in=gpu]').innerHTML = GPUS_TFLOPS.map((g) => `<option value="${g.id}">${esc(g.label)}</option>`).join('');

  registerOutput(outputId, { title: options.outputTitle ?? 'LLM 메모리 계산 결과', node: out, inlineHost: $('[data-slot=out-inline]') });

  function syncControls() {
    $('[data-in=preset]').value = s.preset;
    $('[data-in=params]').value = String(Math.log10(s.params));
    $('[data-in=prec]').value = s.prec;
    $('[data-in=kvprec]').value = s.kvprec;
    $('[data-in=ctx]').value = String(Math.round(Math.log2(s.ctx)));
    $('[data-in=batch]').value = String(Math.round(Math.log2(s.batch)));
    $('[data-in=layers]').value = String(s.layers);
    $('[data-in=kvHeads]').value = String(s.kvHeads);
    $('[data-in=headDim]').value = String(s.headDim);
    $('[data-in=tokens]').value = String(Math.log10(s.tokens));
    $('[data-in=gpu]').value = s.gpu;
    $('[data-in=gpus]').value = String(s.gpus);
  }

  function render() {
    const bytes = PRECISIONS[s.prec].bytes;
    const kvBytes = PRECISIONS[s.kvprec].bytes;
    const m = memory({ params: s.params, bytes, layers: s.layers, kvHeads: s.kvHeads, headDim: s.headDim, ctx: s.ctx, batch: s.batch, kvBytes });
    const gpu = GPUS_TFLOPS.find((g) => g.id === s.gpu);
    const flops = trainFlops(s.params, s.tokens);
    const secs = trainSeconds(flops, gpu.tflops, s.util, s.gpus);
    const P = PRESETS[s.preset];

    $('[data-out=params]').textContent = fmtN(s.params);
    $('[data-out=ctx]').textContent = s.ctx.toLocaleString();
    $('[data-out=batch]').textContent = String(s.batch);
    $('[data-out=tokens]').textContent = `${fmtN(s.tokens)} (N의 ${fmtRatio(s.tokens / s.params)}배)`;

    $('[data-slot=stats]').innerHTML = [
      stat('가중치', fmtB(m.weights)),
      stat('KV 캐시', fmtB(m.kv)),
      stat('토큰당 KV', fmtB(m.kvPerToken)),
      stat('추론 합계', fmtB(m.infer)),
      stat('학습 (Adam)', fmtB(m.train)),
    ].join('');

    // log-scale bars
    const bars = [
      { label: '가중치', sub: `${fmtN(s.params)} × ${bytes}B`, v: m.weights, cls: 'w' },
      { label: 'KV 캐시', sub: `문맥 ${s.ctx.toLocaleString()} × 배치 ${s.batch}`, v: m.kv, cls: 'kv' },
      { label: '추론 합계', sub: '가중치 + KV 캐시', v: m.infer, cls: 'inf', part: m.weights },
      { label: '학습 (Adam)', sub: `${fmtN(s.params)} × 16B`, v: m.train, cls: 'tr' },
    ];
    const lo = 1e4;
    const hi = Math.max(1e12, 10 ** Math.ceil(Math.log10(Math.max(...bars.map((b) => b.v)) * 1.2)));
    const pos = (v) => (Math.max(0, Math.log10(Math.max(v, lo)) - Math.log10(lo)) / (Math.log10(hi) - Math.log10(lo))) * 100;
    const ticks = [];
    for (let e = Math.log10(lo); e <= Math.log10(hi); e++) ticks.push(e);
    $('[data-slot=bars]').innerHTML =
      bars
        .map((b) => {
          const fits = GPU_SIZES.find((g) => b.v <= g * GB);
          const tag = fits ? `<b class="ok">${fits}GB ✓</b>` : `<b class="no">80GB ✗ · ${Math.ceil(b.v / (80 * GB))}장</b>`;
          const inner = b.part != null && b.v > 0 ? `<i class="part" style="width:${((pos(b.part) / Math.max(pos(b.v), 1e-9)) * 100).toFixed(1)}%"></i>` : '';
          return `<div class="w09-bar ${b.cls}">
            <span class="l">${esc(b.label)}<small>${esc(b.sub)}</small></span>
            <span class="b"><span class="fill" style="width:${pos(b.v).toFixed(2)}%">${inner}</span>${GPU_SIZES.map((g) => `<span class="gpu" style="left:${pos(g * GB).toFixed(2)}%"></span>`).join('')}</span>
            <span class="v">${fmtB(b.v)} ${tag}</span>
          </div>`;
        })
        .join('') +
      `<div class="w09-axis"><span></span><span class="ax">${ticks.map((e) => `<i style="left:${pos(10 ** e).toFixed(2)}%">${fmtB(10 ** e, true)}</i>`).join('')}${GPU_SIZES.map((g) => `<i class="g" style="left:${pos(g * GB).toFixed(2)}%">${g}G</i>`).join('')}</span><span></span></div>`;

    // compute
    const chin = CHINCHILLA * s.params;
    $('[data-slot=compute]').innerHTML = [
      stat('6·N·D', `${fmtSci(flops)} FLOPs`),
      stat('Chinchilla 어림 D', fmtN(chin)),
      stat('토큰/파라미터', fmtRatio(s.tokens / s.params)),
      stat(`예상 시간 (${s.gpus.toLocaleString()}장 · 활용률 40%)`, fmtTime(secs)),
    ].join('');

    $('[data-slot=formula]').textContent = [
      `가중치 = N × 바이트 = ${s.params.toLocaleString()} × ${bytes} = ${fmtB(m.weights)}`,
      `KV 캐시 = 2(K,V) × 층 ${s.layers} × KV 헤드 ${s.kvHeads} × 헤드 차원 ${s.headDim} × 문맥 ${s.ctx} × ${kvBytes}바이트 × 배치 ${s.batch}`,
      `        = ${fmtB(m.kvPerToken)}/토큰 × ${(s.ctx * s.batch).toLocaleString()}토큰 = ${fmtB(m.kv)}`,
      `학습 메모리 ≈ N × 16바이트 (fp16 가중치 2 + 기울기 2 + fp32 원본 4 + Adam m 4 + v 4) = ${fmtB(m.train)}  · 활성값 제외`,
      `학습 계산량 ≈ 6 × N × D = 6 × ${fmtSci(s.params)} × ${fmtSci(s.tokens)} = ${fmtSci(flops)} FLOPs`,
      `시간 ≈ FLOPs ÷ (${gpu.tflops} TFLOPS × 10¹² × 활용률 ${s.util} × GPU ${s.gpus}장) = ${fmtTime(secs)}   (모두 추정치)`,
      s.preset !== 'custom' ? `프리셋: ${P.note}` : '프리셋 값을 직접 바꾼 설정',
    ].join('\n');

    // verdict
    const v = [];
    if (m.infer > 80 * GB) {
      v.push(callout('danger', `추론만 해도 80GB GPU 한 장에 들어가지 않는다 (${fmtB(m.infer)})`, `가중치만 ${fmtN(s.params)} × ${bytes}바이트 = ${fmtB(m.weights)}다. 최소 ${Math.ceil(m.infer / (80 * GB))}장에 나눠 싣거나(텐서·파이프라인 병렬), 정밀도를 int8·int4로 낮춘다(양자화).`));
    } else if (m.infer > 8 * GB) {
      const fit = GPU_SIZES.find((g) => m.infer <= g * GB);
      v.push(callout('more', `추론은 ${fit}GB GPU부터 들어간다 (${fmtB(m.infer)})`, `8GB 노트북 GPU에는 넘친다. 정밀도를 낮추면 어떻게 되는지 본다.`));
    } else {
      v.push(callout('ok', `추론은 8GB GPU에도 들어간다 (${fmtB(m.infer)})`, s.params < 1e6 ? '우리 미니 GPT는 휴대폰 사진 한 장보다 작다. 브라우저에서 학습할 수 있었던 이유다.' : '가중치 정밀도를 낮추면 품질이 조금 떨어지는 대신 메모리가 줄어든다.'));
    }
    if (m.kv > m.weights) {
      v.push(callout('danger', `KV 캐시(${fmtB(m.kv)})가 가중치(${fmtB(m.weights)})보다 크다`, `KV 캐시는 문맥 길이 × 배치에 비례해 커진다. 토큰 하나에 ${fmtB(m.kvPerToken)}씩 쌓인다. 긴 문맥과 많은 동시 사용자가 GPU 메모리를 먹는 주범이다. KV 헤드 수를 줄이는 GQA가 그래서 나왔다.`));
    }
    if (m.train > 80 * GB) {
      v.push(callout('danger', `Adam 학습은 ${fmtB(m.train)} — 80GB GPU ${Math.ceil(m.train / (80 * GB))}장 이상`, `학습하려면 파라미터마다 기울기와 Adam 상태까지 약 16바이트가 필요하다. 활성값까지 더하면 더 든다. 그래서 큰 모델은 분산 학습을 하고, 미세조정은 10주차 LoRA로 줄인다.`));
    }
    const ratio = s.tokens / s.params;
    if (s.preset === 'mini') {
      const epochs = s.tokens / 6470;
      v.push(callout('more', `토큰/파라미터 ${fmtRatio(ratio)} — 그러나 같은 글을 약 ${Math.round(epochs)}번 반복`, `Chinchilla 어림(약 20)과 비슷해 보이지만, 우리 학습 문서는 6,470자뿐이다. ${fmtN(s.tokens)}토큰은 같은 글을 약 ${Math.round(epochs)}번(epoch) 다시 본 것이다. 어림식은 <b>서로 다른</b> 토큰을 가정한다. 반복해서 보면 외운다 — 퍼플렉서티 측정기의 과대적합이 그 결과다.`));
    } else if (secs > 365 * 86400) {
      v.push(callout('danger', `GPU ${s.gpus.toLocaleString()}장으로 ${fmtTime(secs)}`, `6·N·D = ${fmtSci(flops)} FLOPs다. GPU 개수를 늘려 본다. 대형 모델은 수천 장을 몇 달 동안 쓴다.`));
    } else if (ratio < CHINCHILLA / 4) {
      v.push(callout('more', `토큰/파라미터 ${fmtRatio(ratio)} — 데이터가 모자란다`, `Chinchilla 어림은 파라미터 1개당 약 20토큰이다. 이보다 훨씬 적으면 같은 계산량으로 더 작은 모델을 더 많은 데이터로 학습하는 편이 낫다.`));
    }
    $('[data-slot=verdict]').innerHTML = v.join('');
  }

  const markCustom = () => (s.preset = 'custom');
  const on = (type, fn) => root.addEventListener(type, fn, { signal: ctrl.signal });
  on('input', (e) => {
    const t = e.target;
    const k = t.dataset.in;
    if (k === 'params') {
      s.params = Math.round(10 ** Number(t.value));
      markCustom();
    } else if (k === 'ctx') s.ctx = 2 ** Number(t.value);
    else if (k === 'batch') s.batch = 2 ** Number(t.value);
    else if (k === 'tokens') s.tokens = Math.round(10 ** Number(t.value));
    else if (k === 'layers' || k === 'kvHeads' || k === 'headDim' || k === 'gpus') {
      const n = Math.floor(Number(t.value));
      if (!(n >= 1)) return;
      s[k] = Math.min(n, Number(t.max));
      if (k !== 'gpus') markCustom();
    } else return;
    $('[data-in=preset]').value = s.preset;
    render();
  });
  on('change', (e) => {
    const t = e.target;
    if (t.matches('[data-in=preset]')) {
      applyPreset(t.value);
      syncControls();
    } else if (t.matches('[data-in=prec]')) s.prec = t.value;
    else if (t.matches('[data-in=kvprec]')) s.kvprec = t.value;
    else if (t.matches('[data-in=gpu]')) s.gpu = t.value;
    else return;
    render();
  });
  on('click', (e) => {
    const act = e.target.closest('[data-act]')?.dataset.act;
    if (!act) return;
    if (act === 'chin') s.tokens = CHINCHILLA * s.params;
    if (act === '70b') {
      applyPreset('b70');
      Object.assign(s, { prec: 'fp16', kvprec: 'fp16', batch: 1 });
    }
    if (act === 'longctx') {
      applyPreset('b7');
      Object.assign(s, { prec: 'fp16', kvprec: 'fp16', ctx: 131072, batch: 1 });
    }
    if (act === 'int4') {
      applyPreset('b7');
      Object.assign(s, { prec: 'int4', kvprec: 'fp16', batch: 1 });
    }
    syncControls();
    render();
  });

  syncControls();
  render();
}

export function unmount(el) {
  const st = state.get(el);
  if (!st) return;
  st.ctrl.abort();
  unregisterOutput(st.outputId);
  state.delete(el);
  el.replaceChildren();
}

// ---------------------------------------------------------------- helpers

export function fmtN(n) {
  if (n >= 1e12) return `${+(n / 1e12).toFixed(n >= 1e13 ? 0 : 1)}T`;
  if (n >= 1e9) return `${+(n / 1e9).toFixed(n >= 1e10 ? 0 : 1)}B`;
  if (n >= 1e6) return `${+(n / 1e6).toFixed(n >= 1e7 ? 0 : 1)}M`;
  if (n >= 1e3) return `${+(n / 1e3).toFixed(n >= 1e4 ? 0 : 1)}K`;
  return String(Math.round(n));
}

export function fmtB(b, short = false) {
  const u = [
    [1e12, 'TB'],
    [1e9, 'GB'],
    [1e6, 'MB'],
    [1e3, 'KB'],
  ].find(([d]) => b >= d);
  if (!u) return `${Math.round(b)}B`;
  const v = b / u[0];
  return `${short ? +v.toFixed(0) : +v.toFixed(v >= 100 ? 0 : v >= 10 ? 1 : 2)}${u[1]}`;
}

export function fmtSci(x) {
  const e = Math.floor(Math.log10(x));
  return `${(x / 10 ** e).toFixed(2)}×10^${e}`;
}

const fmtRatio = (r) => (r >= 100 ? Math.round(r).toLocaleString() : r >= 1 ? r.toFixed(1) : r.toPrecision(2));

export function fmtTime(sec) {
  if (sec < 1) return `${(sec * 1000).toFixed(0)}ms`;
  if (sec < 120) return `${sec.toFixed(1)}초`;
  if (sec < 7200) return `${(sec / 60).toFixed(1)}분`;
  if (sec < 172800) return `${(sec / 3600).toFixed(1)}시간`;
  if (sec < 365 * 86400) return `${(sec / 86400).toFixed(1)}일`;
  return `${(sec / (365 * 86400)).toFixed(1)}년`;
}

const callout = (kind, title, html) =>
  `<div class="callout${kind === 'ok' ? ' callout--ok' : kind === 'danger' ? ' callout--danger' : ' callout--more'}"><span class="callout__title">${esc(title)}</span><p>${html}</p></div>`;

const stat = (label, value) => `<div class="stat"><span class="stat__label">${label}</span><span class="stat__value">${value}</span></div>`;

function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
}
