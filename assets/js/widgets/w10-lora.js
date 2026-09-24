// w10 LoRA 계산기
// One concept: LoRA freezes W and trains only B·A (rank r), so the trainable
// parameter count drops from d_in·d_out to r·(d_in + d_out) per layer — but the
// frozen weights still have to sit in GPU memory (hence QLoRA's 4-bit weights).
// Failure modes: full fine-tuning of an 8B model does not fit one 24 GB GPU;
// a rank too large makes LoRA bigger than the layer it adapts.

import { registerOutput, unregisterOutput } from '../site/result.js';

const lin = (key, name, group, din, dout) => ({ key, name, group, din, dout });

/**
 * Model presets. `modules` are the linear weights of ONE transformer layer
 * (LoRA candidates); `other` = everything else (embeddings, output head,
 * biases, norms), which LoRA never touches.
 */
export const PRESETS = {
  mini: {
    label: '우리 미니 GPT (D=48, L=2)',
    note: '7주차 모델 · 어휘 495자 · 문맥 32',
    L: 2,
    other: 50895,
    modules: [
      lin('q', 'q (qkv의 1/3)', 'q', 48, 48),
      lin('k', 'k (qkv의 1/3)', 'k', 48, 48),
      lin('v', 'v (qkv의 1/3)', 'v', 48, 48),
      lin('o', 'proj (o)', 'o', 48, 48),
      lin('fc', 'MLP fc', 'mlp', 48, 192),
      lin('fc2', 'MLP fc2', 'mlp', 192, 48),
    ],
  },
  gpt2: {
    label: 'GPT-2 small (D=768, L=12)',
    note: '어휘 50,257 · 문맥 1,024 · 입출력 임베딩 공유',
    L: 12,
    other: 39505152,
    modules: [
      lin('q', 'q (c_attn의 1/3)', 'q', 768, 768),
      lin('k', 'k (c_attn의 1/3)', 'k', 768, 768),
      lin('v', 'v (c_attn의 1/3)', 'v', 768, 768),
      lin('o', 'c_proj (o)', 'o', 768, 768),
      lin('fc', 'MLP c_fc', 'mlp', 768, 3072),
      lin('fc2', 'MLP c_proj', 'mlp', 3072, 768),
    ],
  },
  llama8b: {
    label: 'Llama 3 8B급 (D=4096, L=32) · 근사',
    note: 'FFN 14,336 · 어휘 128,256 · GQA(k·v 헤드 8개) · 입출력 임베딩 분리',
    L: 32,
    other: 1050939392,
    modules: [
      lin('q', 'q_proj', 'q', 4096, 4096),
      lin('k', 'k_proj', 'k', 4096, 1024),
      lin('v', 'v_proj', 'v', 4096, 1024),
      lin('o', 'o_proj', 'o', 4096, 4096),
      lin('gate', 'gate_proj', 'mlp', 4096, 14336),
      lin('up', 'up_proj', 'mlp', 4096, 14336),
      lin('down', 'down_proj', 'mlp', 14336, 4096),
    ],
  },
};

// Bytes per frozen weight. int4 (QLoRA) applies to the linear layers only;
// embeddings / head / norms stay 16-bit. Quantization constants are ignored.
export const PRECISIONS = {
  fp32: { label: 'fp32 (4바이트)', bytes: 4 },
  fp16: { label: 'fp16 (2바이트)', bytes: 2 },
  bf16: { label: 'bf16 (2바이트)', bytes: 2 },
  int4: { label: 'int4 · QLoRA (0.5바이트)', bytes: 0.5 },
};

export const GROUPS = [
  { id: 'q', label: 'q' },
  { id: 'k', label: 'k' },
  { id: 'v', label: 'v' },
  { id: 'o', label: 'o' },
  { id: 'mlp', label: 'MLP' },
];

export const FULL_FT_BYTES = 16; // fp16 weight 2 + grad 2 + fp32 master 4 + Adam m 4 + v 4
export const TRAIN_BYTES = 16; // LoRA adapters in fp32: weight 4 + grad 4 + Adam 8

/** Parameter / memory estimate for one configuration. */
export function estimate(preset, r, targets, precision) {
  const P = typeof preset === 'string' ? PRESETS[preset] : preset;
  const linearPerLayer = P.modules.reduce((s, m) => s + m.din * m.dout, 0);
  const linear = linearPerLayer * P.L;
  const total = linear + P.other;
  const rows = P.modules.map((m) => {
    const on = targets.includes(m.group);
    const orig = m.din * m.dout;
    const lora = r * (m.din + m.dout);
    return { ...m, on, orig, lora, loraAll: on ? lora * P.L : 0, rankCap: Math.min(m.din, m.dout) };
  });
  const lora = rows.reduce((s, m) => s + m.loraAll, 0);
  const bytes = PRECISIONS[precision].bytes;
  const frozenBytes = precision === 'int4' ? linear * bytes + P.other * 2 : total * bytes;
  return {
    total,
    linear,
    lora,
    ratio: lora / total,
    rows,
    mem: {
      full: total * FULL_FT_BYTES,
      frozen: frozenBytes,
      loraTrain: lora * TRAIN_BYTES,
      lora: frozenBytes + lora * TRAIN_BYTES,
      infer: frozenBytes,
    },
  };
}

// ---------------------------------------------------------------- widget

const GPUS = [8, 16, 24, 48, 80];

const template = document.createElement('template');
template.innerHTML = `
  <div class="widget w10-lora">
    <h3 class="widget__title">LoRA 계산기</h3>
    <div class="widget__controls">
      <label class="field">
        <span class="field__label">모델</span>
        <select data-in="preset"></select>
      </label>
      <label class="field">
        <span class="field__label">rank r <output data-out="r"></output></span>
        <input type="range" data-in="r" min="1" max="128" step="1">
      </label>
      <label class="field">
        <span class="field__label">동결 가중치 정밀도</span>
        <select data-in="prec"></select>
      </label>
      <label class="field">
        <span class="field__label">GPU 메모리</span>
        <select data-in="gpu"></select>
      </label>
    </div>
    <fieldset class="w10-targets">
      <legend>LoRA를 붙일 층 (target modules)</legend>
      <div data-slot="targets"></div>
    </fieldset>
    <div class="btn-row">
      <button type="button" class="btn small ghost" data-act="r1">r = 1 (너무 작다?)</button>
      <button type="button" class="btn small ghost" data-act="mini-big">미니 GPT r = 32 (LoRA가 더 크다)</button>
      <button type="button" class="btn small ghost" data-act="qlora">8B QLoRA 4비트</button>
      <button type="button" class="btn small ghost" data-act="reset">↺ 처음 설정</button>
    </div>
    <div data-slot="out-inline"></div>
  </div>`;

const outTemplate = document.createElement('template');
outTemplate.innerHTML = `
  <div class="w10-lora-out">
    <div data-slot="verdict" aria-live="polite"></div>
    <div class="stat-row" data-slot="stats"></div>
    <h4>학습 메모리 추정 <small class="w10-muted">— 가중치 + 기울기 + Adam 상태 · 활성값 제외 · 1GB = 10⁹바이트</small></h4>
    <div class="w10-mem" data-slot="mem"></div>
    <h4>층별 파라미터 <small class="w10-muted">— 원래 d_in × d_out → LoRA r × (d_in + d_out)</small></h4>
    <div class="w10-table-wrap"><table class="w10-mods" data-slot="mods"></table></div>
    <details class="w10-formula">
      <summary>계산식 보기</summary>
      <pre data-slot="formula"></pre>
    </details>
  </div>`;

const state = new WeakMap();
let seq = 0;
const DEFAULTS = { preset: 'llama8b', r: 16, targets: ['q', 'k', 'v', 'o', 'mlp'], prec: 'bf16', gpu: 24 };

/**
 * @param {HTMLElement} el
 * @param {{ preset?: string, r?: number, outputTitle?: string }} options
 */
export function mount(el, options = {}) {
  const root = template.content.firstElementChild.cloneNode(true);
  const out = outTemplate.content.firstElementChild.cloneNode(true);
  el.replaceChildren(root);
  const $ = (sel) => root.querySelector(sel) ?? out.querySelector(sel);
  const ctrl = new AbortController();
  const outputId = `w10-lora:${++seq}`;
  state.set(el, { ctrl, outputId });

  const s = { ...DEFAULTS, targets: [...DEFAULTS.targets], preset: options.preset ?? DEFAULTS.preset, r: options.r ?? DEFAULTS.r };

  $('[data-in=preset]').innerHTML = Object.entries(PRESETS).map(([id, p]) => `<option value="${id}">${esc(p.label)}</option>`).join('');
  $('[data-in=prec]').innerHTML = Object.entries(PRECISIONS).map(([id, p]) => `<option value="${id}">${esc(p.label)}</option>`).join('');
  $('[data-in=gpu]').innerHTML = GPUS.map((g) => `<option value="${g}">${g}GB${g === 24 ? ' (RTX 4090급)' : g === 80 ? ' (A100·H100급)' : ''}</option>`).join('');
  $('[data-slot=targets]').innerHTML = GROUPS.map((g) => `<label><input type="checkbox" data-target="${g.id}"> ${g.label}</label>`).join('');

  registerOutput(outputId, { title: options.outputTitle ?? 'LoRA 계산 결과', node: out, inlineHost: $('[data-slot=out-inline]') });

  function syncControls() {
    $('[data-in=preset]').value = s.preset;
    $('[data-in=r]').value = String(s.r);
    $('[data-in=prec]').value = s.prec;
    $('[data-in=gpu]').value = String(s.gpu);
    root.querySelectorAll('[data-target]').forEach((c) => (c.checked = s.targets.includes(c.dataset.target)));
  }

  function render() {
    const P = PRESETS[s.preset];
    const e = estimate(P, s.r, s.targets, s.prec);
    const gpuB = s.gpu * 1e9;
    $('[data-out=r]').textContent = String(s.r);

    $('[data-slot=stats]').innerHTML = [
      stat('전체 파라미터', fmtN(e.total)),
      stat('전체 미세조정 학습', fmtN(e.total)),
      stat('LoRA 학습', fmtN(e.lora)),
      stat('LoRA 비율', fmtPct(e.ratio)),
    ].join('');

    // memory bars
    const bars = [
      { label: '전체 미세조정', sub: `${fmtN(e.total)} × 16B`, v: e.mem.full, cls: 'full' },
      { label: `LoRA · ${PRECISIONS[s.prec].label.split(' ')[0]}`, sub: `동결 ${fmtB(e.mem.frozen)} + 학습 ${fmtB(e.mem.loraTrain)}`, v: e.mem.lora, cls: 'lora', part: e.mem.frozen },
      { label: '추론만 (참고)', sub: '가중치만', v: e.mem.infer, cls: 'infer' },
    ];
    const max = Math.max(gpuB, ...bars.map((b) => b.v)) * 1.05;
    const gpuPct = (gpuB / max) * 100;
    $('[data-slot=mem]').innerHTML =
      bars
        .map((b) => {
          const fits = b.v <= gpuB;
          const w = (b.v / max) * 100;
          const inner = b.part != null ? `<i class="frozen" style="width:${((b.part / b.v) * 100).toFixed(2)}%"></i>` : '';
          return `<div class="w10-mbar ${b.cls}${fits ? '' : ' over'}">
            <span class="l">${esc(b.label)}<small>${esc(b.sub)}</small></span>
            <span class="b"><span class="fill" style="width:${w.toFixed(2)}%">${inner}</span><span class="gpu" style="left:${gpuPct.toFixed(2)}%" title="GPU ${s.gpu}GB"></span></span>
            <span class="v">${fmtB(b.v)} ${fits ? '<b class="ok">✓</b>' : '<b class="no">✗</b>'}</span>
          </div>`;
        })
        .join('') + `<p class="w10-muted w10-gpu-note">세로선 = GPU ${s.gpu}GB. ✓는 “가중치·기울기·옵티마이저 상태만 따지면 들어간다”는 뜻이다. 실제로는 활성값(배치·문장 길이에 비례)이 더 든다.</p>`;

    // module table
    const cap = e.rows.reduce((a, m) => Math.min(a, m.rankCap), Infinity);
    $('[data-slot=mods]').innerHTML = `<thead><tr><th scope="col">층</th><th scope="col">d_in × d_out</th><th scope="col">원래 (층 1개)</th><th scope="col">LoRA (층 1개)</th><th scope="col">LoRA × ${P.L}층</th></tr></thead><tbody>${e.rows
      .map(
        (m) => `<tr class="${m.on ? '' : 'off'}${m.on && m.lora >= m.orig ? ' bad' : ''}"><th scope="row">${esc(m.name)}</th><td>${m.din.toLocaleString()} × ${m.dout.toLocaleString()}</td><td>${m.orig.toLocaleString()}</td><td>${m.lora.toLocaleString()}${m.on && m.lora >= m.orig ? ' ⚠' : ''}</td><td>${m.on ? m.loraAll.toLocaleString() : '—'}</td></tr>`,
      )
      .join('')}<tr class="sum"><th scope="row">임베딩·출력층·편향·정규화</th><td colspan="2">${P.other.toLocaleString()}</td><td colspan="2">동결 (학습 안 함)</td></tr></tbody>`;

    // formula with numbers
    const on = e.rows.filter((m) => m.on);
    $('[data-slot=formula]').textContent = [
      `LoRA 층 하나: h = W·x + (α/r)·B·(A·x),  A: r×d_in,  B: d_out×r,  B는 0으로 시작`,
      `LoRA 파라미터 = r × (d_in + d_out) × 층 수`,
      ...on.map((m) => `  ${m.name}: ${s.r} × (${m.din} + ${m.dout}) × ${P.L} = ${m.loraAll.toLocaleString()}`),
      `  합계 = ${e.lora.toLocaleString()}  (전체 ${e.total.toLocaleString()}의 ${fmtPct(e.ratio)})`,
      ``,
      `전체 미세조정 메모리 ≈ 파라미터 × 16바이트 (fp16 가중치 2 + 기울기 2 + fp32 원본 4 + Adam m 4 + v 4)`,
      `  = ${e.total.toLocaleString()} × 16 = ${fmtB(e.mem.full)}`,
      `LoRA 메모리 ≈ 동결 가중치 + LoRA 파라미터 × 16바이트`,
      s.prec === 'int4'
        ? `  동결 = 선형층 ${e.linear.toLocaleString()} × 0.5 + 나머지 ${P.other.toLocaleString()} × 2 = ${fmtB(e.mem.frozen)}  (임베딩·출력층은 16비트로 둔다)`
        : `  동결 = ${e.total.toLocaleString()} × ${PRECISIONS[s.prec].bytes} = ${fmtB(e.mem.frozen)}`,
      `  학습 = ${e.lora.toLocaleString()} × 16 = ${fmtB(e.mem.loraTrain)}`,
      `  합계 = ${fmtB(e.mem.lora)}   (모두 추정치 · 활성값 제외)`,
    ].join('\n');

    // verdict
    const v = [];
    if (!s.targets.length) {
      v.push(callout('danger', '학습할 층이 없다', 'LoRA를 붙일 층을 하나도 고르지 않았다. 학습 파라미터가 0이면 모델은 그대로다.'));
    }
    const bad = on.filter((m) => m.lora >= m.orig);
    if (bad.length) {
      v.push(callout('danger', `LoRA가 원래 층보다 크다 (r = ${s.r})`, `${esc(bad[0].name)}은 ${bad[0].din}×${bad[0].dout} = ${bad[0].orig.toLocaleString()}개인데 LoRA는 ${s.r}×(${bad[0].din}+${bad[0].dout}) = ${bad[0].lora.toLocaleString()}개다. 게다가 B·A의 rank는 min(d_in, d_out) = ${bad[0].rankCap}을 넘을 수 없어 r을 키워도 표현력은 늘지 않는다. 작은 모델에는 그냥 전체 미세조정이 낫다.`));
    } else if (on.length && s.r > cap) {
      v.push(callout('more', 'r이 층의 최대 rank를 넘었다', `B·A의 rank는 min(d_in, d_out)을 넘을 수 없다. 이 모델의 가장 작은 층은 ${cap}이다.`));
    }
    if (e.mem.full > gpuB) {
      v.push(callout('danger', `전체 미세조정은 GPU ${s.gpu}GB 한 장에 들어가지 않는다`, `${fmtN(e.total)}개 × 16바이트 = ${fmtB(e.mem.full)}. 가중치를 학습하려면 기울기와 Adam 상태까지 파라미터마다 따라다니기 때문이다. GPU 여러 장에 나눠 싣거나(분산 학습) LoRA를 쓴다.`));
    }
    if (on.length && e.mem.lora > gpuB) {
      v.push(callout('danger', `LoRA도 ${fmtB(e.mem.lora)}로 넘친다`, `학습 파라미터는 ${fmtPct(e.ratio)}뿐인데도 넘치는 이유는 <b>동결 가중치 ${fmtB(e.mem.frozen)}</b>가 메모리에 그대로 올라가기 때문이다. 정밀도를 bf16이나 int4(QLoRA)로 낮춘다.`));
    } else if (on.length && e.mem.full > gpuB) {
      v.push(callout('ok', `LoRA는 ${fmtB(e.mem.lora)}로 들어간다`, `학습 파라미터가 전체의 ${fmtPct(e.ratio)}라 기울기·Adam 상태가 ${fmtB(e.mem.loraTrain)}로 줄었다. 남은 대부분은 동결 가중치 ${fmtB(e.mem.frozen)}다.${s.prec === 'int4' ? ' QLoRA는 이 동결 가중치를 4비트로 줄였다.' : ''}`));
    }
    if (on.length && s.r <= 2 && !bad.length) {
      v.push(callout('more', `r = ${s.r}: ΔW는 방향 ${s.r}개의 조합만 표현한다`, '말투나 형식처럼 단순한 변화에는 충분할 수 있지만, 여러 가지를 한꺼번에 바꾸려면 부족할 수 있다(Challenge C5에서 rank 2 목표를 r = 1로 배워 본다). 보통 8~64에서 시작해 검증 손실을 보고 고른다.'));
    }
    if (!v.length) v.push(callout('ok', '모두 들어간다', `이 모델은 작아서 전체 미세조정도 ${fmtB(e.mem.full)}면 된다. LoRA는 큰 모델에서 의미가 커진다.`));
    $('[data-slot=verdict]').innerHTML = v.join('');
  }

  const on = (type, fn) => root.addEventListener(type, fn, { signal: ctrl.signal });
  on('input', (e) => {
    if (!e.target.matches('[data-in=r]')) return;
    s.r = Number(e.target.value);
    render();
  });
  on('change', (e) => {
    const t = e.target;
    if (t.matches('[data-in=preset]')) s.preset = t.value;
    else if (t.matches('[data-in=prec]')) s.prec = t.value;
    else if (t.matches('[data-in=gpu]')) s.gpu = Number(t.value);
    else if (t.matches('[data-target]')) s.targets = [...root.querySelectorAll('[data-target]:checked')].map((c) => c.dataset.target);
    else return;
    render();
  });
  on('click', (e) => {
    const act = e.target.closest('[data-act]')?.dataset.act;
    if (!act) return;
    if (act === 'r1') s.r = 1;
    if (act === 'mini-big') Object.assign(s, { preset: 'mini', r: 32, targets: ['q', 'k', 'v', 'o'], prec: 'fp32' });
    if (act === 'qlora') Object.assign(s, { preset: 'llama8b', r: 16, prec: 'int4', targets: ['q', 'k', 'v', 'o', 'mlp'] });
    if (act === 'reset') Object.assign(s, { ...DEFAULTS, targets: [...DEFAULTS.targets] });
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
  if (n >= 1e9) return `${(n / 1e9).toFixed(2)}B`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(n >= 1e8 ? 0 : 1)}M`;
  if (n >= 1e4) return `${(n / 1e3).toFixed(1)}K`;
  return n.toLocaleString();
}

export function fmtB(b) {
  if (b >= 1e9) return `${(b / 1e9).toFixed(b >= 1e11 ? 0 : 1)}GB`;
  if (b >= 1e6) return `${(b / 1e6).toFixed(1)}MB`;
  if (b >= 1e3) return `${(b / 1e3).toFixed(1)}KB`;
  return `${b}B`;
}

function fmtPct(x) {
  if (x === 0) return '0%';
  if (x < 0.001) return `${(x * 100).toFixed(3)}%`;
  if (x < 0.1) return `${(x * 100).toFixed(2)}%`;
  return `${(x * 100).toFixed(1)}%`;
}

const callout = (kind, title, html) =>
  `<div class="callout${kind === 'ok' ? ' callout--ok' : kind === 'danger' ? ' callout--danger' : ' callout--more'}"><span class="callout__title">${esc(title)}</span><p>${html}</p></div>`;

const stat = (label, value) => `<div class="stat"><span class="stat__label">${label}</span><span class="stat__value">${value}</span></div>`;

function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
}
