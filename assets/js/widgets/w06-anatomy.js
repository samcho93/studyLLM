// w06 트랜스포머 해부도
// One concept: a GPT is "embeddings + L identical blocks + head". Change the
// config and watch the tensor shape at every arrow and where the parameters go.
// The formulas mirror core/gpt.js exactly (createModel / paramCount), so the
// mini-GPT preset shows the same number the week-7 trainer prints.

import { registerOutput, unregisterOutput } from '../site/result.js';

const L_MAX = 48;
const D_LIST = [16, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1280, 1600, 2048, 4096];
const H_LIST = [1, 2, 3, 4, 5, 6, 8, 12, 16, 20, 25, 32];
const V_LIST = [65, 256, 495, 1000, 4096, 8192, 32000, 50257, 128256];
const T_LIST = [8, 16, 32, 48, 64, 128, 256, 512, 1024, 2048, 4096];

export const PRESETS = {
  mini: { label: '이 과정의 미니 GPT', vocabSize: 495, blockSize: 48, nLayer: 2, nHead: 4, nEmbd: 48, tie: false },
  gpt2: { label: 'GPT-2 small (124M)', vocabSize: 50257, blockSize: 1024, nLayer: 12, nHead: 12, nEmbd: 768, tie: true },
  fail: { label: '큰 어휘 + 작은 모델 (실패 재현)', vocabSize: 50257, blockSize: 48, nLayer: 2, nHead: 4, nEmbd: 48, tie: false },
};

/** Parameter groups exactly as core/gpt.js creates them (tie: GPT-2 style shared head, no head bias). */
export function paramBreakdown({ vocabSize: V, blockSize: T, nLayer: L, nEmbd: D, tie = false }) {
  const perBlock = {
    ln: 4 * D, // ln1 γ,β + ln2 γ,β
    attnQkv: D * 3 * D + 3 * D,
    attnProj: D * D + D,
    mlpFc: D * 4 * D + 4 * D,
    mlpProj: 4 * D * D + D,
  };
  const block = Object.values(perBlock).reduce((a, b) => a + b, 0); // 12D² + 13D
  const groups = [
    { key: 'wte', label: '토큰 임베딩 wte', formula: 'V·D', n: V * D },
    { key: 'wpe', label: '위치 임베딩 wpe', formula: 'T·D', n: T * D },
    { key: 'attn', label: `어텐션 qkv + proj (×${L})`, formula: 'L·(4D² + 4D)', n: L * (perBlock.attnQkv + perBlock.attnProj) },
    { key: 'mlp', label: `피드포워드 fc + proj (×${L})`, formula: 'L·(8D² + 5D)', n: L * (perBlock.mlpFc + perBlock.mlpProj) },
    { key: 'ln', label: `층 정규화 (×${L} + 최종)`, formula: 'L·4D + 2D', n: L * perBlock.ln + 2 * D },
    { key: 'head', label: tie ? '출력 헤드 (wte 공유)' : '출력 헤드 W + b', formula: tie ? '0' : 'D·V + V', n: tie ? 0 : D * V + V },
  ];
  const total = groups.reduce((a, g) => a + g.n, 0);
  return { perBlock, block, groups, total, approx: 12 * L * D * D };
}

/** The arrows of the diagram for batch B: [{ id, name, op, shape, params }]. */
export function shapeRows({ vocabSize: V, blockSize: T, nHead: H, nEmbd: D }, B = 1) {
  const hs = D / H;
  const hsTxt = Number.isInteger(hs) ? hs : hs.toFixed(2);
  return {
    pre: [
      { name: '입력 토큰 id', op: '글자 → 번호', shape: [B, T] },
      { name: '토큰 임베딩 + 위치 임베딩', op: 'wte[id] + wpe[0…T−1]', shape: [B, T, D] },
    ],
    attn: [
      { name: '층 정규화 ①', op: '토큰마다 평균 0 · 분산 1', shape: [B, T, D] },
      { name: 'qkv 선형', op: 'x · W_qkv + b', shape: [B, T, 3 * D] },
      { name: '헤드로 나누기', op: `Q · K · V 각각`, shape: [B, H, T, hsTxt] },
      { name: '어텐션 점수', op: 'QKᵀ/√hs + 인과 마스크 → 소프트맥스', shape: [B, H, T, T], hot: true },
      { name: '가중합 · 헤드 합치기', op: '점수 · V → 이어 붙이기', shape: [B, T, D] },
      { name: '출력 투영', op: 'x · W_proj + b', shape: [B, T, D] },
    ],
    mlp: [
      { name: '층 정규화 ②', op: '토큰마다 평균 0 · 분산 1', shape: [B, T, D] },
      { name: '확장 fc', op: 'x · W_fc + b (4배)', shape: [B, T, 4 * D] },
      { name: 'GELU', op: '활성화 함수', shape: [B, T, 4 * D] },
      { name: '축소 proj', op: 'x · W_proj2 + b', shape: [B, T, D] },
    ],
    post: [
      { name: '최종 층 정규화', op: 'lnf', shape: [B, T, D] },
      { name: '출력 헤드 → 로짓', op: 'x · W_head + b', shape: [B, T, V] },
      { name: '다음 토큰 확률', op: '마지막 위치만 소프트맥스', shape: [B, V] },
    ],
  };
}

/** Memory estimates in bytes. */
export function memory(cfg, total) {
  const { nLayer: L, nHead: H, blockSize: T } = cfg;
  return {
    fp32: total * 4,
    fp16: total * 2,
    train: total * 16, // weights + grads + AdamW m, v (fp32)
    attn: L * H * T * T * 4, // attention maps kept for backward, B = 1
  };
}

const COLORS = { wte: 'var(--color-accent)', wpe: 'var(--color-more)', attn: 'var(--color-success)', mlp: 'var(--color-accent2)', ln: 'var(--color-border-strong)', head: 'var(--color-danger)' };

const template = document.createElement('template');
template.innerHTML = `
  <div class="widget w06-an">
    <h3 class="widget__title">트랜스포머 해부도</h3>
    <div class="btn-row w06-presets" role="group" aria-label="설정 불러오기">
      <button type="button" class="btn small" data-preset="mini">🐣 이 과정의 미니 GPT</button>
      <button type="button" class="btn small" data-preset="gpt2">🏛 GPT-2 small</button>
      <button type="button" class="btn small ghost" data-preset="fail">⚠ 큰 어휘 + 작은 모델</button>
    </div>
    <div class="widget__controls">
      <label class="field">
        <span class="field__label">블록 수 L (nLayer) <output data-out="L"></output></span>
        <input type="range" data-in="L" min="1" max="${L_MAX}" step="1">
      </label>
      <label class="field">
        <span class="field__label">임베딩 차원 D (nEmbd) <output data-out="D"></output></span>
        <input type="range" data-in="D" min="0" max="${D_LIST.length - 1}" step="1">
      </label>
      <label class="field">
        <span class="field__label">헤드 수 H (nHead) <output data-out="H"></output></span>
        <input type="range" data-in="H" min="0" max="${H_LIST.length - 1}" step="1">
      </label>
      <label class="field">
        <span class="field__label">어휘 크기 V (vocabSize) <output data-out="V"></output></span>
        <input type="range" data-in="V" min="0" max="${V_LIST.length - 1}" step="1">
      </label>
      <label class="field">
        <span class="field__label">문맥 길이 T (blockSize) <output data-out="T"></output></span>
        <input type="range" data-in="T" min="0" max="${T_LIST.length - 1}" step="1">
      </label>
      <label class="field w06-check">
        <span><input type="checkbox" data-in="tie"> 출력 헤드가 토큰 임베딩을 공유 (GPT-2 방식 · 헤드 편향 없음)</span>
      </label>
    </div>
    <p class="w06-note">화살표 위의 <code>[…]</code>는 그 지점을 지나는 텐서의 모양이다 (배치 B = 1). 오른쪽 숫자는 그 층의 파라미터 수다.</p>
    <div class="w06-diagram" data-slot="diagram" aria-live="polite"></div>
    <div data-slot="out-inline"></div>
  </div>`;

const outTemplate = document.createElement('template');
outTemplate.innerHTML = `
  <div class="w06-an-out">
    <div data-slot="verdict" aria-live="polite"></div>
    <div class="stat-row" data-slot="stats"></div>
    <h4>파라미터는 어디에 있나</h4>
    <div class="w06-stack" data-slot="stack" role="img"></div>
    <div class="legend" data-slot="legend"></div>
    <div class="w06-table-wrap"><table class="w06-table" data-slot="table"></table></div>
    <h4>메모리 어림 <small class="w06-muted">— 파라미터 수 × 바이트</small></h4>
    <div class="stat-row" data-slot="mem"></div>
    <details class="w06-modern">
      <summary>요즘 LLM(Llama 등)은 무엇이 다른가</summary>
      <p>이 위젯의 공식은 GPT-2 구조(= <code>core/gpt.js</code>) 기준이다. 최근 모델은 뼈대는 같고 부품이 조금 다르다. 아래는 <b>대략적인 차이</b>이며 정확한 수는 각 모델의 설정 파일로 따로 계산해야 한다.</p>
      <ul>
        <li><b>RMSNorm</b>: 층 정규화에서 평균 빼기와 β를 없앤다 → 정규화 파라미터가 D개로 준다.</li>
        <li><b>SwiGLU</b>: 피드포워드를 행렬 3개(게이트 포함)로 만들고 은닉 크기를 약 8/3·D로 줄여 파라미터 수를 비슷하게 맞춘다.</li>
        <li><b>RoPE</b>: 위치 임베딩 표(wpe)를 더하지 않고 Q·K를 위치만큼 회전시킨다 → wpe 파라미터 0.</li>
        <li><b>GQA</b>: K·V 헤드를 Q 헤드보다 적게 두어 qkv 행렬과 추론 캐시를 줄인다.</li>
      </ul>
      <p class="w06-muted">예: Llama 3 8B의 공개 설정(V 128,256 · D 4,096 · L 32 · K·V 헤드 8 · 피드포워드 14,336)으로 세면 약 80억 개다. 이 위젯에 같은 L·D를 넣으면 다른 값이 나온다.</p>
    </details>
  </div>`;

const state = new WeakMap();
let seq = 0;

/**
 * @param {HTMLElement} el
 * @param {{ preset?: keyof PRESETS, outputTitle?: string }} options
 */
export function mount(el, options = {}) {
  const root = template.content.firstElementChild.cloneNode(true);
  const out = outTemplate.content.firstElementChild.cloneNode(true);
  el.replaceChildren(root);
  const $ = (sel) => root.querySelector(sel) ?? out.querySelector(sel);
  const ctrl = new AbortController();
  const outputId = `w06-anatomy:${++seq}`;
  state.set(el, { ctrl, outputId });

  const s = { ...PRESETS[options.preset ?? 'mini'], preset: options.preset ?? 'mini' };

  registerOutput(outputId, { title: options.outputTitle ?? '트랜스포머 해부도 결과', node: out, inlineHost: $('[data-slot=out-inline]') });

  function syncInputs() {
    $('[data-in=L]').value = String(s.nLayer);
    $('[data-in=D]').value = String(nearest(D_LIST, s.nEmbd));
    $('[data-in=H]').value = String(nearest(H_LIST, s.nHead));
    $('[data-in=V]').value = String(nearest(V_LIST, s.vocabSize));
    $('[data-in=T]').value = String(nearest(T_LIST, s.blockSize));
    $('[data-in=tie]').checked = s.tie;
    root.querySelectorAll('[data-preset]').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.preset === s.preset)));
  }

  function render() {
    const cfg = { vocabSize: s.vocabSize, blockSize: s.blockSize, nLayer: s.nLayer, nHead: s.nHead, nEmbd: s.nEmbd, tie: s.tie };
    const D = s.nEmbd;
    const divisible = D % s.nHead === 0;
    $('[data-out=L]').textContent = String(s.nLayer);
    $('[data-out=D]').textContent = String(D);
    $('[data-out=H]').textContent = `${s.nHead}${divisible ? ` · hs ${D / s.nHead}` : ' · ✗'}`;
    $('[data-out=V]').textContent = s.vocabSize.toLocaleString();
    $('[data-out=T]').textContent = s.blockSize.toLocaleString();

    const pb = paramBreakdown(cfg);
    const rows = shapeRows(cfg, 1);
    $('[data-slot=diagram]').innerHTML = diagram(rows, pb, cfg, divisible);

    // verdict
    const embShare = (pb.groups[0].n + pb.groups[1].n + pb.groups[5].n) / pb.total;
    const mem = memory(cfg, pb.total);
    $('[data-slot=verdict]').innerHTML = verdict({ cfg, pb, embShare, divisible, mem, preset: s.preset });

    // stats
    const blocksTotal = pb.block * s.nLayer;
    $('[data-slot=stats]').innerHTML = [
      stat('총 파라미터', fmtBig(pb.total)),
      stat('블록 하나', `${pb.block.toLocaleString()}`),
      stat('12·D² 근사', `${(12 * D * D).toLocaleString()} (${pct(12 * D * D / pb.block)})`),
      stat('블록 전체 비중', pct(blocksTotal / pb.total)),
      stat('임베딩 + 헤드 비중', pct(embShare)),
    ].join('');

    // stacked bar + legend + table
    $('[data-slot=stack]').setAttribute('aria-label', pb.groups.map((g) => `${g.label} ${pct(g.n / pb.total)}`).join(', '));
    $('[data-slot=stack]').innerHTML = pb.groups
      .filter((g) => g.n > 0)
      .map((g) => `<i style="flex-grow:${g.n};background:${COLORS[g.key]}" title="${esc(`${g.label}: ${g.n.toLocaleString()} (${pct(g.n / pb.total)})`)}"></i>`)
      .join('');
    $('[data-slot=legend]').innerHTML = pb.groups.map((g) => `<span><i style="background:${COLORS[g.key]}"></i>${esc(g.label.replace(/ \(.*\)$/, ''))}</span>`).join('');
    $('[data-slot=table]').innerHTML =
      `<thead><tr><th scope="col">부분</th><th scope="col">공식</th><th scope="col">개수</th><th scope="col">비중</th></tr></thead><tbody>` +
      pb.groups
        .map((g) => `<tr><td><i class="w06-sw" style="background:${COLORS[g.key]}"></i>${esc(g.label)}</td><td class="m">${g.formula}</td><td class="m">${g.n.toLocaleString()}</td><td class="m">${pct(g.n / pb.total)}</td></tr>`)
        .join('') +
      `<tr class="sum"><td>합계</td><td class="m">—</td><td class="m">${pb.total.toLocaleString()}</td><td class="m">100%</td></tr></tbody>`;

    $('[data-slot=mem]').innerHTML = [
      stat('가중치 fp32 (4B)', fmtBytes(mem.fp32)),
      stat('가중치 fp16 (2B)', fmtBytes(mem.fp16)),
      stat('학습 AdamW (16B)', fmtBytes(mem.train)),
      stat('어텐션 점수 L·H·T²', fmtBytes(mem.attn)),
    ].join('');
  }

  const custom = () => {
    s.preset = 'custom';
    syncInputs();
    render();
  };
  const on = (sel, type, fn) => $(sel).addEventListener(type, fn, { signal: ctrl.signal });
  on('[data-in=L]', 'input', (e) => ((s.nLayer = Number(e.target.value)), custom()));
  on('[data-in=D]', 'input', (e) => ((s.nEmbd = D_LIST[Number(e.target.value)]), custom()));
  on('[data-in=H]', 'input', (e) => ((s.nHead = H_LIST[Number(e.target.value)]), custom()));
  on('[data-in=V]', 'input', (e) => ((s.vocabSize = V_LIST[Number(e.target.value)]), custom()));
  on('[data-in=T]', 'input', (e) => ((s.blockSize = T_LIST[Number(e.target.value)]), custom()));
  on('[data-in=tie]', 'change', (e) => ((s.tie = e.target.checked), custom()));
  root.addEventListener(
    'click',
    (e) => {
      const key = e.target.closest('[data-preset]')?.dataset.preset;
      if (!key) return;
      Object.assign(s, PRESETS[key], { preset: key });
      syncInputs();
      render();
    },
    { signal: ctrl.signal },
  );

  syncInputs();
  try {
    render();
  } catch (err) {
    $('[data-slot=diagram]').innerHTML = `<div class="widget__error">해부도를 그리지 못했다 (${esc(err.message)}).</div>`;
  }
}

export function unmount(el) {
  const st = state.get(el);
  if (!st) return;
  st.ctrl.abort();
  unregisterOutput(st.outputId);
  state.delete(el);
  el.replaceChildren();
}

// ---------------------------------------------------------------- rendering

function diagram(rows, pb, cfg, divisible) {
  const { nLayer: L, nEmbd: D, vocabSize: V, blockSize: T, tie } = cfg;
  const pbk = pb.perBlock;
  const params = {
    '토큰 임베딩 + 위치 임베딩': `${(V * D + T * D).toLocaleString()}`,
    '층 정규화 ①': (2 * D).toLocaleString(),
    'qkv 선형': pbk.attnQkv.toLocaleString(),
    '출력 투영': pbk.attnProj.toLocaleString(),
    '층 정규화 ②': (2 * D).toLocaleString(),
    '확장 fc': pbk.mlpFc.toLocaleString(),
    '축소 proj': pbk.mlpProj.toLocaleString(),
    '최종 층 정규화': (2 * D).toLocaleString(),
    '출력 헤드 → 로짓': tie ? '0 (공유)' : (D * V + V).toLocaleString(),
  };
  const node = (r, extra = '') => {
    const bad = !divisible && r.name === '헤드로 나누기';
    return `<div class="w06-node${r.hot ? ' hot' : ''}${bad ? ' bad' : ''}${extra}">
        <span class="nm">${esc(r.name)}</span><span class="op">${esc(r.op)}</span>${params[r.name] ? `<span class="pc">${params[r.name]}</span>` : ''}
      </div>`;
  };
  const edge = (shape, note = '') => `<div class="w06-edge"><code>[${shape.map((v) => (typeof v === 'number' ? v.toLocaleString() : v)).join(', ')}]</code>${note ? `<span>${note}</span>` : ''}</div>`;
  const seq = (list) => list.map((r) => node(r) + edge(r.shape, r.hot ? `T² = ${(T * T).toLocaleString()}칸 × 헤드 ${cfg.nHead}` : '')).join('');
  const add = (label, withEdge) => `<div class="w06-node add"><span class="nm">⊕ 잔차 연결</span><span class="op">${label}</span></div>${withEdge ? edge([1, T, D]) : ''}`;
  return `
    ${seq(rows.pre)}
    <div class="w06-block">
      <div class="w06-block__tag">트랜스포머 블록 × ${L} <small>(블록 하나 ${pb.block.toLocaleString()}개)</small></div>
      <div class="w06-sub"><span class="w06-sub__tag">어텐션 부분</span>${seq(rows.attn)}</div>
      ${add('x ← x + 어텐션(층 정규화(x))', true)}
      <div class="w06-sub"><span class="w06-sub__tag">피드포워드 부분</span>${seq(rows.mlp)}</div>
      ${add('x ← x + 피드포워드(층 정규화(x))', false)}
    </div>
    ${edge([1, T, D], '다음 블록으로, 마지막 블록이면 아래로')}
    ${rows.post.map((r, i) => node(r) + (i < rows.post.length - 1 ? edge(r.shape) : edge(r.shape, '← 이 V개 점수가 다음 토큰 확률'))).join('')}
    ${divisible ? '' : `<div class="widget__error">임베딩 차원(D = ${D})은 헤드 수(H = ${cfg.nHead})로 나누어떨어지지 않는다. 헤드 하나의 크기 hs = D/H가 정수가 아니라서 <code>createModel</code>이 오류를 낸다.</div>`}`;
}

function verdict({ cfg, pb, embShare, divisible, mem, preset }) {
  const { nEmbd: D, nHead: H, vocabSize: V, blockSize: T, nLayer: L } = cfg;
  const share = (key) => pb.groups.find((g) => g.key === key).n / pb.total;
  const vocabShare = share('wte') + share('head');
  if (!divisible) {
    return `<div class="callout callout--danger"><span class="callout__title">✗ 헤드로 나눌 수 없다</span><p>임베딩 차원(D = ${D})을 헤드 ${H}개로 똑같이 나눌 수 없다. 그런데 파라미터 수는 H와 상관없이 그대로다. 헤드는 qkv 행렬을 <b>나눠 쓰는</b> 방식일 뿐, 새 가중치를 만들지 않는다.</p></div>`;
  }
  if (preset === 'gpt2') {
    const untied = pb.total + D * V + V; // our head adds D·V weights + V bias (GPT-2 has neither)
    return `<div class="callout callout--ok"><span class="callout__title">GPT-2 small = ${pb.total.toLocaleString()}개</span><p>공개된 GPT-2 small(124M)과 같은 수다. GPT-2는 출력 헤드가 토큰 임베딩 행렬을 그대로 다시 쓴다(가중치 공유). 우리 <code>core/gpt.js</code>는 헤드를 따로 두므로, 공유를 끄면 ${untied.toLocaleString()}개가 된다.</p></div>`;
  }
  if (mem.attn > 256e6) {
    return `<div class="callout callout--danger"><span class="callout__title">⚠ 문맥 길이의 제곱으로 폭발</span><p>어텐션 점수는 블록마다 헤드마다 T × T 표다. T = ${T.toLocaleString()}이면 순전파 한 번(B = 1)에 ${L}·${H}·T² = ${fmtBytes(mem.attn)}를 저장해야 역전파할 수 있다. 파라미터는 wpe만큼만 늘었는데 메모리는 T²로 커진다.</p></div>`;
  }
  if (vocabShare >= 0.6) {
    return `<div class="callout callout--danger"><span class="callout__title">⚠ 모델이 아니라 어휘 표다 · 임베딩+헤드 ${pct(vocabShare)}</span><p>어휘(V = ${V.toLocaleString()})에 비해 임베딩 차원(D = ${D})이 작아 파라미터 대부분이 “토큰 → 벡터” 표와 “벡터 → 토큰” 표에 있다. 실제로 문맥을 섞고 계산하는 블록은 ${pct((pb.block * L) / pb.total)}뿐이다. 작은 모델에 큰 어휘를 쓰면 이렇게 된다.</p></div>`;
  }
  if (share('wpe') >= 0.3) {
    return `<div class="callout"><span class="callout__title">위치 임베딩 표가 커졌다 · wpe ${pct(share('wpe'))}</span><p>학습형 위치 임베딩은 위치마다 D차원 한 줄이라 T·D개다. T를 늘리면 이 표가 블록보다 커질 수 있다. 사인파 인코딩이나 RoPE는 이 표가 아예 없다.</p></div>`;
  }
  if (embShare >= 0.35) {
    return `<div class="callout"><span class="callout__title">작은 모델은 절반 가까이가 임베딩 · ${pct(embShare)}</span><p>블록 하나는 약 12·D²개인데 임베딩과 헤드는 V·D에 비례한다. D가 작으면 D²보다 V·D가 크게 남는다. L이나 D를 키워 보면 블록 비중이 금방 커진다.</p></div>`;
  }
  return `<div class="callout callout--ok"><span class="callout__title">블록이 대부분 · 12·L·D² 근사 ${pct(pb.approx / pb.total)}</span><p>모델이 커지면 파라미터 대부분이 블록 안의 행렬(어텐션 4D², 피드포워드 8D²)에 있다. 그래서 “블록 하나 ≈ 12·D²”라는 어림셈이 잘 맞는다.</p></div>`;
}

// ---------------------------------------------------------------- helpers

const stat = (label, value) => `<div class="stat"><span class="stat__label">${label}</span><span class="stat__value">${value}</span></div>`;

function nearest(list, v) {
  let best = 0;
  list.forEach((x, i) => {
    if (Math.abs(x - v) < Math.abs(list[best] - v)) best = i;
  });
  return best;
}

function pct(x) {
  return `${(x * 100).toFixed(x < 0.1 ? 1 : 0)}%`;
}

export function fmtBig(n) {
  if (n >= 1e9) return `${(n / 1e9).toFixed(2)}B`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
  return n.toLocaleString();
}

export function fmtBytes(b) {
  const u = ['B', 'KB', 'MB', 'GB', 'TB'];
  let i = 0;
  while (b >= 1000 && i < u.length - 1) {
    b /= 1000;
    i++;
  }
  return `${b.toFixed(b < 10 && i > 0 ? 1 : 0)} ${u[i]}`;
}

function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
}
