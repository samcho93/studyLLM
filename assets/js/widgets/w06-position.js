// w06 위치 정보가 없으면
// One concept: self-attention by itself does not know token order. Shuffle the
// tokens (keeping the last one in place) and the last token's output is exactly
// the same — unless a position vector is added to each token embedding.
// A single causal attention layer (1 head, D = 16, seeded weights) in plain math.

import { registerOutput, unregisterOutput } from '../site/result.js';
import { mulberry32, gaussian } from '../core/rng.js';

export const D = 16;
const HEAT_T = 48; // our mini GPT blockSize
const HEAT_D = 48; // our mini GPT nEmbd

const SENTENCES = ['5-3=', '어텐션은 순서를 모른다', '실습실은 평일 개방', '모델이 글을 쓴다'];
const POS_MODES = [
  { id: 'learned', label: '학습형 위치 임베딩 wpe (학습 전 무작위 값)' },
  { id: 'sin', label: '사인파 위치 인코딩 (원조 트랜스포머)' },
];

/** Seeded token embedding: the same character always gets the same vector. */
export function tokenEmbedding(ch) {
  const rand = mulberry32((ch.codePointAt(0) * 2654435761) >>> 0);
  return Array.from({ length: D }, () => gaussian(rand));
}

/** Seeded projection matrices W_q, W_k, W_v [D][D] with std 1/√D. */
export function makeWeights(seed = 6) {
  const rand = mulberry32(seed);
  const mat = () => Array.from({ length: D }, () => Array.from({ length: D }, () => gaussian(rand) / Math.sqrt(D)));
  return { Wq: mat(), Wk: mat(), Wv: mat() };
}

/** Sinusoidal encoding (Vaswani et al. 2017): PE[pos][2i] = sin(pos/10000^(2i/d)), PE[pos][2i+1] = cos(…). */
export function sinusoid(T, d) {
  return Array.from({ length: T }, (_, pos) =>
    Array.from({ length: d }, (_, j) => {
      const angle = pos / Math.pow(10000, (2 * Math.floor(j / 2)) / d);
      return j % 2 === 0 ? Math.sin(angle) : Math.cos(angle);
    }),
  );
}

/** A learned position table before training: just random numbers, one row per position. */
export function learnedPositions(T, seed = 11) {
  const rand = mulberry32(seed);
  return Array.from({ length: T }, () => Array.from({ length: D }, () => gaussian(rand)));
}

/**
 * Permutation of every position except the last one. perm[newPos] = oldPos.
 * seed 0 reverses the tokens before the last; any other seed shuffles them.
 */
export function shuffleKeepLast(n, seed) {
  const identity = Array.from({ length: n }, (_, i) => i);
  if (n < 3) return identity;
  if (seed === 0) return [...identity.slice(0, n - 1).reverse(), n - 1];
  const rand = mulberry32(seed);
  for (let tries = 0; tries < 20; tries++) {
    const p = identity.slice(0, n - 1);
    for (let i = p.length - 1; i > 0; i--) {
      const j = Math.floor(rand() * (i + 1));
      [p[i], p[j]] = [p[j], p[i]];
    }
    p.push(n - 1);
    if (p.some((v, i) => v !== i)) return p;
  }
  return identity;
}

/** Layer norm without γ/β (per token: mean 0, variance 1), as at the start of a block. */
export function norm(v) {
  const mean = v.reduce((a, b) => a + b, 0) / v.length;
  const vr = v.reduce((a, b) => a + (b - mean) ** 2, 0) / v.length;
  return v.map((x) => (x - mean) / Math.sqrt(vr + 1e-5));
}

const matVec = (v, W) => W[0].map((_, c) => v.reduce((s, x, r) => s + x * W[r][c], 0));

/**
 * Pre-norm causal self-attention (layer norm → q, k, v), output of the LAST token only.
 * @param {string[]} tokens
 * @param {number[][]|null} pos  position vectors (row per position) or null
 */
export function attendLast(tokens, pos, W) {
  const xs = tokens.map((ch, i) => {
    const e = tokenEmbedding(ch);
    return norm(pos ? e.map((v, j) => v + pos[i][j]) : e);
  });
  const n = xs.length;
  const q = matVec(xs[n - 1], W.Wq);
  const ks = xs.map((x) => matVec(x, W.Wk));
  const vs = xs.map((x) => matVec(x, W.Wv));
  const scores = ks.map((k) => k.reduce((s, kv, j) => s + kv * q[j], 0) / Math.sqrt(D));
  const max = Math.max(...scores);
  const exps = scores.map((s) => Math.exp(s - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  const weights = exps.map((e) => e / sum);
  const out = Array.from({ length: D }, (_, j) => vs.reduce((s, v, i) => s + weights[i] * v[j], 0));
  return { weights, out };
}

/** Everything the widget shows for one sentence / shuffle / position mode. */
export function compare(text, seed, mode) {
  const tokens = [...text];
  const perm = shuffleKeepLast(tokens.length, seed);
  const shuffled = perm.map((i) => tokens[i]);
  const W = makeWeights();
  const posTable = mode === 'sin' ? sinusoid(tokens.length, D) : learnedPositions(tokens.length);
  const run = (pos) => {
    const a = attendLast(tokens, pos, W);
    const b = attendLast(shuffled, pos, W);
    // weight each ORIGINAL token received, before and after shuffling
    const inv = new Array(tokens.length);
    perm.forEach((old, now) => (inv[old] = now));
    const rows = tokens.map((ch, i) => ({ ch, i, j: inv[i], wa: a.weights[i], wb: b.weights[inv[i]] }));
    const diff = Math.max(...a.out.map((v, j) => Math.abs(v - b.out[j])));
    return { outA: a.out, outB: b.out, rows, diff };
  };
  return { tokens, shuffled, perm, none: run(null), withPos: run(posTable) };
}

const template = document.createElement('template');
template.innerHTML = `
  <div class="widget w06-pos">
    <h3 class="widget__title">위치 정보가 없으면</h3>
    <div class="widget__controls">
      <label class="field">
        <span class="field__label">문장 (글자 = 토큰)</span>
        <select data-in="sentence"></select>
      </label>
      <label class="field">
        <span class="field__label">직접 입력 (3~12자)</span>
        <input type="text" data-in="custom" maxlength="12" spellcheck="false" placeholder="예: 나는 너를 본다">
      </label>
      <label class="field">
        <span class="field__label">더할 위치 정보</span>
        <select data-in="mode"></select>
      </label>
    </div>
    <div class="btn-row">
      <button type="button" class="btn small primary" data-act="reverse">↔ 앞 토큰 뒤집기</button>
      <button type="button" class="btn small" data-act="shuffle">🔀 무작위로 섞기</button>
      <span class="w06-muted w06-seed" data-out="seed"></span>
    </div>
    <p class="w06-note">층 정규화 → 어텐션 층 하나(헤드 1개, D = ${D}, 가중치는 seed 고정 무작위)에 원래 문장과 <b>마지막 글자만 두고 앞을 섞은</b> 문장을 넣는다. 인과 마스크 때문에 마지막 토큰은 앞의 모든 토큰을 본다. 그 마지막 토큰의 출력을 비교한다.</p>
    <div data-slot="out-inline"></div>
    <h4 class="w06-h4">사인파 위치 인코딩 <small class="w06-muted">— 행: 위치 0~${HEAT_T - 1} · 열: 차원 0~${HEAT_D - 1} · 남색 +1 · 황토색 −1</small></h4>
    <div class="w06-heat" data-slot="heat" role="img" aria-label="위치 0부터 ${HEAT_T - 1}까지 사인파 위치 인코딩 값의 히트맵. 앞쪽 차원은 위치마다 빠르게 바뀌고, 뒤쪽 차원은 천천히 바뀐다."></div>
    <p class="w06-note">앞쪽 열(차원)은 위치가 하나만 바뀌어도 값이 크게 바뀌고, 뒤쪽 열은 아주 천천히 바뀐다. 시계의 초침·분침·시침처럼 여러 속도의 파동을 겹쳐 모든 위치에 서로 다른 무늬를 준다. GPT-2와 우리 미니 GPT는 이 식 대신 위치마다 벡터 한 줄(<code>wpe</code>)을 두고 <b>학습</b>으로 채운다.</p>
  </div>`;

const outTemplate = document.createElement('template');
outTemplate.innerHTML = `
  <div class="w06-pos-out">
    <div class="w06-seqs" data-slot="seqs"></div>
    <div data-slot="verdict" aria-live="polite"></div>
    <div class="w06-panels">
      <section class="w06-panel" data-slot="none"></section>
      <section class="w06-panel" data-slot="with"></section>
    </div>
  </div>`;

const state = new WeakMap();
let seq = 0;

/**
 * @param {HTMLElement} el
 * @param {{ sentence?: string, mode?: 'learned'|'sin', outputTitle?: string }} options
 */
export function mount(el, options = {}) {
  const root = template.content.firstElementChild.cloneNode(true);
  const out = outTemplate.content.firstElementChild.cloneNode(true);
  el.replaceChildren(root);
  const $ = (sel) => root.querySelector(sel) ?? out.querySelector(sel);
  const ctrl = new AbortController();
  const outputId = `w06-position:${++seq}`;
  state.set(el, { ctrl, outputId });

  const s = { text: options.sentence ?? SENTENCES[0], mode: options.mode ?? 'learned', seed: 0 };

  $('[data-in=sentence]').innerHTML = SENTENCES.map((t) => `<option>${esc(t)}</option>`).join('');
  $('[data-in=sentence]').value = SENTENCES.includes(s.text) ? s.text : SENTENCES[0];
  $('[data-in=mode]').innerHTML = POS_MODES.map((m) => `<option value="${m.id}">${esc(m.label)}</option>`).join('');
  $('[data-in=mode]').value = s.mode;

  registerOutput(outputId, { title: options.outputTitle ?? '위치 정보 실험 결과', node: out, inlineHost: $('[data-slot=out-inline]') });

  function render() {
    const tokens = [...s.text];
    $('[data-out=seed]').textContent = s.seed === 0 ? '방법: 뒤집기' : `방법: 무작위 seed ${s.seed}`;
    if (tokens.length < 3) {
      $('[data-slot=seqs]').innerHTML = '';
      $('[data-slot=verdict]').innerHTML = `<div class="widget__error">글자가 3개 이상이어야 앞 토큰을 섞을 수 있다.</div>`;
      $('[data-slot=none]').innerHTML = '';
      $('[data-slot=with]').innerHTML = '';
      return;
    }
    const r = compare(s.text, s.seed, s.mode);
    const modeLabel = s.mode === 'sin' ? '사인파' : '학습형 wpe';
    $('[data-slot=seqs]').innerHTML = `
      <div class="w06-seq"><span class="lab">원래</span>${chips(r.tokens, null)}</div>
      <div class="w06-seq"><span class="lab">섞음</span>${chips(r.shuffled, r.perm)}</div>`;
    $('[data-slot=verdict]').innerHTML = verdict(r, modeLabel, sameMultiset(r));
    $('[data-slot=none]').innerHTML = panel('위치 정보 없음', 'x = 토큰 임베딩', r.none, false);
    $('[data-slot=with]').innerHTML = panel(`위치 정보 있음 · ${modeLabel}`, 'x = 토큰 임베딩 + 위치 벡터', r.withPos, true);
  }

  const on = (sel, type, fn) => $(sel).addEventListener(type, fn, { signal: ctrl.signal });
  on('[data-in=sentence]', 'change', (e) => {
    s.text = e.target.value;
    $('[data-in=custom]').value = '';
    s.seed = 0;
    render();
  });
  on('[data-in=custom]', 'input', (e) => {
    const v = e.target.value.trim();
    s.text = v || $('[data-in=sentence]').value;
    render();
  });
  on('[data-in=mode]', 'change', (e) => {
    s.mode = e.target.value;
    render();
  });
  on('[data-act=shuffle]', 'click', () => {
    s.seed++;
    render();
  });
  on('[data-act=reverse]', 'click', () => {
    s.seed = 0;
    render();
  });

  try {
    $('[data-slot=heat]').innerHTML = heatmap(sinusoid(HEAT_T, HEAT_D));
    render();
  } catch (err) {
    $('[data-slot=verdict]').innerHTML = `<div class="widget__error">계산하지 못했다 (${esc(err.message)}).</div>`;
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

function chips(tokens, perm) {
  return tokens
    .map((ch, i) => {
      const moved = perm && perm[i] !== i;
      const last = i === tokens.length - 1;
      return `<span class="w06-tok${last ? ' last' : ''}${moved ? ' moved' : ''}" title="${esc(perm ? `원래 ${perm[i]}번 → 지금 ${i}번` : `${i}번`)}">${esc(ch === ' ' ? '␣' : ch)}</span>`;
    })
    .join('');
}

function sameMultiset(r) {
  return r.tokens.join('') === r.shuffled.join('');
}

function fmtDiff(d) {
  if (d < 1e-12) return d === 0 ? '0' : `${d.toExponential(0)} ≈ 0`;
  return d.toFixed(3);
}

function panel(title, sub, res, withPos) {
  const rows = res.rows
    .map(
      (r) => `<tr><td class="t">${esc(r.ch === ' ' ? '␣' : r.ch)}</td><td class="m">${r.i}→${r.j}</td>
        <td><span class="w06-wbar"><i style="width:${(r.wa * 100).toFixed(1)}%"></i></span><span class="m">${r.wa.toFixed(3)}</span></td>
        <td><span class="w06-wbar b"><i style="width:${(r.wb * 100).toFixed(1)}%"></i></span><span class="m">${r.wb.toFixed(3)}</span></td></tr>`,
    )
    .join('');
  const same = res.diff < 1e-9;
  return `
    <h4>${esc(title)}</h4>
    <p class="w06-muted w06-psub">${esc(sub)}</p>
    <div class="w06-vec"><span class="lab">원래 출력</span>${strip(res.outA)}</div>
    <div class="w06-vec"><span class="lab">섞은 출력</span>${strip(res.outB)}</div>
    <p class="w06-diff ${same ? 'same' : 'diff'}">최대 차이 |Δ| = <b>${fmtDiff(res.diff)}</b> ${same ? '→ 두 문장을 구별하지 못한다' : '→ 순서가 출력에 반영된다'}</p>
    <details${withPos ? '' : ' open'}>
      <summary>마지막 토큰이 각 토큰에 준 어텐션 가중치</summary>
      <div class="w06-table-wrap"><table class="w06-wt">
        <thead><tr><th scope="col">토큰</th><th scope="col">위치</th><th scope="col">원래</th><th scope="col">섞음</th></tr></thead>
        <tbody>${rows}</tbody>
      </table></div>
    </details>`;
}

function strip(vec) {
  return `<span class="w06-strip">${vec
    .map((v) => {
      const a = Math.min(1, Math.abs(v) / 1.5);
      const color = v >= 0 ? 'var(--color-accent)' : 'var(--color-accent2)';
      return `<i style="background:color-mix(in srgb, ${color} ${Math.round(a * 85)}%, var(--color-surface))" title="${v.toFixed(3)}"></i>`;
    })
    .join('')}</span>`;
}

function verdict(r, modeLabel, identical) {
  if (identical) {
    return `<div class="callout"><span class="callout__title">섞어도 순서가 그대로다</span><p>같은 글자끼리 자리만 바뀌어 문장이 같다. 🔀를 한 번 더 누른다.</p></div>`;
  }
  const noneSame = r.none.diff < 1e-9;
  const pair = `“${esc(r.tokens.join(''))}” ↔ “${esc(r.shuffled.join(''))}”`;
  return `<div class="callout ${noneSame ? 'callout--danger' : ''}"><span class="callout__title">${noneSame ? `✗ 위치 정보가 없으면 두 입력이 똑같아 보인다 · ${pair}` : '확인'}</span><p>위치 정보 없이 계산한 마지막 토큰의 출력 차이: ${fmtDiff(r.none.diff)}. 어텐션 가중치는 토큰을 <b>따라다닐</b> 뿐 자리와 상관이 없다. 위치 정보(${esc(modeLabel)})를 더하면 차이가 ${fmtDiff(r.withPos.diff)}로 커져, 모델이 순서를 구별할 수 있게 된다.</p></div>`;
}

function heatmap(pe) {
  const cells = [];
  pe.forEach((row, pos) =>
    row.forEach((v, j) => {
      const color = v >= 0 ? 'var(--color-accent)' : 'var(--color-accent2)';
      cells.push(`<i style="background:color-mix(in srgb, ${color} ${Math.round(Math.abs(v) * 90)}%, var(--color-surface))" title="pos ${pos} · dim ${j} = ${v.toFixed(2)}"></i>`);
    }),
  );
  return `<div class="w06-heat__grid" style="--cols:${pe[0].length}">${cells.join('')}</div>`;
}

function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
}
