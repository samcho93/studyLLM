// w04 소프트맥스 · 교차 엔트로피 탐색기
// One concept: a model outputs scores (logits); softmax turns them into
// probabilities; the loss is −log p(correct); its gradient is p − onehot.
// Failure shown: a huge logit on a wrong answer → overconfident, huge loss.

import { registerOutput, unregisterOutput } from '../site/result.js';

// "실" 다음에 코퍼스에서 실제로 나온 글자 상위 6개와 그 횟수 (1주차 C2)
const CANDS = [
  { ch: '습', count: 7 },
  { ch: '수', count: 3 },
  { ch: '제', count: 2 },
  { ch: '은', count: 2 },
  { ch: '에', count: 2 },
  { ch: ' ', count: 1 },
];
const ZMIN = -4;
const ZMAX = 12;
const START = [0.5, 1.5, 0.2, -0.3, 0.8, 0]; // an "untrained" guess: 수 is favoured, 정답 습 is not
const LN_K = Math.log(CANDS.length);

const template = document.createElement('template');
template.innerHTML = `
  <div class="widget w04-sm">
    <h3 class="widget__title">소프트맥스 · 교차 엔트로피 탐색기</h3>
    <p class="w04-ctx">앞 글자 <b>실</b> → 다음 글자 후보 6개의 <b>로짓</b>(모델이 낸 점수)을 직접 움직인다</p>
    <div class="w04-logits" data-slot="sliders"></div>
    <div class="widget__controls">
      <label class="field">
        <span class="field__label">정답(실제로 온 다음 글자)</span>
        <select data-in="target"></select>
      </label>
      <label class="field">
        <span class="field__label">온도 T <output data-out="T"></output></span>
        <input type="range" data-in="T" min="-1.3" max="0.7" step="0.01" aria-label="온도 (로그 눈금)">
      </label>
    </div>
    <div class="btn-row">
      <button type="button" class="btn small primary" data-act="step">⬇ 경사 하강 1스텝 (η = 1)</button>
      <button type="button" class="btn small ghost" data-act="uniform">모두 0 (균등)</button>
      <button type="button" class="btn small ghost" data-act="counts">빈도표처럼 (로짓 = log 횟수)</button>
      <button type="button" class="btn small ghost" data-act="overconf">과신한 오답 (수 = 10)</button>
      <button type="button" class="btn small ghost" data-act="cold">T → 0 (argmax)</button>
      <button type="button" class="btn small ghost" data-act="reset">↺ 처음으로</button>
    </div>
    <div data-slot="out-inline"></div>
  </div>`;

const outTemplate = document.createElement('template');
outTemplate.innerHTML = `
  <div class="w04-out w04-sm-out">
    <div data-slot="verdict" aria-live="polite"></div>
    <div class="stat-row" data-slot="stats"></div>
    <p class="w04-formula" data-slot="formula"></p>
    <h4>후보별 확률과 경사 <small class="w04-muted">— 경사 ∂L/∂z = (p − onehot) / T</small></h4>
    <div class="w04-tbl" data-slot="rows"></div>
    <div class="legend" aria-hidden="true">
      <span><i class="w04-lg-p"></i>확률 p</span>
      <span><i class="w04-lg-t"></i>정답의 확률</span>
      <span><i class="w04-lg-up"></i>경사 &lt; 0 → 로짓을 올린다</span>
      <span><i class="w04-lg-down"></i>경사 &gt; 0 → 로짓을 내린다</span>
    </div>
    <h4>손실 곡선 L = −log p(정답) <small class="w04-muted">— 점: 지금 위치</small></h4>
    <div data-slot="curve"></div>
  </div>`;

const state = new WeakMap();
let seq = 0;

/** Stable softmax with temperature (plain numbers). */
export function softmaxT(z, T = 1) {
  const m = Math.max(...z);
  const e = z.map((x) => Math.exp((x - m) / T));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((x) => x / s);
}

/**
 * @param {HTMLElement} el
 * @param {{ outputTitle?: string }} options
 */
export function mount(el, options = {}) {
  const root = template.content.firstElementChild.cloneNode(true);
  const out = outTemplate.content.firstElementChild.cloneNode(true);
  el.replaceChildren(root);
  const $ = (sel) => root.querySelector(sel) ?? out.querySelector(sel);
  const ctrl = new AbortController();
  const outputId = `w04-softmax:${++seq}`;
  state.set(el, { ctrl, outputId });

  const s = { z: START.slice(), target: 0, logT: 0 };

  $('[data-slot=sliders]').innerHTML = CANDS.map(
    (c, i) => `
      <label class="w04-lrow">
        <span class="w04-lch">${esc(show(c.ch))}</span>
        <input type="range" data-z="${i}" min="${ZMIN}" max="${ZMAX}" step="0.1" aria-label="${esc(show(c.ch))}의 로짓">
        <output data-zout="${i}"></output>
      </label>`,
  ).join('');
  $('[data-in=target]').innerHTML = CANDS.map((c, i) => `<option value="${i}">${esc(show(c.ch))} (코퍼스에서 ${c.count}번)</option>`).join('');

  registerOutput(outputId, { title: options.outputTitle ?? '소프트맥스 · 교차 엔트로피 결과', node: out, inlineHost: $('[data-slot=out-inline]') });

  const T = () => Math.pow(10, s.logT);

  function syncInputs() {
    s.z.forEach((v, i) => {
      root.querySelector(`[data-z="${i}"]`).value = String(v);
    });
    $('[data-in=target]').value = String(s.target);
    $('[data-in=T]').value = String(s.logT);
  }

  function render() {
    const t = T();
    const p = softmaxT(s.z, t);
    const grad = p.map((q, i) => (q - (i === s.target ? 1 : 0)) / t);
    const pt = p[s.target];
    const loss = -Math.log(Math.max(pt, 1e-300));
    const top = p.indexOf(Math.max(...p));
    const entropy = -p.reduce((a, q) => a + (q > 0 ? q * Math.log(q) : 0), 0);

    s.z.forEach((v, i) => (root.querySelector(`[data-zout="${i}"]`).textContent = fmt(v, 1)));
    $('[data-out=T]').textContent = t < 0.1 ? t.toFixed(3) : t.toFixed(2);

    $('[data-slot=stats]').innerHTML = [
      stat('손실 L', Number.isFinite(loss) ? loss.toFixed(3) : '∞', loss > LN_K ? 'bad' : loss < 0.1 ? 'good' : ''),
      stat('p(정답)', pct(pt)),
      stat('균등 추측의 L', `ln 6 = ${LN_K.toFixed(3)}`),
      stat('엔트로피', `${entropy.toFixed(2)} nats`),
      stat('Σp', p.reduce((a, b) => a + b, 0).toFixed(3)),
    ].join('');

    $('[data-slot=formula]').innerHTML = `L = −log p(<b>${esc(show(CANDS[s.target].ch))}</b>) = −log ${pt < 1e-4 ? pt.toExponential(2) : pt.toFixed(4)} = <b>${Number.isFinite(loss) ? loss.toFixed(3) : '∞'}</b>`;

    const gmax = Math.max(1, ...grad.map(Math.abs));
    $('[data-slot=rows]').innerHTML =
      `<div class="w04-th"><span>후보</span><span>로짓 z</span><span>확률 p</span><span>경사 ∂L/∂z</span></div>` +
      CANDS.map((c, i) => {
        const g = grad[i];
        const w = (Math.abs(g) / gmax) * 50;
        const dir = Math.abs(g) < 5e-4 ? '·' : g < 0 ? '↑' : '↓';
        return `<div class="w04-tr${i === s.target ? ' is-target' : ''}">
          <span class="w04-c">${esc(show(c.ch))}${i === s.target ? '<small>정답</small>' : ''}</span>
          <span class="w04-z">${fmt(s.z[i], 1)}</span>
          <span class="w04-p"><span class="w04-bar"><i style="width:${(p[i] * 100).toFixed(1)}%"></i></span><em>${pct(p[i])}</em></span>
          <span class="w04-g"><span class="w04-gbar"><i class="${g < 0 ? 'up' : 'down'}" style="${g < 0 ? `right:50%` : `left:50%`};width:${w.toFixed(1)}%"></i></span><em>${dir} ${fmtSigned(g)}</em></span>
        </div>`;
      }).join('');

    $('[data-slot=curve]').innerHTML = lossCurve(pt, loss);
    $('[data-slot=verdict]').innerHTML = verdict({ p, pt, loss, top, t, target: s.target });
  }

  const on = (target, type, fn) => target.addEventListener(type, fn, { signal: ctrl.signal });
  on(root, 'input', (e) => {
    const i = e.target.dataset.z;
    if (i !== undefined) s.z[Number(i)] = Number(e.target.value);
    else if (e.target.matches('[data-in=T]')) s.logT = Number(e.target.value);
    else return;
    render();
  });
  on($('[data-in=target]'), 'change', (e) => {
    s.target = Number(e.target.value);
    render();
  });
  on(root, 'click', (e) => {
    const act = e.target.closest('[data-act]')?.dataset.act;
    if (!act) return;
    if (act === 'step') {
      const t = T();
      const p = softmaxT(s.z, t);
      s.z = s.z.map((v, i) => clamp(v - (p[i] - (i === s.target ? 1 : 0)) / t, ZMIN, ZMAX));
    }
    if (act === 'uniform') s.z = s.z.map(() => 0);
    if (act === 'counts') {
      s.z = CANDS.map((c) => Math.log(c.count)); // exact: softmax(log count) = count / total
      s.logT = 0;
    }
    if (act === 'overconf') {
      s.z = START.slice();
      s.z[1] = 10;
      s.target = 0;
      s.logT = 0;
    }
    if (act === 'cold') s.logT = -1.3;
    if (act === 'reset') {
      s.z = START.slice();
      s.target = 0;
      s.logT = 0;
    }
    syncInputs();
    render();
  });

  syncInputs();
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

function verdict({ p, pt, loss, top, t, target }) {
  const tc = esc(show(CANDS[target].ch));
  const wc = esc(show(CANDS[top].ch));
  if (top !== target && pt < 0.05) {
    return `<div class="callout callout--danger"><span class="callout__title">과신한 오답 · 손실 ${Number.isFinite(loss) ? loss.toFixed(2) : '∞'}</span><p>모델은 “${wc}”를 ${pct(p[top])}로 확신했는데 정답은 “${tc}”였다. 손실은 아무것도 모르는 균등 추측(${LN_K.toFixed(2)})의 ${(loss / LN_K).toFixed(1)}배다. −log는 p가 0에 가까울수록 끝없이 커지므로, 틀린 답을 확신하면 크게 벌을 받는다.</p></div>`;
  }
  if (t <= 0.1) {
    return `<div class="callout callout--more"><span class="callout__title">T → 0: 소프트맥스가 argmax가 된다</span><p>로짓을 T로 나누면 차이가 ${(1 / t).toFixed(0)}배로 벌어져 1등 “${wc}”가 ${pct(p[top])}를 가져간다. 경사도 1/T배로 커진다. 온도는 8주차 디코딩에서 다시 만난다.</p></div>`;
  }
  if (pt > 0.9) {
    return `<div class="callout callout--ok"><span class="callout__title">정답을 확신 · 경사가 거의 0</span><p>p(${tc}) = ${pct(pt)}. 경사 (p − onehot)의 크기가 모두 ${(1 - pt).toFixed(2)} 이하라 더 배울 것이 거의 없다. 학습은 틀린 만큼만 고친다.</p></div>`;
  }
  return `<div class="callout"><span class="callout__title">경사 = p − onehot</span><p>정답 “${tc}”의 로짓은 (1 − p) = ${(1 - pt).toFixed(2)}만큼 올리고(↑), 나머지는 <b>자기 확률만큼</b> 내린다(↓). 가장 크게 내려가는 것은 가장 헷갈리게 만든 “${esc(show(CANDS[maxWrong(p, target)].ch))}”다. <b>⬇ 경사 하강 1스텝</b>을 눌러 본다.</p></div>`;
}

function maxWrong(p, target) {
  let best = -1;
  p.forEach((q, i) => {
    if (i !== target && (best < 0 || q > p[best])) best = i;
  });
  return best;
}

/** −ln p curve with the current point. */
function lossCurve(pt, loss) {
  const W = 300;
  const H = 120;
  const L0 = 36;
  const B = 100;
  const YMAX = 6;
  const x = (p) => L0 + p * (W - L0 - 10);
  const y = (l) => B - (Math.min(l, YMAX) / YMAX) * (B - 10);
  const pts = [];
  for (let i = 0; i <= 60; i++) {
    const p = 0.0025 + (i / 60) * 0.9975;
    pts.push(`${x(p).toFixed(1)},${y(-Math.log(p)).toFixed(1)}`);
  }
  const cx = x(pt);
  const cy = y(Number.isFinite(loss) ? loss : YMAX);
  const off = loss > YMAX ? ' (위로 넘침)' : '';
  return `<svg class="w04-curve" viewBox="0 0 ${W} ${H}" role="img" aria-label="손실 곡선: p(정답) ${pct(pt)}에서 손실 ${Number.isFinite(loss) ? loss.toFixed(2) : '무한대'}${off}">
    <line class="ax" x1="${L0}" y1="${B}" x2="${W - 6}" y2="${B}"/><line class="ax" x1="${L0}" y1="${B}" x2="${L0}" y2="6"/>
    <line class="ref" x1="${L0}" x2="${W - 6}" y1="${y(LN_K)}" y2="${y(LN_K)}"/>
    <text class="lbl" x="${W - 8}" y="${y(LN_K) - 4}" text-anchor="end">균등 추측 ln 6</text>
    <polyline class="crv" points="${pts.join(' ')}"/>
    <circle class="dot" cx="${cx.toFixed(1)}" cy="${cy.toFixed(1)}" r="5"/>
    <text class="lbl" x="${L0}" y="${H - 4}">p(정답) 0</text><text class="lbl" x="${W - 8}" y="${H - 4}" text-anchor="end">1</text>
    <text class="lbl" x="${L0 - 4}" y="14" text-anchor="end">6</text><text class="lbl" x="${L0 - 4}" y="${B}" text-anchor="end">0</text>
  </svg>`;
}

const stat = (label, value, cls = '') => `<div class="stat${cls ? ` w04-${cls}` : ''}"><span class="stat__label">${label}</span><span class="stat__value">${value}</span></div>`;
const show = (c) => (c === ' ' ? '␣' : c);
const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
const fmt = (v, d) => (v < 0 ? '−' : '') + Math.abs(v).toFixed(d);
const fmtSigned = (v) => (Math.abs(v) < 5e-4 ? '0.000' : (v < 0 ? '−' : '+') + Math.abs(v).toFixed(3));

function pct(x) {
  if (x >= 0.9995) return '100%';
  if (x < 0.001) return x === 0 ? '0%' : `${(x * 100).toExponential(1)}%`;
  return `${(x * 100).toFixed(x < 0.1 ? 1 : 0)}%`;
}

function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
}
