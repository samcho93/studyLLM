// w04 자동 미분 그래프
// One concept: backpropagation is the chain rule applied node by node on a
// computation graph, from the loss back to every weight. Uses core/value.js.
// Graph: L = (tanh(x1·w1 + x2·w2 + b) − y)².
// Failure shown: a too-large learning rate makes the loss bounce (lr 0.4) or
// jumps into the flat part of tanh where the gradient vanishes (lr 1.0).

import { registerOutput, unregisterOutput } from '../site/result.js';
import { Value } from '../core/value.js';

const X1 = 2;
const X2 = -1;
const Y = 0.5;
const INIT = { w1: -0.4, w2: 0.3, b: 0.2 };
const PARAMS = ['w1', 'w2', 'b'];
const HIST_MAX = 120;

// node layout (centre x, y) in a 880 × 330 viewBox
const NW = 104;
const NH = 58;
const POS = {
  x1: [62, 40], w1: [62, 112], x2: [62, 218], w2: [62, 290],
  m1: [190, 76], m2: [190, 254], b: [318, 300],
  s: [318, 165], n: [446, 210],
  h: [574, 210], ny: [574, 300],
  d: [702, 255], L: [826, 255],
};
const NAME = {
  x1: 'x₁', w1: 'w₁', x2: 'x₂', w2: 'w₂', b: 'b',
  m1: 'x₁·w₁', m2: 'x₂·w₂', s: 's', n: 'n', h: 'h', ny: '−y', d: 'd', L: 'L',
};
const DESC = {
  x1: '입력 (고정)', x2: '입력 (고정)', ny: '정답 y = 0.5의 음수 (고정)',
  w1: '가중치 (학습)', w2: '가중치 (학습)', b: '편향 (학습)',
  m1: '곱', m2: '곱', s: '합', n: '합 = 뉴런의 로짓', h: 'tanh 활성화', d: '오차 h − y', L: '손실 = 오차²',
};
const OPSYM = { '+': '+', '×': '×', tanh: 'tanh', '^2': 'x²' };

const template = document.createElement('template');
template.innerHTML = `
  <div class="widget w04-ag">
    <h3 class="widget__title">자동 미분 그래프</h3>
    <p class="w04-ctx">뉴런 하나: <code>L = (tanh(x₁·w₁ + x₂·w₂ + b) − y)²</code> · 노드를 누르면 그 노드의 경사가 어떻게 계산됐는지 연쇄 법칙으로 보여 준다</p>
    <div class="w04-graph-wrap" data-slot="graph"></div>
    <div class="legend" aria-hidden="true">
      <span><i class="w04-lg-param"></i>학습할 파라미터</span>
      <span><i class="w04-lg-const"></i>입력·상수</span>
      <span><i class="w04-lg-loss"></i>손실</span>
      <span><i class="w04-lg-flow"></i>방금 경사가 흐른 길</span>
    </div>
    <div class="widget__controls">
      <label class="field">
        <span class="field__label">학습률 lr <output data-out="lr"></output></span>
        <input type="range" data-in="lr" min="0.01" max="1" step="0.01">
      </label>
    </div>
    <div class="btn-row">
      <button type="button" class="btn small ghost" data-act="fwd">① 순전파</button>
      <button type="button" class="btn small ghost" data-act="bwd">② 역전파</button>
      <button type="button" class="btn small primary" data-act="step">③ 경사 하강 1스텝</button>
      <button type="button" class="btn small primary" data-act="step10">10스텝</button>
      <button type="button" class="btn small ghost" data-act="reset">↺ 처음 가중치</button>
    </div>
    <div class="btn-row w04-presets">
      <button type="button" class="btn small ghost" data-lr="0.05">lr 0.05 (안정)</button>
      <button type="button" class="btn small ghost" data-lr="0.4">lr 0.4 (진동)</button>
      <button type="button" class="btn small ghost" data-lr="1">lr 1.0 (튕겨 나감)</button>
    </div>
    <div data-slot="out-inline"></div>
  </div>`;

const outTemplate = document.createElement('template');
outTemplate.innerHTML = `
  <div class="w04-out w04-ag-out">
    <div data-slot="verdict" aria-live="polite"></div>
    <div class="stat-row" data-slot="stats"></div>
    <h4>손실 기록 <small class="w04-muted" data-slot="spark-note"></small></h4>
    <div data-slot="spark"></div>
    <h4 data-slot="detail-title">선택한 노드</h4>
    <div class="w04-detail" data-slot="detail"></div>
    <div data-slot="update"></div>
  </div>`;

const state = new WeakMap();
let seq = 0;

/** Build the graph for given parameters. Returns { L, nodes: { key: Value } }. */
export function buildGraph(p) {
  const x1 = new Value(X1, [], '', 'x1');
  const w1 = new Value(p.w1, [], '', 'w1');
  const x2 = new Value(X2, [], '', 'x2');
  const w2 = new Value(p.w2, [], '', 'w2');
  const b = new Value(p.b, [], '', 'b');
  const m1 = x1.mul(w1);
  const m2 = x2.mul(w2);
  const s = m1.add(m2);
  const n = s.add(b);
  const h = n.tanh();
  const ny = new Value(-Y, [], '', 'ny');
  const d = h.add(ny);
  const L = d.pow(2);
  const nodes = { x1, w1, x2, w2, b, m1, m2, s, n, h, ny, d, L };
  Object.entries(nodes).forEach(([k, v]) => (v.label = k));
  return { L, nodes };
}

/** One gradient-descent step; returns { params, loss (before the step), grads }. */
export function gdStep(p, lr) {
  const g = buildGraph(p);
  g.L.backward();
  const grads = Object.fromEntries(PARAMS.map((k) => [k, g.nodes[k].grad]));
  return { params: Object.fromEntries(PARAMS.map((k) => [k, p[k] - lr * grads[k]])), loss: g.L.data, grads };
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
  const uid = ++seq;
  const outputId = `w04-autograd:${uid}`;
  const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
  const st = { ctrl, outputId, timer: 0 };
  state.set(el, st);

  const s = {
    params: { ...INIT },
    lr: 0.05,
    graph: null,
    dataShown: new Set(),
    gradShown: new Set(),
    flow: new Set(), // edges "child>parent" lit during backward
    active: null,
    sel: 'w1',
    history: [],
    lastUpdate: null,
    steps: 0,
  };

  $('[data-in=lr]').value = String(s.lr);
  registerOutput(outputId, { title: options.outputTitle ?? '자동 미분 결과', node: out, inlineHost: $('[data-slot=out-inline]') });

  // ---- graph skeleton (drawn once; texts updated by render)
  const keys = Object.keys(POS);
  const edges = [];
  function drawSkeleton() {
    const g = buildGraph(s.params);
    keys.forEach((k) => g.nodes[k].children.forEach((c) => edges.push([c.label, k])));
    const edgeSvg = edges
      .map(([a, b]) => {
        const [ax, ay] = POS[a];
        const [bx, by] = POS[b];
        const x0 = ax + NW / 2;
        const x1 = bx - NW / 2 - 4;
        const mx = (x0 + x1) / 2;
        return `<path class="w04-edge" data-edge="${a}>${b}" d="M${x0} ${ay} C${mx} ${ay} ${mx} ${by} ${x1} ${by}"/>`;
      })
      .join('');
    const nodeSvg = keys
      .map((k) => {
        const [cx, cy] = POS[k];
        const kind = PARAMS.includes(k) ? 'param' : ['x1', 'x2', 'ny'].includes(k) ? 'const' : k === 'L' ? 'loss' : 'op';
        const op = g.nodes[k].op;
        return `<g class="w04-node ${kind}" data-node="${k}" tabindex="0" role="button" aria-label="${esc(`${NAME[k]} 노드: ${DESC[k]}`)}" transform="translate(${cx - NW / 2} ${cy - NH / 2})">
          <rect width="${NW}" height="${NH}" rx="9"/>
          <text class="nm" x="8" y="17">${esc(NAME[k])}</text>${op ? `<text class="op" x="${NW - 8}" y="17" text-anchor="end">${esc(OPSYM[op] ?? op)}</text>` : ''}
          <text class="dv" x="8" y="35" data-v="${k}"></text>
          <text class="gv" x="8" y="51" data-g="${k}"></text>
        </g>`;
      })
      .join('');
    $('[data-slot=graph]').innerHTML = `<svg class="w04-graph" viewBox="0 0 890 336" role="group" aria-label="계산 그래프: 입력과 가중치에서 손실까지">
      <defs><marker id="w04-arr-${uid}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0L10 5L0 10z" class="w04-arrhead"/></marker></defs>
      <g class="w04-edges" style="--arr: url(#w04-arr-${uid})">${edgeSvg}</g>${nodeSvg}</svg>`;
  }

  function forwardNow() {
    s.graph = buildGraph(s.params);
    s.gradShown.clear();
    s.flow.clear();
  }

  function stop() {
    clearTimeout(st.timer);
    s.active = null;
  }

  /** Reveal a list of keys one by one (or at once with reduced motion). */
  function reveal(list, onEach, done) {
    stop();
    if (reduced) {
      list.forEach(onEach);
      done?.();
      render();
      return;
    }
    let i = 0;
    const tick = () => {
      if (ctrl.signal.aborted) return;
      if (i >= list.length) {
        s.active = null;
        done?.();
        render();
        return;
      }
      s.active = list[i];
      onEach(list[i]);
      i++;
      render();
      st.timer = setTimeout(tick, 110);
    };
    tick();
  }

  const topoKeys = () => s.graph.L.topo().map((v) => v.label);

  function doForward(animate = true) {
    forwardNow();
    s.dataShown.clear();
    if (!animate) {
      keys.forEach((k) => s.dataShown.add(k));
      return render();
    }
    reveal(topoKeys(), (k) => s.dataShown.add(k));
  }

  function doBackward() {
    if (!s.graph) forwardNow();
    keys.forEach((k) => s.dataShown.add(k));
    s.graph.L.backward();
    s.gradShown.clear();
    s.flow.clear();
    reveal(topoKeys().reverse(), (k) => {
      s.gradShown.add(k);
      edges.filter(([, b]) => b === k).forEach(([a, b]) => s.flow.add(`${a}>${b}`));
    });
  }

  function doSteps(k) {
    stop();
    let last = null;
    for (let i = 0; i < k; i++) {
      const r = gdStep(s.params, s.lr);
      last = { before: { ...s.params }, grads: r.grads, after: r.params, lr: s.lr };
      s.params = r.params;
      s.steps++;
      if (!Number.isFinite(s.params.w1 + s.params.w2 + s.params.b)) break;
    }
    s.lastUpdate = { ...last, k };
    forwardNow();
    keys.forEach((key) => s.dataShown.add(key));
    pushHistory();
    render();
  }

  function pushHistory() {
    s.history.push(s.graph.L.data);
    if (s.history.length > HIST_MAX) s.history.shift();
  }

  function render() {
    const g = s.graph;
    $('[data-out=lr]').textContent = s.lr.toFixed(2);
    keys.forEach((k) => {
      const v = g.nodes[k];
      const dv = root.querySelector(`[data-v="${k}"]`);
      const gv = root.querySelector(`[data-g="${k}"]`);
      dv.textContent = s.dataShown.has(k) ? `값 ${fmt(v.data)}` : '값 ?';
      gv.textContent = s.gradShown.has(k) ? `∂L ${fmt(v.grad)}` : '∂L ?';
      const node = root.querySelector(`[data-node="${k}"]`);
      node.classList.toggle('is-active', s.active === k);
      node.classList.toggle('is-sel', s.sel === k);
      node.classList.toggle('has-grad', s.gradShown.has(k));
    });
    root.querySelectorAll('[data-edge]').forEach((p) => p.classList.toggle('is-flow', s.flow.has(p.dataset.edge)));

    // stats
    const n = g.nodes.n.data;
    const h = g.nodes.h.data;
    const L = g.L.data;
    const slope = 1 - h * h;
    $('[data-slot=stats]').innerHTML = [
      stat('손실 L', fmt(L, 4)),
      stat('스텝', String(s.steps)),
      stat('n (로짓)', fmt(n, 2)),
      stat('tanh 기울기 1−h²', fmt(slope, 3)),
      stat('lr', s.lr.toFixed(2)),
    ].join('');

    $('[data-slot=spark]').innerHTML = sparkline(s.history);
    $('[data-slot=spark-note]').textContent = `— 최근 ${s.history.length}개 · 처음 ${fmt(s.history[0], 3)} → 지금 ${fmt(s.history.at(-1), 3)}`;
    $('[data-slot=verdict]').innerHTML = verdict(s, slope);
    renderDetail();
    renderUpdate();
  }

  function renderDetail() {
    const k = s.sel;
    const g = s.graph;
    const v = g.nodes[k];
    $('[data-slot=detail-title]').textContent = `선택한 노드 · ${NAME[k]} (${DESC[k]})`;
    const lines = [];
    lines.push(`<p>값 = ${forwardFormula(k, g.nodes)}</p>`);
    if (!s.gradShown.has(k)) {
      lines.push(`<p class="w04-muted">경사 ∂L/∂${esc(NAME[k])}는 아직 모른다. <b>② 역전파</b>를 누르면 L에서부터 거꾸로 채워진다.</p>`);
    } else if (k === 'L') {
      lines.push(`<p>∂L/∂L = <b>1</b> — 역전파의 출발점</p>`);
    } else {
      const users = keys.filter((u) => g.nodes[u].children.includes(v));
      const terms = users.map((u) => {
        const up = g.nodes[u];
        const [local, why] = localDeriv(up, v, u);
        return { u, up: up.grad, local, why };
      });
      lines.push(
        `<p>∂L/∂${esc(NAME[k])} = ${terms.map((t) => `∂L/∂${esc(NAME[t.u])} × ∂${esc(NAME[t.u])}/∂${esc(NAME[k])}`).join(' + ')}</p>`,
      );
      lines.push(
        `<p class="w04-chain">= ${terms.map((t) => `<span>${fmt(t.up)}</span> × <span title="${esc(t.why)}">${fmt(t.local)}</span>`).join(' + ')} = <b>${fmt(v.grad)}</b></p>`,
      );
      lines.push(`<p class="w04-muted">${terms.map((t) => `∂${esc(NAME[t.u])}/∂${esc(NAME[k])}: ${esc(t.why)}`).join(' · ')}</p>`);
      if (PARAMS.includes(k)) {
        lines.push(`<p>경사 하강: ${esc(NAME[k])} ← ${fmt(v.data)} − ${s.lr.toFixed(2)} × ${fmt(v.grad)} = <b>${fmt(v.data - s.lr * v.grad)}</b> ${v.grad > 0 ? '(경사가 +라 줄인다)' : v.grad < 0 ? '(경사가 −라 늘린다)' : ''}</p>`);
      }
    }
    $('[data-slot=detail]').innerHTML = lines.join('');
  }

  function renderUpdate() {
    const u = s.lastUpdate;
    if (!u) {
      $('[data-slot=update]').innerHTML = '';
      return;
    }
    $('[data-slot=update]').innerHTML = `<h4>마지막 업데이트${u.k > 1 ? ` (${u.k}스텝 중 마지막)` : ''} <small class="w04-muted">— w ← w − lr × ∂L/∂w</small></h4>
      <table class="w04-upd"><thead><tr><th>파라미터</th><th>전</th><th>∂L/∂w</th><th>후</th></tr></thead><tbody>${PARAMS.map(
        (k) => `<tr><td>${NAME[k]}</td><td>${fmt(u.before[k])}</td><td>${fmt(u.grads[k])}</td><td>${fmt(u.after[k])}</td></tr>`,
      ).join('')}</tbody></table>`;
  }

  // ---- events
  const on = (target, type, fn) => target.addEventListener(type, fn, { signal: ctrl.signal });
  on($('[data-in=lr]'), 'input', (e) => {
    s.lr = Number(e.target.value);
    render();
  });
  on(root, 'click', (e) => {
    const lrBtn = e.target.closest('[data-lr]');
    if (lrBtn) {
      s.lr = Number(lrBtn.dataset.lr);
      $('[data-in=lr]').value = String(s.lr);
      render();
      return;
    }
    const node = e.target.closest('[data-node]');
    if (node) {
      s.sel = node.dataset.node;
      render();
      return;
    }
    const act = e.target.closest('[data-act]')?.dataset.act;
    if (act === 'fwd') doForward(true);
    if (act === 'bwd') doBackward();
    if (act === 'step') doSteps(1);
    if (act === 'step10') doSteps(10);
    if (act === 'reset') {
      stop();
      s.params = { ...INIT };
      s.history = [];
      s.steps = 0;
      s.lastUpdate = null;
      forwardNow();
      keys.forEach((k) => s.dataShown.add(k));
      pushHistory();
      render();
    }
  });
  on(root, 'keydown', (e) => {
    const node = e.target.closest?.('[data-node]');
    if (node && (e.key === 'Enter' || e.key === ' ')) {
      e.preventDefault();
      s.sel = node.dataset.node;
      render();
    }
  });

  // ---- initial state: forward pass already done, gradients still unknown
  drawSkeleton();
  forwardNow();
  keys.forEach((k) => s.dataShown.add(k));
  pushHistory();
  render();
}

export function unmount(el) {
  const st = state.get(el);
  if (!st) return;
  clearTimeout(st.timer);
  st.ctrl.abort();
  unregisterOutput(st.outputId);
  state.delete(el);
  el.replaceChildren();
}

// ---------------------------------------------------------------- helpers

/** Local derivative ∂parent/∂child for the ops used in this graph. */
function localDeriv(parent, child, key) {
  const [a, b] = parent.children;
  switch (parent.op) {
    case '+':
      return [1, '덧셈은 경사를 그대로 넘긴다 (1)'];
    case '×': {
      const other = a === child ? b : a;
      return [other.data, `곱셈은 상대편 값을 곱한다 (${NAME[other.label]} = ${fmt(other.data)})`];
    }
    case 'tanh':
      return [1 - parent.data ** 2, `tanh의 기울기 1 − h² (h = ${fmt(parent.data)})`];
    case '^2':
      return [2 * child.data, `제곱의 기울기 2·d (d = ${fmt(child.data)})`];
    default:
      return [0, key];
  }
}

function forwardFormula(k, nd) {
  const v = nd[k];
  const f = (x) => fmt(x.data);
  switch (k) {
    case 'm1':
      return `x₁ × w₁ = ${f(nd.x1)} × ${f(nd.w1)} = <b>${f(v)}</b>`;
    case 'm2':
      return `x₂ × w₂ = ${f(nd.x2)} × ${f(nd.w2)} = <b>${f(v)}</b>`;
    case 's':
      return `x₁·w₁ + x₂·w₂ = ${f(nd.m1)} + ${f(nd.m2)} = <b>${f(v)}</b>`;
    case 'n':
      return `s + b = ${f(nd.s)} + ${f(nd.b)} = <b>${f(v)}</b>`;
    case 'h':
      return `tanh(n) = tanh(${f(nd.n)}) = <b>${f(v)}</b>`;
    case 'd':
      return `h − y = ${f(nd.h)} − ${fmt(Y)} = <b>${f(v)}</b>`;
    case 'L':
      return `d² = (${f(nd.d)})² = <b>${fmt(v.data, 4)}</b>`;
    default:
      return `<b>${f(v)}</b> (${DESC[k]})`;
  }
}

function verdict(s, slope) {
  const hist = s.history;
  const L = hist.at(-1);
  const last = hist.slice(-8);
  let ups = 0;
  for (let i = 1; i < last.length; i++) if (last[i] > last[i - 1] + 1e-9) ups++;
  if (hist.length > 1 && slope < 0.02 && L > 0.05) {
    return `<div class="callout callout--danger"><span class="callout__title">튕겨 나가 멈췄다 · tanh 포화</span><p>lr이 커서 n이 한 번에 ${fmt(s.graph.nodes.n.data, 1)}까지 뛰었다. h가 ±1에 붙어 tanh의 기울기 1 − h²가 ${slope.toExponential(1)}뿐이다. 역전파는 이 값을 곱해 내려가므로 뒤쪽 경사가 모두 0에 가깝고, 손실 ${fmt(L, 3)}에서 더 내려가지 못한다(경사 소실).</p></div>`;
  }
  if (ups >= 2) {
    return `<div class="callout callout--danger"><span class="callout__title">진동 · lr ${s.lr.toFixed(2)}은 너무 크다</span><p>최근 ${last.length}스텝 중 ${ups}번 손실이 <b>올라갔다</b>. 경사 방향은 맞지만 한 걸음이 너무 커서 골짜기를 넘어갔다 왔다 한다. lr을 줄이면 매끄럽게 내려간다.</p></div>`;
  }
  if (L < 1e-4) {
    return `<div class="callout callout--ok"><span class="callout__title">수렴 · 손실 ${L.toExponential(1)}</span><p>${s.steps}스텝 만에 h가 정답 y = ${Y}에 닿았다. 경사가 0이 되어 더 움직이지 않는다. ↺로 돌아가 lr을 바꿔 비교한다.</p></div>`;
  }
  if (hist.length === 1 && !s.gradShown.size) {
    return `<div class="callout"><span class="callout__title">순전파는 끝났다 · 경사는 아직 ?</span><p>입력 → 손실 방향으로 값이 모두 계산되었다(L = ${fmt(L, 3)}). <b>② 역전파</b>를 누르면 L에서 출발해 각 노드의 ∂L이 거꾸로 채워진다. 그다음 <b>③ 경사 하강</b>.</p></div>`;
  }
  if (hist.length === 1) {
    return `<div class="callout"><span class="callout__title">경사가 채워졌다</span><p>파라미터 노드(w₁, w₂, b)를 눌러 연쇄 법칙 계산을 확인한다. ∂L이 +면 그 값을 줄여야, −면 늘려야 손실이 준다. <b>③ 경사 하강 1스텝</b>을 누른다.</p></div>`;
  }
  return `<div class="callout callout--ok"><span class="callout__title">손실이 내려간다 · ${fmt(hist[0], 3)} → ${fmt(L, 4)}</span><p>한 스텝 = 순전파 → 역전파 → w ← w − lr × ∂L/∂w. 이 세 줄이 7주차 미니 GPT 학습의 전부다(파라미터 3개 대신 수십만 개).</p></div>`;
}

function sparkline(hist) {
  const W = 300;
  const H = 90;
  const P = 6;
  if (!hist.length) return '';
  const max = Math.max(...hist.filter(Number.isFinite), 1e-6);
  const n = Math.max(hist.length - 1, 1);
  const x = (i) => P + (i / n) * (W - 2 * P);
  const y = (v) => H - P - (Math.min(v, max) / max) * (H - 2 * P);
  const pts = hist.map((v, i) => `${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(' ');
  const lastI = hist.length - 1;
  return `<svg class="w04-spark" viewBox="0 0 ${W} ${H}" role="img" aria-label="손실 기록 ${hist.length}개, 최대 ${fmt(max, 3)}, 마지막 ${fmt(hist[lastI], 4)}">
    <line class="ax" x1="${P}" y1="${H - P}" x2="${W - P}" y2="${H - P}"/>
    <polyline class="crv" points="${pts}"/>
    ${hist.length < 40 ? hist.map((v, i) => `<circle class="pt" cx="${x(i).toFixed(1)}" cy="${y(v).toFixed(1)}" r="2.2"/>`).join('') : ''}
    <circle class="dot" cx="${x(lastI).toFixed(1)}" cy="${y(hist[lastI]).toFixed(1)}" r="4"/>
    <text class="lbl" x="${P + 2}" y="12">${fmt(max, 3)}</text><text class="lbl" x="${W - P}" y="${H - P - 4}" text-anchor="end">0</text>
  </svg>`;
}

const stat = (label, value) => `<div class="stat"><span class="stat__label">${label}</span><span class="stat__value">${value}</span></div>`;

function fmt(v, d = 3) {
  if (v === undefined || v === null || Number.isNaN(v)) return '—';
  if (!Number.isFinite(v)) return v > 0 ? '∞' : '−∞';
  const r = Math.abs(v) < 0.5 * 10 ** -d ? 0 : v;
  return (r < 0 ? '−' : '') + Math.abs(r).toFixed(d);
}

function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
}
