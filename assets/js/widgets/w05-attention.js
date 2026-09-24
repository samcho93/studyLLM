// w05 어텐션 계산기
// One concept: scaled dot-product attention. Every token builds a query, key and
// value; scores = QKᵀ/√d, a causal mask hides the future, softmax turns each row
// into weights, and the output is the weighted average of the values.
// Failure modes: mask off (the model peeks at the future), no √d scaling with a
// large d or a large score multiplier (softmax saturates into one-hot).

import { registerOutput, unregisterOutput } from '../site/result.js';
import { mulberry32, gaussian } from '../core/rng.js';

// ---------------------------------------------------------------- pure model (no DOM, testable in Node)

export const THETA = Math.PI / 3; // position angle per step (period 6, sentences ≤ 7 tokens)
export const D_OPTIONS = [4, 8, 16, 32, 64];
export const RAND_STD = 0.7;
export const DIM_NAMES = ['내용', '시간', '위치c', '위치s'];

/** Hand-designed 2-d meaning: [내용어(+)·기능어(−), 시간·수량(+)·사물(−)]. Position fills dims 3–4. */
export const SENTENCES = [
  {
    id: 'lab',
    label: '실습실 은 평일 오후 9시 까지',
    tokens: [['실습실', 1.0, -0.8], ['은', -1.0, 0.0], ['평일', 0.8, 0.9], ['오후', 0.7, 1.0], ['9시', 0.9, 1.2], ['까지', -0.9, 0.6]],
  },
  {
    id: 'apply',
    label: '신청 은 이용일 이틀 전 까지',
    tokens: [['신청', 0.9, -0.6], ['은', -1.0, 0.0], ['이용일', 0.8, 0.8], ['이틀', 0.6, 1.1], ['전', 0.5, 0.9], ['까지', -0.9, 0.6]],
  },
  {
    id: 'attn',
    label: '어텐션 은 토큰 의 순서 를 모른다',
    tokens: [['어텐션', 1.0, -0.8], ['은', -1.0, 0.0], ['토큰', 1.0, -0.7], ['의', -1.0, -0.2], ['순서', 0.8, 0.4], ['를', -1.0, -0.1], ['모른다', -0.5, 0.2]],
  },
];

export const PRESETS = [
  { id: 'prev', label: '앞 토큰 보기 헤드' },
  { id: 'same', label: '같은 종류 보기 헤드' },
  { id: 'random', label: '무작위 (d 조절)' },
];

/** Position part of the embedding: a point on the unit circle, angle t·θ. */
export function posEnc(t) {
  return [Math.cos(t * THETA), Math.sin(t * THETA)];
}

/** X = [meaning(2) | position(2)] for each token. */
export function buildX(tokens) {
  return tokens.map((tok, t) => [tok.e[0], tok.e[1], ...posEnc(t)]);
}

export const zeros = (r, c) => Array.from({ length: r }, () => new Array(c).fill(0));
export const eye = (n) => zeros(n, n).map((row, i) => row.map((_, j) => (i === j ? 1 : 0)));

export function matmul(A, B) {
  const n = A.length;
  const m = B[0].length;
  const k = B.length;
  const C = zeros(n, m);
  for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) {
    let s = 0;
    for (let p = 0; p < k; p++) s += A[i][p] * B[p][j];
    C[i][j] = s;
  }
  return C;
}

export const transpose = (A) => A[0].map((_, j) => A.map((row) => row[j]));

/**
 * Projection matrices for a head. Row-vector convention: Q = X · Wq.
 *  prev : Wq rotates the position back by one step, Wk keeps it → q_t · k_s peaks at s = t − 1
 *  same : Wq = Wk = gain on the meaning dims → tokens of the same kind score high
 *  random: N(0, 0.7²) entries, query/key width d (value stays 4-d)
 */
export function headWeights(preset, { d = 4, seed = 1 } = {}) {
  if (preset === 'prev') {
    const g = Math.sqrt(8);
    const c = Math.cos(THETA);
    const s = Math.sin(THETA);
    const Wq = zeros(4, 4);
    const Wk = zeros(4, 4);
    Wq[2][2] = g * c; Wq[2][3] = -g * s;
    Wq[3][2] = g * s; Wq[3][3] = g * c;
    Wk[2][2] = g; Wk[3][3] = g;
    return { Wq, Wk, Wv: eye(4), d: 4 };
  }
  if (preset === 'same') {
    const Wq = zeros(4, 4);
    Wq[0][0] = 2; Wq[1][1] = 2;
    return { Wq, Wk: Wq.map((r) => r.slice()), Wv: eye(4), d: 4 };
  }
  // Draw the full 4×64 matrices and keep the first d columns, so moving the d
  // slider adds dimensions instead of reshuffling the whole head.
  const rand = mulberry32(seed);
  const rnd = (c) => zeros(4, c).map((row) => row.map(() => RAND_STD * gaussian(rand)));
  const cut = (M) => M.map((row) => row.slice(0, d));
  const Wq = rnd(64);
  const Wk = rnd(64);
  return { Wq: cut(Wq), Wk: cut(Wk), Wv: rnd(4), d };
}

/** Stable softmax that treats −Infinity as "weight 0". */
export function softmaxRow(xs) {
  let max = -Infinity;
  for (const x of xs) if (x > max) max = x;
  const ex = xs.map((x) => (x === -Infinity ? 0 : Math.exp(x - max)));
  const sum = ex.reduce((a, b) => a + b, 0);
  return ex.map((e) => e / sum);
}

/**
 * Full scaled dot-product attention, keeping every intermediate for display.
 * @returns {{ Q, K, V, raw, scaled, masked, A, O, d, div }}
 */
export function attention(X, W, { mask = true, scale = true, mult = 1 } = {}) {
  const Q = matmul(X, W.Wq);
  const K = matmul(X, W.Wk);
  const V = matmul(X, W.Wv);
  const d = W.Wq[0].length;
  const div = scale ? Math.sqrt(d) : 1;
  const raw = matmul(Q, transpose(K));
  const scaled = raw.map((row) => row.map((x) => (x / div) * mult));
  const masked = scaled.map((row, t) => row.map((x, s) => (mask && s > t ? -Infinity : x)));
  const A = masked.map(softmaxRow);
  const O = matmul(A, V);
  return { Q, K, V, raw, scaled, masked, A, O, d, div };
}

/** Row statistics over rows that have ≥ 2 visible keys. */
export function attnStats(A, mask) {
  const T = A.length;
  let maxW = 0, normH = 0, rows = 0, future = 0, futureRows = 0;
  for (let t = 0; t < T; t++) {
    const n = mask ? t + 1 : T;
    if (t < T - 1) {
      let f = 0;
      for (let s = t + 1; s < T; s++) f += A[t][s];
      future += f;
      futureRows++;
    }
    if (n < 2) continue;
    let h = 0, m = 0;
    for (let s = 0; s < n; s++) {
      const w = A[t][s];
      if (w > 0) h -= w * Math.log2(w);
      if (w > m) m = w;
    }
    maxW += m;
    normH += h / Math.log2(n);
    rows++;
  }
  return {
    meanMax: rows ? maxW / rows : 1,
    meanNormH: rows ? normH / rows : 0,
    future: futureRows ? future / futureRows : 0,
  };
}

/** Standard deviation of the visible (unmasked) scores. */
export function scoreStd(masked) {
  const xs = masked.flat().filter((x) => x !== -Infinity);
  const m = xs.reduce((a, b) => a + b, 0) / xs.length;
  return Math.sqrt(xs.reduce((a, b) => a + (b - m) ** 2, 0) / xs.length);
}

/** Mean weight each row puts on the token just before it (rows 1…T−1). */
export function prevShare(A) {
  let s = 0;
  for (let t = 1; t < A.length; t++) s += A[t][t - 1];
  return A.length > 1 ? s / (A.length - 1) : 0;
}

// ---------------------------------------------------------------- widget

const template = document.createElement('template');
template.innerHTML = `
  <div class="widget w05-at">
    <h3 class="widget__title">어텐션 계산기</h3>
    <div class="widget__controls">
      <label class="field">
        <span class="field__label">문장 (토큰 단위)</span>
        <select data-in="sent"></select>
      </label>
      <label class="field">
        <span class="field__label">헤드 (Wq · Wk · Wv)</span>
        <select data-in="preset"></select>
      </label>
      <label class="field">
        <span class="field__label">점수 배율 ×m (= 1/온도) <output data-out="mult"></output></span>
        <input type="range" data-in="mult" min="-2" max="3" step="0.5">
      </label>
      <label class="field">
        <span class="field__label">쿼리·키 차원 d <output data-out="d"></output></span>
        <input type="range" data-in="d" min="0" max="${D_OPTIONS.length - 1}" step="1">
      </label>
    </div>
    <div class="w05-toggles">
      <label class="w05-tg"><input type="checkbox" data-in="mask"> 인과 마스크</label>
      <label class="w05-tg"><input type="checkbox" data-in="scale"> ÷√d 스케일링</label>
      <label class="w05-tg"><input type="checkbox" data-in="multi"> 멀티헤드 (2개 나란히)</label>
    </div>
    <div class="btn-row">
      <button type="button" class="btn small ghost" data-act="reset">↺ 처음 설정</button>
      <button type="button" class="btn small ghost" data-act="peek">❌ 마스크 끄기</button>
      <button type="button" class="btn small ghost" data-act="saturate">❌ d = 64 · 스케일링 끄기</button>
      <button type="button" class="btn small ghost" data-act="reseed" data-slot="reseed">🎲 새 무작위 헤드</button>
    </div>
    <div class="w05-edit">
      <div class="w05-edit__head">
        <b>임베딩 편집</b> <small class="w05-muted">— 토큰을 고르면 그 토큰이 쿼리가 되고, 의미 두 칸을 바꿀 수 있다</small>
      </div>
      <div class="w05-chips" data-slot="chips" role="group" aria-label="토큰 선택"></div>
      <div class="w05-edit__grid">
        <label class="field">
          <span class="field__label">내용어(+) · 기능어(−) <output data-out="e0"></output></span>
          <input type="range" data-in="e0" min="-1.5" max="1.5" step="0.1">
        </label>
        <label class="field">
          <span class="field__label">시간(+) · 사물(−) <output data-out="e1"></output></span>
          <input type="range" data-in="e1" min="-1.5" max="1.5" step="0.1">
        </label>
        <div class="w05-pos" data-slot="pos"></div>
      </div>
    </div>
    <div data-slot="out-inline"></div>
  </div>`;

const outTemplate = document.createElement('template');
outTemplate.innerHTML = `
  <div class="w05-at-out">
    <div data-slot="verdict" aria-live="polite"></div>
    <div class="stat-row" data-slot="stats"></div>
    <div class="w05-steps" role="group" aria-label="계산 단계">
      <button type="button" class="btn small ghost" data-step="raw">① QKᵀ</button>
      <button type="button" class="btn small ghost" data-step="scaled">② ÷√d ×m</button>
      <button type="button" class="btn small ghost" data-step="masked">③ 마스크</button>
      <button type="button" class="btn small ghost" data-step="soft">④ 소프트맥스</button>
    </div>
    <h4 data-slot="mat-title"></h4>
    <div class="w05-mat-wrap"><table class="w05-mat" data-slot="mat"></table></div>
    <div class="legend" aria-hidden="true">
      <span><i class="w05-lg-w"></i>가중치 (진할수록 큼)</span>
      <span><i class="w05-lg-mask"></i>가림 (−∞ → 0)</span>
      <span><i class="w05-lg-fut"></i>미래 엿보기</span>
    </div>
    <h4 data-slot="row-title"></h4>
    <div class="w05-bars" data-slot="bars"></div>
    <div class="w05-mat-wrap"><table class="w05-sum" data-slot="sum"></table></div>
    <div data-slot="multi"></div>
    <details class="w05-details">
      <summary>X · Q · K · V 행렬 보기</summary>
      <div data-slot="qkv"></div>
    </details>
  </div>`;

const state = new WeakMap();
let seq = 0;

/**
 * @param {HTMLElement} el
 * @param {{ sentence?: string, preset?: string, outputTitle?: string }} options
 */
export function mount(el, options = {}) {
  const root = template.content.firstElementChild.cloneNode(true);
  const out = outTemplate.content.firstElementChild.cloneNode(true);
  el.replaceChildren(root);
  const $ = (sel) => root.querySelector(sel) ?? out.querySelector(sel);
  const ctrl = new AbortController();
  const outputId = `w05-attention:${++seq}`;
  state.set(el, { ctrl, outputId });

  const DEFAULTS = { preset: options.preset ?? 'prev', multLog: 0, dIdx: 0, mask: true, scale: true, multi: false, step: 'soft' };
  const s = {
    sentId: options.sentence ?? 'lab',
    tokens: [],
    sel: 4,
    seed: 4, // a draw where the d / scaling contrast is clear
    ...DEFAULTS,
  };

  function loadSentence(id) {
    const sent = SENTENCES.find((x) => x.id === id) ?? SENTENCES[0];
    s.sentId = sent.id;
    s.tokens = sent.tokens.map(([t, a, b]) => ({ t, e: [a, b], e0: [a, b] }));
    s.sel = Math.min(s.sel, s.tokens.length - 1);
  }
  loadSentence(s.sentId);

  $('[data-in=sent]').innerHTML = SENTENCES.map((o) => `<option value="${o.id}">${esc(o.label)}</option>`).join('');
  $('[data-in=preset]').innerHTML = PRESETS.map((o) => `<option value="${o.id}">${esc(o.label)}</option>`).join('');

  registerOutput(outputId, { title: options.outputTitle ?? '어텐션 계산 결과', node: out, inlineHost: $('[data-slot=out-inline]') });

  function syncControls() {
    $('[data-in=sent]').value = s.sentId;
    $('[data-in=preset]').value = s.preset;
    $('[data-in=mult]').value = String(s.multLog);
    $('[data-in=d]').value = String(s.dIdx);
    $('[data-in=d]').disabled = s.preset !== 'random';
    $('[data-in=mask]').checked = s.mask;
    $('[data-in=scale]').checked = s.scale;
    $('[data-in=multi]').checked = s.multi;
    $('[data-slot=reseed]').hidden = s.preset !== 'random';
  }

  function render() {
    const T = s.tokens.length;
    const labels = s.tokens.map((x) => x.t);
    const X = buildX(s.tokens);
    const d = s.preset === 'random' ? D_OPTIONS[s.dIdx] : 4;
    const W = headWeights(s.preset, { d, seed: s.seed });
    const mult = 2 ** s.multLog;
    const r = attention(X, W, { mask: s.mask, scale: s.scale, mult });
    const st = attnStats(r.A, s.mask);

    // controls text
    $('[data-out=mult]').textContent = `×${fmtMult(mult)}`;
    $('[data-out=d]').textContent = s.preset === 'random' ? `${d}` : '4 (고정)';
    const tok = s.tokens[s.sel];
    $('[data-in=e0]').value = String(tok.e[0]);
    $('[data-in=e1]').value = String(tok.e[1]);
    $('[data-out=e0]').textContent = fmt(tok.e[0], 1);
    $('[data-out=e1]').textContent = fmt(tok.e[1], 1);
    const p = posEnc(s.sel);
    const edited = tok.e[0] !== tok.e0[0] || tok.e[1] !== tok.e0[1];
    $('[data-slot=pos]').innerHTML = `<span class="w05-muted">위치 ${s.sel} → [${fmt(p[0])}, ${fmt(p[1])}] (고정)</span>${
      edited ? ` <button type="button" class="btn small ghost" data-act="undo-emb">↺ 원래 값</button>` : ''
    }`;
    $('[data-slot=chips]').innerHTML = s.tokens
      .map((x, i) => {
        const ed = x.e[0] !== x.e0[0] || x.e[1] !== x.e0[1];
        return `<button type="button" class="w05-chip${i === s.sel ? ' sel' : ''}${ed ? ' edited' : ''}" data-tok="${i}" aria-pressed="${i === s.sel}">${esc(x.t)}</button>`;
      })
      .join('');

    // verdict + stats
    $('[data-slot=verdict]').innerHTML = verdict(s, r, st, labels, d);
    $('[data-slot=stats]').innerHTML = [
      stat('d (쿼리·키)', `${d}`),
      stat('나누는 값', s.scale ? `√${d} = ${fmt(Math.sqrt(d), 2)}` : '1 (끔)'),
      stat('점수 표준편차', fmt(scoreStd(r.masked), 2)),
      stat('평균 최대 가중치', pct(st.meanMax)),
      stat('엔트로피 (최대 대비)', pct(st.meanNormH)),
      stat('미래로 간 가중치', pct(st.future)),
    ].join('');

    // step matrix
    out.querySelectorAll('[data-step]').forEach((b) => {
      b.classList.toggle('primary', b.dataset.step === s.step);
      b.classList.toggle('ghost', b.dataset.step !== s.step);
      b.setAttribute('aria-pressed', String(b.dataset.step === s.step));
    });
    const titles = {
      raw: `① 점수 QKᵀ — 쿼리 행 · 키 열 · 칸 = q<sub>t</sub>·k<sub>s</sub>`,
      scaled: `② ÷${s.scale ? `√${d}` : '1 (스케일링 끔)'} ×${fmtMult(mult)} — 크기를 맞춘 점수`,
      masked: s.mask ? '③ 인과 마스크 — 자기보다 뒤(미래)의 칸을 −∞로' : '③ 마스크 꺼짐 — 미래 칸이 그대로 남는다',
      soft: '④ 소프트맥스 — 각 행의 합이 1인 어텐션 가중치',
    };
    $('[data-slot=mat-title]').innerHTML = `${titles[s.step]} <small class="w05-muted">— 행을 누르면 그 쿼리를 자세히 본다</small>`;
    $('[data-slot=mat]').innerHTML = matrixHtml(s, r, labels);

    // selected query row
    const t = s.sel;
    $('[data-slot=row-title]').innerHTML = `쿼리 “${esc(labels[t])}”(${t}번)가 본 곳 → 값의 가중 평균`;
    $('[data-slot=bars]').innerHTML = labels
      .map((lab, j) => {
        const hidden = r.masked[t][j] === -Infinity;
        const w = r.A[t][j];
        const fut = !s.mask && j > t;
        return `<div class="w05-bar${hidden ? ' masked' : ''}${fut ? ' future' : ''}"><span class="t">${esc(lab)}</span><span class="b"><i style="width:${(w * 100).toFixed(1)}%"></i></span><span class="v">${hidden ? '가림' : pct(w, 1)}</span></div>`;
      })
      .join('');
    const vHead = r.V[0].map((_, k) => `<th scope="col">${esc(DIM_NAMES[k] ?? `v${k}`)}</th>`).join('');
    const rows = labels
      .map((lab, j) => {
        const w = r.A[t][j];
        if (r.masked[t][j] === -Infinity) return '';
        return `<tr><th scope="row">${esc(lab)}</th><td class="w">${fmt(w, 3)}</td>${r.V[j].map((v) => `<td>${fmt(w * v)}</td>`).join('')}</tr>`;
      })
      .join('');
    $('[data-slot=sum]').innerHTML = `<thead><tr><th scope="col">키</th><th scope="col">가중치 w</th>${vHead}</tr></thead><tbody>${rows}</tbody><tfoot><tr><th scope="row">출력 o</th><td>Σ = ${fmt(r.A[t].reduce((a, b) => a + b, 0), 2)}</td>${r.O[t]
      .map((v) => `<td><b>${fmt(v)}</b></td>`)
      .join('')}</tr></tfoot>`;

    // multi-head view
    $('[data-slot=multi]').innerHTML = s.multi ? multiHtml(s, X, labels, mult) : '';

    // X, Q, K, V
    const small = (M, name, cols) => `<h5>${name} <small class="w05-muted">(${M.length}×${M[0].length})</small></h5>${
      M[0].length <= 8
        ? `<div class="w05-mat-wrap"><table class="w05-mat plain">${plainMat(M, labels, cols)}</table></div>`
        : `<p class="w05-muted">열이 ${M[0].length}개라 생략한다. d가 커도 점수 행렬은 여전히 ${T}×${T}이다.</p>`
    }`;
    $('[data-slot=qkv]').innerHTML =
      small(X, 'X = [의미 | 위치]', DIM_NAMES) + small(r.Q, 'Q = X·Wq') + small(r.K, 'K = X·Wk') + small(r.V, 'V = X·Wv', s.preset === 'random' ? null : DIM_NAMES);
  }

  const on = (sel, type, fn) => $(sel).addEventListener(type, fn, { signal: ctrl.signal });
  const update = () => {
    syncControls();
    render();
  };
  on('[data-in=sent]', 'change', (e) => {
    loadSentence(e.target.value);
    update();
  });
  on('[data-in=preset]', 'change', (e) => {
    s.preset = e.target.value;
    update();
  });
  on('[data-in=mult]', 'input', (e) => {
    s.multLog = Number(e.target.value);
    render();
  });
  on('[data-in=d]', 'input', (e) => {
    s.dIdx = Number(e.target.value);
    render();
  });
  on('[data-in=mask]', 'change', (e) => {
    s.mask = e.target.checked;
    render();
  });
  on('[data-in=scale]', 'change', (e) => {
    s.scale = e.target.checked;
    render();
  });
  on('[data-in=multi]', 'change', (e) => {
    s.multi = e.target.checked;
    render();
  });
  on('[data-in=e0]', 'input', (e) => {
    s.tokens[s.sel].e[0] = Number(e.target.value);
    render();
  });
  on('[data-in=e1]', 'input', (e) => {
    s.tokens[s.sel].e[1] = Number(e.target.value);
    render();
  });
  root.addEventListener(
    'click',
    (e) => {
      const chip = e.target.closest('[data-tok]');
      if (chip) {
        s.sel = Number(chip.dataset.tok);
        render();
        root.querySelector(`[data-tok="${s.sel}"]`)?.focus();
        return;
      }
      const act = e.target.closest('[data-act]')?.dataset.act;
      if (!act) return;
      if (act === 'reset') {
        Object.assign(s, DEFAULTS, { preset: options.preset ?? 'prev' });
        loadSentence(s.sentId);
      }
      if (act === 'peek') {
        s.mask = false;
        s.step = 'soft';
      }
      if (act === 'saturate') {
        s.preset = 'random';
        s.dIdx = D_OPTIONS.length - 1;
        s.scale = false;
        s.step = 'soft';
      }
      if (act === 'reseed') s.seed++;
      if (act === 'undo-emb') {
        const tk = s.tokens[s.sel];
        tk.e = tk.e0.slice();
      }
      update();
    },
    { signal: ctrl.signal },
  );
  out.addEventListener(
    'click',
    (e) => {
      const b = e.target.closest('[data-step],[data-row]');
      if (!b) return;
      if (b.dataset.step) s.step = b.dataset.step;
      if (b.dataset.row) s.sel = Number(b.dataset.row);
      render();
      if (b.dataset.row) out.querySelector(`[data-row="${s.sel}"]`)?.focus();
    },
    { signal: ctrl.signal },
  );

  try {
    update();
  } catch (err) {
    root.insertAdjacentHTML('beforeend', `<div class="widget__error">어텐션을 계산하지 못했다 (${esc(err.message)}).</div>`);
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

// ---------------------------------------------------------------- rendering helpers

function matrixHtml(s, r, labels) {
  const M = r[s.step === 'soft' ? 'A' : s.step];
  const head = `<thead><tr><th scope="col" class="corner">쿼리＼키</th>${labels.map((l) => `<th scope="col">${esc(l)}</th>`).join('')}</tr></thead>`;
  const body = M.map((row, t) => {
    const cells = row
      .map((x, j) => {
        const fut = j > t;
        const hidden = r.masked[t][j] === -Infinity && s.step !== 'raw' && s.step !== 'scaled';
        const cls = [hidden ? 'masked' : '', fut && !s.mask ? 'future' : ''].filter(Boolean).join(' ');
        if (s.step === 'soft') {
          const w = hidden ? 0 : Math.round(Math.min(1, x) * 85);
          return `<td class="${cls}" style="--w:${w}%" title="${esc(`${labels[t]} → ${labels[j]}: ${(x * 100).toFixed(1)}%`)}">${hidden ? '0' : fmt(x, 2)}</td>`;
        }
        if (hidden) return `<td class="${cls}">−∞</td>`;
        return `<td class="${cls} num${fut && s.mask && s.step !== 'masked' ? ' dim' : ''}">${fmt(x, 1)}</td>`;
      })
      .join('');
    return `<tr class="${t === s.sel ? 'sel' : ''}"><th scope="row"><button type="button" data-row="${t}" aria-pressed="${t === s.sel}">${esc(labels[t])}</button></th>${cells}</tr>`;
  }).join('');
  return head + `<tbody>${body}</tbody>`;
}

function plainMat(M, labels, cols) {
  const head = `<thead><tr><th scope="col" class="corner"></th>${M[0].map((_, k) => `<th scope="col">${esc(cols?.[k] ?? String(k))}</th>`).join('')}</tr></thead>`;
  return head + `<tbody>${M.map((row, i) => `<tr><th scope="row">${esc(labels[i])}</th>${row.map((x) => `<td class="num">${fmt(x)}</td>`).join('')}</tr>`).join('')}</tbody>`;
}

function multiHtml(s, X, labels, mult) {
  const heads = ['prev', 'same'].map((id) => {
    const r = attention(X, headWeights(id), { mask: s.mask, scale: s.scale, mult });
    return { id, label: PRESETS.find((p) => p.id === id).label, r };
  });
  const mini = (h) => {
    const body = h.r.A.map((row, t) =>
      `<tr><th scope="row">${esc(labels[t])}</th>${row
        .map((w, j) => {
          const hidden = h.r.masked[t][j] === -Infinity;
          return `<td class="${hidden ? 'masked' : ''}${!s.mask && j > t ? ' future' : ''}" style="--w:${hidden ? 0 : Math.round(w * 85)}%" title="${esc(`${labels[t]} → ${labels[j]}: ${(w * 100).toFixed(1)}%`)}"></td>`;
        })
        .join('')}</tr>`,
    ).join('');
    return `<figure class="w05-mini"><figcaption>${esc(h.label)}</figcaption><table class="w05-mat mini"><tbody>${body}</tbody></table></figure>`;
  };
  const t = s.sel;
  const concat = [...heads[0].r.O[t], ...heads[1].r.O[t]];
  return `<h4>멀티헤드 — 같은 입력, 다른 Wq·Wk·Wv</h4>
    <div class="w05-minis">${heads.map(mini).join('')}</div>
    <p class="w05-note">“${esc(labels[t])}”의 두 헤드 출력을 이어 붙이면 [${concat.map((v) => fmt(v)).join(', ')}] (8차원)이다. 실제 모델은 이것에 W<sub>O</sub>(8×4)를 곱해 다시 4차원으로 되돌린다. 한 헤드는 “바로 앞 토큰”을, 다른 헤드는 “같은 종류의 토큰”을 본다.</p>`;
}

function verdict(s, r, st, labels, d) {
  const t = s.sel;
  const top = r.A[t].reduce((bi, w, j, a) => (w > a[bi] ? j : bi), 0);
  const focus = `쿼리 “${esc(labels[t])}”는 “${esc(labels[top])}”에 ${pct(r.A[t][top], 0)}를 준다.`;
  if (!s.mask) {
    return `<div class="callout callout--danger"><span class="callout__title">🙈 미래 엿보기 = 반칙</span><p>마스크가 꺼져 각 토큰이 아직 생성되지 않은 뒤쪽 토큰을 본다. 행마다 평균 ${pct(st.future)}의 가중치가 미래(빨간 테두리 칸)로 갔다. 학습 때는 정답을 보고 맞히는 셈이라 손실은 낮아지지만, 생성할 때는 미래 토큰이 없으므로 이 모델은 쓸 수 없다.</p></div>`;
  }
  if (st.meanMax > 0.9) {
    const why = !s.scale && d > 4
      ? `√d로 나누지 않아 d = ${d}차원 내적의 크기가 그대로 커졌다.`
      : s.multLog > 0
        ? `점수 배율 ×${fmtMult(2 ** s.multLog)}로 점수 차이가 벌어졌다(온도를 낮춘 것과 같다).`
        : '점수 차이가 너무 크다.';
    return `<div class="callout callout--danger"><span class="callout__title">🔥 소프트맥스 포화 — 거의 원-핫</span><p>${why} 행마다 가장 큰 가중치가 평균 ${pct(st.meanMax)}다. 한 토큰만 보고 나머지는 무시하며, 학습 때 기울기가 거의 0이 되어 가중치가 잘 바뀌지 않는다.</p></div>`;
  }
  if (st.meanNormH > 0.97) {
    return `<div class="callout"><span class="callout__title">🌫 골고루 퍼졌다</span><p>점수 차이가 작아 모든 토큰을 거의 같은 비율로 본다(엔트로피 ${pct(st.meanNormH)}). 이러면 출력은 앞 토큰들의 단순 평균이다. 점수 배율을 올리면 어떻게 되는가?</p></div>`;
  }
  if (s.preset === 'prev') {
    return `<div class="callout callout--ok"><span class="callout__title">👈 앞 토큰 보기 헤드</span><p>각 쿼리가 바로 앞 토큰에 평균 ${pct(prevShare(r.A))}를 준다. Wq가 위치 벡터를 한 칸 뒤로 돌려 두었기 때문이다. 첫 토큰은 볼 것이 자기뿐이라 100%다. ${focus}</p></div>`;
  }
  if (s.preset === 'same') {
    return `<div class="callout callout--ok"><span class="callout__title">🧲 같은 종류 보기 헤드</span><p>Wq · Wk가 의미 두 칸만 남기므로 의미 벡터가 비슷한 토큰끼리 점수가 높다. ${focus} 임베딩을 바꿔 “${esc(labels[t])}”를 조사처럼 만들면 어디를 보게 되는가?</p></div>`;
  }
  return `<div class="callout"><span class="callout__title">🎲 무작위 헤드 · d = ${d}</span><p>학습 전의 헤드는 이런 모습이다. 특별한 규칙 없이 점수가 흩어져 있다. ${focus} 스케일링을 끄고 d를 64로 올려 본다.</p></div>`;
}

const stat = (label, value) => `<div class="stat"><span class="stat__label">${label}</span><span class="stat__value">${value}</span></div>`;

function fmt(x, digits = 2) {
  if (x === -Infinity) return '−∞';
  const v = Math.abs(x) < 0.5 * 10 ** -digits ? 0 : x;
  return v.toFixed(digits).replace('-', '−');
}

function pct(x, digits = 0) {
  return `${(x * 100).toFixed(digits)}%`;
}

function fmtMult(m) {
  return m >= 1 ? String(+m.toFixed(2)) : `1/${+(1 / m).toFixed(2)}`;
}

function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
}
