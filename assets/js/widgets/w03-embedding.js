// w03 임베딩 공간 탐색기
// One concept: a word becomes a vector, and "similar words" = vectors pointing the
// same way. One-hot vectors are all orthogonal (cosine 0); counting neighbours in a
// window gives similarity; PPMI removes the pull of frequent words; SVD squeezes the
// sparse rows into a dense k-dim embedding table (vocab × k).

import { registerOutput, unregisterOutput } from '../site/result.js';
import { loadCorpus } from '../core/text.js';
import {
  corpusWords,
  buildVocab,
  tokenizeWords,
  oneHot,
  cooccurrence,
  ppmi,
  truncatedSVD,
  cosineMatrix,
  project2D,
  row,
} from '../core/cooc.js';

const REPS = [
  { id: 'onehot', label: '① 원-핫 (단어마다 칸 하나)' },
  { id: 'count', label: '② 동시 출현 횟수' },
  { id: 'ppmi', label: '③ PPMI' },
  { id: 'svd', label: '④ PPMI + SVD (k차원 밀집 벡터)' },
];
const DIMS = [2, 5, 10, 20, 30, 50, 100];
const QUICK = ['확률', '임베딩', '검색', '청크', '모델', '실습실'];
const HEAT_WORDS = ['토큰', '확률', '온도', '벡터', '코사인', '검색', '청크', '실습실'];
const TOP_K = 8;
const LABELS = 30; // frequent words labelled on the map (plus the selected word and its neighbours)
const W = 400;
const H = 320;

const template = document.createElement('template');
template.innerHTML = `
  <div class="widget w03-emb">
    <h3 class="widget__title">임베딩 공간 탐색기</h3>
    <div data-slot="status" class="widget__status" role="status" aria-live="polite">
      <span class="spinner" aria-hidden="true"></span> <span>문서셋 불러오는 중…</span>
    </div>
    <div class="widget__controls">
      <label class="field">
        <span class="field__label">표현 방식</span>
        <select data-in="rep"></select>
      </label>
      <label class="field">
        <span class="field__label">창 크기 (앞뒤로 볼 단어 수) <output data-out="win"></output></span>
        <input type="range" data-in="win" min="1" max="5" step="1">
      </label>
      <label class="field">
        <span class="field__label">SVD 차원 k <output data-out="k"></output></span>
        <input type="range" data-in="k" min="0" max="${DIMS.length - 1}" step="1">
      </label>
      <label class="field">
        <span class="field__label">단어 (입력 후 Enter)</span>
        <input type="text" data-in="word" maxlength="20" spellcheck="false" autocomplete="off">
      </label>
    </div>
    <div class="btn-row">
      ${QUICK.map((w) => `<button type="button" class="btn small ghost" data-word="${w}">${w}</button>`).join('')}
      <span class="w03-busy" data-slot="busy" hidden><span class="spinner" aria-hidden="true"></span> 계산 중…</span>
    </div>
    <p class="w03-msg" data-slot="msg" role="status" aria-live="polite"></p>
    <div data-slot="out-inline"></div>
  </div>`;

const outTemplate = document.createElement('template');
outTemplate.innerHTML = `
  <div class="w03-out">
    <div data-slot="verdict" aria-live="polite"></div>
    <div class="stat-row" data-slot="stats"></div>
    <h4 data-slot="map-title">2D 지도</h4>
    <div class="w03-map" data-slot="map"></div>
    <div class="legend" aria-hidden="true">
      <span><i class="w03-lg-sel"></i>고른 단어</span>
      <span><i class="w03-lg-nb"></i>코사인 이웃 ${TOP_K}개</span>
      <span><i class="w03-lg-dot"></i>다른 단어</span>
    </div>
    <h4 data-slot="nb-title">가까운 단어</h4>
    <table class="w03-nb" data-slot="nb"></table>
    <p class="w03-note" data-slot="hubs"></p>
    <h4 data-slot="vec-title">이 단어의 벡터</h4>
    <div class="w03-vec" data-slot="vec"></div>
    <h4>코사인 유사도 표 <small class="w03-muted">— 비슷한 주제끼리 진한 블록이 생기는가?</small></h4>
    <div class="w03-heat-wrap"><table class="w03-heat" data-slot="heat"></table></div>
  </div>`;

const state = new WeakMap();
let seq = 0;

/**
 * @param {HTMLElement} el
 * @param {{ rep?: string, window?: number, k?: number, word?: string, outputTitle?: string }} options
 */
export async function mount(el, options = {}) {
  const root = template.content.firstElementChild.cloneNode(true);
  const out = outTemplate.content.firstElementChild.cloneNode(true);
  el.replaceChildren(root);
  const $ = (sel) => root.querySelector(sel) ?? out.querySelector(sel);
  const ctrl = new AbortController();
  const outputId = `w03-embedding:${++seq}`;
  state.set(el, { ctrl, outputId });

  const s = {
    rep: options.rep ?? 'svd',
    win: options.window ?? 3,
    kIdx: Math.max(0, DIMS.indexOf(options.k ?? 30)),
    word: options.word ?? '확률',
    docs: null,
    vocab: null,
    cache: new Map(),
    job: 0,
  };

  $('[data-in=rep]').innerHTML = REPS.map((r) => `<option value="${r.id}">${r.label}</option>`).join('');
  $('[data-in=rep]').value = s.rep;
  $('[data-in=win]').value = String(s.win);
  $('[data-in=k]').value = String(s.kIdx);
  $('[data-in=word]').value = s.word;

  registerOutput(outputId, { title: options.outputTitle ?? '임베딩 공간 탐색기 결과', node: out, inlineHost: $('[data-slot=out-inline]') });

  // ---- cached computation: counts per window, PPMI per window, SVD per (window, k)
  function memo(key, fn) {
    if (!s.cache.has(key)) s.cache.set(key, fn());
    return s.cache.get(key);
  }
  const counts = (w) => memo(`count|${w}`, () => cooccurrence(s.docs, s.vocab, w));
  const ppmiM = (w) => memo(`ppmi|${w}`, () => ppmi(counts(w)));
  function matrixFor(rep, w, k) {
    if (rep === 'onehot') return memo('onehot', () => oneHot(s.vocab.size));
    if (rep === 'count') return counts(w);
    if (rep === 'ppmi') return ppmiM(w);
    return memo(`svd|${w}|${k}`, () => truncatedSVD(ppmiM(w), k).vectors);
  }
  function view(rep, w, k) {
    const key = rep === 'onehot' ? 'view|onehot' : `view|${rep}|${w}|${rep === 'svd' ? k : ''}`;
    return memo(key, () => {
      const M = matrixFor(rep, w, k);
      const S = cosineMatrix(M);
      const n = M.rows;
      let nz = 0;
      for (let i = 0; i < M.data.length; i++) if (M.data[i] !== 0) nz++;
      let sum = 0;
      for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) sum += S.data[i * n + j];
      const nbs = [];
      const hubs = new Float64Array(n);
      for (let i = 0; i < n; i++) {
        const list = neighbours(S, i, TOP_K);
        nbs.push(list);
        list.forEach((j) => hubs[j]++);
      }
      const proj = rep === 'onehot' ? circle(n) : project2D(M);
      return { M, S, nbs, hubs, density: nz / M.data.length, meanCos: sum / ((n * (n - 1)) / 2), proj, dims: M.cols };
    });
  }

  async function update() {
    if (!s.vocab) return;
    const k = DIMS[s.kIdx];
    $('[data-out=win]').textContent = `±${s.win}`;
    $('[data-out=k]').textContent = s.rep === 'svd' ? `${k}` : `— (${s.rep === 'onehot' ? '원-핫' : 'SVD 아님'})`;
    $('[data-in=k]').disabled = s.rep !== 'svd';
    $('[data-in=win]').disabled = s.rep === 'onehot';
    root.querySelectorAll('.btn-row [data-word]').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.word === s.word)));

    const key = s.rep === 'onehot' ? 'view|onehot' : `view|${s.rep}|${s.win}|${s.rep === 'svd' ? k : ''}`;
    const job = ++s.job;
    if (!s.cache.has(key)) {
      $('[data-slot=busy]').hidden = false;
      await new Promise((r) => setTimeout(r, 20)); // let the spinner paint before the heavy work
      if (job !== s.job || ctrl.signal.aborted) return;
    }
    try {
      const v = view(s.rep, s.win, k);
      render(v, k);
    } catch (err) {
      $('[data-slot=verdict]').innerHTML = `<div class="widget__error">계산 중 오류가 났다 (${esc(err.message)}).</div>`;
    } finally {
      if (job === s.job) $('[data-slot=busy]').hidden = true;
    }
  }

  function render(v, k) {
    const { words, counts: freq, index } = s.vocab;
    const i = index.get(s.word);
    const nb = v.nbs[i];
    const sims = nb.map((j) => v.S.data[i * words.length + j]);
    const repName = REPS.find((r) => r.id === s.rep).label.replace(/^\S+ /, '');

    // stats
    $('[data-slot=stats]').innerHTML = [
      stat('어휘 (빈도 ≥ 2)', words.length.toLocaleString()),
      stat('벡터 차원', v.dims.toLocaleString()),
      stat('0이 아닌 칸', fmtPct(v.density)),
      stat('평균 코사인 (모든 쌍)', v.meanCos.toFixed(3)),
      stat('2D에 담긴 정보', s.rep === 'onehot' ? '—' : fmtPct(v.proj.explained)),
    ].join('');

    // verdict
    $('[data-slot=verdict]').innerHTML = verdict({ s, v, i, nb, sims, k, words, freq });

    // map
    $('[data-slot=map-title]').innerHTML = `2D 지도 · ${esc(repName)} <small class="w03-muted">— ${
      s.rep === 'onehot' ? '원-핫은 모든 단어가 서로 같은 거리라 원 위에 늘어놓았다' : '단위 길이로 맞춘 벡터를 주성분 2개에 투영 (PCA)'
    }</small>`;
    $('[data-slot=map]').innerHTML = scatter(v.proj.points, i, nb, sims, words, freq);

    // neighbour table
    $('[data-slot=nb-title]').innerHTML = `“${esc(s.word)}” 기준 코사인 유사도가 높은 단어 ${TOP_K}개 <small class="w03-muted">— 단어를 누르면 그 단어로 바뀐다</small>`;
    $('[data-slot=nb]').innerHTML =
      `<thead><tr><th scope="col">#</th><th scope="col">단어</th><th scope="col">코사인</th><th scope="col">빈도</th></tr></thead><tbody>` +
      nb
        .map((j, r) => {
          const sim = sims[r];
          const w = Math.max(0, Math.min(1, sim)) * 100;
          const rare = freq[j] <= 2 ? ' <span class="w03-rare" title="코퍼스에 두 번만 나온 단어 — 근거가 약하다">⚠ 드묾</span>' : '';
          return `<tr><td>${r + 1}</td><td><button type="button" class="w03-link" data-word="${esc(words[j])}">${esc(words[j])}</button>${rare}</td><td><div class="w03-simcell"><span class="w03-simbar"><i style="width:${w.toFixed(1)}%"></i></span><span class="w03-num">${sim.toFixed(2)}</span></div></td><td class="w03-num">${freq[j]}</td></tr>`;
        })
        .join('') +
      '</tbody>';

    // hubs
    if (s.rep === 'onehot') {
      $('[data-slot=hubs]').textContent = '유사도가 전부 0이라 순위를 매길 수 없다. 위 표는 빈도가 높은 순서로 늘어놓았을 뿐이다.';
    } else {
      const top = [...v.hubs.keys()].sort((a, b) => v.hubs[b] - v.hubs[a]).slice(0, 4);
      $('[data-slot=hubs]').innerHTML = `이웃 목록 단골(허브): ${top
        .map((j) => `<b>${esc(words[j])}</b> ${v.hubs[j]}회`)
        .join(' · ')} <span class="w03-muted">— ${words.length}개 단어의 이웃 ${TOP_K}개를 모두 모아 센 값</span>`;
    }

    // the vector itself
    $('[data-slot=vec-title]').innerHTML = `“${esc(s.word)}”의 벡터 <small class="w03-muted">— ${v.dims}개의 숫자</small>`;
    $('[data-slot=vec]').innerHTML = vectorView(s.rep, row(v.M, i), words, freq, i);

    // heatmap
    const ids = HEAT_WORDS.map((w) => index.get(w)).filter((x) => x !== undefined);
    $('[data-slot=heat]').innerHTML = heatmap(v.S, ids, words, i);
  }

  function selectWord(raw) {
    const msg = $('[data-slot=msg]');
    const typed = String(raw).trim();
    if (!typed) return;
    const w = s.vocab.index.has(typed.toLowerCase()) ? typed.toLowerCase() : tokenizeWords(typed)[0];
    if (!w || !s.vocab.index.has(w)) {
      msg.innerHTML = `어휘에 없는 단어다: “${esc(typed)}”. 코퍼스에 두 번 이상 나온 단어만 벡터가 있다 (조사를 뗀 형태로 입력한다: “모델은” → “모델”).`;
      return;
    }
    msg.textContent = w !== typed ? `“${typed}” → 조사를 떼어 “${w}”로 찾았다.` : '';
    s.word = w;
    $('[data-in=word]').value = w;
    update();
  }

  const on = (sel, type, fn) => $(sel).addEventListener(type, fn, { signal: ctrl.signal });
  on('[data-in=rep]', 'change', (e) => {
    s.rep = e.target.value;
    update();
  });
  on('[data-in=win]', 'input', (e) => {
    s.win = Number(e.target.value);
    update();
  });
  on('[data-in=k]', 'input', (e) => {
    s.kIdx = Number(e.target.value);
    update();
  });
  on('[data-in=word]', 'keydown', (e) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      selectWord(e.target.value);
    }
  });
  on('[data-in=word]', 'change', (e) => selectWord(e.target.value));
  const pick = (e) => {
    const b = e.target.closest('[data-word]');
    if (b && s.vocab) selectWord(b.dataset.word);
  };
  // the output node sits inside root on narrow screens: let only one listener handle it
  root.addEventListener('click', (e) => !out.contains(e.target) && pick(e), { signal: ctrl.signal });
  out.addEventListener('click', pick, { signal: ctrl.signal });

  try {
    const corpus = await loadCorpus();
    if (ctrl.signal.aborted) return;
    s.docs = corpusWords(corpus);
    s.vocab = buildVocab(s.docs, 2);
    if (!s.vocab.index.has(s.word)) s.word = s.vocab.words[0];
    $('[data-slot=status]').hidden = true;
    await update();
  } catch (err) {
    $('[data-slot=status]').innerHTML = `<div class="widget__error">문서셋을 불러오지 못했다 (${esc(err.message)}). <code>file://</code>로 열었다면 <code>python -m http.server</code>로 연다.</div>`;
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

// ---------------------------------------------------------------- helpers

/** Top-k most similar rows by a precomputed cosine matrix; ties keep frequency order. */
function neighbours(S, i, k) {
  const n = S.rows;
  const ids = [];
  for (let j = 0; j < n; j++) if (j !== i) ids.push(j);
  ids.sort((a, b) => S.data[i * n + b] - S.data[i * n + a] || a - b);
  return ids.slice(0, k);
}

function circle(n) {
  const points = [];
  for (let i = 0; i < n; i++) {
    const t = (2 * Math.PI * i) / n - Math.PI / 2;
    points.push([Math.cos(t), Math.sin(t)]);
  }
  return { points, explained: 0 };
}

function verdict({ s, v, i, nb, sims, k, words, freq }) {
  const rareCount = nb.filter((j) => freq[j] <= 2).length;
  const rareNote = rareCount
    ? `<p class="w03-muted">⚠ 이웃 ${TOP_K}개 중 ${rareCount}개는 코퍼스에 두 번만 나온 단어다. 코퍼스가 ${s.vocab.totalTokens.toLocaleString()}단어뿐이라 한두 번 겹친 것만으로 이웃이 된다. 엉뚱한 이웃이 섞이는 것은 버그가 아니라 데이터 부족이다.</p>`
    : '';
  if (s.rep === 'onehot') {
    return `<div class="callout callout--danger"><span class="callout__title">모든 이웃의 유사도가 0이다</span><p>원-핫 벡터는 ${words.length}칸 중 자기 칸 하나만 1이다. 두 단어가 겹치는 칸이 없으니 내적이 0이고 코사인도 0이다. “확률”과 “온도”도, “확률”과 “실습실”도 똑같이 무관하다. 1주차 n-gram이 “비슷한 문맥”을 몰랐던 것과 같은 이유다.</p></div>`;
  }
  if (s.rep === 'count') {
    const r = row(v.M, i);
    let best = -1;
    for (let j = 0; j < r.length; j++) if (best < 0 || r[j] > r[best]) best = j;
    return `<div class="callout"><span class="callout__title">이웃이 생겼다 · 하지만 흔한 단어가 벡터를 지배한다</span><p>“${esc(words[i])}” 벡터에서 가장 큰 칸은 “${esc(words[best])}”(${r[best]}회)이다. “${esc(words[best])}”는 코퍼스에 ${freq[best]}번 나온다. 횟수를 그대로 쓰면 “흔한 단어 옆에 있었다”는 사실이 벡터를 차지해, 흔한 단어 근처의 단어끼리 모두 비슷해진다. 허브 목록과 평균 코사인(${v.meanCos.toFixed(3)})을 PPMI와 비교한다.</p></div>${rareNote}`;
  }
  if (s.rep === 'ppmi') {
    return `<div class="callout callout--ok"><span class="callout__title">“우연보다 자주” 만난 단어만 남겼다</span><p>PPMI는 빈도로 예상되는 것보다 더 자주 함께 나온 쌍에만 값을 준다. 흔한 단어의 힘이 빠지고 주제가 드러난다. 대신 벡터는 ${v.dims}차원이고 ${fmtPct(v.density)}만 0이 아니다(희소). 어휘가 5만 개면 벡터도 5만 차원이 된다.</p></div>${rareNote}`;
  }
  const avg = sims.reduce((a, b) => a + b, 0) / sims.length;
  if (avg >= 0.95) {
    return `<div class="callout callout--danger"><span class="callout__title">차원이 너무 적다 · 모두가 이웃이다</span><p>k = ${k}차원에 ${words.length}개 단어를 밀어 넣으니 이웃 ${TOP_K}개의 평균 코사인이 ${avg.toFixed(2)}다. 방향이 몇 개뿐이라 전혀 다른 단어도 같은 방향을 가리킨다. k를 20~30으로 올려 본다.</p></div>`;
  }
  return `<div class="callout callout--ok"><span class="callout__title">${k}차원 밀집 벡터 = 임베딩 표의 한 행</span><p>SVD로 ${words.length}차원 PPMI 행을 ${k}개의 숫자로 압축했다. 자주 함께 움직이는 차원이 합쳐져, 직접 만난 적이 없어도 문맥이 비슷하면 가까워진다. 이 ${words.length} × ${k} 표가 LLM 첫 층의 임베딩 표와 같은 모양이다. LLM은 이 표를 세지 않고 역전파로 학습한다.</p></div>${rareNote}`;
}

function scatter(points, sel, nb, sims, words, freq) {
  const pad = 26;
  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  for (const [x, y] of points) {
    minX = Math.min(minX, x);
    maxX = Math.max(maxX, x);
    minY = Math.min(minY, y);
    maxY = Math.max(maxY, y);
  }
  const scale = Math.min((W - 2 * pad) / (maxX - minX || 1), (H - 2 * pad) / (maxY - minY || 1));
  const ox = (W - (maxX - minX) * scale) / 2;
  const oy = (H - (maxY - minY) * scale) / 2;
  const P = points.map(([x, y]) => [ox + (x - minX) * scale, H - (oy + (y - minY) * scale)]);

  const nbSet = new Set(nb);
  const parts = [];
  // links: selected → neighbours
  nb.forEach((j, r) => {
    const op = 0.25 + 0.75 * Math.max(0, Math.min(1, sims[r]));
    parts.push(`<line class="w03-link-line" x1="${f(P[sel][0])}" y1="${f(P[sel][1])}" x2="${f(P[j][0])}" y2="${f(P[j][1])}" stroke-opacity="${op.toFixed(2)}"/>`);
  });
  // all dots
  P.forEach(([x, y], j) => {
    if (j === sel || nbSet.has(j)) return;
    parts.push(`<circle class="w03-dot" cx="${f(x)}" cy="${f(y)}" r="2.2" data-word="${esc(words[j])}"><title>${esc(words[j])} (빈도 ${freq[j]})</title></circle>`);
  });
  // labels: selected, neighbours, then frequent words — skip if overlapping
  const order = [sel, ...nb, ...Array.from({ length: Math.min(LABELS, words.length) }, (_, j) => j).filter((j) => j !== sel && !nbSet.has(j))];
  const boxes = [];
  for (const j of order) {
    const [x, y] = P[j];
    const kind = j === sel ? 'sel' : nbSet.has(j) ? 'nb' : 'fq';
    const fs = kind === 'sel' ? 13 : 11;
    const wpx = textWidth(words[j], fs);
    const box = [x + 5, y - fs + 2, x + 5 + wpx, y + 3];
    if (box[2] > W) {
      box[0] = x - 5 - wpx;
      box[2] = x - 5;
    }
    const clash = boxes.some((b) => !(box[2] < b[0] || box[0] > b[2] || box[3] < b[1] || box[1] > b[3]));
    const dot = `<circle class="w03-dot ${kind}" cx="${f(x)}" cy="${f(y)}" r="${kind === 'sel' ? 5 : kind === 'nb' ? 3.6 : 2.6}"/>`;
    if (clash && kind === 'fq') {
      parts.push(`<g data-word="${esc(words[j])}" class="w03-pt">${dot}<title>${esc(words[j])} (빈도 ${freq[j]})</title></g>`);
      continue;
    }
    boxes.push(box);
    const anchor = box[0] < x ? 'end' : 'start';
    const tx = anchor === 'end' ? x - 5 : x + 5;
    // selected word and neighbours are always labelled, even if they overlap
    const label = `<text class="w03-lab ${kind}" x="${f(tx)}" y="${f(y + 4)}" text-anchor="${anchor}">${esc(words[j])}</text>`;
    parts.push(`<g data-word="${esc(words[j])}" class="w03-pt">${dot}${label}<title>${esc(words[j])} (빈도 ${freq[j]})</title></g>`);
  }
  return `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="단어 벡터의 2차원 지도. 고른 단어와 이웃이 선으로 이어져 있다.">${parts.join('')}</svg>`;
}

function textWidth(w, fs) {
  let u = 0;
  for (const ch of w) u += /[가-힣]/.test(ch) ? 1 : 0.62;
  return u * fs;
}

function vectorView(rep, vec, words, freq, i) {
  const n = vec.length;
  if (rep === 'onehot') {
    const lo = Math.max(0, i - 4);
    const hi = Math.min(n, lo + 9);
    const cells = [];
    for (let j = lo; j < hi; j++) cells.push(`<span class="w03-cell${j === i ? ' on' : ''}" title="${j}번 칸 = ${esc(words[j])}">${vec[j]}</span>`);
    return `<p class="w03-mono">[${lo > 0 ? '0, …, ' : ''}${cells.join(', ')}${hi < n ? ', …' : ''}]</p><p class="w03-muted">${n}칸 중 ${i}번 칸만 1이고 나머지 ${n - 1}칸은 0이다. 다른 단어와 1이 겹치는 칸이 하나도 없다.</p>`;
  }
  if (rep === 'svd') {
    const max = Math.max(...Array.from(vec, Math.abs)) || 1;
    const cells = Array.from(vec, (x, d) => {
      const w = Math.round((Math.abs(x) / max) * 80);
      return `<span class="w03-comp ${x < 0 ? 'neg' : 'pos'}" style="--w:${w}%" title="${d + 1}번째 차원 = ${x.toFixed(3)}"></span>`;
    }).join('');
    const head = Array.from(vec.slice(0, 6), (x) => x.toFixed(2)).join(', ');
    return `<div class="w03-strip" style="--n:${n}">${cells}</div><p class="w03-mono">[${head}${n > 6 ? ', …' : ''}]</p><p class="w03-muted">모든 칸에 값이 있다(밀집). 파랑은 양수, 빨강은 음수다. 각 차원에는 이름이 없다. “몇 번째 차원 = 무슨 뜻”으로 읽을 수 없고, 이웃과의 방향 관계로만 의미를 읽는다.</p>`;
  }
  // count / ppmi: dimensions are context words — show the largest ones
  const order = Array.from(vec.keys())
    .filter((j) => vec[j] > 0)
    .sort((a, b) => vec[b] - vec[a]);
  const top = order.slice(0, 8);
  const max = vec[top[0]] || 1;
  const bars = top
    .map(
      (j) =>
        `<div class="w03-bar"><span class="t">${esc(words[j])}</span><span class="b"><i style="width:${((vec[j] / max) * 100).toFixed(1)}%"></i></span><span class="v">${rep === 'count' ? vec[j] : vec[j].toFixed(2)} <small>(빈도 ${freq[j]})</small></span></div>`,
    )
    .join('');
  return `${bars}<p class="w03-muted">차원 하나 = 문맥 단어 하나. ${n}칸 중 ${order.length}칸만 0이 아니다. ${
    rep === 'count' ? '큰 칸은 대개 코퍼스 전체에서 흔한 단어다.' : '흔한 단어는 내려가고, 이 단어 곁에만 유독 자주 나온 단어가 올라왔다.'
  }</p>`;
}

function heatmap(S, ids, words, sel) {
  const n = S.rows;
  const head = `<thead><tr><th scope="col" class="corner">코사인</th>${ids.map((j) => `<th scope="col"${j === sel ? ' class="sel"' : ''}>${esc(words[j])}</th>`).join('')}</tr></thead>`;
  const body = ids
    .map((a) => {
      const cells = ids
        .map((b) => {
          const v = S.data[a * n + b];
          const pct = Math.round(Math.min(1, Math.abs(v)) * 60);
          return `<td class="${v < 0 ? 'neg' : ''}" style="--w:${pct}%" title="${esc(`cos(${words[a]}, ${words[b]}) = ${v.toFixed(3)}`)}">${v.toFixed(2).replace(/^(-?)0\./, '$1.')}</td>`;
        })
        .join('');
      return `<tr><th scope="row"${a === sel ? ' class="sel"' : ''}>${esc(words[a])}</th>${cells}</tr>`;
    })
    .join('');
  return head + `<tbody>${body}</tbody>`;
}

const f = (x) => x.toFixed(1);

const stat = (label, value) => `<div class="stat"><span class="stat__label">${label}</span><span class="stat__value">${value}</span></div>`;

function fmtPct(x) {
  if (x === 0) return '0%';
  if (x < 0.001) return `${(x * 100).toExponential(1)}%`;
  return `${(x * 100).toFixed(x < 0.1 ? 1 : 0)}%`;
}

function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
}
