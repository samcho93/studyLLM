// w01 n-gram 언어 모델
// One concept: a language model is "count what came next, turn counts into
// probabilities, sample". n = 2 babbles; large n copies the source verbatim.

import { registerOutput, unregisterOutput } from '../site/result.js';
import { loadCorpus, corpusText, showWs } from '../core/text.js';
import { countNgrams, nextDist, generate, BOS } from '../core/ngram.js';
import { mulberry32 } from '../core/rng.js';

const SOURCES = [
  { id: 'all', label: '코퍼스 전체 (17문서 · 약 6,800자)' },
  { id: 'llm', label: 'LLM 교안만 (8문서)' },
  { id: 'facility-rules', label: '실습실 이용 규정 한 편 (284자)' },
];
const COPY_MIN = 10; // a run this long found verbatim in the source counts as "copied"
const HEAT_N = 12;

const template = document.createElement('template');
template.innerHTML = `
  <div class="widget w01-ng">
    <h3 class="widget__title">n-gram 언어 모델</h3>
    <div data-slot="status" class="widget__status" role="status" aria-live="polite">
      <span class="spinner" aria-hidden="true"></span> <span>문서셋 불러오는 중…</span>
    </div>
    <div class="widget__controls">
      <label class="field">
        <span class="field__label">학습 문서</span>
        <select data-in="source"></select>
      </label>
      <label class="field">
        <span class="field__label">n (앞 글자 n−1개를 본다) <output data-out="n"></output></span>
        <input type="range" data-in="n" min="1" max="6" step="1">
      </label>
      <label class="field">
        <span class="field__label">시작 글자</span>
        <input type="text" data-in="start" maxlength="20" spellcheck="false">
      </label>
      <label class="field">
        <span class="field__label">생성 길이 <output data-out="len"></output></span>
        <input type="range" data-in="len" min="20" max="160" step="10">
      </label>
    </div>
    <div class="btn-row">
      <button type="button" class="btn small primary" data-act="again">🎲 다시 뽑기</button>
      <button type="button" class="btn small ghost" data-act="n2">n = 2 (바이그램)</button>
      <button type="button" class="btn small ghost" data-act="n5">n = 5 (외우기 재현)</button>
      <span class="w01-seed" data-out="seed"></span>
    </div>
    <div data-slot="out-inline"></div>
  </div>`;

const outTemplate = document.createElement('template');
outTemplate.innerHTML = `
  <div class="w01-ng-out">
    <div data-slot="verdict" aria-live="polite"></div>
    <h4>생성된 글 <small class="w01-muted">— 글자를 누르면 그 순간의 후보 확률이 아래에 나온다</small></h4>
    <div class="w01-gen" data-slot="gen" tabindex="0"></div>
    <div class="legend" aria-hidden="true">
      <span><i class="w01-lg-prompt"></i>시작 글자</span>
      <span><i class="w01-lg-hi"></i>확률 ≥ 50%</span>
      <span><i class="w01-lg-mid"></i>10~50%</span>
      <span><i class="w01-lg-lo"></i>&lt; 10%</span>
      <span><i class="w01-lg-copy"></i>원문 그대로 ${COPY_MIN}자 이상</span>
    </div>
    <div class="stat-row" data-slot="stats"></div>
    <h4 data-slot="dist-title">다음 글자 후보</h4>
    <div class="w01-dist" data-slot="dist"></div>
    <div data-slot="heat-wrap">
      <h4>바이그램 빈도표 (자주 나오는 글자 ${HEAT_N}개) <small class="w01-muted">— 행: 앞 글자 · 열: 다음 글자 · 진할수록 P(다음 | 앞)이 크다</small></h4>
      <div class="w01-heat-wrap"><table class="w01-heat" data-slot="heat"></table></div>
    </div>
  </div>`;

const state = new WeakMap();
let seq = 0;

/**
 * @param {HTMLElement} el
 * @param {{ n?: number, source?: string, start?: string, outputTitle?: string }} options
 */
export async function mount(el, options = {}) {
  const root = template.content.firstElementChild.cloneNode(true);
  const out = outTemplate.content.firstElementChild.cloneNode(true);
  el.replaceChildren(root);
  const $ = (sel) => root.querySelector(sel) ?? out.querySelector(sel);
  const ctrl = new AbortController();
  const outputId = `w01-bigram:${++seq}`;
  state.set(el, { ctrl, outputId });

  const s = {
    source: options.source ?? 'all',
    n: options.n ?? 2,
    start: options.start ?? '실습실은',
    len: 80,
    seed: 1,
    pick: 0, // which generation step's distribution to show
    corpus: null,
    model: null,
    modelKey: '',
    text: '',
  };

  $('[data-in=source]').innerHTML = SOURCES.map((o) => `<option value="${o.id}">${o.label}</option>`).join('');
  $('[data-in=source]').value = s.source;
  $('[data-in=n]').value = String(s.n);
  $('[data-in=start]').value = s.start;
  $('[data-in=len]').value = String(s.len);

  registerOutput(outputId, { title: options.outputTitle ?? 'n-gram 언어 모델 결과', node: out, inlineHost: $('[data-slot=out-inline]') });

  function sourceText() {
    const docs = s.corpus.documents;
    if (s.source === 'all') return corpusText(s.corpus);
    if (s.source === 'llm') return corpusText(s.corpus, docs.filter((d) => d.course === 'studyLLM').map((d) => d.id));
    return corpusText(s.corpus, [s.source]);
  }

  function ensureModel() {
    const key = `${s.source}|${s.n}`;
    if (key === s.modelKey) return;
    s.text = sourceText();
    s.model = countNgrams([...s.text], s.n);
    s.modelKey = key;
  }

  function render() {
    if (!s.corpus) return;
    ensureModel();
    $('[data-out=n]').textContent = `${s.n} · ${nName(s.n)}`;
    $('[data-out=len]').textContent = `${s.len}자`;
    $('[data-out=seed]').textContent = `seed ${s.seed}`;

    const startToks = [...s.start];
    const gen = generate(s.model, { start: startToks, length: s.len, rand: mulberry32(s.seed) });
    const genText = gen.tokens.slice(startToks.length).join('');
    const copied = copyMask(genText, s.text);
    const copyRate = genText.length ? copied.filter(Boolean).length / genText.length : 0;
    const deadEnd = gen.steps.length < s.len;
    s.pick = Math.min(s.pick, Math.max(0, gen.steps.length - 1));

    // generated text, one span per character
    const pieces = [`<span class="w01-prompt">${esc(showWs(s.start))}</span>`];
    gen.steps.forEach((st, i) => {
      const cls = st.p >= 0.5 ? 'hi' : st.p >= 0.1 ? 'mid' : 'lo';
      const ch = st.chosen === '\n' ? '↵\n' : st.chosen;
      pieces.push(
        `<button type="button" class="w01-ch ${cls}${copied[i] ? ' copy' : ''}${i === s.pick ? ' sel' : ''}" data-step="${i}" title="${esc(`${i + 1}번째: P=${(st.p * 100).toFixed(1)}% · 후보 ${st.options}개`)}">${esc(ch)}</button>`,
      );
    });
    if (deadEnd) pieces.push(`<span class="w01-dead" title="이 문맥 다음에 온 글자가 학습 문서에 없다">⛔ 막다른 길</span>`);
    $('[data-slot=gen]').innerHTML = pieces.join('');

    // stats
    const V = s.model.vocab.length;
    const contexts = s.model.counts.size;
    let filled = 0;
    s.model.counts.forEach((row) => (filled += row.size));
    const possible = Math.pow(V, s.n - 1) * V;
    const avgOptions = gen.steps.length ? gen.steps.reduce((a, st) => a + st.options, 0) / gen.steps.length : 0;
    $('[data-slot=stats]').innerHTML = [
      stat('어휘(글자 종류)', V.toLocaleString()),
      stat('본 문맥 수', contexts.toLocaleString()),
      stat('채워진 칸', `${fmtPct(filled / possible)}`),
      stat('평균 후보 수', avgOptions.toFixed(1)),
      stat('원문 복사율', fmtPct(copyRate)),
    ].join('');

    // verdict
    $('[data-slot=verdict]').innerHTML = verdict({ n: s.n, copyRate, deadEnd, avgOptions, start: s.start, genLen: gen.steps.length });

    // next-token distribution at the picked step
    const st = gen.steps[s.pick];
    const ctx = st ? gen.tokens.slice(0, startToks.length + s.pick) : startToks;
    const dist = nextDist(s.model, ctx);
    const ctxShown = s.n === 1 ? '(문맥 없음 — 앞 글자를 보지 않는다)' : `“${showWs(ctx.slice(-(s.n - 1)).join('')).replaceAll(BOS, '⟨s⟩')}” 다음`;
    $('[data-slot=dist-title]').innerHTML = `다음 글자 후보 · ${st ? `${s.pick + 1}번째 글자를 고를 때` : '처음'} · ${esc(ctxShown)}`;
    const total = dist.reduce((a, d) => a + d.count, 0);
    $('[data-slot=dist]').innerHTML = dist.length
      ? dist
          .slice(0, 10)
          .map(
            (d) => `<div class="w01-bar${st && d.token === st.chosen ? ' chosen' : ''}"><span class="t">${esc(showWs(d.token))}</span><span class="b"><i style="width:${(d.p * 100).toFixed(1)}%"></i></span><span class="v">${d.count}/${total} · ${(d.p * 100).toFixed(1)}%</span></div>`,
          )
          .join('') + (dist.length > 10 ? `<p class="w01-muted">… 외 ${dist.length - 10}개 후보</p>` : '')
      : '<p class="w01-muted">이 문맥은 학습 문서에 한 번도 나오지 않았다. 빈도표의 이 행은 전부 0이다.</p>';

    // bigram heatmap (n = 2 only)
    $('[data-slot=heat-wrap]').hidden = s.n !== 2;
    if (s.n === 2) $('[data-slot=heat]').innerHTML = heatmap(s.model, s.text);
  }

  const on = (sel, type, fn) => $(sel).addEventListener(type, fn, { signal: ctrl.signal });
  on('[data-in=source]', 'change', (e) => {
    s.source = e.target.value;
    s.pick = 0;
    render();
  });
  on('[data-in=n]', 'input', (e) => {
    s.n = Number(e.target.value);
    s.pick = 0;
    render();
  });
  on('[data-in=start]', 'input', (e) => {
    s.start = e.target.value;
    s.pick = 0;
    render();
  });
  on('[data-in=len]', 'input', (e) => {
    s.len = Number(e.target.value);
    render();
  });
  root.addEventListener(
    'click',
    (e) => {
      const act = e.target.closest('[data-act]')?.dataset.act;
      if (act === 'again') s.seed++;
      if (act === 'n2' || act === 'n5') {
        s.n = act === 'n2' ? 2 : 5;
        $('[data-in=n]').value = String(s.n);
      }
      if (act) {
        s.pick = 0;
        render();
      }
    },
    { signal: ctrl.signal },
  );
  out.addEventListener(
    'click',
    (e) => {
      const b = e.target.closest('[data-step]');
      if (!b) return;
      s.pick = Number(b.dataset.step);
      render();
    },
    { signal: ctrl.signal },
  );

  try {
    s.corpus = await loadCorpus();
    if (ctrl.signal.aborted) return;
    $('[data-slot=status]').hidden = true;
    render();
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

const nName = (n) => ['', '유니그램', '바이그램', '트라이그램', '4-gram', '5-gram', '6-gram'][n] ?? `${n}-gram`;

/** For each generated character: is it inside a run of ≥ COPY_MIN chars found verbatim in the source? */
function copyMask(gen, source) {
  const mask = new Array(gen.length).fill(false);
  let i = 0;
  while (i < gen.length) {
    let L = 0;
    while (i + L < gen.length && source.includes(gen.slice(i, i + L + 1))) L++;
    if (L >= COPY_MIN) {
      for (let j = i; j < i + L; j++) mask[j] = true;
      i += L;
    } else i++;
  }
  return mask;
}

function heatmap(model, text) {
  const freq = new Map();
  for (const c of text) if (!/\s/.test(c)) freq.set(c, (freq.get(c) ?? 0) + 1);
  const top = [...freq].sort((a, b) => b[1] - a[1]).slice(0, HEAT_N).map(([c]) => c);
  const head = `<thead><tr><th scope="col" class="corner">앞＼다음</th>${top.map((c) => `<th scope="col">${esc(c)}</th>`).join('')}</tr></thead>`;
  const rows = top
    .map((a) => {
      const row = model.counts.get(a) ?? new Map();
      let total = 0;
      row.forEach((c) => (total += c));
      const cells = top
        .map((b) => {
          const c = row.get(b) ?? 0;
          const p = total ? c / total : 0;
          const pct = Math.round(Math.min(1, p * 3) * 70); // stretch: most bigram probs are small
          return `<td style="--w:${pct}%" title="${esc(`P(${b} | ${a}) = ${c}/${total} = ${(p * 100).toFixed(1)}%`)}">${c || ''}</td>`;
        })
        .join('');
      return `<tr><th scope="row">${esc(a)}</th>${cells}</tr>`;
    })
    .join('');
  return head + `<tbody>${rows}</tbody>`;
}

function verdict({ n, copyRate, deadEnd, avgOptions, start, genLen }) {
  if (deadEnd && genLen === 0) {
    return `<div class="callout callout--danger"><span class="callout__title">⛔ 시작하자마자 막혔다</span><p>“${esc(start.slice(-(n - 1)) || start)}” 다음에 온 글자가 학습 문서에 한 번도 없다. 빈도표에 없는 문맥에서 n-gram 모델은 아무것도 말할 수 없다.</p></div>`;
  }
  if (n === 1) {
    return `<div class="callout callout--danger"><span class="callout__title">글자 수프</span><p>유니그램은 앞 글자를 보지 않고 “자주 나오는 글자”만 뽑는다. 글자 빈도는 맞지만 단어도 문장도 없다.</p></div>`;
  }
  if (copyRate >= 0.5) {
    return `<div class="callout callout--danger"><span class="callout__title">원문 베끼기 · 복사율 ${fmtPct(copyRate)}</span><p>앞 ${n - 1}글자가 같은 문맥이 학습 문서에 거의 하나뿐이라 후보가 평균 ${avgOptions.toFixed(1)}개밖에 없다. 모델이 “언어”를 배운 것이 아니라 문서를 외웠다. 새 문장을 만들 수 없다.</p></div>`;
  }
  if (deadEnd) {
    return `<div class="callout callout--danger"><span class="callout__title">⛔ 중간에 막혔다</span><p>생성 도중 학습 문서에 없던 ${n - 1}글자 문맥에 들어섰다. n이 클수록 이런 빈칸이 많다.</p></div>`;
  }
  if (n === 2) {
    return `<div class="callout"><span class="callout__title">단어 조각은 맞고 문장은 틀린다</span><p>바로 앞 글자 하나만 보므로 “실습”, “모델” 같은 짧은 조각은 자연스럽지만, 세 글자만 지나도 앞 내용을 잊어 문장이 엉뚱하게 흘러간다.</p></div>`;
  }
  return `<div class="callout callout--ok"><span class="callout__title">그럴듯해졌다 · 복사율 ${fmtPct(copyRate)}</span><p>앞 ${n - 1}글자를 보니 단어가 자연스러워졌다. 복사율을 확인한다. n을 한 칸 더 올리면 어떻게 될까?</p></div>`;
}

const stat = (label, value) => `<div class="stat"><span class="stat__label">${label}</span><span class="stat__value">${value}</span></div>`;

function fmtPct(x) {
  if (x === 0) return '0%';
  if (x < 0.001) return `${(x * 100).toExponential(1)}%`;
  return `${(x * 100).toFixed(x < 0.1 ? 1 : 0)}%`;
}

function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
}
