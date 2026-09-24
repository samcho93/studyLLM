// w09 퍼플렉서티 측정기
// One concept: a language model is measured by how surprised it is by real text.
// loss = mean −log p(next char) · perplexity = e^loss ("how many candidates it is
// torn between on average") · bits/char = loss / ln 2.
// Failure modes: the memorised model has a tiny perplexity on the documents it
// saw and a huge one on the held-out document (overfitting); a fluent but false
// sentence still gets a low perplexity (low PPL ≠ truth).

import { registerOutput, unregisterOutput } from '../site/result.js';
import { loadCorpus, corpusText, showWs } from '../core/text.js';
import { countNgrams, prob } from '../core/ngram.js';
import { deserialize, forward } from '../core/gpt.js';
import { noGrad } from '../core/tensor.js';
import { mulberry32 } from '../core/rng.js';

export const VAL_ID = 'rag-vectordb'; // held out from every model in this course
export const BIGRAM_K = 0.1;

export const MODELS = [
  { id: 'bigram', label: '바이그램 빈도표 (1주차 · add-0.1)', short: '바이그램' },
  { id: '300', label: '미니 GPT · 300스텝 (덜 학습)', short: 'GPT-300', file: 'mini-gpt-300.json', step: 300 },
  { id: '800', label: '미니 GPT · 800스텝', short: 'GPT-800', file: 'mini-gpt-800.json', step: 800 },
  { id: '3000', label: '미니 GPT · 3000스텝 (오래 학습)', short: 'GPT-3000', file: 'mini-gpt-3000.json', step: 3000 },
];

export const SENTENCES = {
  's-true': { label: '문장 ① 코퍼스 그대로 (참)', text: '실습실은 수업 시간 외에도 평일 오후 9시까지 개방된다.' },
  's-false': { label: '문장 ② 숫자 하나만 바꿈 (거짓: 9시 → 3시)', text: '실습실은 수업 시간 외에도 평일 오후 3시까지 개방된다.' },
  's-shuffle': { label: '문장 ③ 문장 ①의 글자를 섞음', text: null },
};

/** Deterministic shuffle of a sentence (same characters, different order). */
export function scramble(text, seed = 9) {
  const cs = [...text];
  const rand = mulberry32(seed);
  for (let i = cs.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [cs[i], cs[j]] = [cs[j], cs[i]];
  }
  return cs.join('');
}
SENTENCES['s-shuffle'].text = scramble(SENTENCES['s-true'].text);

// ---------------------------------------------------------------- scoring (DOM-free)

/** Bigram (add-k) trained on every document except the held-out one. */
export function buildBigram(corpus) {
  const ids = corpus.documents.filter((d) => d.id !== VAL_ID).map((d) => d.id);
  return countNgrams([...corpusText(corpus, ids)], 2);
}

/**
 * Per-character surprise under the bigram: −ln P(c | previous char).
 * The first character is predicted from the start marker (same as ngram.nll).
 * @returns {{ ch: string, s: number|null }[]}
 */
export function scoreBigram(bigram, text) {
  const ctx = [];
  return [...text].map((ch) => {
    const p = prob(bigram, ctx, ch, { k: BIGRAM_K });
    ctx.push(ch);
    return { ch, s: -Math.log(p) };
  });
}

/**
 * Per-character surprise under a mini GPT checkpoint.
 * The text is cut into windows of blockSize characters; inside a window every
 * character is predicted from the characters before it (context restarts at each
 * window). The first character has no context and is not scored; characters
 * outside the vocabulary are skipped.
 * @param {{ model: object, vocab: object }} ckpt  result of deserialize()
 * @returns {{ ch: string, s: number|null, oov?: boolean }[]}
 */
export function scoreGpt(ckpt, text) {
  const { model, vocab } = ckpt;
  const T = model.config.blockSize;
  const chars = [...text];
  const out = chars.map((ch) => ({ ch, s: null, oov: !vocab.stoi.has(ch) }));
  const known = [];
  chars.forEach((ch, i) => vocab.stoi.has(ch) && known.push(i));
  const ids = known.map((i) => vocab.stoi.get(chars[i]));
  const N = ids.length;
  noGrad(() => {
    for (let start = 0; start < N - 1; start += T) {
      const len = Math.min(T, N - 1 - start);
      const x = Int32Array.from(ids.slice(start, start + len));
      const { logits } = forward(model, x, 1, len);
      const V = logits.shape[1];
      for (let t = 0; t < len; t++) {
        const row = logits.data.subarray(t * V, (t + 1) * V);
        let mx = -Infinity;
        for (let v = 0; v < V; v++) if (row[v] > mx) mx = row[v];
        let se = 0;
        for (let v = 0; v < V; v++) se += Math.exp(row[v] - mx);
        out[known[start + t + 1]].s = Math.log(se) + mx - row[ids[start + t + 1]];
      }
    }
  });
  return out;
}

/** loss (nats/char), perplexity, bits/char over the scored characters. */
export function summarize(scores) {
  const xs = scores.filter((c) => c.s != null).map((c) => c.s);
  const n = xs.length;
  const loss = n ? xs.reduce((a, b) => a + b, 0) / n : NaN;
  let worst = null;
  scores.forEach((c, i) => c.s != null && (!worst || c.s > worst.s) && (worst = { ...c, i }));
  return { n, loss, ppl: Math.exp(loss), bpc: loss / Math.LN2, worst };
}

// ---------------------------------------------------------------- widget

const DATA = (f) => new URL(`../../data/${f}`, import.meta.url);

const template = document.createElement('template');
template.innerHTML = `
  <div class="widget w09-ppl">
    <h3 class="widget__title">퍼플렉서티 측정기</h3>
    <div data-slot="status" class="widget__status" role="status" aria-live="polite">
      <span class="spinner" aria-hidden="true"></span> <span>문서셋과 체크포인트 불러오는 중…</span>
    </div>
    <div class="widget__controls">
      <label class="field">
        <span class="field__label">모델</span>
        <select data-in="model"></select>
      </label>
      <label class="field">
        <span class="field__label">측정할 글</span>
        <select data-in="text"></select>
      </label>
    </div>
    <label class="field w09-custom" data-slot="custom-wrap" hidden>
      <span class="field__label">직접 입력 (한두 문장)</span>
      <input type="text" data-in="custom" maxlength="200" spellcheck="false">
    </label>
    <div class="btn-row">
      <button type="button" class="btn small ghost" data-act="overfit">과대적합 보기 (오래 학습 · 검증 문서)</button>
      <button type="button" class="btn small ghost" data-act="memo">외운 문서 (오래 학습 · 학습 문서)</button>
      <button type="button" class="btn small ghost" data-act="false">참 vs 거짓 문장</button>
      <button type="button" class="btn small ghost" data-act="shuffle">섞은 문장</button>
    </div>
    <div data-slot="out-inline"></div>
  </div>`;

const outTemplate = document.createElement('template');
outTemplate.innerHTML = `
  <div class="w09-ppl-out">
    <div data-slot="verdict" aria-live="polite"></div>
    <div class="stat-row" data-slot="stats"></div>
    <h4>글자별 놀람 −ln p <small class="w09-muted">— 글자에 마우스를 올리면 값이 나온다</small></h4>
    <div class="w09-strip" data-slot="strip" tabindex="0" aria-label="글자별 놀람 색 띠 (위의 가장 놀란 글자 참고)"></div>
    <div class="legend" aria-hidden="true">
      <span><i class="w09-lg-s0"></i>&lt; 1 (p &gt; 37%)</span>
      <span><i class="w09-lg-s1"></i>1~3</span>
      <span><i class="w09-lg-s2"></i>3~6</span>
      <span><i class="w09-lg-s3"></i>≥ 6 (p &lt; 0.25%)</span>
      <span><i class="w09-lg-na"></i>점수 없음</span>
    </div>
    <p class="w09-muted w09-note" data-slot="strip-note"></p>
    <h4>같은 글, 다른 모델</h4>
    <div class="w09-table-wrap"><table class="w09-cmp" data-slot="cmp"></table></div>
    <h4>학습 곡선 <small class="w09-muted">— 체크포인트에 저장된 meta.history · 50스텝마다 무작위 창 24개로 잰 손실</small></h4>
    <div class="w09-curve" data-slot="curve"></div>
  </div>`;

const state = new WeakMap();
let seq = 0;

/**
 * @param {HTMLElement} el
 * @param {{ model?: string, text?: string, outputTitle?: string }} options
 */
export async function mount(el, options = {}) {
  const root = template.content.firstElementChild.cloneNode(true);
  const out = outTemplate.content.firstElementChild.cloneNode(true);
  el.replaceChildren(root);
  const $ = (sel) => root.querySelector(sel) ?? out.querySelector(sel);
  const ctrl = new AbortController();
  const outputId = `w09-perplexity:${++seq}`;
  state.set(el, { ctrl, outputId });

  const s = {
    model: options.model ?? null, // resolved after loading (latest checkpoint)
    text: options.text ?? VAL_ID,
    custom: '실습실은 주말에도 자유롭게 이용할 수 있다.',
    corpus: null,
    bigram: null,
    ckpts: {}, // id → deserialize() result
    precomputed: null, // assets/data/w09/eval.json (per-document losses)
    cache: new Map(), // `${model}|${text}` → scores
    job: 0,
  };

  registerOutput(outputId, { title: options.outputTitle ?? '퍼플렉서티 측정 결과', node: out, inlineHost: $('[data-slot=out-inline]') });

  const available = () => MODELS.filter((m) => m.id === 'bigram' || s.ckpts[m.id]);
  const latest = () => available().at(-1).id;
  const docTitle = (id) => s.corpus.documents.find((d) => d.id === id)?.title ?? id;

  function fillControls() {
    $('[data-in=model]').innerHTML = MODELS.map((m) => `<option value="${m.id}"${m.id === 'bigram' || s.ckpts[m.id] ? '' : ' disabled'}>${esc(m.label)}${m.id === 'bigram' || s.ckpts[m.id] ? '' : ' — 파일 없음'}</option>`).join('');
    const train = s.corpus.documents.filter((d) => d.id !== VAL_ID);
    $('[data-in=text]').innerHTML = [
      `<optgroup label="검증 문서 (어떤 모델도 보지 못한 글)"><option value="${VAL_ID}">${esc(docTitle(VAL_ID))} (${VAL_ID})</option></optgroup>`,
      `<optgroup label="학습 문서 (모델이 학습한 글)">${train.map((d) => `<option value="${d.id}">${esc(d.title)}</option>`).join('')}</optgroup>`,
      `<optgroup label="문장">${Object.entries(SENTENCES).map(([id, x]) => `<option value="${id}">${esc(x.label)}</option>`).join('')}<option value="custom">직접 입력</option></optgroup>`,
    ].join('');
    $('[data-in=custom]').value = s.custom;
  }

  function syncControls() {
    $('[data-in=model]').value = s.model;
    $('[data-in=text]').value = s.text;
    $('[data-slot=custom-wrap]').hidden = s.text !== 'custom';
  }

  function textOf(id) {
    if (id === 'custom') return s.custom;
    if (SENTENCES[id]) return SENTENCES[id].text;
    return s.corpus.documents.find((d) => d.id === id)?.text ?? '';
  }

  const isDoc = (id) => !SENTENCES[id] && id !== 'custom';

  async function scores(modelId, textId) {
    const key = `${modelId}|${textId}|${textId === 'custom' ? s.custom : ''}`;
    if (s.cache.has(key)) return s.cache.get(key);
    const text = textOf(textId);
    const r = modelId === 'bigram' ? scoreBigram(s.bigram, text) : scoreGpt(s.ckpts[modelId], text);
    if (s.cache.size > 80) s.cache.clear();
    s.cache.set(key, r);
    return r;
  }

  /** Loss of a model on a text: precomputed for documents, live for sentences. */
  async function lossOf(modelId, textId) {
    const pre = s.precomputed?.models?.[modelId]?.docs?.[textId];
    if (isDoc(textId) && pre != null) return pre;
    await tick();
    return summarize(await scores(modelId, textId)).loss;
  }

  async function trainAvg(modelId) {
    const pre = s.precomputed?.models?.[modelId]?.train;
    if (pre != null) return pre;
    // fallback: live, chunked (one document per tick)
    let sum = 0;
    let n = 0;
    for (const d of s.corpus.documents.filter((x) => x.id !== VAL_ID)) {
      await tick();
      const sc = summarize(await scores(modelId, d.id));
      sum += sc.loss * sc.n;
      n += sc.n;
    }
    return sum / n;
  }

  async function render() {
    if (!s.corpus) return;
    const job = ++s.job;
    const stale = () => job !== s.job || ctrl.signal.aborted;
    const m = MODELS.find((x) => x.id === s.model);
    $('[data-slot=stats]').innerHTML = '<div class="widget__status"><span class="spinner" aria-hidden="true"></span> 계산 중…</div>';
    await tick();
    if (stale()) return;

    const sc = await scores(s.model, s.text);
    const sum = summarize(sc);
    if (stale()) return;

    $('[data-slot=stats]').innerHTML = [
      stat('손실 (nats/글자)', fmt(sum.loss, 3)),
      stat('퍼플렉서티 e^손실', fmt(sum.ppl, sum.ppl < 10 ? 2 : 1)),
      stat('bits/글자', fmt(sum.bpc, 2)),
      stat('점수 낸 글자', `${sum.n}자`),
      stat('가장 놀란 글자', sum.worst ? `“${esc(showWs(sum.worst.ch))}” ${sum.worst.s.toFixed(1)}` : '—'),
    ].join('');

    // surprise strip
    $('[data-slot=strip]').innerHTML = sc
      .map((c, i) => {
        const shown = c.ch === '\n' ? '↵\n' : c.ch;
        if (c.s == null) {
          const why = c.oov ? '어휘에 없는 글자 — 건너뜀' : '문맥이 없어 예측하지 않음';
          return `<span class="w09-c na" title="${esc(why)}">${esc(shown)}</span>`;
        }
        const b = c.s < 1 ? 0 : c.s < 3 ? 1 : c.s < 6 ? 2 : 3;
        const tip = `${i + 1}번째 “${showWs(c.ch)}”: −ln p = ${c.s.toFixed(2)} · p = ${fmtP(Math.exp(-c.s))}`;
        return `<span class="w09-c s${b}" title="${esc(tip)}">${esc(shown)}</span>`;
      })
      .join('');
    const T = s.ckpts[s.model]?.model.config.blockSize;
    $('[data-slot=strip-note]').textContent =
      s.model === 'bigram'
        ? '바이그램은 바로 앞 글자 하나만 보고 다음 글자의 확률을 낸다. 첫 글자는 “글의 시작” 표시 뒤의 확률이다.'
        : `미니 GPT는 글을 ${T}글자 창으로 잘라 창 안에서 앞 글자들을 보고 예측한다. 맨 첫 글자는 문맥이 없어 점수에서 뺀다.`;

    // comparison table across models (+ train average) and verdict
    const rows = [];
    for (const mm of available()) {
      const here = mm.id === s.model ? sum.loss : await lossOf(mm.id, s.text);
      const tr = await trainAvg(mm.id);
      if (stale()) return;
      rows.push({ m: mm, here, tr });
    }
    const valCol = s.text === VAL_ID ? null : await Promise.all(rows.map((r) => lossOf(r.m.id, VAL_ID)));
    if (stale()) return;
    const bestHere = Math.min(...rows.map((r) => r.here));
    $('[data-slot=cmp]').innerHTML = `<thead><tr><th scope="col">모델</th><th scope="col">이 글 손실</th><th scope="col">이 글 PPL</th><th scope="col">학습 문서 평균 PPL</th>${valCol ? '<th scope="col">검증 문서 PPL</th>' : ''}</tr></thead><tbody>${rows
      .map(
        (r, i) =>
          `<tr class="${r.m.id === s.model ? 'cur' : ''}"><th scope="row">${esc(r.m.short)}</th><td>${fmt(r.here, 3)}${r.here === bestHere ? ' 🏆' : ''}</td><td>${fmtPpl(Math.exp(r.here))}</td><td>${fmtPpl(Math.exp(r.tr))}</td>${valCol ? `<td>${fmtPpl(Math.exp(valCol[i]))}</td>` : ''}</tr>`,
      )
      .join('')}</tbody>`;

    const cur = rows.find((r) => r.m.id === s.model);
    let trueLoss = null;
    if (s.text === 's-false') trueLoss = await lossOf(s.model, 's-true');
    if (stale()) return;
    $('[data-slot=verdict]').innerHTML = verdict({ m, textId: s.text, sum, cur, rows, trueLoss, title: isDoc(s.text) ? docTitle(s.text) : '' });

    renderCurve();
  }

  function renderCurve() {
    const src = [...MODELS].reverse().find((x) => s.ckpts[x.id]);
    const hist = src ? s.ckpts[src.id].meta.history ?? [] : [];
    const valBigram = s.precomputed?.models?.bigram?.docs?.[VAL_ID] ?? null;
    $('[data-slot=curve]').innerHTML = hist.length
      ? curveSvg(hist, { snapshots: MODELS.filter((x) => x.step && s.ckpts[x.id]).map((x) => x.step), current: MODELS.find((x) => x.id === s.model)?.step ?? null, bigram: valBigram, full: src.step })
      : '<p class="w09-muted">체크포인트에 학습 기록이 없다.</p>';
  }

  const onIn = (type, fn) => root.addEventListener(type, fn, { signal: ctrl.signal });
  let debounce = 0;
  onIn('change', (e) => {
    const t = e.target;
    if (t.matches('[data-in=model]')) s.model = t.value;
    else if (t.matches('[data-in=text]')) s.text = t.value;
    else return;
    syncControls();
    render();
  });
  onIn('input', (e) => {
    if (!e.target.matches('[data-in=custom]')) return;
    s.custom = e.target.value;
    clearTimeout(debounce);
    debounce = setTimeout(render, 250);
  });
  onIn('click', (e) => {
    const act = e.target.closest('[data-act]')?.dataset.act;
    if (!act || !s.corpus) return;
    if (act === 'overfit') Object.assign(s, { model: latest(), text: VAL_ID });
    if (act === 'memo') Object.assign(s, { model: latest(), text: 'facility-rules' });
    if (act === 'false') s.text = 's-false';
    if (act === 'shuffle') s.text = 's-shuffle';
    syncControls();
    render();
  });

  try {
    const [corpus, pre, ...ck] = await Promise.all([
      loadCorpus(),
      fetch(DATA('w09/eval.json'))
        .then((r) => (r.ok ? r.json() : null))
        .catch(() => null),
      ...MODELS.filter((m) => m.file).map((m) =>
        fetch(DATA(`models/${m.file}`))
          .then((r) => (r.ok ? r.json() : null))
          .then((j) => (j ? [m.id, deserialize(j)] : null))
          .catch(() => null),
      ),
    ]);
    if (ctrl.signal.aborted) return;
    s.corpus = corpus;
    s.bigram = buildBigram(corpus);
    s.precomputed = pre;
    ck.filter(Boolean).forEach(([id, c]) => (s.ckpts[id] = c));
    // precomputed numbers are only trusted for checkpoints that were actually loaded
    if (s.precomputed) for (const id of Object.keys(s.precomputed.models ?? {})) if (id !== 'bigram' && !s.ckpts[id]) delete s.precomputed.models[id];
    if (!s.model || !available().some((m) => m.id === s.model)) s.model = latest();
    fillControls();
    syncControls();
    const missing = MODELS.filter((m) => m.file && !s.ckpts[m.id]);
    const status = $('[data-slot=status]');
    if (missing.length) status.innerHTML = `<span class="w09-muted">체크포인트 ${missing.map((m) => esc(m.file)).join(', ')}이(가) 없어 있는 모델만 쓴다.</span>`;
    else status.hidden = true;
    render();
  } catch (err) {
    $('[data-slot=status]').innerHTML = `<div class="widget__error">데이터를 불러오지 못했다 (${esc(err.message)}). <code>file://</code>로 열었다면 <code>python -m http.server</code>로 연다.</div>`;
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

const tick = () => new Promise((r) => setTimeout(r, 0));

function verdict({ m, textId, sum, cur, rows, trueLoss, title }) {
  const ppl = sum.ppl;
  const gap = cur.here - cur.tr;
  if (textId === 's-shuffle') {
    return callout('danger', `글자를 섞으면 PPL ${fmt(ppl, 0)}`, `같은 글자들인데 순서만 바꿨다. 모델은 글자 “주머니”가 아니라 <b>순서</b>를 배웠기 때문에 매 글자마다 크게 놀란다. 1주차 BoW와 언어 모델의 차이가 숫자로 보인다.`);
  }
  if (textId === 's-false' && trueLoss != null) {
    const d = sum.loss - trueLoss;
    return callout(
      'danger',
      `거짓 문장도 PPL ${fmtPpl(ppl)} (참 문장 ${fmtPpl(Math.exp(trueLoss))})`,
      `“9시”를 “3시”로 바꾼 거짓 문장이다. 손실 차이는 ${d >= 0 ? '+' : ''}${d.toFixed(3)}뿐이다. 퍼플렉서티는 “이 글이 얼마나 <b>그럴듯한 글자 배열</b>인가”를 잴 뿐, <b>사실인지</b>는 재지 않는다. 낮은 PPL ≠ 참.${m.id === '3000' || m.id === '800' ? ' 오래 학습한 모델은 원문을 외워서 차이가 조금 더 크다 — 이해가 아니라 암기다.' : ''}`,
    );
  }
  if (textId === VAL_ID && m.id !== 'bigram') {
    const big = rows.find((r) => r.m.id === 'bigram');
    if (gap > 1.5) {
      return callout('danger', `과대적합: 학습 문서 PPL ${fmtPpl(Math.exp(cur.tr))} → 검증 문서 PPL ${fmt(ppl, 0)}`, `이 모델은 학습 문서에서는 평균 ${fmtPpl(Math.exp(cur.tr))}개 후보 중에서 고민하지만, 처음 보는 “${esc(title)}” 문서에서는 ${fmt(ppl, 0)}개 중에서 헤맨다. 1주차 바이그램(PPL ${fmt(Math.exp(big.here), 0)})보다도 나쁘다. 6.5천 자를 반복해 보며 <b>문서를 외웠기</b> 때문이다. 학습 곡선에서 검증 손실이 가장 낮았던 지점을 찾아본다.`);
    }
    return callout('more', `검증 문서 PPL ${fmt(ppl, 1)} · 학습 문서 PPL ${fmtPpl(Math.exp(cur.tr))}`, `학습 문서보다 높지만 차이가 아직 크지 않다. 바이그램(PPL ${fmt(Math.exp(big.here), 1)})과 비교해 본다. 이 정도 데이터로는 GPT도 바이그램을 크게 이기지 못한다.`);
  }
  if (textId === VAL_ID) {
    return callout('more', `기준선: 바이그램 검증 PPL ${fmt(ppl, 1)}`, `1주차 C5에서 구한 NLL ${sum.loss.toFixed(3)}의 e^ 값이다. 바로 앞 글자만 보고 “평균 ${fmt(ppl, 0)}개 후보 중에서 고민”한다. 미니 GPT들이 이 기준선을 이기는지 모델을 바꿔 본다.`);
  }
  if (!['bigram'].includes(m.id) && ppl < 2 && SENTENCES[textId] == null && textId !== 'custom') {
    return callout('danger', `외운 문서: PPL ${fmt(ppl, 2)}`, `“${esc(title)}” 문서는 모델이 학습한 글이다. PPL이 ${fmt(ppl, 2)}이면 거의 매 글자를 확신한다는 뜻이다. 잘 배운 것처럼 보이지만 검증 문서 PPL(표의 마지막 열)을 보면 외운 것이다. 학습 문서로 잰 점수는 모델을 평가하지 못한다.`);
  }
  if (textId === 'custom' || textId === 's-true') {
    return callout('ok', `PPL ${fmtPpl(ppl)} · ${fmt(sum.bpc, 2)} bits/글자`, `이 문장에서 모델은 평균 ${fmtPpl(ppl)}개 후보 중에서 고민했다. 빨간 글자가 모델이 가장 놀란 곳이다. 문장 ②(거짓)나 ③(섞음)과 비교해 본다.`);
  }
  return callout('more', `학습 문서 PPL ${fmt(ppl, 1)}`, `모델이 학습한 문서다. 이 숫자는 “얼마나 잘 외웠나”에 가깝다. 검증 문서로 바꿔 차이를 본다.`);
}

/** Loss curve (train/val) with snapshot markers and the minimum-validation step. */
export function curveSvg(hist, { snapshots = [], current = null, bigram = null, full = null } = {}) {
  const W = 560;
  const H = 230;
  const L = 38;
  const R = 12;
  const Tp = 14;
  const B = 34;
  const maxStep = Math.max(...hist.map((h) => h.step));
  const maxLoss = Math.ceil(Math.max(...hist.map((h) => Math.max(h.train, h.val)), bigram ?? 0));
  const x = (st) => L + ((W - L - R) * st) / maxStep;
  const y = (v) => Tp + ((H - Tp - B) * (maxLoss - v)) / maxLoss;
  const path = (k) => hist.map((h, i) => `${i ? 'L' : 'M'}${x(h.step).toFixed(1)} ${y(h[k]).toFixed(1)}`).join('');
  const best = hist.reduce((a, h) => (h.val < a.val ? h : a));
  const yt = [];
  for (let v = 0; v <= maxLoss; v += maxLoss > 8 ? 2 : 1) yt.push(v);
  const xt = [];
  const stepTick = maxStep > 1500 ? 500 : maxStep > 500 ? 200 : 100;
  for (let st = 0; st <= maxStep; st += stepTick) xt.push(st);
  return `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(`학습 손실과 검증 손실 곡선. 검증 손실 최저는 ${best.step}스텝의 ${best.val}`)}">
    ${yt.map((v) => `<line class="w09g-grid" x1="${L}" x2="${W - R}" y1="${y(v)}" y2="${y(v)}"/><text class="w09g-t" x="${L - 6}" y="${y(v) + 4}" text-anchor="end">${v}</text>`).join('')}
    ${xt.map((st) => `<text class="w09g-t" x="${x(st)}" y="${H - B + 16}" text-anchor="middle">${st}</text>`).join('')}
    <text class="w09g-t" x="${W - R}" y="${H - 4}" text-anchor="end">학습 스텝${full ? ` (기록: ${full}스텝 체크포인트)` : ''}</text>
    ${snapshots.map((st) => `<line class="w09g-snap${st === current ? ' cur' : ''}" x1="${x(st)}" x2="${x(st)}" y1="${Tp}" y2="${H - B}"/><text class="w09g-t${st === current ? ' cur' : ''}" x="${Math.min(x(st) + 3, W - R - 30)}" y="${Tp + 10}">${st}</text>`).join('')}
    ${bigram != null ? `<line class="w09g-base" x1="${L}" x2="${W - R}" y1="${y(bigram)}" y2="${y(bigram)}"/><text class="w09g-t" x="${W - R}" y="${y(bigram) - 4}" text-anchor="end">바이그램 검증 ${bigram.toFixed(2)}</text>` : ''}
    <path class="w09g-train" d="${path('train')}"/>
    <path class="w09g-val" d="${path('val')}"/>
    <circle class="w09g-best" cx="${x(best.step)}" cy="${y(best.val)}" r="5"/>
    <text class="w09g-t strong" x="${x(best.step) + 8}" y="${y(best.val) + 16}">검증 최저 ${best.val.toFixed(2)} @ ${best.step}스텝</text>
    <g class="w09g-key"><line class="w09g-train" x1="${L + 10}" x2="${L + 30}" y1="${H - 8}" y2="${H - 8}"/><text class="w09g-t" x="${L + 34}" y="${H - 4}">학습</text>
    <line class="w09g-val" x1="${L + 70}" x2="${L + 90}" y1="${H - 8}" y2="${H - 8}"/><text class="w09g-t" x="${L + 94}" y="${H - 4}">검증 (rag-vectordb)</text></g>
  </svg>`;
}

const callout = (kind, title, html) =>
  `<div class="callout${kind === 'ok' ? ' callout--ok' : kind === 'danger' ? ' callout--danger' : ' callout--more'}"><span class="callout__title">${esc(title)}</span><p>${html}</p></div>`;

const stat = (label, value) => `<div class="stat"><span class="stat__label">${label}</span><span class="stat__value">${value}</span></div>`;

function fmt(v, d) {
  if (!Number.isFinite(v)) return '—';
  if (v >= 1000) return Math.round(v).toLocaleString();
  return v.toFixed(d);
}

const fmtPpl = (v) => fmt(v, v < 10 ? 2 : 1);

function fmtP(p) {
  if (p >= 0.01) return `${(p * 100).toFixed(1)}%`;
  return `${(p * 100).toExponential(1)}%`;
}

function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
}
