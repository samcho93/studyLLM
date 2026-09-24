// w08 샘플링 조절판
// One concept: the model gives a next-token distribution; the decoding strategy
// (greedy / temperature / top-k / top-p / repetition penalty) decides which token
// is actually written. Same model, different knobs → loops, gibberish or fluent text.
//
// Generation runs on the main thread but is time-sliced (a few tokens per frame)
// and cancelled whenever a control changes; logits are cached per context.

import { registerOutput, unregisterOutput } from '../site/result.js';
import { loadCorpus, corpusText, showWs } from '../core/text.js';
import { deserialize, nextLogits } from '../core/gpt.js';
import { softmax, topK, topP, repetitionPenalty, decodeStep, entropyBits } from '../core/sampling.js';
import { mulberry32 } from '../core/rng.js';

const MODEL_DIR = new URL('../../data/models/', import.meta.url);
const MODELS = [
  { id: '300', file: 'mini-gpt-300.json', label: '미니 GPT · 300스텝 (균형 잡힌 모델)' },
  { id: '3000', file: 'mini-gpt-3000.json', label: '미니 GPT · 3000스텝 (코퍼스를 외운 모델)' },
];
const DEFAULTS = { greedy: false, T: 1, k: 0, p: 1, penalty: 1 };
const PRESETS = {
  greedy: { greedy: true, penalty: 1 },
  hot: { greedy: false, T: 2, k: 0, p: 1, penalty: 1 },
  narrow: { greedy: false, T: 1, k: 0, p: 0.1, penalty: 1 },
  good: { greedy: false, T: 0.8, k: 0, p: 0.9, penalty: 1.2 },
  reset: { ...DEFAULTS },
};
const SAMPLE_COUNT = 5;
const SAMPLE_LEN = 40;
const DIST_ROWS = 12;
const REP_N = 4; // a 4-character chunk seen earlier counts as a repeat
const COPY_MIN = 10; // a run this long found verbatim in the corpus counts as "copied"
const SLICE_MS = 14; // work per frame before yielding to the UI

const template = document.createElement('template');
template.innerHTML = `
  <div class="widget w08-sp">
    <h3 class="widget__title">샘플링 조절판</h3>
    <div data-slot="status" class="widget__status" role="status" aria-live="polite">
      <span class="spinner" aria-hidden="true"></span> <span>미니 GPT 체크포인트 불러오는 중…</span>
    </div>
    <div class="widget__controls">
      <label class="field">
        <span class="field__label">모델</span>
        <select data-in="model"></select>
      </label>
      <label class="field">
        <span class="field__label">프롬프트 (시작 글)</span>
        <input type="text" data-in="prompt" maxlength="40" spellcheck="false">
      </label>
      <div class="field">
        <span class="field__label" id="w08-strategy-label">전략</span>
        <div class="w08-seg" role="group" aria-labelledby="w08-strategy-label">
          <button type="button" class="btn small" data-strategy="greedy" aria-pressed="false">그리디 (1등만)</button>
          <button type="button" class="btn small" data-strategy="sample" aria-pressed="true">샘플링 (확률대로)</button>
        </div>
      </div>
      <label class="field" data-sampling-only>
        <span class="field__label">온도 T <output data-out="T"></output></span>
        <input type="range" data-in="T" min="0.1" max="2" step="0.1">
      </label>
      <label class="field" data-sampling-only>
        <span class="field__label">top-k <output data-out="k"></output></span>
        <input type="range" data-in="k" min="0" max="50" step="1">
      </label>
      <label class="field" data-sampling-only>
        <span class="field__label">top-p <output data-out="p"></output></span>
        <input type="range" data-in="p" min="0.1" max="1" step="0.05">
      </label>
      <label class="field">
        <span class="field__label">반복 페널티 <output data-out="penalty"></output></span>
        <input type="range" data-in="penalty" min="1" max="2" step="0.1">
      </label>
      <label class="field">
        <span class="field__label">생성 길이 <output data-out="len"></output></span>
        <input type="range" data-in="len" min="20" max="120" step="10">
      </label>
      <div class="field">
        <label class="field__label" for="w08-seed">seed <small class="w08-muted" data-out="seed-note"></small></label>
        <div class="w08-seed-row">
          <input type="number" id="w08-seed" data-in="seed" min="1" max="9999" step="1">
          <button type="button" class="btn small" data-act="dice" aria-label="seed 하나 올리기">🎲 다시 뽑기</button>
        </div>
      </div>
    </div>
    <div class="btn-row w08-presets" aria-label="실패 사례와 추천 설정">
      <button type="button" class="btn small ghost" data-preset="greedy">그리디 → 반복 루프</button>
      <button type="button" class="btn small ghost" data-preset="hot">T = 2.0 → 횡설수설</button>
      <button type="button" class="btn small ghost" data-preset="narrow">top-p = 0.1 → 사실상 그리디</button>
      <button type="button" class="btn small ghost" data-preset="good">추천: T 0.8 · top-p 0.9 · 페널티 1.2</button>
      <button type="button" class="btn small ghost" data-preset="reset">초기화</button>
    </div>
    <div class="btn-row w08-steps">
      <button type="button" class="btn small" data-act="rewind">⏮ 처음부터</button>
      <button type="button" class="btn small primary" data-act="step">▶ 한 글자</button>
      <button type="button" class="btn small" data-act="finish">⏭ 끝까지</button>
      <span class="w08-muted w08-stepinfo" data-out="stepinfo" aria-live="polite"></span>
    </div>
    <div data-slot="out-inline"></div>
  </div>`;

const outTemplate = document.createElement('template');
outTemplate.innerHTML = `
  <div class="w08-out">
    <div data-slot="verdict" aria-live="polite"></div>
    <h4>① 다음 글자 분포 <small class="w08-muted" data-slot="dist-title"></small></h4>
    <div class="legend" aria-hidden="true">
      <span><i class="w08-lg-orig"></i>모델 원래 (T = 1)</span>
      <span><i class="w08-lg-temp"></i>페널티 · 온도 적용</span>
      <span><i class="w08-lg-fin"></i>top-k · top-p 거른 뒤 (실제로 뽑는 분포)</span>
      <span><i class="w08-lg-cut"></i>잘려 나간 후보</span>
    </div>
    <div class="w08-dist" data-slot="dist"></div>
    <div class="stat-row" data-slot="dist-stats"></div>
    <h4>② 생성된 글 <small class="w08-muted">— 글자를 누르면 그 글자를 고를 때의 분포가 ①에 나온다</small></h4>
    <div class="w08-gen" data-slot="gen" tabindex="0"></div>
    <div class="legend" aria-hidden="true">
      <span><i class="w08-lg-prompt"></i>프롬프트</span>
      <span><i class="w08-lg-hi"></i>모델 원래 확률 ≥ 50%</span>
      <span><i class="w08-lg-mid"></i>10~50%</span>
      <span><i class="w08-lg-lo"></i>&lt; 10%</span>
      <span><i class="w08-lg-rep"></i>앞에서 나온 ${REP_N}글자 반복</span>
    </div>
    <div class="stat-row" data-slot="stats"></div>
    <h4>③ 같은 설정으로 ${SAMPLE_COUNT}번 <small class="w08-muted">— seed만 바꿔 ${SAMPLE_LEN}글자씩</small></h4>
    <ol class="w08-samples" data-slot="samples"></ol>
    <p class="w08-sum" data-slot="samples-sum"></p>
    <h4>④ 반복 탐지</h4>
    <div class="w08-reps" data-slot="reps"></div>
  </div>`;

const state = new WeakMap();
let seq = 0;

/**
 * @param {HTMLElement} el
 * @param {{ outputTitle?: string, prompt?: string, preset?: keyof PRESETS, model?: string }} options
 */
export async function mount(el, options = {}) {
  const root = template.content.firstElementChild.cloneNode(true);
  const out = outTemplate.content.firstElementChild.cloneNode(true);
  el.replaceChildren(root);
  const $ = (sel) => root.querySelector(sel) ?? out.querySelector(sel);
  const ctrl = new AbortController();
  const outputId = `w08-sampler:${++seq}`;
  state.set(el, { ctrl, outputId });

  // unique ids when the widget is mounted twice (student page + deck)
  const labelId = `w08-strategy-label-${seq}`;
  $('#w08-strategy-label').id = labelId;
  $('.w08-seg').setAttribute('aria-labelledby', labelId);
  const seedId = `w08-seed-${seq}`;
  $('#w08-seed').id = seedId;
  $('label[for=w08-seed]').htmlFor = seedId;

  const s = {
    modelId: options.model ?? '300',
    prompt: options.prompt ?? '언어 모델',
    ...DEFAULTS,
    ...(options.preset ? PRESETS[options.preset] : {}),
    seed: 1,
    len: 80,
    stepMode: false,
    cursor: 0, // generated tokens shown in step mode
    pick: null, // step index shown in ① (null = automatic)
  };
  const models = new Map(); // id → { model, vocab, meta, cache }
  let source = ''; // corpus text for the copy check
  let job = 0;
  let view = null; // last finished main run
  let timer = 0;

  $('[data-in=model]').innerHTML = MODELS.map((m) => `<option value="${m.id}">${esc(m.label)}</option>`).join('');
  registerOutput(outputId, { title: options.outputTitle ?? '샘플링 조절판 결과', node: out, inlineHost: $('[data-slot=out-inline]') });

  function syncControls() {
    $('[data-in=model]').value = s.modelId;
    if ($('[data-in=prompt]').value !== s.prompt) $('[data-in=prompt]').value = s.prompt;
    for (const key of ['T', 'k', 'p', 'penalty', 'len']) $(`[data-in=${key}]`).value = String(s[key]);
    $('[data-in=seed]').value = String(s.seed);
    $('[data-out=T]').textContent = s.T.toFixed(1);
    $('[data-out=k]').textContent = s.k === 0 ? '끔 (0)' : String(s.k);
    $('[data-out=p]').textContent = s.p >= 1 ? '끔 (1.0)' : s.p.toFixed(2);
    $('[data-out=penalty]').textContent = s.penalty === 1 ? '끔 (1.0)' : s.penalty.toFixed(1);
    $('[data-out=len]').textContent = `${s.len}자`;
    $('[data-out=seed-note]').textContent = s.greedy ? '— 그리디는 쓰지 않는다' : '';
    root.querySelectorAll('[data-strategy]').forEach((b) => b.setAttribute('aria-pressed', String((b.dataset.strategy === 'greedy') === s.greedy)));
    root.querySelectorAll('[data-sampling-only]').forEach((f) => {
      f.classList.toggle('is-off', s.greedy);
      f.querySelector('input').disabled = s.greedy;
    });
  }

  function opts() {
    return { greedy: s.greedy, temperature: s.T, k: s.k, p: s.p, penalty: s.penalty };
  }

  async function ensureModel(id) {
    if (models.has(id)) return models.get(id);
    const info = MODELS.find((m) => m.id === id);
    const res = await fetch(new URL(info.file, MODEL_DIR));
    if (!res.ok) throw new Error(`${info.file}을(를) 찾을 수 없다 (HTTP ${res.status})`);
    const { model, vocab, meta } = deserialize(await res.json());
    const entry = { model, vocab, meta, cache: new Map() };
    models.set(id, entry);
    return entry;
  }

  const setStatus = (html) => {
    const st = $('[data-slot=status]');
    st.hidden = !html;
    if (html) st.innerHTML = html;
  };

  // ---------------------------------------------------------------- generation (time-sliced)

  function schedule(delay = 90) {
    clearTimeout(timer);
    timer = setTimeout(regenerate, delay);
  }

  async function regenerate() {
    const my = ++job;
    const M = models.get(s.modelId);
    if (!M) return;
    const { ids: promptIds, dropped } = encodePrompt(M.vocab, s.prompt);
    const target = s.stepMode ? s.cursor : s.len;
    const main = createRun(M, promptIds, opts(), s.seed, true);
    const alive = () => my === job && !ctrl.signal.aborted;
    const slice = async (fn) => {
      let t0 = performance.now();
      while (fn()) {
        if (performance.now() - t0 > SLICE_MS) {
          await frame();
          if (!alive()) return false;
          t0 = performance.now();
        }
      }
      return alive();
    };

    // show a spinner only when generation is slow (cache misses), so ▶ does not flicker
    const spin = setTimeout(() => alive() && setStatus(`<span class="spinner" aria-hidden="true"></span> <span>생성 중… (${M.vocab.size}개 글자 점수를 한 글자마다 계산한다)</span>`), 200);
    const ok = await slice(() => main.steps.length < target && (advance(M, main), true));
    clearTimeout(spin);
    if (!ok) return;

    // near-greedy check needs the greedy path too (cheap: shares the logits cache)
    let greedyRun = null;
    if (!s.greedy && main.steps.length && isNearGreedy(main.steps)) {
      greedyRun = createRun(M, promptIds, { ...opts(), greedy: true }, s.seed, false);
      if (!(await slice(() => greedyRun.steps.length < main.steps.length && (advance(M, greedyRun), true)))) return;
    }

    // the next-token distribution at the end (step mode shows what comes next)
    const nextLogitsAtEnd = s.stepMode ? cachedLogits(M, main.ids) : null;
    view = { M, main, promptIds, dropped, greedyRun, nextLogitsAtEnd };
    setStatus('');
    renderMain();

    // five samples with different seeds, rendered as they arrive
    const n = Math.min(SAMPLE_LEN, s.len);
    const samples = [];
    $('[data-slot=samples]').innerHTML = '';
    $('[data-slot=samples-sum]').innerHTML = '<span class="spinner" aria-hidden="true"></span> 샘플 만드는 중…';
    for (let i = 0; i < SAMPLE_COUNT; i++) {
      const run = createRun(M, promptIds, opts(), s.seed + i, false);
      if (!(await slice(() => run.steps.length < n && (advance(M, run), true)))) return;
      samples.push({ seed: s.seed + i, text: M.vocab.decode(run.ids.slice(promptIds.length)), logp: mean(run.steps.map((st) => Math.log(st.pModel))) });
      renderSamples(samples);
      await frame();
      if (!alive()) return;
    }
  }

  // ---------------------------------------------------------------- rendering

  function renderMain() {
    const { M, main, promptIds, dropped, greedyRun } = view;
    const promptText = M.vocab.decode(promptIds);
    const genIds = main.ids.slice(promptIds.length);
    const genChars = genIds.map((i) => M.vocab.itos[i]);
    const genText = genChars.join('');
    const allChars = [...promptText, ...genChars];
    const repMask = repeatMask(allChars, REP_N).slice(promptIds.length);
    const loops = findLoops(genText);
    const copyRate = source && genText.length ? copyMask(genText, source).filter(Boolean).length / genChars.length : null;

    // step info
    const last = main.steps.at(-1);
    $('[data-out=stepinfo]').textContent = s.stepMode
      ? `한 글자씩 · ${main.steps.length}/${s.len}자${last ? ` · 방금 “${showWs(M.vocab.itos[last.index])}”을(를) 뽑았다 (뽑는 분포에서 ${pct(last.pFinal)})` : ''}`
      : '';

    // ② generated text
    const pick = currentPick();
    const pieces = [`<span class="w08-prompt">${esc(showWs(promptText))}</span>`];
    main.steps.forEach((st, i) => {
      const cls = st.pModel >= 0.5 ? 'hi' : st.pModel >= 0.1 ? 'mid' : 'lo';
      const ch = genChars[i] === '\n' ? '↵\n' : genChars[i];
      const title = `${i + 1}번째 · 모델 원래 확률 ${pct(st.pModel)} · 실제로 뽑은 분포에서 ${pct(st.pFinal)} · 남은 후보 ${st.kept}개`;
      pieces.push(`<button type="button" class="w08-ch ${cls}${repMask[i] ? ' rep' : ''}${i === pick ? ' sel' : ''}${s.stepMode && i === main.steps.length - 1 ? ' last' : ''}" data-step="${i}" title="${esc(title)}">${esc(ch)}</button>`);
    });
    if (s.stepMode) pieces.push(`<span class="w08-caret" aria-hidden="true">▍</span>`);
    $('[data-slot=gen]').innerHTML = pieces.join('');

    // main stats
    const steps = main.steps;
    const avgLogP = steps.length ? mean(steps.map((st) => Math.log(st.pModel))) : null;
    const d2 = distinct2([genText]);
    const repRate = repMask.length > REP_N ? repMask.filter(Boolean).length / repMask.length : 0;
    $('[data-slot=stats]').innerHTML = [
      stat('평균 log P / 글자', avgLogP === null ? '—' : avgLogP.toFixed(2), '모델 원래 확률의 로그 평균. 0에 가까울수록 모델이 보기에 자연스럽다 (유창성)'),
      stat('distinct-2', steps.length > 1 ? d2.toFixed(2) : '—', '서로 다른 2글자 조각 수 / 전체 2글자 조각 수. 1에 가까울수록 반복이 적다 (다양성)'),
      stat('반복 글자 비율', fmtPct(repRate), `앞에서 이미 나온 ${REP_N}글자 조각에 속한 글자의 비율`),
      stat('평균 남은 후보', steps.length ? mean(steps.map((st) => st.kept)).toFixed(1) : '—', `거르기 뒤에 남은 후보 수의 평균 (어휘 ${M.vocab.size}개)`),
      stat('원문 복사율', copyRate === null ? '—' : fmtPct(copyRate), `코퍼스에 그대로 있는 ${COPY_MIN}글자 이상 구간의 비율`),
    ].join('');

    // ④ repetition detector
    const repeated = topRepeats(genText);
    $('[data-slot=reps]').innerHTML =
      (loops.length
        ? `<p><b class="w08-bad">연속 반복(루프) ${loops.length}곳</b> — ${loops.map((l) => `<code>${esc(showWs(l.unit))}</code> × ${l.times}`).join(' · ')}</p>`
        : `<p>같은 조각이 3번 이상 <b>연달아</b> 나오는 루프는 없다.</p>`) +
      (repeated.length
        ? `<p>여러 번 나온 조각 (${REP_N}글자 이상): ${repeated.map((r) => `<code>${esc(showWs(r.text))}</code> × ${r.count}`).join(' · ')}</p>`
        : `<p>${REP_N}글자 이상 조각이 두 번 나온 곳도 없다.</p>`);

    // verdict
    $('[data-slot=verdict]').innerHTML = verdict({ loops, avgLogP, avgKept: steps.length ? mean(steps.map((st) => st.kept)) : 0, greedyRun, main, copyRate, genText, dropped, M });

    renderDist();
  }

  function currentPick() {
    const n = view.main.steps.length;
    if (s.pick !== null && s.pick < n) return s.pick;
    if (s.stepMode) return null; // show the upcoming token
    return n ? 0 : null;
  }

  function renderDist() {
    const { M, main, promptIds, nextLogitsAtEnd } = view;
    const B = M.model.config.blockSize;
    const pick = currentPick();
    let logits;
    let ctxLen;
    let chosen = -1;
    if (pick === null) {
      if (!nextLogitsAtEnd) {
        $('[data-slot=dist]').innerHTML = '<p class="w08-muted">생성된 글자가 없다.</p>';
        $('[data-slot=dist-stats]').innerHTML = '';
        return;
      }
      logits = nextLogitsAtEnd;
      ctxLen = main.ids.length;
    } else {
      logits = main.steps[pick].logits;
      ctxLen = promptIds.length + pick;
      chosen = main.steps[pick].index;
    }
    const ctx = main.ids.slice(0, ctxLen);
    const history = ctx.slice(-B);
    const st = stages(logits, history, opts());
    const seen = new Set(s.penalty !== 1 ? history : []);
    const tail = M.vocab.decode(ctx.slice(-10));
    $('[data-slot=dist-title]').innerHTML = `— ${pick === null ? `${main.steps.length + 1}번째 글자를 고를 차례` : `${pick + 1}번째 글자를 고를 때`} · 문맥 “…${esc(showWs(tail))}” 다음`;

    const order = st.temp.map((q, i) => i).sort((a, b) => st.temp[b] - st.temp[a] || st.orig[b] - st.orig[a]);
    const rows = order.slice(0, DIST_ROWS);
    if (chosen >= 0 && !rows.includes(chosen)) rows.push(chosen);
    const bar = (cls, q) => `<i class="${cls}" style="width:${(q * 100).toFixed(2)}%"></i>`;
    const hidden = order.slice(DIST_ROWS).filter((i) => i !== chosen);
    const hiddenKept = hidden.filter((i) => st.fin[i] > 0).length;
    $('[data-slot=dist]').innerHTML =
      rows
        .map((i, r) => {
          const cut = st.fin[i] === 0;
          const label = M.vocab.itos[i] === '\n' ? '↵' : showWs(M.vocab.itos[i]);
          const gap = r === DIST_ROWS ? '<div class="w08-gap" aria-hidden="true">⋮</div>' : '';
          return `${gap}<div class="w08-row${cut ? ' cut' : ''}${i === chosen ? ' chosen' : ''}">
            <span class="t">${esc(label)}${seen.has(i) ? '<sup title="앞에 나온 글자라 반복 페널티로 점수가 깎였다">↺</sup>' : ''}</span>
            <span class="b">${bar('o', st.orig[i])}${bar('m', st.temp[i])}${bar('f', st.fin[i])}</span>
            <span class="v">${pct(st.orig[i])} → ${pct(st.temp[i])} → ${cut ? '<s>제거</s>' : `<b>${pct(st.fin[i])}</b>`}${i === chosen ? ' ◀' : ''}</span>
          </div>`;
        })
        .join('') + `<p class="w08-muted w08-more">${hidden.length ? `… 나머지 ${hidden.length}개 후보${hiddenKept ? ` (이 중 ${hiddenKept}개는 아직 남아 있다)` : ' (모두 잘려 나갔다)'}` : ''}</p>`;

    const kept = st.fin.filter((q) => q > 0).length;
    $('[data-slot=dist-stats]').innerHTML = [
      stat('엔트로피 · 원래', `${entropyBits(st.orig).toFixed(2)} bit`, '모델이 낸 분포가 얼마나 퍼져 있는가'),
      stat('· 페널티·온도 후', `${entropyBits(st.temp).toFixed(2)} bit`, '온도가 1보다 크면 커지고 작으면 작아진다'),
      stat('· 최종', `${entropyBits(st.fin).toFixed(2)} bit`, '실제로 뽑는 분포. 0이면 결과가 정해져 있다'),
      stat('남은 후보 수', `${kept} / ${M.vocab.size}`, 'top-k · top-p로 거른 뒤 뽑힐 수 있는 글자 수'),
      stat('실질 후보 수', (2 ** entropyBits(st.fin)).toFixed(1), '2^엔트로피. 확률이 고르게 퍼진 후보 몇 개와 맞먹는가'),
    ].join('');
  }

  function renderSamples(samples) {
    const counts = new Map();
    samples.forEach((x) => counts.set(x.text, (counts.get(x.text) ?? 0) + 1));
    $('[data-slot=samples]').innerHTML = samples
      .map((x) => `<li><span class="w08-seedtag">seed ${x.seed}</span><span class="w08-sample${counts.get(x.text) > 1 ? ' dup' : ''}">${esc(x.text.replaceAll('\n', '↵'))}</span><span class="w08-muted w08-lp" title="모델 원래 확률의 로그 평균">log P ${x.logp.toFixed(2)}</span></li>`)
      .join('');
    if (samples.length < SAMPLE_COUNT) {
      $('[data-slot=samples-sum]').innerHTML = `<span class="spinner" aria-hidden="true"></span> 샘플 ${samples.length}/${SAMPLE_COUNT}…`;
      return;
    }
    const unique = counts.size;
    const d2 = distinct2(samples.map((x) => x.text));
    const msg = s.greedy
      ? `그리디는 난수를 쓰지 않으므로 seed가 달라도 <b>${SAMPLE_COUNT}번 모두 같다</b>.`
      : unique === 1
        ? `<b>${SAMPLE_COUNT}번 모두 같다</b> — 분포가 너무 좁아 난수가 끼어들 틈이 없다.`
        : `서로 다른 결과 <b>${unique}/${SAMPLE_COUNT}</b>`;
    $('[data-slot=samples-sum]').innerHTML = `${msg} · ${SAMPLE_COUNT}개를 합친 distinct-2 <b>${d2.toFixed(2)}</b> <span class="w08-muted">(같은 글이 겹치면 낮아진다)</span>`;
  }

  function verdict({ loops, avgLogP, avgKept, greedyRun, main, copyRate, genText, dropped, M }) {
    const notes = [];
    if (dropped) notes.push(`<p class="w08-muted">프롬프트에서 어휘에 없는 글자 ${dropped}개는 뺐다.</p>`);
    const box = (cls, title, body) => `<div class="callout ${cls}"><span class="callout__title">${title}</span><p>${body}</p></div>`;
    if (!main.steps.length) return notes.join('') + box('', '처음부터 한 글자씩', '▶ 한 글자를 누르면 ①의 분포에서 글자 하나를 뽑아 붙인다. 뽑기 전에 어떤 글자가 나올지 먼저 예상해 본다.');
    const big = loops.find((l) => l.times * [...l.unit].length >= 6) ?? loops[0];
    const loopNote = big ? ` 그래서 “${esc(showWs(big.unit))}”를 ${big.times}번 연달아 되풀이했다.` : '';
    if (copyRate !== null && copyRate >= 0.5) {
      return notes.join('') + box('callout--more', `📋 원문 베끼기 · 복사율 ${fmtPct(copyRate)}`, '이 모델은 코퍼스를 외웠다. 대부분의 자리에서 분포가 한 글자에 몰려 있어(엔트로피 ≈ 0) 샘플링을 해도 원문이 그대로 나온다. 원문에 갈림길이 있는 자리에서만 결과가 갈린다. 온도를 1.5 이상으로 올려야 비로소 무너진다.');
    }
    if (greedyRun) {
      const g = [...M.vocab.decode(greedyRun.ids.slice(greedyRun.promptLen))];
      const same = [...genText].filter((c, i) => c === g[i]).length;
      const effective = mean(main.steps.map((st) => 2 ** st.hFinal));
      return notes.join('') + box('callout--more', `🎯 사실상 그리디 · 실질 후보 ${effective.toFixed(1)}개`, `샘플링을 켰지만 뽑는 분포가 거의 한 글자에 몰려 있다(평균 남은 후보 ${avgKept.toFixed(1)}개). 그리디로 만든 글과 ${[...genText].length}글자 중 ${same}글자가 같은 자리에 같다.${loopNote} 다양성을 원하면 top-p를 0.9 근처로, 온도를 0.7 이상으로 둔다.`);
    }
    if (big) {
      const why = s.greedy
        ? '그리디는 같은 문맥에서 항상 같은 글자를 고른다. 반복에 들어서면 문맥도 반복이 되어 같은 선택을 이어 간다. 반복 페널티를 1.2쯤 주거나 샘플링으로 바꾼다.'
        : '분포가 좁아 매번 같은 후보가 뽑혔다. 온도를 올리거나 반복 페널티를 1.2쯤 주면 빠져나온다.';
      return notes.join('') + box('callout--danger', `🔁 반복 루프 · “${esc(showWs(big.unit))}” × ${big.times}`, why);
    }
    if (avgLogP !== null && avgLogP < -3) {
      return notes.join('') + box('callout--danger', `🌀 횡설수설 · 평균 log P ${avgLogP.toFixed(2)}`, `모델 스스로도 가능성이 낮다고 본 글자(빨간 글자)를 자주 뽑았다. 온도 ${s.T.toFixed(1)}이 분포를 평평하게 펴서 꼬리의 엉뚱한 후보가 살아났다. 온도를 낮추거나 top-p로 꼬리를 자른다.`);
    }
    if (s.penalty >= 1.5) {
      const spaces = [...genText].filter((c) => c === ' ').length / Math.max(1, [...genText].length);
      return notes.join('') + box('callout--more', `✂️ 페널티 ${s.penalty.toFixed(1)} · 공백 비율 ${fmtPct(spaces)}`, '반복은 사라졌지만 공백·조사처럼 원래 자주 써야 하는 글자까지 점수가 깎인다. 띄어쓰기가 줄고 낱말이 들러붙는지 본다.');
    }
    return notes.join('') + box('callout--ok', s.greedy ? '그리디 · 이번에는 루프가 없다' : '균형 잡힌 설정', s.greedy ? '길이를 늘리거나 프롬프트를 바꿔 본다. 그리디는 언젠가 같은 조각을 되풀이하기 쉽다.' : '빨간 글자(모델이 드물다고 본 글자)가 적고 루프도 없다. ③에서 seed마다 다른 글이 나오는지 확인한다.');
  }

  // ---------------------------------------------------------------- events

  const on = (sel, type, fn) => $(sel).addEventListener(type, fn, { signal: ctrl.signal });
  const changed = (delay) => {
    s.pick = null;
    syncControls();
    schedule(delay);
  };
  for (const key of ['T', 'p', 'penalty']) on(`[data-in=${key}]`, 'input', (e) => { s[key] = Number(e.target.value); changed(); });
  on('[data-in=k]', 'input', (e) => { s.k = Number(e.target.value); changed(); });
  on('[data-in=len]', 'input', (e) => {
    s.len = Number(e.target.value);
    s.cursor = Math.min(s.cursor, s.len);
    changed();
  });
  on('[data-in=prompt]', 'input', (e) => {
    s.prompt = e.target.value;
    s.cursor = 0;
    changed(250);
  });
  on('[data-in=seed]', 'change', (e) => {
    s.seed = clampInt(e.target.value, 1, 9999, 1);
    changed(0);
  });
  on('[data-in=model]', 'change', async (e) => {
    const id = e.target.value;
    setStatus(`<span class="spinner" aria-hidden="true"></span> <span>${esc(MODELS.find((m) => m.id === id).label)} 불러오는 중…</span>`);
    try {
      await ensureModel(id);
      if (ctrl.signal.aborted) return;
      s.modelId = id;
      changed(0);
    } catch (err) {
      e.target.value = s.modelId;
      setStatus(`<div class="widget__error">이 체크포인트를 불러오지 못했다 (${esc(err.message)}). 지금 모델로 계속한다.</div>`);
    }
  });
  root.addEventListener(
    'click',
    (e) => {
      const b = e.target.closest('button');
      if (!b || !root.contains(b)) return;
      if (b.dataset.strategy) {
        s.greedy = b.dataset.strategy === 'greedy';
        changed(0);
      } else if (b.dataset.preset) {
        Object.assign(s, PRESETS[b.dataset.preset]);
        s.stepMode = false;
        if (b.dataset.preset === 'reset') {
          s.seed = 1;
          s.len = 80;
        }
        changed(0);
      } else if (b.dataset.act === 'dice') {
        s.seed = (s.seed % 9999) + 1;
        changed(0);
      } else if (b.dataset.act === 'step') {
        if (!s.stepMode) {
          s.stepMode = true;
          s.cursor = 0;
        }
        s.cursor = Math.min(s.len, s.cursor + 1);
        if (s.cursor >= s.len) s.stepMode = false;
        s.pick = null;
        schedule(0);
      } else if (b.dataset.act === 'rewind') {
        s.stepMode = true;
        s.cursor = 0;
        s.pick = null;
        schedule(0);
      } else if (b.dataset.act === 'finish') {
        s.stepMode = false;
        s.pick = null;
        schedule(0);
      }
    },
    { signal: ctrl.signal },
  );
  out.addEventListener(
    'click',
    (e) => {
      const b = e.target.closest('[data-step]');
      if (!b || !view) return;
      s.pick = Number(b.dataset.step);
      out.querySelectorAll('.w08-ch.sel').forEach((x) => x.classList.remove('sel'));
      b.classList.add('sel');
      renderDist();
    },
    { signal: ctrl.signal },
  );

  syncControls();
  try {
    await ensureModel(s.modelId);
    if (ctrl.signal.aborted) return;
    setStatus('');
    loadCorpus()
      .then((c) => {
        source = corpusText(c);
        if (view && !ctrl.signal.aborted) renderMain();
      })
      .catch(() => {});
    regenerate();
  } catch (err) {
    setStatus(`<div class="widget__error">미니 GPT 체크포인트를 불러오지 못했다 (${esc(err.message)}). <code>file://</code>로 열었다면 <code>python -m http.server</code>로 연다.</div>`);
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

// ---------------------------------------------------------------- engine helpers

const frame = () => (globalThis.scheduler?.yield ? globalThis.scheduler.yield() : new Promise((r) => setTimeout(r, 0)));

/** Sampling that behaves like greedy: ≤ 1.5 candidates left, or the final distribution is almost one-hot. */
export function isNearGreedy(steps) {
  return mean(steps.map((st) => st.kept)) <= 1.5 || mean(steps.map((st) => st.hFinal)) < 0.25;
}

function encodePrompt(vocab, prompt) {
  const ids = vocab.encode(prompt);
  const dropped = [...prompt].length - ids.length;
  // an empty context cannot be fed to the model: start from a newline
  return { ids: ids.length ? ids : vocab.encode('\n'), dropped };
}

function cacheKey(M, ids) {
  return ids.slice(-M.model.config.blockSize).join(',');
}

function cachedLogits(M, ids) {
  const key = cacheKey(M, ids);
  let hit = M.cache.get(key);
  if (!hit) {
    if (M.cache.size > 5000) M.cache.clear();
    hit = Float32Array.from(nextLogits(M.model, ids));
    M.cache.set(key, hit);
  }
  return hit;
}

/** Same loop as gpt.generate(): identical seeds give identical text in the Challenges. */
function createRun(M, promptIds, decode, seed, keepLogits) {
  return { ids: [...promptIds], promptLen: promptIds.length, steps: [], rand: mulberry32(seed), decode, keepLogits };
}

function advance(M, run) {
  const logits = cachedLogits(M, run.ids);
  const arr = Array.from(logits);
  const { index, probs } = decodeStep(arr, { ...run.decode, history: run.ids.slice(-M.model.config.blockSize), rand: run.rand });
  let kept = 0;
  for (const q of probs) if (q > 0) kept++;
  run.steps.push({ index, pModel: softmax(arr)[index], pFinal: probs[index], kept, hFinal: entropyBits(probs), logits: run.keepLogits ? logits : null });
  run.ids.push(index);
}

// ---------------------------------------------------------------- pure helpers (tested in Node)

/** The three distributions shown in ①: model, after penalty+temperature, after top-k/top-p. */
export function stages(logits, history, { greedy, temperature, k, p, penalty }) {
  const arr = Array.from(logits);
  const orig = softmax(arr);
  const adjusted = repetitionPenalty(arr, history, penalty);
  const temp = softmax(adjusted, greedy ? 1 : temperature);
  let fin;
  if (greedy) {
    let best = 0;
    for (let i = 1; i < adjusted.length; i++) if (adjusted[i] > adjusted[best]) best = i;
    fin = adjusted.map((_, i) => (i === best ? 1 : 0));
  } else fin = topP(topK(temp, k), p);
  return { orig, temp, fin };
}

/** Units of 1–8 characters repeated back to back at least `minTimes` times. */
export function findLoops(text, minTimes = 3) {
  const cs = [...text];
  const found = [];
  let i = 0;
  while (i < cs.length) {
    let best = null;
    for (let L = 1; L <= 8; L++) {
      const unit = cs.slice(i, i + L).join('');
      if (unit.length < L || !unit.trim()) continue;
      let times = 1;
      while (cs.slice(i + times * L, i + (times + 1) * L).join('') === unit) times++;
      if (times >= minTimes && (!best || times * L > best.times * [...best.unit].length)) best = { unit, times, at: i };
    }
    if (best) {
      found.push(best);
      i += best.times * [...best.unit].length;
    } else i++;
  }
  return found;
}

/** For each character: is it inside an n-character chunk that already appeared earlier? */
export function repeatMask(chars, n = REP_N) {
  const mask = new Array(chars.length).fill(false);
  const seen = new Set();
  for (let i = 0; i + n <= chars.length; i++) {
    const g = chars.slice(i, i + n).join('');
    if (!g.trim()) continue;
    if (seen.has(g)) for (let j = i; j < i + n; j++) mask[j] = true;
    seen.add(g);
  }
  return mask;
}

/** Distinct 2-gram ratio over several texts (2-grams never cross text boundaries). */
export function distinct2(texts) {
  const grams = [];
  for (const t of texts) {
    const cs = [...t];
    for (let i = 0; i + 1 < cs.length; i++) grams.push(cs[i] + cs[i + 1]);
  }
  return grams.length ? new Set(grams).size / grams.length : 0;
}

/** Maximal chunks (≥ REP_N chars, not only spaces) that occur at least twice, most frequent first. */
export function topRepeats(text, limit = 4) {
  const counts = new Map();
  for (let L = REP_N; L <= 12; L++) {
    for (let i = 0; i + L <= text.length; i++) {
      const g = text.slice(i, i + L);
      if (!g.trim() || g.includes('\n')) continue;
      counts.set(g, (counts.get(g) ?? 0) + 1);
    }
  }
  const rep = [...counts].filter(([, c]) => c >= 2);
  // keep only chunks that are not part of a longer chunk with the same count
  const maximal = rep.filter(([g, c]) => !rep.some(([h, d]) => h.length > g.length && d === c && h.includes(g)));
  return maximal
    .sort((a, b) => b[1] * b[0].length - a[1] * a[0].length)
    .slice(0, limit)
    .map(([t, count]) => ({ text: t, count }));
}

/** For each generated character: is it inside a run of ≥ COPY_MIN chars found verbatim in the corpus? */
export function copyMask(gen, source) {
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

const mean = (xs) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : 0);

function clampInt(v, lo, hi, dflt) {
  const n = Math.round(Number(v));
  return Number.isFinite(n) ? Math.min(hi, Math.max(lo, n)) : dflt;
}

function pct(x) {
  if (x >= 0.995) return `${(x * 100).toFixed(0)}%`;
  if (x >= 0.1) return `${(x * 100).toFixed(1)}%`;
  if (x >= 0.001) return `${(x * 100).toFixed(1)}%`;
  return x > 0 ? '<0.1%' : '0%';
}

function fmtPct(x) {
  return `${(x * 100).toFixed(0)}%`;
}

const stat = (label, value, title = '') => `<div class="stat" title="${esc(title)}"><span class="stat__label">${label}</span><span class="stat__value">${value}</span></div>`;

function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
}
