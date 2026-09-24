// w02 BPE 병합 스텝퍼
// One concept: BPE builds its vocabulary by repeatedly merging the most frequent
// adjacent pair. Each merge grows the vocabulary by one and shrinks the text.
// Failure modes on screen: byte fragments of Korean syllables, unseen words
// falling back to single characters/bytes ([UNK] in char mode), running out of
// pairs on a short text, and late merges that memorize whole 어절.

import { registerOutput, unregisterOutput } from '../site/result.js';
import { loadCorpus, corpusText, sentences } from '../core/text.js';
import { trainSteps, encode, showSymbol, pretokenize, WORD_START } from '../core/bpe.js';

const MAX_MERGES = 400;
const CHUNK_MS = 12; // training runs in slices so the page never freezes
const SOURCES = [
  { id: 'all', label: '코퍼스 전체 (17문서 · 6,828자)' },
  { id: 'facility-rules', label: '실습실 이용 규정 한 편 (284자)' },
  { id: 'llm', label: 'LLM 교안만 (8문서)' },
  { id: 'custom', label: '직접 입력' },
];
const TESTS = [
  { id: 'unseen', label: '처음 보는 문장', text: '메타버스 강의실에서 챗봇으로 토크나이저를 배운다.' },
  { id: 'seen', label: '코퍼스 문장', text: '실습실은 수업 시간 외에도 평일 오후 9시까지 개방된다.' },
  { id: 'en', label: '영어 문장', text: 'Tokenizers split rare words into pieces.' },
];
const CUSTOM_DEFAULT =
  '토큰은 LLM이 글을 읽는 단위다. 토크나이저는 글을 토큰으로 자르고 토큰마다 번호를 붙인다. ' +
  '토큰이 길면 문장이 짧아지고 토큰이 짧으면 문장이 길어진다. 토큰 수는 곧 비용이다.';

const template = document.createElement('template');
template.innerHTML = `
  <div class="widget w02-bpe">
    <h3 class="widget__title">BPE 병합 스텝퍼</h3>
    <div data-slot="status" class="widget__status" role="status" aria-live="polite">
      <span class="spinner" aria-hidden="true"></span> <span>문서셋 불러오는 중…</span>
    </div>
    <div class="widget__controls">
      <label class="field">
        <span class="field__label">학습 텍스트</span>
        <select data-in="source"></select>
      </label>
      <fieldset class="field w02-unit">
        <legend class="field__label">시작 단위</legend>
        <div class="w02-seg" role="radiogroup" aria-label="시작 단위">
          <label><input type="radio" name="w02-unit" value="char" checked> 글자 단위</label>
          <label><input type="radio" name="w02-unit" value="byte"> UTF-8 바이트 단위</label>
        </div>
      </fieldset>
    </div>
    <label class="field w02-custom" data-slot="custom" hidden>
      <span class="field__label">직접 입력한 학습 텍스트 <output data-out="customLen"></output></span>
      <textarea data-in="custom" spellcheck="false" maxlength="30000"></textarea>
    </label>
    <label class="field w02-merges">
      <span class="field__label">병합 횟수 <output data-out="merges"></output></span>
      <input type="range" data-in="merges" min="0" max="${MAX_MERGES}" step="1">
    </label>
    <div class="btn-row">
      <button type="button" class="btn small ghost" data-act="prev" aria-label="병합 한 단계 되돌리기">◀ 한 단계</button>
      <button type="button" class="btn small ghost" data-act="next" aria-label="병합 한 단계 더 하기">한 단계 ▶</button>
      <button type="button" class="btn small primary" data-act="play" aria-pressed="false">▶ 재생</button>
      <button type="button" class="btn small ghost" data-act="zero">0회 (시작 단위 그대로)</button>
      <button type="button" class="btn small ghost" data-act="max">400회</button>
      <span class="w02-train" data-out="train" aria-live="polite"></span>
    </div>
    <label class="field w02-test">
      <span class="field__label">테스트 문장 (고쳐 써 본다)</span>
      <input type="text" data-in="test" maxlength="200" spellcheck="false">
    </label>
    <div class="btn-row" data-slot="presets"></div>
    <div data-slot="out-inline"></div>
  </div>`;

const outTemplate = document.createElement('template');
outTemplate.innerHTML = `
  <div class="w02-out">
    <div data-slot="verdict" aria-live="polite"></div>
    <h4>새로 배운 병합 <small class="w02-muted">— 최근 것이 위</small></h4>
    <ol class="w02-merges-list" data-slot="merges"></ol>
    <h4>테스트 문장 <small class="w02-muted" data-slot="test-info"></small></h4>
    <div class="w02-toks" data-slot="test"></div>
    <h4>코퍼스 문장 <small class="w02-muted" data-slot="seen-info"></small></h4>
    <div class="w02-toks" data-slot="seen"></div>
    <div class="legend" aria-hidden="true">
      <span><i class="w02-lg-merged"></i>병합된 토큰</span>
      <span><i class="w02-lg-single"></i>낱글자·낱바이트</span>
      <span><i class="w02-lg-frag"></i>음절이 깨진 바이트 조각</span>
      <span><i class="w02-lg-unk"></i>어휘에 없음 [UNK]</span>
    </div>
    <div class="stat-row" data-slot="stats"></div>
    <h4>어휘 크기 ↔ 전체 토큰 수 <small class="w02-muted">— 병합할수록 오른쪽 아래로 간다</small></h4>
    <div class="w02-chart" data-slot="chart"></div>
  </div>`;

const state = new WeakMap();
let seq = 0;

/**
 * @param {HTMLElement} el
 * @param {{ source?: string, unit?: 'char'|'byte', merges?: number, test?: string, outputTitle?: string }} options
 */
export async function mount(el, options = {}) {
  const root = template.content.firstElementChild.cloneNode(true);
  const out = outTemplate.content.firstElementChild.cloneNode(true);
  el.replaceChildren(root);
  const $ = (sel) => root.querySelector(sel) ?? out.querySelector(sel);
  const ctrl = new AbortController();
  const outputId = `w02-bpe:${++seq}`;
  const s = {
    source: options.source ?? 'all',
    unit: options.unit ?? 'char',
    target: options.merges ?? 100,
    test: options.test ?? TESTS[0].text,
    custom: CUSTOM_DEFAULT,
    corpus: null,
    timer: 0,
    playing: 0,
    raf: 0,
  };
  const runs = new Map(); // `${unit}\u0001${text}` → run
  state.set(el, { ctrl, outputId, s });

  // unique radio name per instance (student page + demo slide can coexist)
  root.querySelectorAll('input[type=radio]').forEach((r) => {
    r.name = `w02-unit-${seq}`;
    r.checked = r.value === s.unit;
  });
  $('[data-in=source]').innerHTML = SOURCES.map((o) => `<option value="${o.id}">${esc(o.label)}</option>`).join('');
  $('[data-in=source]').value = s.source;
  $('[data-in=merges]').value = String(s.target);
  $('[data-in=test]').value = s.test;
  $('[data-in=custom]').value = s.custom;
  $('[data-slot=custom]').hidden = s.source !== 'custom';
  $('[data-slot=presets]').innerHTML = TESTS.map(
    (t) => `<button type="button" class="btn small ghost" data-test="${t.id}">${esc(t.label)}</button>`,
  ).join('');

  registerOutput(outputId, { title: options.outputTitle ?? 'BPE 병합 결과', node: out, inlineHost: $('[data-slot=out-inline]') });

  // ------------------------------------------------------------ training (cached, chunked)

  function sourceText() {
    if (s.source === 'custom') return s.custom;
    const docs = s.corpus.documents;
    if (s.source === 'all') return corpusText(s.corpus);
    if (s.source === 'llm') return corpusText(s.corpus, docs.filter((d) => d.course === 'studyLLM').map((d) => d.id));
    return corpusText(s.corpus, [s.source]);
  }

  /** Get (and start if needed) the training run for (text, unit). */
  function getRun(text, unit) {
    const key = `${unit}\u0001${text}`;
    let run = runs.get(key);
    if (run) return run;
    const gen = trainSteps(text, { merges: MAX_MERGES, unit, minCount: 2 });
    const first = gen.next().value;
    run = {
      key,
      text,
      unit,
      gen,
      done: false,
      base: new Set(first.words.flatMap((w) => w.syms)),
      trainWords: new Set(first.words.map((w) => w.word)),
      steps: [{ step: 0, merge: null, vocabSize: first.vocabSize, tokens: first.tokens }],
    };
    runs.set(key, run);
    // keep the cache small: drop the oldest finished runs
    if (runs.size > 8) for (const [k, r] of runs) if (runs.size > 8 && r.text !== text) runs.delete(k);
    pump();
    return run;
  }

  /** Advance whichever run the screen needs, CHUNK_MS at a time. */
  function pump() {
    if (s.timer || ctrl.signal.aborted) return;
    s.timer = setTimeout(() => {
      s.timer = 0;
      if (ctrl.signal.aborted) return;
      const text = sourceText();
      const cur = runs.get(`${s.unit}\u0001${text}`);
      const other = runs.get(`${s.unit === 'char' ? 'byte' : 'char'}\u0001${text}`);
      const run = [cur, other].find((r) => r && !r.done);
      if (!run) return;
      const t0 = performance.now();
      while (performance.now() - t0 < CHUNK_MS) {
        const { value, done } = run.gen.next();
        if (done) {
          run.done = true;
          run.gen = null;
          break;
        }
        run.steps.push({ step: value.step, merge: value.merge, vocabSize: value.vocabSize, tokens: value.tokens });
      }
      // once the current unit is done, train the other unit too (for the chart)
      if (run === cur && cur.done && !other) getRun(text, s.unit === 'char' ? 'byte' : 'char');
      schedule();
      if ([cur, other].some((r) => r && !r.done)) pump();
    }, 0);
  }

  function schedule() {
    if (s.raf) return;
    s.raf = requestAnimationFrame(() => {
      s.raf = 0;
      render();
    });
  }

  // ------------------------------------------------------------ render

  function render() {
    if (!s.corpus || ctrl.signal.aborted) return;
    const text = sourceText();
    if (!text.trim()) {
      $('[data-slot=verdict]').innerHTML = callout('danger', '학습 텍스트가 비었다', '글을 한 문장 이상 넣어야 셀 쌍이 생긴다.');
      return;
    }
    const run = getRun(text, s.unit);
    const other = runs.get(`${s.unit === 'char' ? 'byte' : 'char'}\u0001${text}`);
    const avail = run.steps.length - 1;
    const k = Math.min(s.target, avail);
    const st = run.steps[k];
    const merges = run.steps.slice(1, k + 1).map((x) => x.merge);
    const tok = { merges, unit: s.unit };
    const vocab = new Set(run.base);
    merges.forEach((m) => vocab.add(m.merged));

    $('[data-in=merges]').value = String(s.target);
    $('[data-out=merges]').textContent = k < s.target ? `${k}회 (요청 ${s.target})` : `${k}회`;
    $('[data-out=train]').textContent = run.done
      ? avail < MAX_MERGES
        ? `학습 완료 · ${avail}회에서 합칠 쌍이 없음`
        : `학습 완료 · 최대 ${MAX_MERGES}회`
      : `학습 중 ${avail} / ${MAX_MERGES}…`;
    $('[data-out=customLen]').textContent = `${s.custom.length.toLocaleString()}자`;

    // newest merges
    const recent = run.steps.slice(Math.max(1, k - 7), k + 1).reverse();
    $('[data-slot=merges]').innerHTML = recent.length
      ? recent
          .map(
            (x, i) =>
              `<li class="${i === 0 ? 'new' : ''}"><span class="n">#${x.step}</span><span class="m"><code>${sym(x.merge.a, s.unit)}</code> + <code>${sym(x.merge.b, s.unit)}</code> → <code class="r">${sym(x.merge.merged, s.unit)}</code></span><span class="c">×${x.merge.count}</span></li>`,
          )
          .join('')
      : `<li class="w02-muted">아직 병합하지 않았다. 토큰 = ${s.unit === 'byte' ? 'UTF-8 바이트 하나' : '글자 하나'}. ▶ 한 단계를 누른다.</li>`;

    // test sentence + corpus sentence
    const seen = sentences(text)[0]?.slice(0, 80) ?? '';
    const test = tokenize(tok, vocab, run, s.test);
    const seenT = tokenize(tok, vocab, run, seen);
    $('[data-slot=test]').innerHTML = test.html || '<span class="w02-muted">문장을 입력한다</span>';
    $('[data-slot=test-info]').textContent = `— ${test.n}토큰 / ${[...s.test].length}자`;
    $('[data-slot=seen]').innerHTML = seenT.html;
    $('[data-slot=seen-info]').textContent = `— 학습 텍스트의 첫 문장 · ${seenT.n}토큰 / ${[...seen].length}자`;

    // stats
    const base0 = run.steps[0];
    const saved = 1 - st.tokens / base0.tokens;
    $('[data-slot=stats]').innerHTML = [
      stat('어휘 크기', st.vocabSize.toLocaleString()),
      stat('학습 텍스트 토큰', `${st.tokens.toLocaleString()}${k ? ` <small>(−${(saved * 100).toFixed(0)}%)</small>` : ''}`),
      stat('글자 / 토큰', (text.length / st.tokens).toFixed(2)),
      stat('테스트 문장 토큰', `${test.n} <small>/ ${[...s.test].length}자</small>`),
    ].join('');

    // failure modes
    $('[data-slot=verdict]').innerHTML = verdict({ run, k, avail, test, st, textLen: text.length });

    // chart
    $('[data-slot=chart]').innerHTML = chart(run, other, k);
  }

  function tokenize(tok, vocab, run, sentence) {
    const toks = encode(tok, sentence);
    const words = pretokenize(sentence);
    const unseenWords = words.filter((w) => !run.trainWords.has(w));
    let frag = 0;
    let single = 0;
    const unk = [];
    const html = toks
      .map((t, i) => {
        const shown = showSymbol(t, s.unit);
        const body = t.startsWith(WORD_START) ? t.slice(1) : t;
        let cls = `c${i % 3}`;
        let title = `${i + 1}번째 토큰`;
        if (s.unit === 'char' && !vocab.has(t)) {
          cls = 'unk';
          unk.push(t);
          title = `“${t}”는 학습 텍스트에 없던 글자다. 실제 모델에서는 [UNK] 하나가 된다`;
        } else if (s.unit === 'byte' && /<[0-9A-F]{2}>/.test(shown)) {
          cls = 'frag';
          frag++;
          title = '한글 음절(3바이트)의 일부만 들어 있는 바이트 조각이다';
        } else if ([...body].length === 1 && body !== '') {
          cls += ' single';
          single++;
          title = '더 합쳐지지 못한 낱개 단위';
        } else if (body === '') {
          cls = 'ws';
          title = '단어 시작 표시(▁ = 앞의 공백)만 남은 토큰';
        }
        const label = shown.startsWith(WORD_START) ? `<i class="ws-mark">▁</i>${esc(shown.slice(1))}` : esc(shown);
        return `<span class="w02-tok ${cls}" title="${esc(title)}">${label || '<i class="ws-mark">▁</i>'}</span>`;
      })
      .join('');
    return { html, n: toks.length, chars: [...sentence].length, frag, single, unk, unseenWords };
  }

  // ------------------------------------------------------------ events

  const on = (sel, type, fn) => $(sel).addEventListener(type, fn, { signal: ctrl.signal });
  const setTarget = (v) => {
    s.target = Math.max(0, Math.min(MAX_MERGES, v));
    render();
  };
  on('[data-in=source]', 'change', (e) => {
    s.source = e.target.value;
    $('[data-slot=custom]').hidden = s.source !== 'custom';
    stopPlay();
    render();
  });
  root.querySelectorAll('input[type=radio]').forEach((r) =>
    r.addEventListener(
      'change',
      (e) => {
        if (!e.target.checked) return;
        s.unit = e.target.value;
        render();
      },
      { signal: ctrl.signal },
    ),
  );
  let debounce = 0;
  on('[data-in=custom]', 'input', (e) => {
    clearTimeout(debounce);
    debounce = setTimeout(() => {
      s.custom = e.target.value;
      render();
    }, 350);
  });
  on('[data-in=merges]', 'input', (e) => {
    stopPlay();
    setTarget(Number(e.target.value));
  });
  on('[data-in=test]', 'input', (e) => {
    s.test = e.target.value;
    render();
  });
  root.addEventListener(
    'click',
    (e) => {
      const b = e.target.closest('button');
      if (!b) return;
      if (b.dataset.test) {
        s.test = TESTS.find((t) => t.id === b.dataset.test).text;
        $('[data-in=test]').value = s.test;
        render();
        return;
      }
      const act = b.dataset.act;
      if (act === 'play') return s.playing ? stopPlay() : startPlay();
      if (!act) return;
      stopPlay();
      const run = runs.get(`${s.unit}\u0001${sourceText()}`);
      const avail = run ? run.steps.length - 1 : 0;
      const cur = Math.min(s.target, avail);
      if (act === 'prev') setTarget(cur - 1);
      if (act === 'next') setTarget(cur + 1);
      if (act === 'zero') setTarget(0);
      if (act === 'max') setTarget(MAX_MERGES);
    },
    { signal: ctrl.signal },
  );

  function startPlay() {
    const run = runs.get(`${s.unit}\u0001${sourceText()}`);
    const avail = run ? run.steps.length - 1 : 0;
    if (run?.done && s.target >= avail) s.target = 0; // replay from the start
    s.playing = setInterval(() => {
      const r = runs.get(`${s.unit}\u0001${sourceText()}`);
      if (!r || ctrl.signal.aborted) return stopPlay();
      const have = r.steps.length - 1;
      const next = Math.min(s.target, have) + (s.target < 40 ? 1 : 3);
      if (r.done && next > have) {
        s.target = Math.min(MAX_MERGES, have);
        render();
        return stopPlay();
      }
      if (next > have) return; // wait for training to catch up
      setTarget(next);
    }, 90);
    const b = $('[data-act=play]');
    b.textContent = '⏸ 정지';
    b.setAttribute('aria-pressed', 'true');
  }

  function stopPlay() {
    if (!s.playing) return;
    clearInterval(s.playing);
    s.playing = 0;
    const b = $('[data-act=play]');
    b.textContent = '▶ 재생';
    b.setAttribute('aria-pressed', 'false');
  }
  ctrl.signal.addEventListener('abort', () => {
    stopPlay();
    clearTimeout(s.timer);
    clearTimeout(debounce);
    cancelAnimationFrame(s.raf);
  });

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

function verdict({ run, k, avail, test, st, textLen }) {
  const unitName = run.unit === 'byte' ? '바이트' : '글자';
  const list = [];
  if (run.done && avail < MAX_MERGES && k === avail) {
    list.push(
      callout(
        'danger',
        `합칠 쌍이 바닥났다 · ${avail}회에서 멈춤`,
        `남은 인접 쌍이 모두 학습 텍스트에 한 번씩만 나온다. 두 번 이상 나온 쌍만 합치므로 텍스트가 짧으면 어휘를 더 키울 수 없다. 실제 토크나이저를 수십억 글자로 학습하는 이유다.`,
      ),
    );
  }
  if (test.unk.length) {
    list.push(
      callout(
        'danger',
        `어휘에 없는 글자 ${test.unk.length}개 → [UNK]`,
        `“${esc([...new Set(test.unk)].join(', '))}”는 학습 텍스트에 한 번도 없던 글자라 글자 단위 어휘에 들어 있지 않다. 모델은 이 글자를 “모르는 것” 하나로만 본다. <b>UTF-8 바이트 단위</b>로 바꾸면 어떤 글자든 256개 바이트로 표현되므로 [UNK]가 사라진다.`,
      ),
    );
  }
  if (test.frag) {
    list.push(
      callout(
        'danger',
        `한글 음절이 바이트 조각으로 깨졌다 · 조각 ${test.frag}개`,
        `한글 음절 하나는 UTF-8로 3바이트다. 아직 합쳐지지 않은 &lt;EC&gt;&lt;A7&gt; 같은 조각은 사람이 읽을 수 없고 토큰 수만 늘린다. 테스트 문장이 ${test.n}토큰(글자당 ${ratio(test)}토큰)이다. 영어 중심 어휘에서 한국어가 비싼 이유가 바로 이것이다.`,
      ),
    );
  }
  if (test.unseenWords.length && test.single >= 3 && k > 0) {
    list.push(
      callout(
        'more',
        `처음 보는 어절 ${test.unseenWords.length}개는 낱${unitName}로 돌아간다`,
        `“${esc(test.unseenWords.map((w) => w.slice(1)).slice(0, 4).join('”, “'))}”는 학습 텍스트에 없던 어절이다. 배운 병합 규칙이 맞는 부분만 합치고 나머지는 낱${unitName}로 남는다(낱개 ${test.single}개). 그래도 모르는 “단어”가 되지는 않는다. 이것이 BPE가 단어 단위 어휘보다 나은 점이다.`,
      ),
    );
  }
  const last = run.steps[k]?.merge;
  const lastShown = last ? showSymbol(last.merged, run.unit) : '';
  if (last && last.count <= 3 && !lastShown.includes('<') && [...lastShown].length >= 4) {
    list.push(
      callout(
        'danger',
        `외우기 시작 · 방금 병합은 ×${last.count}`,
        `“${esc(lastShown)}”처럼 학습 텍스트에 ${last.count}번 나온 어절을 통째로 어휘에 넣고 있다. 이런 토큰은 다른 문서에서 거의 쓰이지 않는다. 병합을 더 해도 새 문장의 토큰 수는 별로 줄지 않는다.`,
      ),
    );
  }
  if (k === 0) {
    list.unshift(
      callout(
        '',
        `병합 0회 = ${unitName} 단위 토큰화`,
        `토큰 하나가 ${unitName} 하나다. 학습 텍스트 ${textLen.toLocaleString()}자가 ${st.tokens.toLocaleString()}토큰이다${run.unit === 'byte' ? ' — 한글이 3바이트라 글자보다 훨씬 많다' : ''}. ▶ 재생을 눌러 가장 자주 붙어 나오는 쌍부터 합쳐 본다.`,
      ),
    );
  }
  if (!list.length) {
    list.push(
      callout(
        'ok',
        `병합 ${k}회 · 토큰 ${(100 * (1 - st.tokens / run.steps[0].tokens)).toFixed(0)}% 절약`,
        `자주 나오는 조각(어미, 조사, 자주 쓰는 단어)이 토큰 하나가 되었다. 어휘는 ${k}개 늘었고 같은 텍스트가 더 적은 토큰으로 표현된다.`,
      ),
    );
  }
  return list.slice(0, 3).join('');
}

function ratio(test) {
  return test.chars ? (test.n / test.chars).toFixed(1) : '0';
}

function chart(run, other, k) {
  const W = 340;
  const H = 180;
  const P = { l: 46, r: 12, t: 12, b: 34 };
  const series = [run, other].filter((r) => r && r.steps.length > 1);
  if (!series.includes(run) || !run.steps[k]) return '<p class="w02-muted w02-chart-note">병합을 학습하는 중… 곡선은 두 번째 병합부터 그려진다.</p>';
  const xs = series.flatMap((r) => [r.steps[0].vocabSize, r.steps.at(-1).vocabSize]);
  const x0 = Math.min(...xs);
  const x1 = Math.max(...xs, x0 + 1);
  const y1 = Math.max(...series.map((r) => r.steps[0].tokens)) * 1.05;
  const X = (v) => P.l + ((v - x0) / (x1 - x0)) * (W - P.l - P.r);
  const Y = (v) => H - P.b - (v / y1) * (H - P.t - P.b);
  const path = (r) => r.steps.map((p, i) => `${i ? 'L' : 'M'}${X(p.vocabSize).toFixed(1)} ${Y(p.tokens).toFixed(1)}`).join('');
  const yTicks = niceTicks(y1, 4);
  const cur = run.steps[k];
  const name = (u) => (u === 'byte' ? '바이트 단위' : '글자 단위');
  const lines = series
    .map((r) => {
      const end = r.steps.at(-1);
      const main = r === run;
      return `<path d="${path(r)}" class="${main ? 'ln-main' : 'ln-other'}"/>
        <text x="${Math.min(W - P.r, X(end.vocabSize)).toFixed(1)}" y="${(Y(end.tokens) - 6).toFixed(1)}" text-anchor="end" class="lab">${name(r.unit)}</text>`;
    })
    .join('');
  const cx = X(cur.vocabSize);
  const cy = Y(cur.tokens);
  return `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(`어휘 크기와 전체 토큰 수. 현재 ${name(run.unit)} 병합 ${k}회: 어휘 ${cur.vocabSize}, 토큰 ${cur.tokens}`)}">
    ${yTicks
      .map(
        (t) =>
          `<line x1="${P.l}" x2="${W - P.r}" y1="${Y(t).toFixed(1)}" y2="${Y(t).toFixed(1)}" class="grid"/><text x="${P.l - 5}" y="${(Y(t) + 3.5).toFixed(1)}" text-anchor="end" class="tick">${t.toLocaleString()}</text>`,
      )
      .join('')}
    <line x1="${P.l}" x2="${W - P.r}" y1="${H - P.b}" y2="${H - P.b}" class="axis"/>
    <text x="${P.l}" y="${H - P.b + 14}" class="tick">${x0}</text>
    <text x="${W - P.r}" y="${H - P.b + 14}" text-anchor="end" class="tick">${x1}</text>
    <text x="${(P.l + W - P.r) / 2}" y="${H - 4}" text-anchor="middle" class="ax">어휘 크기 →</text>
    <text x="4" y="${P.t + 2}" class="ax" transform="rotate(-90 4 ${P.t + 2})" text-anchor="end" dy="8">전체 토큰 수</text>
    ${lines}
    <line x1="${cx.toFixed(1)}" x2="${cx.toFixed(1)}" y1="${P.t}" y2="${H - P.b}" class="cross"/>
    <circle cx="${cx.toFixed(1)}" cy="${cy.toFixed(1)}" r="5" class="dot"><title>병합 ${k}회: 어휘 ${cur.vocabSize} · 토큰 ${cur.tokens}</title></circle>
    <text x="${Math.min(cx + 8, W - 90).toFixed(1)}" y="${Math.max(cy - 10, P.t + 10).toFixed(1)}" class="val">${k}회 · ${cur.tokens.toLocaleString()}토큰</text>
  </svg>
  <p class="w02-muted w02-chart-note">${other && other.steps.length > 1 ? '실선 = 지금 단위, 점선 = 다른 단위(같은 텍스트).' : '다른 단위 곡선은 학습이 끝나면 점선으로 나타난다.'} 어휘가 1 늘 때마다 토큰이 줄지만, 줄어드는 폭은 점점 작아진다.</p>`;
}

function niceTicks(max, n) {
  const raw = max / n;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 5, 10].map((m) => m * mag).find((v) => v >= raw);
  const out = [];
  for (let v = 0; v <= max; v += step) out.push(v);
  return out;
}

const sym = (x, unit) => esc(showSymbol(x, unit));

const callout = (kind, title, body) =>
  `<div class="callout${kind ? ` callout--${kind}` : ''}"><span class="callout__title">${esc(title)}</span><p>${body}</p></div>`;

const stat = (label, value) => `<div class="stat"><span class="stat__label">${label}</span><span class="stat__value">${value}</span></div>`;

function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
}
