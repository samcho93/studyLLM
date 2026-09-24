// w12 환각 해부
// One concept: a language model always produces a confident-looking distribution,
// whether or not it "knows". Confidence (average probability) does not track truth.
// Tab ① dissects our own mini GPT: per-character probability + a grounding check
// that finds the longest corpus span for every stretch of generated text.
// Tab ② shows the fix: retrieve a paragraph and put it in the prompt (RAG).

import { registerOutput, unregisterOutput } from '../site/result.js';
import { loadCorpus, showWs } from '../core/text.js';
import { deserialize, nextLogits } from '../core/gpt.js';
import { softmax } from '../core/sampling.js';
import { mulberry32, sampleIndex } from '../core/rng.js';
import { PROVIDERS, generate as llmGenerate, ragPrompt, setKey, hasKey, clearKey } from '../core/llm.js';

const MODELS = [
  { id: 'mini-gpt-3000', label: 'mini-gpt-3000 · 외운 모델 (3,000스텝, 학습 손실 ≈ 0.12)' },
  { id: 'mini-gpt-300', label: 'mini-gpt-300 · 덜 학습된 모델 (300스텝, 학습 손실 ≈ 1.8)' },
];

// topic: the document the prompt is about. gold: regex any of which marks the true fact.
const PRESETS = [
  { id: 'k1', group: 1, prompt: '실습실은 평일 오후', topic: ['facility-rules'], gold: ['9시까지'], truth: '실습실은 수업 시간 외에도 평일 오후 9시까지 개방된다. (실습실 이용 규정)' },
  { id: 'k2', group: 1, prompt: '한 사람이 연속으로 사용할 수 있는 시간은', topic: ['facility-rules'], gold: ['12시간'], truth: '한 사람이 연속으로 사용할 수 있는 시간은 최대 12시간이다. (실습실 이용 규정)' },
  { id: 'k3', group: 1, prompt: '수료 요건은', topic: ['dept-overview'], gold: ['80%'], truth: '수료 요건은 출석률 80% 이상과 팀 프로젝트 최종 발표 통과다. (학과 소개)' },
  { id: 'u1', group: 2, prompt: 'HNSW는', topic: ['rag-vectordb'], gold: ['그래프'], truth: 'HNSW는 벡터들을 여러 층의 그래프로 연결해 두고 위층에서 아래층으로 내려가며 탐색한다. (벡터 데이터베이스와 인덱스 — 학습에서 뺀 문서)' },
  { id: 'u2', group: 2, prompt: '벡터 데이터베이스는', topic: ['rag-vectordb'], gold: ['가까운 벡터'], truth: '벡터스토어는 임베딩 벡터와 원문, 메타데이터를 함께 저장하고 가장 가까운 벡터를 찾아 주는 저장소다. (학습에서 뺀 문서)' },
  { id: 'u3', group: 2, prompt: 'Flat 인덱스는', topic: ['rag-vectordb'], gold: ['하나씩 비교', '모든 벡터'], truth: 'Flat 인덱스는 질의 벡터를 저장된 모든 벡터와 하나씩 비교한다. (학습에서 뺀 문서)' },
  { id: 'f1', group: 3, prompt: '실습실은 토요일에', topic: ['facility-rules'], gold: ['사전 신청'], truth: '토요일(주말)에는 사전 신청한 학생만 이용할 수 있다. 모델이 이 점을 짚어야 맞다.' },
  { id: 'f2', group: 3, prompt: '실습실은 별관 5층', topic: ['dept-overview'], gold: ['본관'], truth: '별관 5층은 없다. 실습실은 본관 3층 305호와 306호다. 틀린 전제를 바로잡아야 맞다.' },
  { id: 'f3', group: 3, prompt: 'LLM 원리와 활용의 기말고사는', topic: ['course-llm-syllabus'], gold: ['없다', '없으며'], truth: '기말고사는 없다. 평가는 실습 과제 40% · 보고서 20% · 종합 프로젝트 30% · 참여 10%다.' },
];
const GROUPS = { 1: '① 학습한 사실 (외운 것)', 2: '② 학습하지 않은 문서 (rag-vectordb)', 3: '③ 존재하지 않는 사실 (틀린 전제)' };
const MATCH_MIN = 6; // a verbatim run this long counts as "found in the corpus"
const TOP_K = 2;

const template = document.createElement('template');
template.innerHTML = `
  <div class="widget w12-hal">
    <h3 class="widget__title">환각 해부</h3>
    <div data-slot="status" class="widget__status" role="status" aria-live="polite">
      <span class="spinner" aria-hidden="true"></span> <span>문서셋 불러오는 중…</span>
    </div>
    <div class="w12-tabs" role="tablist" aria-label="모드">
      <button type="button" class="w12-tab" role="tab" data-tab="gpt" aria-selected="true">① 미니 GPT의 환각</button>
      <button type="button" class="w12-tab" role="tab" data-tab="rag" aria-selected="false">② 근거를 붙이면</button>
    </div>

    <div class="w12-panel" data-panel="gpt" role="tabpanel">
      <div class="widget__controls">
        <label class="field">
          <span class="field__label">모델</span>
          <select data-in="model"></select>
        </label>
        <label class="field">
          <span class="field__label">시작 글자 (예제)</span>
          <select data-in="preset"></select>
        </label>
        <label class="field">
          <span class="field__label">직접 입력 <small class="w12-muted">바꾸면 예제 대신 쓴다</small></span>
          <input type="text" data-in="prompt" maxlength="40" spellcheck="false">
        </label>
        <label class="field">
          <span class="field__label">생성 길이 <output data-out="len"></output></span>
          <input type="range" data-in="len" min="20" max="80" step="10">
        </label>
        <label class="field">
          <span class="field__label">고르는 방식</span>
          <select data-in="decode">
            <option value="greedy">greedy (항상 1등)</option>
            <option value="sample">샘플링 (T = 1, seed)</option>
          </select>
        </label>
      </div>
      <div class="btn-row">
        <button type="button" class="btn small" data-act="again" disabled>🎲 다시 뽑기</button>
        <button type="button" class="btn small primary" data-act="compare">▦ 예제 9개 한꺼번에 비교</button>
        <span class="w12-seed" data-out="seed"></span>
      </div>
    </div>

    <div class="w12-panel" data-panel="rag" role="tabpanel" hidden>
      <div class="widget__controls">
        <label class="field">
          <span class="field__label">학과 질문</span>
          <select data-in="question"></select>
        </label>
        <label class="field w12-checkfield">
          <span class="field__label">검색기 설정</span>
          <span class="w12-check"><input type="checkbox" data-in="title"> 문단 앞에 문서 제목을 붙여 검색한다</span>
        </label>
      </div>
      <details class="w12-llm">
        <summary>🔑 실제 LLM으로 해 보기 (선택 · 키가 없으면 수업용 작성 예시)</summary>
        <div class="widget__controls">
          <label class="field">
            <span class="field__label">공급자</span>
            <select data-in="provider"></select>
          </label>
          <label class="field">
            <span class="field__label">모델</span>
            <select data-in="llm-model"></select>
          </label>
          <div class="field">
            <span class="field__label">API 키 <small class="w12-muted" data-out="keystate"></small></span>
            <div class="w12-keyrow">
              <input type="password" data-in="key" autocomplete="off" spellcheck="false" placeholder="sessionStorage에만 저장" aria-label="API 키">
              <button type="button" class="btn small" data-act="savekey">저장</button>
              <button type="button" class="btn small ghost" data-act="clearkey">지우기</button>
            </div>
          </div>
        </div>
        <div class="btn-row">
          <button type="button" class="btn small primary" data-act="ask">▶ 실제 LLM에 두 번 묻기 (근거 없이 · 근거 포함)</button>
        </div>
      </details>
    </div>
    <div data-slot="out-inline"></div>
  </div>`;

const outTemplate = document.createElement('template');
outTemplate.innerHTML = `
  <div class="w12-out">
    <div data-view="gpt">
      <div data-slot="verdict" aria-live="polite"><div class="w12-busy"><span class="spinner" aria-hidden="true"></span> 문서셋과 모델 불러오는 중…</div></div>
      <h4>생성된 글 <small class="w12-muted">— 배경색 = 그 글자의 확률 · 밑줄 = 근거 대조 결과</small></h4>
      <div class="w12-gen" data-slot="gen"></div>
      <div class="legend" aria-hidden="true">
        <span><i class="w12-lg-prompt"></i>시작 글자</span>
        <span><i class="w12-lg-hi"></i>확률 ≥ 50%</span>
        <span><i class="w12-lg-mid"></i>10~50%</span>
        <span><i class="w12-lg-lo"></i>&lt; 10%</span>
        <span><span class="w12-ch m-topic">주제 문서 일치</span></span>
        <span><span class="w12-ch m-other">다른 문서 일치</span></span>
        <span><span class="w12-ch m-none">원문에 없음</span></span>
      </div>
      <div class="stat-row" data-slot="stats"></div>
      <h4>근거 대조 <small class="w12-muted">— 생성된 글을 학습 문서에서 찾은 가장 긴 일치 구간 (${MATCH_MIN}자 이상)</small></h4>
      <ol class="w12-spans" data-slot="spans"></ol>
      <p class="w12-truth" data-slot="truth"></p>
      <div data-slot="compare"></div>
    </div>
    <div data-view="rag" hidden>
      <div data-slot="rverdict" aria-live="polite"></div>
      <p class="w12-src" data-slot="rsrc"></p>
      <h4>검색된 문단 (키워드 겹침 상위 ${TOP_K}개)</h4>
      <ol class="w12-hits" data-slot="hits"></ol>
      <div class="w12-cols">
        <section class="w12-ans" data-slot="ans-no"></section>
        <section class="w12-ans" data-slot="ans-ctx"></section>
      </div>
    </div>
  </div>`;

const state = new WeakMap();
const modelCache = new Map();
let grounded = null;
let seq = 0;

/**
 * @param {HTMLElement} el
 * @param {{ tab?: 'gpt'|'rag', model?: string, preset?: string, outputTitle?: string }} options
 */
export async function mount(el, options = {}) {
  const root = template.content.firstElementChild.cloneNode(true);
  const out = outTemplate.content.firstElementChild.cloneNode(true);
  el.replaceChildren(root);
  const $ = (sel) => root.querySelector(sel) ?? out.querySelector(sel);
  const ctrl = new AbortController();
  const outputId = `w12-hallucination:${++seq}`;
  state.set(el, { ctrl, outputId });

  const s = {
    tab: options.tab ?? 'gpt',
    model: options.model ?? 'mini-gpt-3000',
    preset: options.preset ?? 'k1',
    custom: '',
    len: 50,
    decode: 'greedy',
    seed: 1,
    runId: 0,
    cmpId: 0,
    result: null,
    compare: null,
    corpus: null,
    trainDocs: [],
    paragraphs: [],
    question: 0,
    useTitle: true,
    provider: 'anthropic',
    llmModel: PROVIDERS.anthropic.defaultModel,
    live: new Map(), // key: question|retrieved ids → { no, ctx, model } or { error }
    busy: false,
  };

  // ---- controls
  $('[data-in=model]').innerHTML = MODELS.map((m) => `<option value="${m.id}">${esc(m.label)}</option>`).join('');
  $('[data-in=preset]').innerHTML = Object.entries(GROUPS)
    .map(([g, label]) => `<optgroup label="${esc(label)}">${PRESETS.filter((p) => p.group === Number(g)).map((p) => `<option value="${p.id}">${esc(p.prompt)}</option>`).join('')}</optgroup>`)
    .join('');
  $('[data-in=model]').value = s.model;
  $('[data-in=preset]').value = s.preset;
  $('[data-in=prompt]').placeholder = presetById(s.preset).prompt;
  $('[data-in=len]').value = String(s.len);
  $('[data-in=provider]').innerHTML = Object.entries(PROVIDERS).map(([id, p]) => `<option value="${id}">${esc(p.label)}</option>`).join('');
  $('[data-in=title]').checked = s.useTitle;

  registerOutput(outputId, { title: options.outputTitle ?? '환각 해부 결과', node: out, inlineHost: $('[data-slot=out-inline]') });

  const prompt = () => s.custom.trim() || presetById(s.preset).prompt;
  const activePreset = () => (s.custom.trim() ? null : presetById(s.preset));

  // ---------------------------------------------------------------- tab ①
  async function runGpt() {
    if (!s.corpus) return;
    const id = ++s.runId;
    s.cmpId++; // any input change cancels a running comparison
    s.compare = null;
    $('[data-slot=compare]').replaceChildren();
    $('[data-out=len]').textContent = `${s.len}자`;
    $('[data-out=seed]').textContent = s.decode === 'sample' ? `seed ${s.seed}` : '';
    $('[data-act=again]').disabled = s.decode !== 'sample';
    setGptBusy('모델 불러오는 중…');
    let m;
    try {
      m = await loadModel(s.model);
    } catch (err) {
      if (id === s.runId) setGptError(`모델 파일을 불러오지 못했다 (${err.message}).`);
      return;
    }
    if (id !== s.runId || ctrl.signal.aborted) return;
    const text = prompt();
    const res = await generateTraced(m, text, s.len, { decode: s.decode, seed: s.seed, isStale: () => id !== s.runId || ctrl.signal.aborted, onProgress: (k) => setGptBusy(`생성 중… ${k}/${s.len}`) });
    if (!res) return;
    s.result = { ...res, preset: activePreset(), model: s.model, meta: m.meta };
    renderGpt();
  }

  function setGptBusy(msg) {
    $('[data-slot=verdict]').innerHTML = `<div class="w12-busy"><span class="spinner" aria-hidden="true"></span> ${esc(msg)}</div>`;
  }
  function setGptError(msg) {
    $('[data-slot=verdict]').innerHTML = `<div class="widget__error">${esc(msg)} <code>file://</code>로 열었다면 <code>python -m http.server</code>로 연다.</div>`;
  }

  function renderGpt() {
    const r = s.result;
    if (!r) return;
    const gen = r.chars.map((c) => c.ch).join('');
    const an = analyze(gen, s.trainDocs, r.preset?.topic ?? null);
    const conf = r.chars.length ? r.chars.reduce((a, c) => a + c.p, 0) / r.chars.length : 0;
    const goldHit = r.preset ? r.preset.gold.some((g) => gen.includes(g)) : null;

    // text: background = probability, underline = grounding class
    const pieces = [`<span class="w12-prompt">${esc(showWs(r.prompt))}</span>`];
    r.chars.forEach((c, i) => {
      const cls = c.p >= 0.5 ? 'hi' : c.p >= 0.1 ? 'mid' : 'lo';
      const alt = c.top.map(([t, p]) => `${showWs(t)} ${(p * 100).toFixed(1)}%`).join(' · ');
      const ch = c.ch === '\n' ? '↵\n' : c.ch;
      pieces.push(`<span class="w12-ch ${cls} m-${an.mask[i]}" title="${esc(`P = ${(c.p * 100).toFixed(1)}% · 후보: ${alt}`)}">${esc(ch)}</span>`);
    });
    $('[data-slot=gen]').innerHTML = pieces.join('');

    const matched = an.mask.filter((k) => k !== 'none').length / Math.max(1, gen.length);
    const onTopic = an.mask.filter((k) => k === 'topic').length / Math.max(1, gen.length);
    const lossNow = r.meta.history?.at(-1);
    $('[data-slot=stats]').innerHTML = [
      stat('확신도 (평균 확률)', pct1(conf)),
      stat('원문 일치', pct(matched)),
      r.preset ? stat('그중 주제 문서', pct(onTopic)) : '',
      stat('정답 사실 포함', goldHit === null ? '—' : goldHit ? '✅ 예' : '❌ 아니오'),
      lossNow ? stat('학습 / 검증 손실', `${lossNow.train.toFixed(2)} / ${lossNow.val.toFixed(2)}`) : '',
    ].join('');

    $('[data-slot=spans]').innerHTML = an.spans
      .map((sp) => {
        const label = sp.kind === 'none' ? '원문에 없음' : sp.kind === 'topic' ? `원문 일치 · ${sp.title}` : `원문 일치 · 다른 문서: ${sp.title}`;
        return `<li class="w12-span k-${sp.kind}"><span class="w12-tag">${esc(label)}</span> <q>${esc(showWs(sp.text))}</q></li>`;
      })
      .join('');
    $('[data-slot=truth]').innerHTML = r.preset
      ? `<b>실제 문서에는</b> ${esc(r.preset.truth)}`
      : '<b>직접 입력</b> — 정답을 모르므로 “정답 사실 포함”은 판정하지 않는다. 근거 대조만 본다.';
    $('[data-slot=verdict]').innerHTML = gptVerdict({ conf, goldHit, matched, onTopic, preset: r.preset, model: r.model });
  }

  async function runCompare() {
    if (!s.corpus) return;
    const id = ++s.cmpId;
    const rows = [];
    let m;
    try {
      m = await loadModel(s.model);
    } catch (err) {
      setGptError(`모델 파일을 불러오지 못했다 (${err.message}).`);
      return;
    }
    const host = $('[data-slot=compare]');
    for (const [i, p] of PRESETS.entries()) {
      host.innerHTML = `<div class="w12-busy"><span class="spinner" aria-hidden="true"></span> 예제 ${i + 1}/${PRESETS.length} 생성 중… (${esc(p.prompt)})</div>`;
      const res = await generateTraced(m, p.prompt, s.len, { decode: 'greedy', seed: 1, isStale: () => id !== s.cmpId || ctrl.signal.aborted });
      if (!res) return;
      const gen = res.chars.map((c) => c.ch).join('');
      const an = analyze(gen, s.trainDocs, p.topic);
      rows.push({
        p,
        gen,
        conf: res.chars.reduce((a, c) => a + c.p, 0) / res.chars.length,
        onTopic: an.mask.filter((k) => k === 'topic').length / gen.length,
        matched: an.mask.filter((k) => k !== 'none').length / gen.length,
        goldHit: p.gold.some((g) => gen.includes(g)),
      });
    }
    s.compare = rows;
    renderCompare();
  }

  function renderCompare() {
    const rows = s.compare;
    if (!rows) return;
    const byGroup = [1, 2, 3].map((g) => {
      const rs = rows.filter((r) => r.p.group === g);
      return { g, conf: rs.reduce((a, r) => a + r.conf, 0) / rs.length, hits: rs.filter((r) => r.goldHit).length, n: rs.length };
    });
    $('[data-slot=compare]').innerHTML = `
      <h4>예제 9개 비교 · ${esc(s.model)} · greedy ${s.len}자</h4>
      <div class="w12-table-wrap"><table class="w12-table">
        <thead><tr><th scope="col">시작 글자</th><th scope="col">확신도</th><th scope="col">주제 문서 일치</th><th scope="col">정답</th></tr></thead>
        <tbody>${rows
          .map(
            (r) => `<tr class="g${r.p.group}"><td title="${esc(r.gen)}"><small class="w12-muted">${'①②③'[r.p.group - 1]}</small> ${esc(r.p.prompt)}</td><td><span class="w12-meter"><i style="width:${(r.conf * 100).toFixed(0)}%"></i></span> ${pct1(r.conf)}</td><td>${pct(r.onTopic)}</td><td>${r.goldHit ? '✅' : '❌'}</td></tr>`,
          )
          .join('')}</tbody>
      </table></div>
      <p class="w12-note">${byGroup.map((b) => `${'①②③'[b.g - 1]} 평균 확신도 <b>${pct1(b.conf)}</b> · 정답 ${b.hits}/${b.n}`).join('<br>')}</p>
      <p class="w12-note">확신도 열과 정답 열을 나란히 본다. 확신도가 높다고 정답인 것이 아니다. 모델의 확률은 “학습 글과 얼마나 닮았나”를 잴 뿐 “사실인가”를 재지 않는다.</p>`;
  }

  // ---------------------------------------------------------------- tab ②
  function retrieved() {
    const q = grounded.questions[s.question];
    return retrieve(s.paragraphs, q.q, { useTitle: s.useTitle, k: TOP_K });
  }

  function renderRag() {
    if (!grounded || !s.paragraphs.length) return;
    const q = grounded.questions[s.question];
    const hits = retrieved();
    const key = hits.map((h) => h.p.id).join(',');
    const golds = (q.gold ?? []).map((g) => new RegExp(g));
    const goldInHits = q.gold ? hits.some((h) => golds.every((re) => re.test(h.p.text))) : null;

    $('[data-slot=hits]').innerHTML = hits
      .map(
        (h, i) => `<li class="w12-hit${q.gold && golds.every((re) => re.test(h.p.text)) ? ' has-gold' : ''}">
          <div class="w12-hit-head"><b>[${i + 1}]</b> ${esc(h.p.title)} <code>${esc(h.p.id)}</code> <span class="w12-muted">겹친 두 글자 ${h.score}개</span></div>
          <p>${highlight(h.p.text, golds)}</p></li>`,
      )
      .join('');

    const live = s.live.get(`${s.question}|${key}`);
    const keyed = safeHasKey(s.provider);
    const canned = q.withContext[key];
    const noAns = live?.no ?? q.noContext;
    const ctxAns = live?.ctx ?? canned ?? null;
    const srcLabel = live?.no ? `실제 응답 · ${live.model}` : '수업용 작성 예시';
    $('[data-slot=rsrc]').innerHTML = live?.error
      ? `<span class="w12-err">${esc(live.error)}</span>`
      : live?.no
        ? `🔌 실제 호출 결과 · ${esc(PROVIDERS[live.provider].label)} · ${esc(live.model)} (temperature 0)`
        : `📝 ${esc(grounded.note.split('.')[0])}. ${keyed ? '키가 저장되어 있다. “실제 LLM에 두 번 묻기”를 누르면 같은 질문을 실제로 보낸다.' : '키를 넣으면 같은 질문을 실제 LLM에 보낼 수 있다.'}`;

    const noPrompt = `[system] ${grounded.systemNoContext}\n\n[user] ${q.q}`;
    const rp = ragPrompt(q.q, hits.map((h) => ({ text: `(${h.p.title}) ${h.p.text}` })));
    const ctxPrompt = `[system] ${rp.system}\n\n[user] ${rp.messages[0].content}`;
    $('[data-slot=ans-no]').innerHTML = answerCard('근거 없이 묻기', noPrompt, noAns, srcLabel, check(q, noAns));
    $('[data-slot=ans-ctx]').innerHTML = ctxAns
      ? answerCard('근거를 붙여 묻기 (RAG)', ctxPrompt, ctxAns, live?.ctx ? srcLabel : '수업용 작성 예시', check(q, ctxAns))
      : answerCard('근거를 붙여 묻기 (RAG)', ctxPrompt, '이 검색 결과 조합에는 작성 예시가 없다. 키를 넣고 실제로 물어본다.', '—', null);

    $('[data-slot=rverdict]').innerHTML = ragVerdict({ q, goldInHits, noOk: check(q, noAns), ctxOk: ctxAns ? check(q, ctxAns) : null, useTitle: s.useTitle });
    $('[data-act=ask]').disabled = !keyed || s.busy;
    $('[data-out=keystate]').textContent = keyed ? '· 저장됨 (이 탭에서만)' : '· 없음';
  }

  async function askLive() {
    const q = grounded.questions[s.question];
    const hits = retrieved();
    const key = `${s.question}|${hits.map((h) => h.p.id).join(',')}`;
    s.busy = true;
    $('[data-slot=rsrc]').innerHTML = '<span class="spinner" aria-hidden="true"></span> 실제 LLM에 묻는 중… (두 번 호출)';
    $('[data-act=ask]').disabled = true;
    try {
      const common = { provider: s.provider, model: s.llmModel, maxTokens: 300, temperature: 0, signal: ctrl.signal };
      const rp = ragPrompt(q.q, hits.map((h) => ({ text: `(${h.p.title}) ${h.p.text}` })));
      const [no, ctx] = await Promise.all([
        llmGenerate({ ...common, system: grounded.systemNoContext, messages: [{ role: 'user', content: q.q }] }),
        llmGenerate({ ...common, system: rp.system, messages: rp.messages }),
      ]);
      s.live.set(key, { no: no.text.trim(), ctx: ctx.text.trim(), model: s.llmModel, provider: s.provider });
    } catch (err) {
      if (err.name === 'AbortError') return;
      s.live.set(key, { error: `실제 호출 실패: ${err.message} — 아래는 수업용 작성 예시다.` });
    } finally {
      s.busy = false;
    }
    if (!ctrl.signal.aborted) renderRag();
  }

  // ---------------------------------------------------------------- wiring
  function showTab() {
    root.querySelectorAll('[data-tab]').forEach((b) => b.setAttribute('aria-selected', String(b.dataset.tab === s.tab)));
    root.querySelectorAll('[data-panel]').forEach((p) => (p.hidden = p.dataset.panel !== s.tab));
    out.querySelectorAll('[data-view]').forEach((v) => (v.hidden = v.dataset.view !== s.tab));
  }

  function refreshLlmModels() {
    $('[data-in=llm-model]').innerHTML = PROVIDERS[s.provider].models.map((m) => `<option>${esc(m)}</option>`).join('');
    $('[data-in=llm-model]').value = s.llmModel;
  }

  const on = (sel, type, fn) => $(sel).addEventListener(type, fn, { signal: ctrl.signal });
  root.querySelectorAll('[data-tab]').forEach((b) =>
    b.addEventListener(
      'click',
      () => {
        s.tab = b.dataset.tab;
        showTab();
        if (s.tab === 'rag') renderRag();
      },
      { signal: ctrl.signal },
    ),
  );
  on('[data-in=model]', 'change', (e) => {
    s.model = e.target.value;
    runGpt();
  });
  on('[data-in=preset]', 'change', (e) => {
    s.preset = e.target.value;
    s.custom = '';
    $('[data-in=prompt]').value = '';
    $('[data-in=prompt]').placeholder = presetById(s.preset).prompt;
    runGpt();
  });
  let typing = 0;
  on('[data-in=prompt]', 'input', (e) => {
    s.custom = e.target.value;
    clearTimeout(typing);
    typing = setTimeout(runGpt, 350);
  });
  on('[data-in=len]', 'change', (e) => {
    s.len = Number(e.target.value);
    runGpt();
  });
  on('[data-in=len]', 'input', (e) => ($('[data-out=len]').textContent = `${e.target.value}자`));
  on('[data-in=decode]', 'change', (e) => {
    s.decode = e.target.value;
    runGpt();
  });
  on('[data-act=again]', 'click', () => {
    s.seed++;
    runGpt();
  });
  on('[data-act=compare]', 'click', runCompare);
  on('[data-in=question]', 'change', (e) => {
    s.question = Number(e.target.value);
    renderRag();
  });
  on('[data-in=title]', 'change', (e) => {
    s.useTitle = e.target.checked;
    renderRag();
  });
  on('[data-in=provider]', 'change', (e) => {
    s.provider = e.target.value;
    s.llmModel = PROVIDERS[s.provider].defaultModel;
    refreshLlmModels();
    renderRag();
  });
  on('[data-in=llm-model]', 'change', (e) => (s.llmModel = e.target.value));
  const keyInput = $('[data-in=key]');
  on('[data-act=savekey]', 'click', () => {
    const v = keyInput.value.trim();
    keyInput.value = '';
    if (!v) return;
    try {
      setKey(s.provider, v);
    } catch {
      /* storage blocked: stay in example mode */
    }
    renderRag();
  });
  on('[data-act=clearkey]', 'click', () => {
    keyInput.value = '';
    try {
      clearKey(s.provider);
    } catch {
      /* storage blocked */
    }
    s.live.clear();
    renderRag();
  });
  keyInput.addEventListener('keydown', (e) => e.key === 'Enter' && $('[data-act=savekey]').click(), { signal: ctrl.signal });
  on('[data-act=ask]', 'click', askLive);
  refreshLlmModels();
  showTab();

  // ---------------------------------------------------------------- load
  try {
    const [corpus, g] = await Promise.all([loadCorpus(), loadGrounded()]);
    if (ctrl.signal.aborted) return;
    s.corpus = corpus;
    s.trainDocs = corpus.documents.filter((d) => d.id !== 'rag-vectordb');
    s.paragraphs = splitParagraphs(corpus);
    $('[data-in=question]').innerHTML = g.questions.map((q, i) => `<option value="${i}">${esc(q.q)}</option>`).join('');
    $('[data-slot=status]').hidden = true;
    renderRag();
    runGpt();
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

// ---------------------------------------------------------------- pure helpers (also used by the Node checks)

/** Split every document into paragraphs (blank-line separated). */
export function splitParagraphs(corpus) {
  return corpus.documents.flatMap((d) =>
    d.text.split(/\n\s*\n/).map((text, i) => ({ id: `${d.id}#${i}`, doc: d.id, title: d.title, text: text.trim() })),
  );
}

/** Set of 2-character pieces inside each word (lowercased). */
export function bigrams(s) {
  const out = new Set();
  for (const w of s.toLowerCase().split(/[^0-9a-z가-힣%]+/)) {
    const c = [...w];
    for (let i = 0; i < c.length - 1; i++) out.add(c[i] + c[i + 1]);
  }
  return out;
}

/** Keyword retriever: score = number of shared bigrams. Ties keep corpus order. */
export function retrieve(paragraphs, question, { useTitle = true, k = 2 } = {}) {
  const q = bigrams(question);
  return paragraphs
    .map((p) => {
      const g = bigrams(useTitle ? `${p.title} ${p.text}` : p.text);
      let score = 0;
      q.forEach((x) => g.has(x) && score++);
      return { p, score };
    })
    .sort((a, b) => b.score - a.score)
    .slice(0, k);
}

/**
 * Grounding check. Walk the generated text; at each position take the longest
 * run found verbatim in one training document. Runs ≥ MATCH_MIN become matched
 * spans ('topic' if the document is the prompt's topic, else 'other').
 */
export function analyze(gen, docs, topic = null) {
  const chars = [...gen];
  const mask = new Array(chars.length).fill('none');
  const spans = [];
  let i = 0;
  let pending = '';
  const flush = () => {
    if (pending) spans.push({ kind: 'none', text: pending });
    pending = '';
  };
  while (i < chars.length) {
    let best = { len: 0, doc: null };
    for (const d of docs) {
      let L = best.len;
      if (!d.text.includes(chars.slice(i, i + L + 1).join(''))) continue;
      while (i + L < chars.length && d.text.includes(chars.slice(i, i + L + 1).join(''))) L++;
      if (L > best.len) best = { len: L, doc: d };
    }
    if (best.len >= MATCH_MIN) {
      flush();
      const kind = !topic || topic.includes(best.doc.id) ? 'topic' : 'other';
      for (let j = i; j < i + best.len; j++) mask[j] = kind;
      spans.push({ kind, text: chars.slice(i, i + best.len).join(''), doc: best.doc.id, title: best.doc.title });
      i += best.len;
    } else {
      pending += chars[i];
      i++;
    }
  }
  flush();
  return { mask, spans };
}

/**
 * Generate with per-character probabilities (softmax at T = 1, before any decoding choice).
 * Yields to the event loop every few steps so the page stays responsive.
 */
export async function generateTraced(m, prompt, length, { decode = 'greedy', seed = 1, isStale = () => false, onProgress = null } = {}) {
  const { model, vocab } = m;
  const ids = vocab.encode(prompt);
  if (!ids.length) ids.push(vocab.stoi.get(' ') ?? 0);
  const rand = mulberry32(seed);
  const chars = [];
  for (let i = 0; i < length; i++) {
    if (i % 4 === 0) {
      await new Promise((r) => setTimeout(r, 0));
      if (isStale()) return null;
      onProgress?.(i);
    }
    const probs = softmax(nextLogits(model, ids));
    let pick = 0;
    if (decode === 'sample') pick = sampleIndex(probs, rand);
    else for (let j = 1; j < probs.length; j++) if (probs[j] > probs[pick]) pick = j;
    const top = probs
      .map((p, j) => [j, p])
      .sort((a, b) => b[1] - a[1])
      .slice(0, 3)
      .map(([j, p]) => [vocab.itos[j], p]);
    chars.push({ ch: vocab.itos[pick], p: probs[pick], top });
    ids.push(pick);
  }
  return { prompt, chars };
}

function loadModel(name) {
  if (!modelCache.has(name)) {
    const p = (async () => {
      const res = await fetch(new URL(`../../data/models/${name}.json`, import.meta.url));
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      return deserialize(await res.json());
    })();
    p.catch(() => modelCache.delete(name));
    modelCache.set(name, p);
  }
  return modelCache.get(name);
}

async function loadGrounded() {
  if (!grounded) {
    const res = await fetch(new URL('../../data/w12/grounded.json', import.meta.url));
    if (!res.ok) throw new Error(`예시 응답 HTTP ${res.status}`);
    grounded = await res.json();
  }
  return grounded;
}

const presetById = (id) => PRESETS.find((p) => p.id === id) ?? PRESETS[0];

function safeHasKey(provider) {
  try {
    return hasKey(provider);
  } catch {
    return false;
  }
}

/** Does the answer contain the gold fact? For "not in the corpus" questions, a refusal is correct. */
export function check(q, answer) {
  if (!q.gold) return /찾을 수 없|알 수 없|확인할 수 없|나와 있지 않|정보가 없/.test(answer);
  return q.gold.every((g) => new RegExp(g).test(answer));
}

function gptVerdict({ conf, goldHit, matched, onTopic, preset, model }) {
  const c = pct1(conf);
  if (conf < 0.6) {
    return `<div class="callout"><span class="callout__title">덜 학습된 모델 · 확신도 ${c}</span><p>${esc(model)}는 말이 되지 않는 글을 내면서 확신도도 낮다. 적어도 “잘 모른다”는 신호가 확률에 드러난다. mini-gpt-3000으로 바꾸면 이 신호가 어떻게 되는지 본다.</p></div>`;
  }
  if (!preset) {
    return `<div class="callout"><span class="callout__title">확신도 ${c} · 원문 일치 ${pct(matched)}</span><p>직접 입력한 문장이다. 원문에 없는 구간(빨간 밑줄)이 있는데도 확신도가 높다면, 모델이 지어낸 글이다.</p></div>`;
  }
  if (goldHit && preset.group === 1) {
    return `<div class="callout callout--ok"><span class="callout__title">외운 사실을 그대로 읊었다 · 확신도 ${c}</span><p>학습 문서의 문장이 글자 그대로 나온다(주제 문서 일치 ${pct(onTopic)}). 이것은 “아는 것”이 아니라 “외운 것”이다. ③의 예제처럼 시작 글자를 조금만 바꾸면 무슨 일이 생기는지 본다.</p></div>`;
  }
  if (goldHit) {
    return `<div class="callout callout--ok"><span class="callout__title">정답 사실이 나왔다 · 확신도 ${c}</span><p>정답 단어가 들어 있다. 문장 전체가 맞는지 근거 대조로 확인한다.</p></div>`;
  }
  const why = matched - onTopic > 0.3
    ? `생성된 글의 ${pct(matched - onTopic)}는 <b>다른 문서</b>에 실제로 있는 문장이다. 외운 문장을 엉뚱한 자리에 이어 붙였다.`
    : onTopic > 0.5
      ? `생성된 글의 ${pct(onTopic)}는 주제 문서의 문장 그대로다. 틀린 전제 뒤에 외운 문장을 붙였을 뿐, 전제를 바로잡지 않았다.`
      : `생성된 글의 ${pct(1 - matched)}는 어떤 학습 문서에도 없는 글이다. 모델이 지어냈다.`;
  const what = preset.group === 2 ? '학습에서 뺀 문서의 내용을 묻자' : '틀린 전제를 주자';
  return `<div class="callout callout--danger"><span class="callout__title">확신도 ${c}로 틀렸다 · 환각</span><p>${what} 모델은 “모른다”고 하지 않고 계속 썼다. ${why} 다음 토큰 분포는 언제나 나오고, 1등 후보는 언제나 있다.</p></div>`;
}

function ragVerdict({ q, goldInHits, noOk, ctxOk, useTitle }) {
  if (!q.gold) {
    return `<div class="callout callout--more"><span class="callout__title">답이 문서에 없는 질문</span><p>근거 없이 물으면 ${noOk ? '모른다고 답했다' : '그럴듯한 숫자를 지어낸다'}. 근거를 붙이면 ${ctxOk ? '“문서에서 찾을 수 없다”고 답한다. 프롬프트의 “근거에 없으면 모른다고 답한다” 규칙이 일한 것이다' : '규칙이 있는데도 지어냈다. 모델과 프롬프트를 바꿔 본다'}.</p></div>`;
  }
  if (!goldInHits) {
    return `<div class="callout callout--danger"><span class="callout__title">검색이 틀렸다 → 근거를 붙여도 답을 못 한다</span><p>검색된 두 문단 어디에도 정답 문장이 없다${useTitle ? '' : '. 성적 문단에는 “LLM”이라는 말이 없어서 과목 이름으로는 찾을 수 없다. “문서 제목을 붙여 검색”을 켜 본다'}. RAG의 답은 검색보다 좋아질 수 없다.</p></div>`;
  }
  return `<div class="callout callout--ok"><span class="callout__title">근거 없이 ${noOk ? '✅' : '❌'} → 근거를 붙이면 ${ctxOk ? '✅' : '❌'}</span><p>모델은 그대로다. 바뀐 것은 프롬프트에 들어간 문단 하나다. 가중치에 없는 지식(비파라메트릭 지식)을 입력으로 넣어 주는 것 — 이것이 검색증강생성(RAG)이다.</p></div>`;
}

function answerCard(title, promptText, answer, src, ok) {
  const badge = ok === null ? '' : ok ? '<span class="chip ok">정답 포함 ✅</span>' : '<span class="chip warn">정답 없음 ❌</span>';
  return `<h4>${esc(title)} ${badge}</h4>
    <p class="w12-answer">${esc(answer)}</p>
    <p class="w12-muted w12-small">${esc(src)}</p>
    <details><summary>보낸 프롬프트 (${[...promptText].length}자)</summary><pre class="w12-prompt-pre">${esc(promptText)}</pre></details>`;
}

function highlight(text, regexes) {
  const ranges = [];
  for (const re of regexes) {
    const g = new RegExp(re.source, 'g');
    let m;
    while ((m = g.exec(text))) {
      ranges.push([m.index, m.index + m[0].length]);
      if (!m[0].length) g.lastIndex++;
    }
  }
  ranges.sort((a, b) => a[0] - b[0]);
  let html = '';
  let at = 0;
  for (const [a, b] of ranges) {
    if (a < at) continue;
    html += esc(text.slice(at, a)) + `<mark>${esc(text.slice(a, b))}</mark>`;
    at = b;
  }
  return html + esc(text.slice(at));
}

const stat = (label, value) => `<div class="stat"><span class="stat__label">${label}</span><span class="stat__value">${value}</span></div>`;
const pct1 = (x) => `${(x * 100).toFixed(1)}%`;
const pct = (x) => `${(x * 100).toFixed(x > 0 && x < 0.1 ? 1 : 0)}%`;

function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
}
