// w14 챗봇 빌더
// One concept: a chatbot "remembers" only what the app resends. The API is
// stateless, so the app chooses which past turns go into every request
// (none / full / last N / summary + last N) under a token budget, with the
// system prompt pinned. Replies stream in piece by piece and can be stopped.

import { registerOutput, unregisterOutput } from '../site/result.js';
import { loadCorpus } from '../core/text.js';
import { PROVIDERS, setKey, hasKey, clearKey, generate, LLMError } from '../core/llm.js';
import { buildContext, mockReply, simulateStream, streamChat, splitSystem, estimateTokens, SUMMARY_HEADER, STREAM_PROVIDERS } from '../core/chat.js';

const FACT_SHEET = `알고 있는 사실:
- 실습실: 본관 3층 305·306호, 평일 오후 9시까지 개방. 주말·공휴일은 이틀 전까지 행정실에 사전 신청
- 실습실 안 음식물(음료 포함) 금지. GPU 서버는 공용 캘린더에 등록하고 연속 최대 12시간
- 수업: 평일 9시~18시(점심 12~13시). 수료: 출석 80% 이상 + 팀 프로젝트 발표 통과
- LLM 원리와 활용, RAG 시스템 구축: 둘 다 2학기 3학점, 주 1회 3시간`;

export const PRESETS = {
  kind: {
    label: '친절한 조교',
    text: `너는 AI응용소프트웨어과의 '학과 안내 도우미'다. 신입생을 돕는 친절한 조교처럼 반말로 다정하게 답한다.
규칙:
1. 학과·수업·실습실·교과목에 관한 질문에만 답한다.
2. 관련 없는 질문은 정중히 거절하고, 대신 물어볼 수 있는 것을 알려 준다.
3. 아래 사실에 없는 내용은 지어내지 말고 학과 행정실에 문의하라고 한다.

${FACT_SHEET}`,
  },
  terse: {
    label: '간결한 안내원',
    text: `너는 AI응용소프트웨어과 안내원이다. 한두 문장으로 간결하게 답한다.
학과와 관련 없는 질문은 거절한다. 아래 사실에 없는 내용은 모른다고 답한다.

${FACT_SHEET}`,
  },
  open: {
    label: '가드레일 없음',
    text: '너는 친절한 도우미다. 무엇이든 성실하게 답한다.',
  },
};

export const SCENARIO = [
  '안녕! 나는 민수야. 이번에 AI응용소프트웨어과에 입학했어.',
  'LLM 원리와 활용 과목은 뭘 배워?',
  '실습실은 몇 시까지 열어?',
  'GPU 서버는 한 번에 얼마나 오래 쓸 수 있어?',
  '아까 말한 과목 몇 학점이야?',
  '내 이름 기억해?',
  '오늘 저녁 메뉴 추천해 줘.',
];

const SUGGEST = ['아까 말한 과목 몇 학점이야?', '내 이름 기억해?', '실습실에서 커피 마셔도 돼?', '장학금 있어?', '오늘 저녁 메뉴 추천해 줘.'];

const STRATEGIES = [
  { id: 'none', label: '기억 없음 (새 질문만)' },
  { id: 'full', label: '전체 기록' },
  { id: 'window', label: '최근 N턴' },
  { id: 'summary', label: '요약 + 최근 N턴' },
];
const STATUS = {
  kept: { label: '보냄', cls: 'kept' },
  window: { label: '창 밖 · 버림', cls: 'window' },
  summarized: { label: '요약으로 압축', cls: 'sum' },
  budget: { label: '예산 초과 · 버림', cls: 'budget' },
};
const CHECKS = [
  { key: 'name', label: '이름 기억' },
  { key: 'course', label: '“아까 그 과목” 기억' },
  { key: 'guard', label: '주제 밖 질문 거절' },
];

const template = document.createElement('template');
template.innerHTML = `
  <div class="widget w14-cb">
    <h3 class="widget__title">챗봇 빌더 · 학과 안내 도우미</h3>
    <div data-slot="status" class="widget__status" role="status" aria-live="polite">
      <span class="spinner" aria-hidden="true"></span> <span>문서셋 불러오는 중…</span>
    </div>
    <div class="w14-grid">
      <div class="w14-settings">
        <div class="field">
          <span class="field__label">페르소나 · 시스템 프롬프트 <output data-out="systok"></output></span>
          <div class="btn-row" role="group" aria-label="페르소나 프리셋" data-slot="presets"></div>
          <textarea data-in="system" rows="7" spellcheck="false" aria-label="시스템 프롬프트"></textarea>
        </div>
        <div class="widget__controls w14-controls">
          <label class="field">
            <span class="field__label">기억 전략</span>
            <select data-in="strategy"></select>
          </label>
          <label class="field">
            <span class="field__label">최근 N턴 <output data-out="lastN"></output></span>
            <input type="range" data-in="lastN" min="1" max="6" step="1">
          </label>
          <label class="field">
            <span class="field__label">토큰 예산 (컨텍스트 창) <output data-out="budget"></output></span>
            <input type="range" data-in="budget" min="250" max="1000" step="10">
          </label>
          <label class="field">
            <span class="field__label">온도 <output data-out="temp"></output></span>
            <input type="range" data-in="temp" min="0" max="1.5" step="0.1">
          </label>
          <label class="field">
            <span class="field__label">응답 모드</span>
            <select data-in="mode">
              <option value="mock">예시 응답 (키 없음)</option>
              <option value="anthropic">Anthropic Claude (내 키)</option>
              <option value="openai">OpenAI (내 키)</option>
              <option value="gemini">Google Gemini (내 키)</option>
            </select>
          </label>
          <label class="field" data-slot="model-field" hidden>
            <span class="field__label">모델</span>
            <select data-in="model"></select>
          </label>
        </div>
        <div class="w14-key" data-slot="key-row" hidden>
          <label class="field">
            <span class="field__label">API 키 <small data-out="keystate"></small></span>
            <input type="password" data-in="key" autocomplete="off" spellcheck="false" placeholder="키를 붙여 넣고 저장">
          </label>
          <div class="btn-row">
            <button type="button" class="btn small" data-act="save-key">저장</button>
            <button type="button" class="btn small ghost" data-act="clear-key">키 지우기</button>
          </div>
          <p class="w14-note">키는 이 탭의 sessionStorage에만 두고 탭을 닫으면 사라진다. 화면·로그에 표시하지 않는다.</p>
        </div>
      </div>
      <div class="w14-chat">
        <div class="w14-chat-head">
          <span class="w14-avatar" aria-hidden="true">🤖</span>
          <b>학과 안내 도우미</b>
          <span class="chip" data-out="mode-chip"></span>
        </div>
        <div class="w14-msgs" data-slot="msgs" role="log" aria-live="polite" aria-label="대화" tabindex="0"></div>
        <div class="w14-suggest" data-slot="suggest" aria-label="예시 질문"></div>
        <form class="w14-input" data-slot="form">
          <textarea data-in="msg" rows="2" placeholder="질문을 입력한다 (Enter 보내기 · Shift+Enter 줄바꿈)" aria-label="질문"></textarea>
          <div class="w14-input-btns">
            <button type="submit" class="btn small primary" data-act="send">보내기</button>
            <button type="button" class="btn small w14-stop" data-act="stop" disabled>■ 중지</button>
          </div>
        </form>
        <div class="btn-row">
          <button type="button" class="btn small" data-act="scenario">▶ 시나리오 재생 (7턴)</button>
          <button type="button" class="btn small ghost" data-act="clear">대화 지우기</button>
        </div>
      </div>
    </div>
    <div data-slot="out-inline"></div>
  </div>`;

const outTemplate = document.createElement('template');
outTemplate.innerHTML = `
  <div class="w14-out">
    <div data-slot="verdict" aria-live="polite"></div>
    <div class="w14-checks" data-slot="checks"></div>
    <h4 data-slot="bud-title">토큰 예산</h4>
    <div class="w14-bud" data-slot="bud"></div>
    <div class="legend" aria-hidden="true">
      <span><i class="w14-lg-sys"></i>시스템(고정)</span>
      <span><i class="w14-lg-sum"></i>요약</span>
      <span><i class="w14-lg-hist"></i>이전 대화</span>
      <span><i class="w14-lg-cur"></i>새 질문</span>
    </div>
    <h4>턴별 처리</h4>
    <ol class="w14-turns" data-slot="turns"></ol>
    <h4 data-slot="arr-title">messages 배열</h4>
    <div class="w14-arr" data-slot="arr"></div>
    <h4>로그</h4>
    <ol class="w14-log" data-slot="log"></ol>
  </div>`;

const state = new WeakMap();
let seq = 0;

/**
 * @param {HTMLElement} el
 * @param {{ outputTitle?: string, prefill?: number }} options
 */
export async function mount(el, options = {}) {
  const root = template.content.firstElementChild.cloneNode(true);
  const out = outTemplate.content.firstElementChild.cloneNode(true);
  el.replaceChildren(root);
  const $ = (sel) => root.querySelector(sel) ?? out.querySelector(sel);
  const ctrl = new AbortController();
  const outputId = `w14-chatbot:${++seq}`;
  const st = { ctrl, outputId, run: null };
  state.set(el, st);

  const s = {
    preset: 'kind',
    system: PRESETS.kind.text,
    strategy: 'window',
    lastN: 3,
    budget: 600,
    temp: 0.3,
    mode: 'mock',
    model: '',
    items: [], // { role: 'user'|'assistant'|'error', content, stopped?, checks?, ctxKey? }
    log: [],
    corpus: null,
    busy: false,
    scenario: false,
    sentKey: '',
  };

  // ---- controls
  $('[data-slot=presets]').innerHTML = Object.entries(PRESETS)
    .map(([id, p]) => `<button type="button" class="btn small" data-preset="${id}" aria-pressed="false">${esc(p.label)}</button>`)
    .join('');
  $('[data-in=strategy]').innerHTML = STRATEGIES.map((o) => `<option value="${o.id}">${esc(o.label)}</option>`).join('');
  $('[data-slot=suggest]').innerHTML = SUGGEST.map((q) => `<button type="button" class="btn small ghost" data-suggest="${esc(q)}">${esc(q)}</button>`).join('');
  $('[data-in=system]').value = s.system;
  $('[data-in=strategy]').value = s.strategy;
  $('[data-in=lastN]').value = String(s.lastN);
  $('[data-in=budget]').value = String(s.budget);
  $('[data-in=temp]').value = String(s.temp);

  registerOutput(outputId, { title: options.outputTitle ?? '챗봇 빌더 결과', node: out, inlineHost: $('[data-slot=out-inline]') });

  const settingsKey = () => [s.system, s.strategy, s.lastN, s.budget].join('|');
  const history = (items = s.items) => items.filter((m) => m.role === 'user' || (m.role === 'assistant' && m.content)).map((m) => ({ role: m.role, content: m.content }));
  const context = (hist) => buildContext({ system: s.system, history: hist, strategy: s.strategy, lastN: s.lastN, budget: s.budget });
  const turnNo = () => s.items.filter((m) => m.role === 'user').length;

  function addLog(text, level = '') {
    s.log.push({ text, level });
    if (s.log.length > 30) s.log.shift();
  }

  // ---- rendering
  function renderControls() {
    root.querySelectorAll('[data-preset]').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.preset === s.preset)));
    $('[data-out=systok]').textContent = `≈ ${estimateTokens(s.system) + 4} 토큰`;
    $('[data-out=lastN]').textContent = `${s.lastN}턴`;
    $('[data-out=budget]').textContent = `${s.budget} 토큰`;
    $('[data-out=temp]').textContent = s.temp.toFixed(1);
    $('[data-in=lastN]').disabled = !(s.strategy === 'window' || s.strategy === 'summary');
    const real = s.mode !== 'mock';
    $('[data-slot=key-row]').hidden = !real;
    $('[data-slot=model-field]').hidden = !real;
    $('[data-out=mode-chip]').textContent = real ? `${PROVIDERS[s.mode].label}${STREAM_PROVIDERS.includes(s.mode) ? ' · 스트리밍' : ''}` : '예시 응답 · 키 없음';
    if (real) $('[data-out=keystate]').textContent = hasKey(s.mode) ? '· 저장됨' : '· 없음';
    $('[data-act=send]').disabled = s.busy;
    $('[data-act=stop]').disabled = !s.busy;
    $('[data-act=scenario]').textContent = s.scenario ? '■ 시나리오 멈춤' : '▶ 시나리오 재생 (7턴)';
    root.querySelectorAll('[data-suggest]').forEach((b) => (b.disabled = s.busy));
  }

  function renderChat() {
    const box = $('[data-slot=msgs]');
    let turn = 0;
    box.innerHTML = s.items
      .map((m, i) => {
        if (m.role === 'user') turn++;
        const tag = m.role === 'user' ? `<span class="w14-turn">턴 ${turn}</span>` : '';
        const cls = m.role === 'user' ? 'user' : m.role === 'error' ? 'error' : 'bot';
        const extra = m.stopped ? '<span class="w14-stopped">■ 중지됨</span>' : '';
        return `<div class="w14-msg ${cls}" data-i="${i}">${tag}<div class="w14-bubble"><span class="w14-text">${esc(m.content)}</span>${m.streaming ? '<span class="w14-caret" aria-hidden="true"></span>' : ''}${extra}</div></div>`;
      })
      .join('');
    box.scrollTop = box.scrollHeight;
  }

  function renderOutput() {
    const hist = history();
    const lastUser = hist.map((m) => m.role).lastIndexOf('user');
    if (lastUser < 0) {
      $('[data-slot=verdict]').innerHTML = '<div class="callout"><span class="callout__title">아직 보낸 질문이 없다</span><p>질문을 보내면 이 자리에 API로 실제로 보낸 messages 배열이 나온다.</p></div>';
      ['checks', 'bud', 'turns', 'arr'].forEach((k) => ($(`[data-slot=${k}]`).innerHTML = ''));
      renderLog();
      return;
    }
    const ctx = context(hist.slice(0, lastUser + 1));
    const same = s.sentKey === settingsKey();
    const nTurn = hist.slice(0, lastUser + 1).filter((m) => m.role === 'user').length;

    $('[data-slot=arr-title]').innerHTML = same
      ? `messages 배열 <small class="w14-muted">— 턴 ${nTurn}에 실제로 보낸 것 (${ctx.messages.length}개)</small>`
      : `messages 배열 <small class="w14-warn">— 설정이 바뀌었다: 지금 설정으로 턴 ${nTurn}을 보낸다면 (미리보기)</small>`;

    // budget bar
    const t = ctx.tokens;
    const scale = Math.max(ctx.budget, t.total);
    const seg = (v, cls, label) => (v > 0 ? `<i class="${cls}" style="width:${((v / scale) * 100).toFixed(2)}%" title="${esc(`${label} ${v} 토큰`)}"></i>` : '');
    $('[data-slot=bud-title]').innerHTML = `토큰 예산 <small class="${ctx.over ? 'w14-bad' : 'w14-muted'}">— ${t.total} / ${ctx.budget} 토큰${ctx.over ? ' · 초과!' : ''}</small>`;
    $('[data-slot=bud]').innerHTML = `<div class="w14-bar${ctx.over ? ' over' : ''}">${seg(t.system, 'sys', '시스템')}${seg(t.summary, 'sum', '요약')}${seg(t.history, 'hist', '이전 대화')}${seg(t.current, 'cur', '새 질문')}${ctx.over ? `<b class="w14-limit" style="left:${((ctx.budget / scale) * 100).toFixed(2)}%" title="예산 한계"></b>` : ''}</div>
      <div class="stat-row">${stat('시스템', t.system)}${stat('요약', t.summary)}${stat('이전 대화', t.history)}${stat('새 질문', t.current)}${stat('합계', t.total)}</div>`;

    // turns
    $('[data-slot=turns]').innerHTML =
      ctx.turns
        .map((tr) => {
          const u = tr.messages[0]?.content ?? '';
          const S = STATUS[tr.status];
          return `<li class="w14-t ${S.cls}"><span class="w14-st">${S.label}</span><span class="w14-tq">턴 ${tr.index + 1} · ${esc(clip(u, 26))}</span><span class="w14-tt">${tr.tokens}</span></li>`;
        })
        .join('') + `<li class="w14-t cur"><span class="w14-st">새 질문</span><span class="w14-tq">턴 ${nTurn} · ${esc(clip(hist[lastUser].content, 26))}</span><span class="w14-tt">${t.current}</span></li>`;

    // messages array
    $('[data-slot=arr]').innerHTML = ctx.messages
      .map((m) => {
        let body = esc(m.content);
        if (m.role === 'system' && m.content.includes(SUMMARY_HEADER)) {
          const k = m.content.indexOf(SUMMARY_HEADER);
          body = `${esc(m.content.slice(0, k))}<mark class="w14-summ">${esc(m.content.slice(k))}</mark>`;
        }
        return `<div class="w14-m ${m.role}"><div class="w14-mh"><b>"role": "${m.role}"</b><span>≈ ${estimateTokens(m.content) + 4} 토큰</span></div><div class="w14-mc">${body}</div></div>`;
      })
      .join('');

    // checks (latest result of each memory / guardrail test)
    const latest = {};
    s.items.forEach((m) => m.checks && Object.assign(latest, m.checks));
    $('[data-slot=checks]').innerHTML = CHECKS.map((c) => {
      const v = latest[c.key];
      const cls = v === undefined ? '' : v ? 'ok' : 'bad';
      return `<span class="w14-chk ${cls}">${v === undefined ? '–' : v ? '✓' : '✗'} ${c.label}</span>`;
    }).join('');

    $('[data-slot=verdict]').innerHTML = verdict(ctx, s, latest);
    renderLog();
  }

  function renderLog() {
    $('[data-slot=log]').innerHTML = s.log.length
      ? s.log.slice(-10).map((l) => `<li class="${l.level}">${esc(l.text)}</li>`).join('')
      : '<li class="w14-muted">아직 기록이 없다.</li>';
  }

  function renderAll() {
    renderControls();
    renderChat();
    renderOutput();
  }

  // ---- sending
  async function send(text, { delay = 28 } = {}) {
    const q = text.trim();
    if (!q || s.busy || !s.corpus) return;
    s.busy = true;
    s.items.push({ role: 'user', content: q });
    const n = turnNo();
    const ctx = context(history());
    s.sentKey = settingsKey();
    const item = { role: 'assistant', content: '', streaming: true };
    s.items.push(item);
    renderAll();

    const run = new AbortController();
    st.run = run;
    const bubble = () => $('[data-slot=msgs]').lastElementChild?.querySelector('.w14-text');
    const t0 = performance.now();
    let tFirst = 0;
    const onDelta = (piece) => {
      if (!tFirst) tFirst = performance.now() - t0;
      item.content += piece;
      const b = bubble();
      if (b) b.textContent = item.content;
      const box = $('[data-slot=msgs]');
      box.scrollTop = box.scrollHeight;
    };
    const drops = ['window', 'summarized', 'budget'].map((k) => ctx.turns.filter((tr) => tr.status === k).length);
    const head = `턴 ${n} · ${s.mode === 'mock' ? '예시' : s.mode} · ${ctx.messages.length}개 메시지 · ${ctx.tokens.total}/${ctx.budget} 토큰` + (drops[0] ? ` · 창 밖 ${drops[0]}턴` : '') + (drops[1] ? ` · 요약 ${drops[1]}턴` : '') + (drops[2] ? ` · 예산 초과 ${drops[2]}턴` : '');

    try {
      let res;
      if (s.mode === 'mock') {
        if (ctx.over) {
          throw new LLMError(`400 · 입력 ${ctx.tokens.total} 토큰이 컨텍스트 창 ${ctx.budget} 토큰을 넘었다 (예시 오류). 기억 전략을 바꾸거나 예산을 늘린다.`, 'context');
        }
        const r = mockReply(s.corpus, ctx.messages, { temperature: s.temp, seed: n });
        item.checks = r.checks;
        await sleep(delay * 6, run.signal); // time to first token
        res = await simulateStream(r.text, { onDelta, signal: run.signal, delay });
      } else if (STREAM_PROVIDERS.includes(s.mode)) {
        if (ctx.over) addLog(`⚠ 예산(${ctx.budget})을 넘었지만 실제 모델의 창은 더 크므로 그대로 보낸다`, 'warn');
        res = await streamChat({ provider: s.mode, model: s.model, messages: ctx.messages, temperature: s.temp, signal: run.signal, onDelta });
      } else {
        const { system, messages } = splitSystem(ctx.messages);
        const r = await generate({ provider: s.mode, model: s.model, system, messages, temperature: s.temp, maxTokens: 512, signal: run.signal });
        addLog('Gemini는 한 번에 받은 답을 조각내어 보여 준다 (스트리밍 흉내)');
        res = await simulateStream(r.text, { onDelta, signal: run.signal, delay: 15 });
      }
      item.content = res.text;
      item.stopped = res.stopped;
      addLog(`${head} · 첫 조각 ${Math.round(tFirst)}ms · 조각 ${res.chunks}개${res.stopped ? ' · ■ 중지' : ''}`, res.stopped ? 'warn' : '');
    } catch (err) {
      if (err.name === 'AbortError') {
        item.stopped = true;
        addLog(`${head} · ■ 첫 조각 전에 중지`, 'warn');
      } else {
        s.items.splice(s.items.indexOf(item), 1);
        s.items.push({ role: 'error', content: `⚠ ${err instanceof LLMError ? err.message : '알 수 없는 오류가 났다.'}` });
        addLog(`${head} · 오류: ${err.code ?? err.name}`, 'err');
      }
    } finally {
      item.streaming = false;
      if (item.role === 'assistant' && !item.content && s.items.includes(item)) s.items.splice(s.items.indexOf(item), 1);
      s.busy = false;
      st.run = null;
      if (!ctrl.signal.aborted) renderAll();
    }
  }

  /** Instant (no animation) mock turn — used to pre-fill the first screen. */
  function sendInstant(q) {
    s.items.push({ role: 'user', content: q });
    const n = turnNo();
    const ctx = context(history());
    const r = mockReply(s.corpus, ctx.messages, { temperature: s.temp, seed: n });
    s.items.push({ role: 'assistant', content: r.text, checks: r.checks });
    s.sentKey = settingsKey();
    addLog(`턴 ${n} · 예시 · ${ctx.messages.length}개 메시지 · ${ctx.tokens.total}/${ctx.budget} 토큰 (미리 채운 대화)`);
  }

  async function playScenario() {
    if (s.scenario) {
      s.scenario = false;
      st.run?.abort();
      return;
    }
    if (s.busy) return;
    s.items = [];
    s.log = [];
    s.scenario = true;
    addLog(`▶ 시나리오 시작 · ${STRATEGIES.find((x) => x.id === s.strategy).label}${s.strategy === 'window' || s.strategy === 'summary' ? ` (N=${s.lastN})` : ''} · 예산 ${s.budget}`);
    renderAll();
    for (const q of SCENARIO) {
      if (!s.scenario || ctrl.signal.aborted) break;
      await send(q, { delay: 12 });
      await sleep(250, ctrl.signal).catch(() => {});
    }
    if (s.scenario) addLog('■ 시나리오 끝 · 위 점검표를 확인한다');
    s.scenario = false;
    if (!ctrl.signal.aborted) renderAll();
  }

  // ---- events
  const on = (sel, type, fn) => $(sel).addEventListener(type, fn, { signal: ctrl.signal });
  on('[data-in=system]', 'input', (e) => {
    s.system = e.target.value;
    s.preset = Object.keys(PRESETS).find((k) => PRESETS[k].text === s.system) ?? '';
    renderControls();
    renderOutput();
  });
  on('[data-in=strategy]', 'change', (e) => {
    s.strategy = e.target.value;
    renderControls();
    renderOutput();
  });
  on('[data-in=lastN]', 'input', (e) => {
    s.lastN = Number(e.target.value);
    renderControls();
    renderOutput();
  });
  on('[data-in=budget]', 'input', (e) => {
    s.budget = Number(e.target.value);
    renderControls();
    renderOutput();
  });
  on('[data-in=temp]', 'input', (e) => {
    s.temp = Number(e.target.value);
    renderControls();
  });
  on('[data-in=mode]', 'change', (e) => {
    s.mode = e.target.value;
    if (s.mode !== 'mock') {
      const p = PROVIDERS[s.mode];
      $('[data-in=model]').innerHTML = p.models.map((m) => `<option value="${esc(m)}">${esc(m)}</option>`).join('');
      s.model = p.defaultModel;
      $('[data-in=model]').value = s.model;
    }
    renderControls();
  });
  on('[data-in=model]', 'change', (e) => (s.model = e.target.value));
  on('[data-slot=form]', 'submit', (e) => {
    e.preventDefault();
    const ta = $('[data-in=msg]');
    const q = ta.value;
    ta.value = '';
    send(q);
  });
  on('[data-in=msg]', 'keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) {
      e.preventDefault();
      $('[data-slot=form]').requestSubmit();
    }
  });
  root.addEventListener(
    'click',
    (e) => {
      const t = e.target.closest('[data-act],[data-preset],[data-suggest]');
      if (!t) return;
      if (t.dataset.preset) {
        s.preset = t.dataset.preset;
        s.system = PRESETS[s.preset].text;
        $('[data-in=system]').value = s.system;
        renderControls();
        renderOutput();
        return;
      }
      if (t.dataset.suggest) return void send(t.dataset.suggest);
      const act = t.dataset.act;
      if (act === 'stop') {
        s.scenario = false;
        st.run?.abort();
      }
      if (act === 'scenario') playScenario();
      if (act === 'clear' && !s.busy) {
        s.items = [];
        s.log = [];
        renderAll();
      }
      if (act === 'save-key') {
        const input = $('[data-in=key]');
        if (input.value.trim()) setKey(s.mode, input.value);
        input.value = '';
        addLog(`${PROVIDERS[s.mode].label} 키 저장 (sessionStorage)`);
        renderControls();
        renderLog();
      }
      if (act === 'clear-key') {
        clearKey(s.mode);
        addLog(`${PROVIDERS[s.mode].label} 키 삭제`);
        renderControls();
        renderLog();
      }
    },
    { signal: ctrl.signal },
  );

  renderControls();
  try {
    s.corpus = await loadCorpus();
    if (ctrl.signal.aborted) return;
    $('[data-slot=status]').hidden = true;
    SCENARIO.slice(0, options.prefill ?? 3).forEach(sendInstant);
    renderAll();
  } catch (err) {
    $('[data-slot=status]').innerHTML = `<div class="widget__error">문서셋을 불러오지 못했다 (${esc(err.message)}). <code>file://</code>로 열었다면 <code>python -m http.server</code>로 연다.</div>`;
  }
}

export function unmount(el) {
  const st = state.get(el);
  if (!st) return;
  st.ctrl.abort();
  st.run?.abort();
  unregisterOutput(st.outputId);
  state.delete(el);
  el.replaceChildren();
}

// ---------------------------------------------------------------- helpers

function verdict(ctx, s, latest) {
  const last = [...s.items].reverse().find((m) => m.role !== 'user');
  const strat = STRATEGIES.find((x) => x.id === s.strategy).label;
  const lost = ctx.turns.filter((t) => t.status === 'window' || t.status === 'budget').length;
  const why =
    s.strategy === 'none'
      ? '“기억 없음”은 매 요청에 새 질문만 보낸다. API는 이전 요청을 기억하지 않으므로 모델은 처음 보는 대화로 받아들인다.'
      : ctx.turns.some((t) => t.status === 'budget')
        ? `토큰 예산 ${s.budget}에 맞추느라 오래된 턴을 버렸다. 시스템 프롬프트(≈ ${ctx.tokens.system} 토큰)는 고정이라 대화에 쓸 자리가 적다.`
        : s.strategy === 'window'
          ? `“최근 ${s.lastN}턴”만 보내므로 그보다 오래된 턴은 창 밖으로 밀려났다. “요약 + 최근 N턴”으로 바꿔 본다.`
          : '요약에서 해당 내용이 잘려 나갔다.';
  if (last?.role === 'error') {
    return `<div class="callout callout--danger"><span class="callout__title">요청 실패</span><p>${esc(last.content.replace(/^⚠ /, ''))}</p></div>`;
  }
  const c = latest; // latest result of each check in this conversation
  if (c.name === false) return `<div class="callout callout--danger"><span class="callout__title">이름을 잊었다</span><p>“민수”라고 말한 턴이 이번 요청에 들어 있지 않다. ${why}</p></div>`;
  if (c.course === false) return `<div class="callout callout--danger"><span class="callout__title">“아까 그 과목”을 모른다</span><p>과목 이름이 나온 턴이 이번 요청에 없다. ${why}</p></div>`;
  if (c.guard === false) return `<div class="callout callout--danger"><span class="callout__title">주제 밖 질문에 답했다 · 게다가 지어냈다</span><p>시스템 프롬프트에 “관련 없는 질문은 거절한다”는 규칙이 없다. 학과 안내 도우미가 식당 이름까지 지어낸다. 프리셋을 “친절한 조교”로 바꿔 다시 물어본다.</p></div>`;
  if (c.name || c.course) return `<div class="callout callout--ok"><span class="callout__title">기억했다</span><p>필요한 내용이 이번 요청에 ${ctx.summary ? '(최근 턴이나 요약으로) ' : ''}다시 실려 모델이 볼 수 있었다. 모델이 기억한 것이 아니라 앱이 다시 보낸 것이다.</p></div>`;
  if (c.guard) return `<div class="callout callout--ok"><span class="callout__title">주제 밖 질문을 거절했다</span><p>시스템 프롬프트의 규칙이 매 요청 맨 앞에 고정되어 있기 때문이다.</p></div>`;
  return `<div class="callout"><span class="callout__title">${esc(strat)} · ${ctx.messages.length}개 메시지</span><p>시스템 프롬프트 + 이전 대화 ${ctx.turns.filter((t) => t.status === 'kept').length}턴${ctx.summary ? ' + 요약' : ''} + 새 질문을 보낸다.${lost ? ` 이전 턴 ${lost}개는 보내지 않았다.` : ''} ▶ 시나리오 재생으로 7턴 대화를 돌려 본다.</p></div>`;
}

const stat = (label, value) => `<div class="stat"><span class="stat__label">${label}</span><span class="stat__value">${value}</span></div>`;
const clip = (t, n) => ([...t].length > n ? [...t].slice(0, n).join('') + '…' : t);

function sleep(ms, signal) {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) return reject(new DOMException('aborted', 'AbortError'));
    const t = setTimeout(resolve, ms);
    signal?.addEventListener('abort', () => (clearTimeout(t), reject(new DOMException('aborted', 'AbortError'))), { once: true });
  });
}

function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
}
