// w15 에이전트 루프 추적기
// One concept: an agent is an LLM in a loop — think → call a tool → observe →
// repeat until it answers or hits the step limit. Guards (policy check +
// human confirmation) sit between the model and side-effect tools.

import { registerOutput, unregisterOutput } from '../site/result.js';
import { loadCorpus } from '../core/text.js';
import { PROVIDERS, setKey, clearKey } from '../core/llm.js';
import {
  TASKS, AgentRun, createLabTools, createMockModel, anthropicModel, openaiModel,
  detectInjection, daysBetween, weekdayOf,
} from '../core/agent.js';

const TOOLS = [
  { name: 'search_docs', label: 'search_docs', note: '문서 검색' },
  { name: 'read_doc', label: 'read_doc', note: '문서 읽기' },
  { name: 'calculator', label: 'calculator', note: '계산' },
  { name: 'get_date', label: 'get_date', note: '오늘 날짜' },
  { name: 'book_lab', label: 'book_lab', note: '⚠ 예약 (부작용)' },
];
const LLM_PROVIDERS = ['anthropic', 'openai'];

const template = document.createElement('template');
template.innerHTML = `
  <div class="widget w15-ag">
    <h3 class="widget__title">에이전트 루프 추적기</h3>
    <div data-slot="status" class="widget__status" role="status" aria-live="polite">
      <span class="spinner" aria-hidden="true"></span> <span>문서셋 불러오는 중…</span>
    </div>
    <div class="w15-mode" role="radiogroup" aria-label="모델">
      <label><input type="radio" name="w15-mode" value="mock" data-in="mode" checked> <span>🧩 예시 에이전트 <small>규칙 기반 · 키 불필요</small></span></label>
      <label><input type="radio" name="w15-mode" value="llm" data-in="mode"> <span>🔑 실제 LLM <small>내 API 키 · 도구 호출</small></span></label>
    </div>
    <div class="widget__controls">
      <label class="field">
        <span class="field__label">과제</span>
        <select data-in="task"></select>
      </label>
      <label class="field">
        <span class="field__label">최대 단계 <output data-out="max"></output></span>
        <input type="range" data-in="max" min="1" max="12" step="1">
      </label>
    </div>
    <label class="field w15-wide">
      <span class="field__label">과제 문장 <small class="w15-muted" data-out="task-hint"></small></span>
      <textarea data-in="text" rows="2" spellcheck="false"></textarea>
    </label>
    <div class="w15-llm" data-slot="llm" hidden>
      <div class="widget__controls">
        <label class="field">
          <span class="field__label">공급자</span>
          <select data-in="provider"></select>
        </label>
        <label class="field">
          <span class="field__label">API 키 <small class="w15-muted" data-out="keystate"></small></span>
          <input type="password" data-in="key" autocomplete="off" spellcheck="false" placeholder="붙여 넣고 저장한다">
        </label>
      </div>
      <div class="btn-row">
        <button type="button" class="btn small" data-act="save-key">키 저장 (이 탭에서만)</button>
        <button type="button" class="btn small ghost" data-act="clear-key">키 지우기</button>
        <span class="w15-muted" data-out="model"></span>
      </div>
    </div>
    <fieldset class="w15-set">
      <legend>에이전트에게 줄 도구 <small class="w15-muted">— 끈 도구는 목록에서도 빠진다 (최소 권한)</small></legend>
      <div class="w15-checks" data-slot="tools"></div>
    </fieldset>
    <fieldset class="w15-set">
      <legend>실패 실험</legend>
      <div class="w15-checks">
        <label><input type="checkbox" data-in="injection"> 🧪 오염 문서 섞기 <small>(프롬프트 주입)</small></label>
        <label><input type="checkbox" data-in="toolError"> 🧪 도구 오류 <small>(잘못된 id · 예시 에이전트)</small></label>
        <label class="w15-danger"><input type="checkbox" data-in="noGuard"> ☠ 확인 없이 실행 <small>(가드 끔 · 위험)</small></label>
      </div>
      <div class="btn-row w15-quick">
        <span class="w15-muted">바로 보기:</span>
        <button type="button" class="btn small ghost" data-act="q-limit">단계 3으로</button>
        <button type="button" class="btn small ghost" data-act="q-nosearch">검색 끄기</button>
        <button type="button" class="btn small ghost" data-act="q-inject">주입 공격</button>
        <button type="button" class="btn small ghost" data-act="q-error">도구 오류</button>
        <button type="button" class="btn small ghost" data-act="q-reset">↺ 기본값</button>
      </div>
    </fieldset>
    <div class="btn-row">
      <button type="button" class="btn small" data-act="step">▶ 한 단계</button>
      <button type="button" class="btn small primary" data-act="all">▶▶ 끝까지</button>
      <button type="button" class="btn small ghost" data-act="restart">↺ 처음부터</button>
      <button type="button" class="btn small ghost" data-act="stop" hidden>■ 멈춤</button>
    </div>
    <div data-slot="out-inline"></div>
  </div>`;

const outTemplate = document.createElement('template');
outTemplate.innerHTML = `
  <div class="w15-out">
    <div data-slot="verdict" aria-live="polite"></div>
    <div class="stat-row" data-slot="stats"></div>
    <p class="w15-task" data-slot="task"></p>
    <ol class="w15-tl" data-slot="timeline"></ol>
    <div data-slot="confirm"></div>
    <div data-slot="final"></div>
    <div class="w15-env" data-slot="env"></div>
  </div>`;

const state = new WeakMap();
let seq = 0;

/**
 * @param {HTMLElement} el
 * @param {{ task?: string, outputTitle?: string }} options
 */
export async function mount(el, options = {}) {
  const root = template.content.firstElementChild.cloneNode(true);
  const out = outTemplate.content.firstElementChild.cloneNode(true);
  el.replaceChildren(root);
  const $ = (sel) => root.querySelector(sel) ?? out.querySelector(sel);
  const ctrl = new AbortController();
  const outputId = `w15-agent:${++seq}`;
  state.set(el, { ctrl, outputId });
  const uid = seq;
  root.querySelectorAll('input[name="w15-mode"]').forEach((r) => (r.name = `w15-mode-${uid}`));

  const DEFAULTS = { maxSteps: 8, tools: new Set(TOOLS.map((t) => t.name)), injection: false, toolError: false, noGuard: false };
  const s = {
    mode: 'mock',
    taskId: options.task ?? 'book',
    text: '',
    provider: 'anthropic',
    maxSteps: DEFAULTS.maxSteps,
    tools: new Set(DEFAULTS.tools),
    injection: false,
    toolError: false,
    noGuard: false,
    corpus: null,
    run: null,
    registry: null,
    busy: false,
    abort: null,
    note: '',
  };
  s.text = TASKS.find((t) => t.id === s.taskId)?.text ?? TASKS[0].text;

  $('[data-in=task]').innerHTML =
    TASKS.map((t) => `<option value="${t.id}">${esc(t.label)}</option>`).join('') + '<option value="custom">직접 쓰기 (실제 LLM 전용)</option>';
  $('[data-in=provider]').innerHTML = LLM_PROVIDERS.map((p) => `<option value="${p}">${esc(PROVIDERS[p].label)}</option>`).join('');
  $('[data-slot=tools]').innerHTML = TOOLS.map(
    (t) => `<label><input type="checkbox" data-tool="${t.name}" checked> <code>${t.label}</code> <small>${esc(t.note)}</small></label>`,
  ).join('');

  registerOutput(outputId, { title: options.outputTitle ?? '에이전트 루프 추적 결과', node: out, inlineHost: $('[data-slot=out-inline]') });

  // ------------------------------------------------------------ sync UI ← state
  function syncControls() {
    root.querySelectorAll('[data-in=mode]').forEach((r) => (r.checked = r.value === s.mode));
    $('[data-in=task]').value = s.taskId;
    const custom = s.taskId === 'custom';
    if (document.activeElement !== $('[data-in=text]')) $('[data-in=text]').value = s.text;
    $('[data-in=text]').readOnly = s.mode === 'mock';
    $('[data-out=task-hint]').textContent = s.mode === 'mock' ? '— 예시 에이전트는 준비된 과제만 푼다' : '— 자유롭게 고쳐 쓸 수 있다';
    $('[data-in=max]').value = String(s.maxSteps);
    $('[data-out=max]').textContent = `${s.maxSteps}단계`;
    root.querySelectorAll('[data-tool]').forEach((c) => (c.checked = s.tools.has(c.dataset.tool)));
    $('[data-in=injection]').checked = s.injection;
    $('[data-in=toolError]').checked = s.toolError;
    $('[data-in=toolError]').disabled = s.mode !== 'mock';
    $('[data-in=noGuard]').checked = s.noGuard;
    $('[data-slot=llm]').hidden = s.mode !== 'llm';
    $('[data-in=provider]').value = s.provider;
    $('[data-out=keystate]').textContent = readKey(s.provider) ? '— 저장됨' : '— 없음';
    $('[data-out=model]').textContent = `모델: ${PROVIDERS[s.provider].defaultModel}`;
    $('[data-act=stop]').hidden = !(s.mode === 'llm' && s.busy);
    for (const a of ['step', 'all', 'restart']) $(`[data-act=${a}]`).disabled = s.busy;
    if (custom && s.mode === 'mock') $('[data-in=task]').value = 'custom';
  }

  // ------------------------------------------------------------ run management
  function newRun() {
    s.abort?.abort();
    s.abort = null;
    s.busy = false;
    s.note = '';
    const registry = createLabTools(s.corpus, { injection: s.injection });
    registry.setEnabled([...s.tools]);
    let model;
    if (s.mode === 'mock') model = createMockModel(s.taskId, { toolError: s.toolError });
    else {
      const key = readKey(s.provider);
      if (!key) {
        s.run = null;
        s.registry = registry;
        s.note = 'no-key';
        return;
      }
      model = s.provider === 'anthropic' ? anthropicModel({ key, model: PROVIDERS.anthropic.defaultModel }) : openaiModel({ key, model: PROVIDERS.openai.defaultModel });
    }
    s.registry = registry;
    s.run = new AgentRun({
      task: s.text,
      registry,
      model,
      maxSteps: s.maxSteps,
      guard: s.noGuard ? { confirm: false, policy: false } : { confirm: true, policy: true },
    });
    s.run.modelName = model.name;
  }

  async function drive(kind) {
    if (!s.corpus || s.busy) return;
    if (!s.run || s.run.finished) newRun();
    if (!s.run) return render();
    if (s.run.state === 'confirm') return render();
    const run = s.run;
    const abort = new AbortController();
    s.busy = true;
    s.abort = abort;
    render();
    try {
      if (kind === 'step') await run.step(abort.signal);
      else await run.runToEnd({ signal: abort.signal, onUpdate: () => s.run === run && render() });
    } catch (err) {
      if (err.name !== 'AbortError') {
        run.state = 'error';
        run.error = err.message;
      } else if (s.run === run) s.note = 'stopped';
    }
    if (s.run !== run) return; // a newer run replaced this one
    s.busy = false;
    s.abort = null;
    render();
  }

  /** Any control change: start over; the example agent re-runs instantly (no cost). */
  function restartAndMaybeRun() {
    newRun();
    if (s.mode === 'mock') drive('all');
    else render();
  }

  // ------------------------------------------------------------ render
  function render() {
    syncControls();
    const run = s.run;
    $('[data-slot=task]').innerHTML = `<b>과제</b> ${esc(s.text)}`;
    if (!run) {
      $('[data-slot=verdict]').innerHTML = s.note === 'no-key'
        ? `<div class="callout"><span class="callout__title">🔑 API 키가 필요하다</span><p>위에 키를 넣고 저장한 뒤 ▶를 누른다. 키는 이 탭의 sessionStorage에만 남고 탭을 닫으면 사라진다. 키가 없으면 “예시 에이전트”로 같은 루프를 본다.</p></div>`
        : '';
      for (const k of ['stats', 'timeline', 'confirm', 'final', 'env']) $(`[data-slot=${k}]`).innerHTML = '';
      return;
    }
    const env = s.registry.env;
    const calls = run.steps.flatMap((st) => st.calls);
    const errors = calls.filter((c) => c.isError).length;
    $('[data-slot=stats]').innerHTML = [
      stat('상태', STATE_LABEL[s.busy ? 'busy' : run.state] ?? run.state),
      stat('단계', `${run.steps.length} / ${run.maxSteps}`),
      stat('도구 호출', `${calls.length}회${errors ? ` · 오류 ${errors}` : ''}`),
      stat(run.usage.estimated ? '입력 토큰 (추정)' : '입력 토큰', run.usage.input.toLocaleString()),
      stat(run.usage.estimated ? '출력 토큰 (추정)' : '출력 토큰', run.usage.output.toLocaleString()),
    ].join('');

    $('[data-slot=timeline]').innerHTML = run.steps.length
      ? run.steps.map((st) => stepHtml(st)).join('')
      : `<li class="w15-empty">아직 한 단계도 실행하지 않았다. <b>▶ 한 단계</b>를 누르면 모델이 한 번 생각하고 도구를 하나 고른다.${s.mode === 'llm' ? ' (실제 LLM 모드: 누를 때마다 API가 호출된다)' : ''}</li>`;

    // pending confirmation (human in the loop)
    const pend = run.state === 'confirm' ? run.pending?.entry : null;
    $('[data-slot=confirm]').innerHTML = pend
      ? `<div class="w15-confirm" role="group" aria-label="부작용 도구 실행 확인">
          <b>⏸ 실행 전 확인 — 부작용이 있는 도구다</b>
          <p><code>${esc(pend.name)}(${esc(argsText(pend.args))})</code> → ${esc(pend.args.date ?? '')}${pend.args.date ? `(${esc(weekdayOf(pend.args.date))})` : ''} ${esc(pend.args.hours ?? '')}시간 예약</p>
          ${checksHtml(pend.guard?.policy)}
          <div class="btn-row">
            <button type="button" class="btn small primary" data-act="approve">✅ 승인하고 실행</button>
            <button type="button" class="btn small ghost" data-act="reject">✋ 거부</button>
          </div>
        </div>`
      : '';

    $('[data-slot=final]').innerHTML = run.final != null
      ? `<div class="w15-final"><span class="w15-final__t">최종 답</span><p>${esc(run.final)}</p></div>`
      : run.state === 'limit'
        ? `<div class="w15-final is-cut"><span class="w15-final__t">최종 답 없음</span><p>단계 한도(${run.maxSteps})에 걸려 루프가 잘렸다. 사용자에게는 아무 답도 가지 않는다.</p></div>`
        : run.state === 'error'
          ? `<div class="widget__error">${esc(run.error ?? '오류')}</div>`
          : '';

    $('[data-slot=env]').innerHTML = `<b>환경 상태</b> 오늘 ${esc(env.today)}(${esc(weekdayOf(env.today))}) · 읽은 문서 ${env.readDocs.size ? [...env.readDocs].map((d) => `<code>${esc(d)}</code>`).join(' ') : '없음'} · 예약 ${
      env.bookings.length
        ? env.bookings.map((b) => `<span class="w15-bk${bookingBad(b, env) ? ' bad' : ''}">${esc(b.reservation_id)} ${esc(b.date)}(${esc(b.weekday)}) ${b.hours}시간${bookingBad(b, env) ? ' ⚠ 규정 위반' : ''}</span>`).join(' ')
        : '없음'
    }${s.injection ? ' · <span class="w15-bk bad">오염 문서 board-gpu-tip 섞임</span>' : ''}`;

    $('[data-slot=verdict]').innerHTML = verdict(run, env);
  }

  function verdict(run, env) {
    const calls = run.steps.flatMap((st) => st.calls);
    const blocked = calls.filter((c) => c.guard?.decision === 'blocked');
    const injectedSeen = calls.some((c) => c.observation && detectInjection(JSON.stringify(c.observation)).length);
    const bad = env.bookings.filter((b) => bookingBad(b, env));
    if (s.busy) return `<div class="callout"><span class="callout__title">⏳ 실행 중…</span><p>${s.mode === 'llm' ? '모델 응답을 기다린다.' : '예시 에이전트가 다음 단계를 고른다.'}</p></div>`;
    if (s.note === 'stopped') return `<div class="callout"><span class="callout__title">■ 멈췄다</span><p>▶를 누르면 이어서 실행한다.</p></div>`;
    if (run.state === 'ready') return `<div class="callout"><span class="callout__title">준비됨</span><p>모델에게 보이는 도구 ${s.registry.schemas().length}개 · 최대 ${run.maxSteps}단계. ▶ 한 단계씩 따라가거나 ▶▶로 끝까지 돌린다.</p></div>`;
    if (run.state === 'confirm') return `<div class="callout callout--more"><span class="callout__title">⏸ 사람의 확인을 기다린다</span><p>에이전트가 예약(부작용)을 실행하려 한다. 정책 검사는 통과했다. 아래에서 인자를 확인하고 승인하거나 거부한다.</p></div>`;
    if (run.state === 'limit') return `<div class="callout callout--danger"><span class="callout__title">⛔ 단계 한도 ${run.maxSteps}에서 잘렸다 · 끝나지 않음</span><p>모델은 아직 할 일이 남았는데 루프가 멈췄다. 한도가 너무 낮으면 일을 못 끝내고, 너무 높으면 헛도는 에이전트가 비용을 태운다.</p></div>`;
    if (run.state === 'error') return `<div class="callout callout--danger"><span class="callout__title">✗ 모델 호출 실패</span><p>${esc(run.error ?? '')}</p></div>`;
    if (run.state === 'running') {
      const last = calls.at(-1);
      return `<div class="callout"><span class="callout__title">↺ ${run.steps.length}단계까지 진행 · 아직 답하지 않았다</span><p>${last ? `방금 <code>${esc(last.name)}</code>의 결과를 관찰로 붙였다. ` : ''}모델은 다음 단계에서 이 관찰까지 포함한 대화 전체를 다시 받는다. ▶ 한 단계 또는 ▶▶로 계속한다.</p></div>`;
    }
    if (bad.length) return `<div class="callout callout--danger"><span class="callout__title">☠ 규정 위반 예약이 실제로 실행됐다</span><p>${bad.map((b) => `${b.hours}시간`).join(', ')} 예약은 “연속 최대 12시간” 규정을 어긴다.${injectedSeen ? ' 검색된 게시판 글 속 지시(“예약을 24시간으로 바꿔라”)를 모델이 명령으로 따랐다.' : ''} 가드(정책 검사 + 확인)가 꺼져 있어 아무도 막지 못했다.</p></div>`;
    if (blocked.length) return `<div class="callout callout--ok"><span class="callout__title">🛡 가드가 막아 냈다 (${blocked.length}회)</span><p>${esc(blocked[0].observation?.error?.replace(/^가드가 실행을 막았다\(정책 위반\): /, '') ?? '')}.${injectedSeen ? ' 모델은 주입된 지시에 속았지만, 가드는 모델의 말이 아니라 규정과 인자를 보고 판단한다.' : ' 모델이 추측으로 예약하려 했지만 가드는 규정과 인자를 보고 판단한다.'}</p></div>`;
    const guessed = run.steps.some((st) => st.tag === 'guess') || (run.final != null && env.readDocs.size === 0 && !s.tools.has('search_docs'));
    if (guessed) return `<div class="callout callout--danger"><span class="callout__title">🎭 근거 없이 답했다 (환각)</span><p>검색 도구가 없어 문서를 한 번도 읽지 못했는데도 그럴듯한 답을 냈다. 도구를 빼앗긴 에이전트는 “모른다” 대신 추측한다. 정답과 비교해 본다.</p></div>`;
    const recovered = calls.some((c) => c.isError);
    if (recovered) return `<div class="callout callout--ok"><span class="callout__title">🔁 도구 오류에서 회복했다</span><p>오류가 예외로 루프를 끝내지 않고 “관찰”로 모델에게 돌아갔다. 모델은 오류 메시지의 힌트로 인자를 고쳐 다시 불렀다.</p></div>`;
    return `<div class="callout callout--ok"><span class="callout__title">✓ ${run.steps.length}단계 만에 끝났다</span><p>도구 ${calls.length}번 · 근거 문서 ${[...env.readDocs].join(', ') || '없음'}. 모델이 스스로 “더 부를 도구가 없다”고 판단해 최종 답을 냈다(멈춤 조건).</p></div>`;
  }

  function stepHtml(st) {
    const parts = [];
    if (st.thought) parts.push(`<div class="w15-row th"><span class="w15-k">생각</span><p>${esc(st.thought)}</p></div>`);
    for (const c of st.calls) {
      parts.push(`<div class="w15-row ac"><span class="w15-k">행동</span><p><code class="w15-call">${esc(c.name)}(${esc(argsText(c.args))})</code>${toolBadge(c.name)}</p></div>`);
      if (c.guard && c.guard.decision !== 'pending') parts.push(`<div class="w15-row gd"><span class="w15-k">가드</span><div>${guardText(c.guard)}${c.guard.policy ? checksHtml(c.guard.policy) : ''}</div></div>`);
      if (c.observation) {
        const inj = detectInjection(JSON.stringify(c.observation));
        parts.push(`<div class="w15-row ob${c.isError ? ' err' : ''}"><span class="w15-k">관찰</span><div>${obsHtml(c)}${inj.length ? `<span class="w15-inj">⚠ 도구 출력에 지시문이 있다 (${esc(inj.join(', '))}) — 데이터로만 다뤄야 한다</span>` : ''}</div></div>`);
      } else if (c.guard?.decision === 'pending') {
        parts.push(`<div class="w15-row ob"><span class="w15-k">관찰</span><p class="w15-muted">⏸ 사람의 확인을 기다리는 중</p></div>`);
      }
    }
    if (st.final != null) parts.push(`<div class="w15-row fi"><span class="w15-k">답</span><p>${esc(st.final)}</p></div>`);
    const tag = st.tag === 'guess' ? '<span class="w15-tag bad">추측</span>' : st.tag === 'injected' ? '<span class="w15-tag bad">주입에 속음</span>' : st.tag === 'retry' || st.tag === 'recover' ? '<span class="w15-tag">재시도</span>' : '';
    return `<li class="w15-step"><div class="w15-step__h">단계 ${st.n}${tag}</div>${parts.join('')}</li>`;
  }

  // ------------------------------------------------------------ events
  const on = (sel, type, fn) => $(sel).addEventListener(type, fn, { signal: ctrl.signal });
  root.querySelectorAll('[data-in=mode]').forEach((r) =>
    r.addEventListener('change', () => {
      s.mode = r.value;
      if (s.mode === 'mock' && s.taskId === 'custom') {
        s.taskId = 'book';
        s.text = TASKS[0].text;
      }
      restartAndMaybeRun();
    }, { signal: ctrl.signal }),
  );
  on('[data-in=task]', 'change', (e) => {
    s.taskId = e.target.value;
    const t = TASKS.find((x) => x.id === s.taskId);
    if (t) s.text = t.text;
    if (s.taskId === 'custom' && s.mode === 'mock') {
      s.mode = 'llm';
      s.text = '';
    }
    restartAndMaybeRun();
    if (s.taskId === 'custom') $('[data-in=text]').focus();
  });
  on('[data-in=text]', 'change', (e) => {
    s.text = e.target.value.trim();
    const t = TASKS.find((x) => x.text === s.text);
    s.taskId = t ? t.id : 'custom';
    restartAndMaybeRun();
  });
  on('[data-in=max]', 'input', (e) => {
    s.maxSteps = Number(e.target.value);
    restartAndMaybeRun();
  });
  on('[data-slot=tools]', 'change', (e) => {
    const name = e.target.dataset.tool;
    if (!name) return;
    if (e.target.checked) s.tools.add(name);
    else s.tools.delete(name);
    restartAndMaybeRun();
  });
  for (const k of ['injection', 'toolError', 'noGuard']) {
    on(`[data-in=${k}]`, 'change', (e) => {
      s[k] = e.target.checked;
      restartAndMaybeRun();
    });
  }
  on('[data-in=provider]', 'change', (e) => {
    s.provider = e.target.value;
    restartAndMaybeRun();
  });

  root.addEventListener('click', (e) => {
    const act = e.target.closest('[data-act]')?.dataset.act;
    if (!act) return;
    if (act === 'step') return drive('step');
    if (act === 'all') return drive('all');
    if (act === 'restart') {
      newRun();
      return render();
    }
    if (act === 'stop') return s.abort?.abort();
    if (act === 'save-key') {
      const v = $('[data-in=key]').value.trim();
      try {
        if (v) setKey(s.provider, v);
      } catch {
        /* storage blocked: the key simply is not kept */
      }
      $('[data-in=key]').value = '';
      return restartAndMaybeRun();
    }
    if (act === 'clear-key') {
      try {
        clearKey(s.provider);
      } catch {
        /* storage blocked */
      }
      return restartAndMaybeRun();
    }
    // quick failure presets (always on the booking task, since it has the side-effect tool)
    if (act.startsWith('q-')) {
      Object.assign(s, { maxSteps: DEFAULTS.maxSteps, tools: new Set(DEFAULTS.tools), injection: false, toolError: false, noGuard: false });
      if (act !== 'q-reset' && s.taskId === 'custom') s.taskId = 'book';
      if (act === 'q-limit') s.maxSteps = 3;
      if (act === 'q-nosearch') s.tools.delete('search_docs');
      if (act === 'q-inject') {
        s.injection = true;
        s.taskId = 'book';
      }
      if (act === 'q-error') {
        s.toolError = true;
        s.mode = 'mock';
      }
      const t = TASKS.find((x) => x.id === s.taskId);
      if (t) s.text = t.text;
      restartAndMaybeRun();
    }
  }, { signal: ctrl.signal });

  out.addEventListener('click', async (e) => {
    const act = e.target.closest('[data-act]')?.dataset.act;
    if (!s.run || s.run.state !== 'confirm' || (act !== 'approve' && act !== 'reject')) return;
    await s.run.resolve(act === 'approve');
    if (s.mode === 'mock') await drive('all');
    else render();
  }, { signal: ctrl.signal });

  // ------------------------------------------------------------ boot
  try {
    s.corpus = await loadCorpus();
    if (ctrl.signal.aborted) return;
    $('[data-slot=status]').hidden = true;
    restartAndMaybeRun();
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

const STATE_LABEL = { ready: '준비', running: '진행 중', busy: '실행 중', confirm: '확인 대기', done: '완료', limit: '한도 도달', error: '오류' };

function readKey(provider) {
  try {
    return sessionStorage.getItem(`llmlab:key:${provider}`) || '';
  } catch {
    return '';
  }
}

function bookingBad(b, env) {
  const max = env.rules?.maxHours ?? 12;
  const ahead = daysBetween(env.today, b.date);
  const weekend = b.weekday === '토' || b.weekday === '일';
  return b.hours > max || ahead < 0 || (weekend && ahead < (env.rules?.advanceDays ?? 2));
}

function argsText(args) {
  const entries = Object.entries(args ?? {});
  if (!entries.length) return '';
  return entries.map(([k, v]) => `${k}: ${JSON.stringify(v)}`).join(', ');
}

function toolBadge(name) {
  return name === 'book_lab' ? ' <span class="w15-tag warn">부작용</span>' : '';
}

function guardText(g) {
  const map = {
    blocked: '<b class="w15-no">🛡 차단</b> 정책 검사 실패 — 실행하지 않고 오류를 관찰로 돌려준다',
    approved: '<b class="w15-yes">✅ 승인</b> 정책 검사 통과 → 사람이 승인해 실행했다',
    rejected: '<b class="w15-no">✋ 거부</b> 사람이 거부했다',
    auto: '<b class="w15-no">☠ 확인 없이 실행</b>',
  };
  return map[g.decision] ?? '';
}

function checksHtml(policy) {
  if (!policy) return '';
  return `<ul class="w15-checklist">${policy.checks.map((c) => `<li class="${c.ok ? 'ok' : 'no'}">${c.ok ? '✓' : '✗'} ${esc(c.label)}</li>`).join('')}</ul>`;
}

function obsHtml(c) {
  const o = c.observation;
  if (c.isError) return `<p class="w15-err">✗ ${esc(o.error)}</p>`;
  if (c.name === 'search_docs') {
    if (!o.results?.length) return '<p class="w15-muted">검색 결과 없음</p>';
    return `<ol class="w15-hits">${o.results
      .map((r) => `<li><code>${esc(r.id)}</code> <b>${esc(r.title)}</b> <span class="w15-muted">score ${r.score}</span><br><span class="w15-snip">“${esc(r.snippet)}”</span></li>`)
      .join('')}</ol>`;
  }
  if (c.name === 'read_doc') {
    return `<details><summary><code>${esc(o.id)}</code> ${esc(o.title)} <span class="w15-muted">· ${[...o.text].length}자 · ${esc(o.source)}</span></summary><p class="w15-doc">${esc(o.text)}</p></details>`;
  }
  return `<code class="w15-json">${esc(JSON.stringify(o))}</code>`;
}

const stat = (label, value) => `<div class="stat"><span class="stat__label">${label}</span><span class="stat__value">${value}</span></div>`;

function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
}
