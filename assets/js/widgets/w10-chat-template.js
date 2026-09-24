// w10 채팅 템플릿 뷰어
// One concept: a chat model only ever sees ONE string. SFT turns a list of
// messages into that string with special tokens (the chat template), and the
// loss is computed only on the assistant's tokens (the loss mask).
// Failure modes: a base model continues the document instead of answering,
// and a template mismatch between training and inference breaks stopping.

import { registerOutput, unregisterOutput } from '../site/result.js';

// Simplified versions of real templates (no tools, no default system prompt).
export const TEMPLATES = {
  chatml: { label: 'ChatML 형식 (Qwen 등) · 간략화', stop: '<|im_end|>' },
  llama3: { label: 'Llama 3 형식 · 간략화', stop: '<|eot_id|>' },
  gemma: { label: 'Gemma 형식 · 간략화', stop: '<end_of_turn>' },
  none: { label: '템플릿 없음 (그냥 이어 붙이기)', stop: null },
};

const ROLES = { system: '시스템', user: '사용자', assistant: '어시스턴트' };

export const DEFAULT_MESSAGES = [
  { role: 'system', content: '너는 AI응용소프트웨어과 학생을 돕는 조교다. 학과 규정에 근거해 짧게 답한다.' },
  { role: 'user', content: '실습실은 몇 시까지 이용할 수 있어?' },
  { role: 'assistant', content: '평일에는 오후 9시까지 이용할 수 있다. 주말과 공휴일에는 사전 신청한 학생만 쓸 수 있다.' },
];

/**
 * Render messages with a template.
 * @returns {{ segments: {kind: 'special'|'role'|'content'|'sep', text: string, train: boolean}[], notes: {level: string, text: string}[] }}
 *   train = true → this segment's tokens are in the loss (assistant answer + its end token).
 */
export function renderTemplate(messages, id, { addGen = false } = {}) {
  const segments = [];
  const notes = [];
  const sp = (text, train = false) => segments.push({ kind: 'special', text, train });
  const tx = (kind, text, train = false) => {
    if (text) segments.push({ kind, text, train });
  };

  if (id === 'chatml') {
    for (const m of messages) {
      const a = m.role === 'assistant';
      sp('<|im_start|>');
      tx('role', `${m.role}\n`);
      tx('content', m.content, a);
      sp('<|im_end|>', a);
      tx('sep', '\n');
    }
    if (addGen) {
      sp('<|im_start|>');
      tx('role', 'assistant\n');
    }
  } else if (id === 'llama3') {
    sp('<|begin_of_text|>');
    for (const m of messages) {
      const a = m.role === 'assistant';
      sp('<|start_header_id|>');
      tx('role', m.role);
      sp('<|end_header_id|>');
      tx('role', '\n\n');
      tx('content', m.content.trim(), a);
      sp('<|eot_id|>', a);
    }
    if (addGen) {
      sp('<|start_header_id|>');
      tx('role', 'assistant');
      sp('<|end_header_id|>');
      tx('role', '\n\n');
    }
  } else if (id === 'gemma') {
    sp('<bos>');
    let list = messages;
    let prefix = '';
    if (list[0]?.role === 'system') {
      prefix = `${list[0].content.trim()}\n\n`;
      list = list.slice(1);
      notes.push({ level: 'more', text: 'Gemma 형식에는 system 차례가 없다. 시스템 지시문을 첫 사용자 차례 앞에 붙였다.' });
    }
    let expect = 'user';
    let broken = false;
    list.forEach((m, i) => {
      const a = m.role === 'assistant';
      const role = a ? 'model' : m.role;
      if (m.role === 'system') broken = true;
      else if (m.role !== expect) broken = true;
      expect = m.role === 'user' ? 'assistant' : 'user';
      sp('<start_of_turn>');
      tx('role', `${role}\n`);
      tx('content', (i === 0 && m.role === 'user' ? prefix : '') + m.content.trim(), a);
      sp('<end_of_turn>', a);
      tx('sep', '\n');
    });
    if (prefix && list[0]?.role !== 'user') broken = true;
    if (broken) {
      notes.push({ level: 'danger', text: 'Gemma 형식은 user → model → user … 가 번갈아 와야 한다. 실제 템플릿 코드는 이 대화에서 오류를 낸다.' });
    }
    if (addGen) {
      sp('<start_of_turn>');
      tx('role', 'model\n');
    }
  } else {
    // no template: plain concatenation, trained like pretraining (every token)
    messages.forEach((m, i) => {
      tx('content', m.content, true);
      if (i < messages.length - 1) tx('sep', '\n\n', true);
    });
  }
  return { segments, notes };
}

/** "Token" = one character, except a special token counts as one (mini GPT style). */
export function tokenize(segments) {
  const toks = [];
  for (const s of segments) {
    if (s.kind === 'special') toks.push({ text: s.text, special: true, train: s.train });
    else for (const ch of s.text) toks.push({ text: ch, special: false, train: s.train });
  }
  return toks;
}

export const joinSegments = (segments) => segments.map((s) => s.text).join('');

// ---------------------------------------------------------------- widget

const template = document.createElement('template');
template.innerHTML = `
  <div class="widget w10-ct">
    <h3 class="widget__title">채팅 템플릿 뷰어</h3>
    <div class="widget__controls">
      <label class="field">
        <span class="field__label">템플릿</span>
        <select data-in="tpl"></select>
      </label>
      <div class="field w10-checks">
        <span class="field__label">보기</span>
        <label><input type="checkbox" data-in="mask" checked> 손실 마스크 표시</label>
        <label><input type="checkbox" data-in="gen"> 생성 프롬프트 붙이기 (추론용)</label>
      </div>
    </div>
    <div class="w10-msgs" data-slot="msgs"></div>
    <div class="btn-row">
      <button type="button" class="btn small ghost" data-act="add-user">＋ 사용자</button>
      <button type="button" class="btn small ghost" data-act="add-assistant">＋ 어시스턴트</button>
      <button type="button" class="btn small ghost" data-act="add-system">＋ 시스템</button>
      <button type="button" class="btn small ghost" data-act="reset">↺ 기본 대화</button>
    </div>
    <div class="btn-row w10-fail-row">
      <span class="w10-fail-label">실패 사례</span>
      <button type="button" class="btn small" data-fail="base" aria-pressed="false">🧪 베이스 모델에 그냥 묻기</button>
      <button type="button" class="btn small" data-fail="mismatch" aria-pressed="false">⚠ 템플릿 불일치</button>
    </div>
    <div data-slot="status" class="widget__status" role="status" aria-live="polite">
      <span class="spinner" aria-hidden="true"></span> <span>실패 사례 예시 불러오는 중…</span>
    </div>
    <div data-slot="out-inline"></div>
  </div>`;

const outTemplate = document.createElement('template');
outTemplate.innerHTML = `
  <div class="w10-ct-out">
    <div data-slot="verdict" aria-live="polite"></div>
    <h4 data-slot="view-title">모델이 실제로 보는 문자열</h4>
    <div class="w10-view" data-slot="view"></div>
    <div class="legend" aria-hidden="true">
      <span><i class="w10-lg-sp"></i>특수 토큰</span>
      <span><i class="w10-lg-role"></i>역할 표시</span>
      <span><i class="w10-lg-train"></i>학습(손실 계산)</span>
      <span><i class="w10-lg-mask"></i>마스크(손실 제외)</span>
    </div>
    <div class="stat-row" data-slot="stats"></div>
    <h4>토큰별 손실 마스크 <small class="w10-muted">— 칸 하나 = 토큰 하나 (글자 1개 또는 특수 토큰 1개)</small></h4>
    <div class="w10-strip" data-slot="strip"></div>
    <div data-slot="fail"></div>
  </div>`;

const state = new WeakMap();
let seq = 0;

/**
 * @param {HTMLElement} el
 * @param {{ template?: string, outputTitle?: string, fail?: 'base'|'mismatch' }} options
 */
export async function mount(el, options = {}) {
  const root = template.content.firstElementChild.cloneNode(true);
  const out = outTemplate.content.firstElementChild.cloneNode(true);
  el.replaceChildren(root);
  const $ = (sel) => root.querySelector(sel) ?? out.querySelector(sel);
  const ctrl = new AbortController();
  const outputId = `w10-chat-template:${++seq}`;
  state.set(el, { ctrl, outputId });

  const s = {
    tpl: options.template ?? 'chatml',
    mask: true,
    gen: false,
    fail: options.fail ?? null,
    messages: DEFAULT_MESSAGES.map((m) => ({ ...m })),
    data: null,
    dataError: null,
  };

  $('[data-in=tpl]').innerHTML = Object.entries(TEMPLATES)
    .map(([id, t]) => `<option value="${id}">${esc(t.label)}</option>`)
    .join('');
  $('[data-in=tpl]').value = s.tpl;

  registerOutput(outputId, { title: options.outputTitle ?? '채팅 템플릿 결과', node: out, inlineHost: $('[data-slot=out-inline]') });

  function renderMessages() {
    $('[data-slot=msgs]').innerHTML = s.messages
      .map(
        (m, i) => `<div class="w10-msg w10-r-${m.role}">
          <select data-msg-role="${i}" aria-label="${i + 1}번째 메시지 역할">${Object.entries(ROLES)
            .map(([r, ko]) => `<option value="${r}"${r === m.role ? ' selected' : ''}>${r} · ${ko}</option>`)
            .join('')}</select>
          <textarea data-msg-text="${i}" rows="2" aria-label="${i + 1}번째 메시지 내용" spellcheck="false">${esc(m.content)}</textarea>
          <button type="button" class="btn small ghost w10-del" data-msg-del="${i}" aria-label="${i + 1}번째 메시지 삭제" title="삭제">✕</button>
        </div>`,
      )
      .join('');
  }

  function render() {
    const { segments, notes } = renderTemplate(s.messages, s.tpl, { addGen: s.gen });
    const toks = tokenize(segments);
    const trained = s.gen ? 0 : toks.filter((t) => t.train).length;
    const specials = toks.filter((t) => t.special).length;
    const chars = [...joinSegments(segments)].length;
    const hasAssistant = s.messages.some((m) => m.role === 'assistant');
    const last = s.messages.at(-1);
    const showMask = s.mask && !s.gen;

    $('[data-slot=view-title]').textContent = s.gen ? '추론 때 모델에 넣는 프롬프트 (여기서부터 이어 쓴다)' : '학습 예시: 모델이 실제로 보는 문자열';
    $('[data-slot=view]').innerHTML = viewHtml(segments, showMask) + (s.gen ? '<span class="w10-cursor" aria-label="모델이 이어 쓸 자리">▍</span>' : '');
    $('[data-slot=view]').classList.toggle('masked', showMask);

    $('[data-slot=stats]').innerHTML = [
      stat('글자 수', chars.toLocaleString()),
      stat('토큰 수', toks.length.toLocaleString()),
      stat('특수 토큰', specials),
      stat('학습 토큰', s.gen ? '— (추론)' : trained.toLocaleString()),
      stat('학습 비율', s.gen ? '—' : fmtPct(toks.length ? trained / toks.length : 0)),
    ].join('');

    $('[data-slot=strip]').innerHTML =
      toks
        .slice(0, 900)
        .map((t) => `<i class="${t.special ? 'sp ' : ''}${!s.gen && t.train ? 'tr' : ''}" title="${esc(t.text === '\n' ? '↵ 줄바꿈' : t.text)}${!s.gen && t.train ? ' · 학습' : ' · 제외'}"></i>`)
        .join('') + (toks.length > 900 ? `<span class="w10-muted"> … 외 ${toks.length - 900}개</span>` : '');

    // verdict
    const v = [];
    if (s.tpl === 'none') {
      v.push(callout('danger', '누가 한 말인지 표시가 없다', `메시지를 그냥 이어 붙이면 시스템 지시문, 질문, 답이 한 덩어리 글이 된다. 모델은 어디까지가 질문이고 어디서부터 자기 차례인지, 언제 멈춰야 하는지 알 수 없다. 사전학습처럼 모든 토큰(${toks.length}개)에 손실을 계산하니 질문을 흉내 내는 법까지 배운다.`));
    } else if (s.gen) {
      if (last?.role === 'assistant') {
        v.push(callout('danger', '이미 답이 있는 대화에 생성 프롬프트를 붙였다', '추론 때는 마지막 사용자 차례 뒤에 어시스턴트 머리말만 붙이고, 그다음은 모델이 쓴다. 마지막 어시스턴트 메시지를 지워 본다.'));
      } else {
        v.push(callout('ok', '추론용 프롬프트', `어시스턴트 머리말까지 붙여 모델이 “이제 내 차례”임을 알게 한다. 모델은 답을 쓰고 <code>${esc(TEMPLATES[s.tpl].stop)}</code>를 내면 멈춘다. 추론 때는 학습이 없으므로 손실 마스크도 없다.`));
      }
    } else if (!hasAssistant) {
      v.push(callout('danger', '학습할 토큰이 0개다', '어시스턴트 답이 없는 대화는 SFT 데이터로 쓸 수 없다. 손실을 계산할 자리가 하나도 없기 때문이다. ＋ 어시스턴트로 모범 답을 넣는다.'));
    } else {
      v.push(callout('ok', `학습 예시 · 전체 ${toks.length}토큰 중 ${trained}토큰(${fmtPct(trained / toks.length)})만 학습`, `시스템 지시문과 질문은 문맥으로만 쓰고, 손실은 어시스턴트의 답과 끝 표시 <code>${esc(TEMPLATES[s.tpl].stop)}</code>에서만 계산한다. 끝 표시를 학습해야 모델이 답을 마치고 멈출 줄 안다.`));
    }
    notes.forEach((n) => v.push(callout(n.level === 'danger' ? 'danger' : 'more', n.level === 'danger' ? '템플릿 규칙 위반' : '참고', esc(n.text))));
    $('[data-slot=verdict]').innerHTML = v.join('');

    // failure panel
    root.querySelectorAll('[data-fail]').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.fail === s.fail)));
    $('[data-slot=fail]').innerHTML = s.fail ? failHtml(s) : '';
  }

  const onRoot = (type, fn) => root.addEventListener(type, fn, { signal: ctrl.signal });
  onRoot('change', (e) => {
    const t = e.target;
    if (t.matches('[data-in=tpl]')) s.tpl = t.value;
    else if (t.matches('[data-in=mask]')) s.mask = t.checked;
    else if (t.matches('[data-in=gen]')) s.gen = t.checked;
    else if (t.matches('[data-msg-role]')) {
      s.messages[Number(t.dataset.msgRole)].role = t.value;
      t.closest('.w10-msg').className = `w10-msg w10-r-${t.value}`;
    } else return;
    render();
  });
  onRoot('input', (e) => {
    const t = e.target;
    if (!t.matches('[data-msg-text]')) return;
    s.messages[Number(t.dataset.msgText)].content = t.value;
    render();
  });
  onRoot('click', (e) => {
    const del = e.target.closest('[data-msg-del]');
    const act = e.target.closest('[data-act]')?.dataset.act;
    const fail = e.target.closest('[data-fail]')?.dataset.fail;
    if (del) {
      s.messages.splice(Number(del.dataset.msgDel), 1);
      renderMessages();
    } else if (act === 'reset') {
      s.messages = DEFAULT_MESSAGES.map((m) => ({ ...m }));
      renderMessages();
    } else if (act?.startsWith('add-')) {
      const role = act.slice(4);
      s.messages.push({ role, content: role === 'user' ? '주말에도 쓸 수 있어?' : role === 'assistant' ? '이용일 이틀 전까지 행정실에 신청하면 쓸 수 있다.' : '존댓말을 쓰지 말고 평서체로 답한다.' });
      renderMessages();
      root.querySelector(`[data-msg-text="${s.messages.length - 1}"]`)?.focus();
    } else if (fail) {
      s.fail = s.fail === fail ? null : fail;
    } else return;
    render();
  });

  renderMessages();
  render();

  try {
    const res = await fetch(new URL('../../data/w10/chat-failures.json', import.meta.url));
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    s.data = await res.json();
    if (ctrl.signal.aborted) return;
    $('[data-slot=status]').hidden = true;
  } catch (err) {
    if (ctrl.signal.aborted) return;
    s.dataError = err.message;
    $('[data-slot=status]').innerHTML = `<div class="widget__error">실패 사례 예시를 불러오지 못했다 (${esc(err.message)}). 템플릿 보기는 그대로 쓸 수 있다. <code>file://</code>로 열었다면 <code>python -m http.server</code>로 연다.</div>`;
  }
  render();
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

function showText(text) {
  return esc(text).replace(/\n/g, '<span class="w10-nl">↵</span>\n');
}

function viewHtml(segments, showMask) {
  return segments
    .map((sg) => {
      const cls = [`w10-${sg.kind}`];
      if (showMask) cls.push(sg.train ? 'tr' : 'off');
      return `<span class="${cls.join(' ')}">${showText(sg.text)}</span>`;
    })
    .join('');
}

const SPECIAL_RE = /(<\|[a-z_]+\|>|<start_of_turn>|<end_of_turn>|<bos>)/g;

/** Canned model output with special tokens highlighted. */
function cannedHtml(text) {
  return text
    .split(SPECIAL_RE)
    .map((part, i) => (i % 2 ? `<span class="w10-special">${esc(part)}</span>` : showText(part)))
    .join('');
}

function failHtml(s) {
  if (!s.data) {
    return s.dataError
      ? `<div class="widget__error">실패 사례 예시를 불러오지 못했다 (${esc(s.dataError)}).</div>`
      : '<p class="w10-muted">예시를 불러오는 중…</p>';
  }
  const d = s.data;
  const tag = `<span class="w10-tag">${esc(d.label)}</span>`;
  const q = [{ role: 'system', content: d.system }, { role: 'user', content: d.question }];
  if (s.fail === 'base') {
    const b = d.base;
    return `<div class="w10-fail">
      <h4>🧪 ${esc(b.title)} ${tag}</h4>
      <p class="w10-muted">${esc(d.note)}</p>
      ${b.continuations
        .map(
          (c, i) => `<div class="w10-cont"><span class="w10-cont-l">이어 쓰기 ${i + 1} · ${esc(c.label)}</span>
          <div class="w10-view"><span class="w10-prompt">${showText(d.question)}</span><span class="w10-gen">${showText(c.text)}</span></div></div>`,
        )
        .join('')}
      <div class="callout callout--danger"><span class="callout__title">답하지 않고 이어 쓴다</span><p>${esc(b.why)}</p></div>
      <div class="w10-cont"><span class="w10-cont-l">비교 · ${esc(d.chat.title)}</span>
        <div class="w10-view">${viewHtml(renderTemplate(q, 'chatml', { addGen: true }).segments, false)}<span class="w10-gen">${showText(d.chat.answer)}</span><span class="w10-special">&lt;|im_end|&gt;</span></div></div>
    </div>`;
  }
  const m = d.mismatch;
  return `<div class="w10-fail">
    <h4>⚠ ${esc(m.title)} ${tag}</h4>
    <p class="w10-muted">${esc(d.note)}</p>
    <div class="w10-cont"><span class="w10-cont-l">추론 프롬프트 (Llama 3 형식) · 멈춤 토큰으로 <code>&lt;|eot_id|&gt;</code>를 기다린다</span>
      <div class="w10-view">${viewHtml(renderTemplate(q, m.used, { addGen: true }).segments, false)}<span class="w10-gen">${cannedHtml(m.output)}</span><span class="w10-dead">… 최대 길이까지 계속</span></div></div>
    <div class="callout callout--danger"><span class="callout__title">혼자 묻고 혼자 답한다</span><p>${esc(m.why)}</p></div>
    <p class="w10-muted">해결: 학습과 추론에 같은 템플릿을 쓴다. 실제 라이브러리는 토크나이저 파일에 템플릿을 함께 저장해 두고(<code>chat_template</code>) 그것을 그대로 쓴다.</p>
  </div>`;
}

const callout = (kind, title, html) =>
  `<div class="callout${kind === 'ok' ? ' callout--ok' : kind === 'danger' ? ' callout--danger' : ' callout--more'}"><span class="callout__title">${esc(title)}</span><p>${html}</p></div>`;

const stat = (label, value) => `<div class="stat"><span class="stat__label">${label}</span><span class="stat__value">${value}</span></div>`;

function fmtPct(x) {
  return `${(x * 100).toFixed(1)}%`;
}

function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
}
