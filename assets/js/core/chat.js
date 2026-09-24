// Multi-turn chat helpers (week 14). No DOM — works in pages, workers and Node.
//  - token estimate and context building (memory strategies + token budget)
//  - a rule-based summarizer for old turns
//  - Server-Sent Events (SSE) parsing and streaming calls for Anthropic / OpenAI
//  - a simulated stream and a scripted mock bot for classes without an API key
//
// The LLM API is stateless: every request must carry the whole conversation the
// model should "remember". Everything here is about choosing what to resend.

import { LLMError } from './llm.js';

// ---------------------------------------------------------------- tokens

/** Per-message overhead for role markers (<|user|> … <|end|>, see week 10). */
export const MSG_OVERHEAD = 4;

const HANGUL = /[ㄱ-ㆎ가-힣]/;

/**
 * Rough token estimate without a tokenizer: one Hangul syllable ≈ 1 token,
 * other visible characters ≈ 3 per token, whitespace merges into neighbours.
 */
export function estimateTokens(text) {
  let hangul = 0;
  let other = 0;
  for (const ch of String(text ?? '')) {
    if (HANGUL.test(ch)) hangul++;
    else if (!/\s/.test(ch)) other++;
  }
  return Math.ceil(hangul + other / 3);
}

export const messageTokens = (m) => estimateTokens(m.content) + MSG_OVERHEAD;

// ---------------------------------------------------------------- turns & context

/** Group a flat [{ role, content }] history into turns: { index, messages: [user, assistant?] }. */
export function toTurns(history) {
  const turns = [];
  for (const m of history) {
    if (m.role === 'user' || !turns.length) turns.push({ index: turns.length, messages: [m] });
    else turns[turns.length - 1].messages.push(m);
  }
  return turns;
}

/**
 * Rule-based summarizer (stand-in for "ask the LLM to summarize"):
 * keeps only what the user said, each utterance cut to `maxChars`.
 */
export function mockSummarize(messages, { maxChars = 40 } = {}) {
  return messages
    .filter((m) => m.role === 'user')
    .map((m) => {
      const t = m.content.replace(/\s+/g, ' ').trim();
      return `- 사용자: ${[...t].length > maxChars ? [...t].slice(0, maxChars).join('') + '…' : t}`;
    })
    .join('\n');
}

export const SUMMARY_HEADER = '[이전 대화 요약]';

/**
 * Decide what to send for the newest user message.
 * @param {{ system: string, history: {role:string, content:string}[],
 *   strategy?: 'none'|'full'|'window'|'summary', lastN?: number, budget?: number,
 *   summarize?: (msgs: object[]) => string }} opts
 *   history must end with the new user message.
 * @returns {{ messages: {role:string, content:string}[], turns: object[], summary: string,
 *   tokens: { system:number, summary:number, history:number, current:number, total:number },
 *   budget:number, over:boolean }}
 */
export function buildContext({ system, history, strategy = 'window', lastN = 3, budget = 600, summarize = mockSummarize }) {
  const current = history[history.length - 1];
  const turns = toTurns(history.slice(0, -1)).map((t) => ({
    ...t,
    tokens: t.messages.reduce((a, m) => a + messageTokens(m), 0),
    status: 'kept', // kept | window (outside the window, dropped) | summarized | budget (dropped to fit)
  }));

  if (strategy === 'none') turns.forEach((t) => (t.status = 'window'));
  if (strategy === 'window' || strategy === 'summary') {
    const cut = Math.max(0, turns.length - lastN);
    turns.slice(0, cut).forEach((t) => (t.status = strategy === 'summary' ? 'summarized' : 'window'));
  }

  let summary = '';
  if (strategy === 'summary') {
    const old = turns.filter((t) => t.status === 'summarized').flatMap((t) => t.messages);
    if (old.length) summary = summarize(old);
  }

  const sysText = () => (summary ? `${system}\n\n${SUMMARY_HEADER}\n${summary}` : system);
  const baseSys = estimateTokens(system) + MSG_OVERHEAD;
  const curTok = messageTokens(current);
  const kept = () => turns.filter((t) => t.status === 'kept');
  const total = () => estimateTokens(sysText()) + MSG_OVERHEAD + curTok + kept().reduce((a, t) => a + t.tokens, 0);

  // 'full' sends everything and lets the caller see the overflow; the others trim.
  if (strategy !== 'full') {
    // 1) drop the oldest kept turns first (system prompt and new message are pinned)
    for (const t of turns) {
      if (total() <= budget) break;
      if (t.status === 'kept') t.status = 'budget';
    }
    // 2) still over: shorten the summary from the front (oldest facts go first)
    while (summary && total() > budget) {
      const lines = summary.split('\n');
      lines.shift();
      summary = lines.join('\n');
    }
  }

  const system_ = sysText();
  const messages = [{ role: 'system', content: system_ }, ...kept().flatMap((t) => t.messages), current];
  const sysTok = estimateTokens(system_) + MSG_OVERHEAD;
  const hist = kept().reduce((a, t) => a + t.tokens, 0);
  const tokens = { system: baseSys, summary: sysTok - baseSys, history: hist, current: curTok, total: sysTok + hist + curTok };
  return { messages, turns, summary, tokens, budget, over: tokens.total > budget };
}

/** Split an OpenAI-style array into provider arguments: Anthropic/Gemini take `system` separately. */
export function splitSystem(messages) {
  const system = messages.filter((m) => m.role === 'system').map((m) => m.content).join('\n\n');
  return { system, messages: messages.filter((m) => m.role !== 'system') };
}

// ---------------------------------------------------------------- SSE

/**
 * Incremental Server-Sent Events parser. Network chunks can split a line
 * anywhere, so text is buffered until a newline; a blank line ends an event.
 * @param {(ev: { event: string, data: string }) => void} onEvent
 */
export function createSSEParser(onEvent) {
  let buf = '';
  let event = '';
  let data = [];
  const line = (l) => {
    if (l === '') {
      if (data.length) onEvent({ event: event || 'message', data: data.join('\n') });
      event = '';
      data = [];
      return;
    }
    if (l.startsWith(':')) return; // comment / keep-alive
    const c = l.indexOf(':');
    const field = c < 0 ? l : l.slice(0, c);
    let value = c < 0 ? '' : l.slice(c + 1);
    if (value.startsWith(' ')) value = value.slice(1);
    if (field === 'event') event = value;
    else if (field === 'data') data.push(value);
  };
  return {
    push(chunk) {
      buf += chunk;
      let i;
      while ((i = buf.indexOf('\n')) >= 0) {
        line(buf.slice(0, i).replace(/\r$/, ''));
        buf = buf.slice(i + 1);
      }
    },
    end() {
      if (buf) line(buf.replace(/\r$/, ''));
      buf = '';
      line('');
    },
  };
}

/** Parse a complete recorded SSE text into events. */
export function parseSSE(raw) {
  const out = [];
  const p = createSSEParser((e) => out.push(e));
  p.push(raw);
  p.end();
  return out;
}

/**
 * Interpret one SSE event for a provider.
 * Anthropic: event content_block_delta → data.delta.text; message_stop ends; error → data.error.
 * OpenAI chat completions: data {choices:[{delta:{content}}]}; data [DONE] ends.
 * @returns {{ text?: string, done?: boolean, error?: string }}
 */
export function readDelta(provider, ev) {
  if (provider === 'openai') {
    if (ev.data.trim() === '[DONE]') return { done: true };
    const j = safeJson(ev.data);
    if (!j) return {};
    if (j.error) return { error: j.error.message ?? 'stream error' };
    const text = j.choices?.[0]?.delta?.content;
    return text ? { text } : {};
  }
  if (provider === 'anthropic') {
    const j = safeJson(ev.data);
    if (!j) return {};
    if (j.type === 'error' || ev.event === 'error') return { error: j.error?.message ?? 'stream error' };
    if (j.type === 'message_stop') return { done: true };
    if (j.type === 'content_block_delta' && j.delta?.type === 'text_delta') return { text: j.delta.text };
    return {};
  }
  return {};
}

function safeJson(s) {
  try {
    return JSON.parse(s);
  } catch {
    return null;
  }
}

// ---------------------------------------------------------------- streaming calls

export const STREAM_PROVIDERS = ['anthropic', 'openai'];
const KEY_PREFIX = 'llmlab:key:'; // same storage as llm.js (sessionStorage only)

/**
 * Stream a chat completion. `messages` may include a leading system message.
 * Calls onDelta(textPiece) as pieces arrive. Abort with `signal` (■ 중지).
 * @returns {Promise<{ text: string, chunks: number, stopped: boolean }>}
 */
export async function streamChat({ provider, model, messages, temperature = 0.3, maxTokens = 512, signal, onDelta = () => {} }) {
  if (!STREAM_PROVIDERS.includes(provider)) throw new LLMError(`스트리밍을 지원하지 않는 공급자: ${provider}`, 'bad-provider');
  const key = typeof sessionStorage !== 'undefined' ? sessionStorage.getItem(KEY_PREFIX + provider) : null;
  if (!key) throw new LLMError('API 키가 입력되지 않았다. 키를 입력한 뒤 다시 시도한다.', 'no-key');
  const { system, messages: rest } = splitSystem(messages);

  const req =
    provider === 'anthropic'
      ? {
          url: 'https://api.anthropic.com/v1/messages',
          headers: {
            'content-type': 'application/json',
            'x-api-key': key,
            'anthropic-version': '2023-06-01',
            'anthropic-dangerous-direct-browser-access': 'true',
          },
          // Anthropic accepts temperature 0–1 and rejects an empty system string
          body: { model, ...(system ? { system } : {}), messages: rest, max_tokens: maxTokens, temperature: Math.min(temperature, 1), stream: true },
        }
      : {
          url: 'https://api.openai.com/v1/chat/completions',
          headers: { 'content-type': 'application/json', authorization: `Bearer ${key}` },
          body: { model, messages, max_tokens: maxTokens, temperature, stream: true },
        };

  let res;
  try {
    res = await fetch(req.url, { method: 'POST', headers: req.headers, body: JSON.stringify(req.body), signal });
  } catch (err) {
    if (err.name === 'AbortError') return { text: '', chunks: 0, stopped: true };
    throw new LLMError('네트워크 오류로 LLM에 연결하지 못했다.', 'network');
  }
  if (!res.ok) {
    let detail = '';
    try {
      const j = await res.json();
      detail = j?.error?.message ?? '';
    } catch {
      /* non-JSON */
    }
    const hint = res.status === 401 || res.status === 403 ? 'API 키가 올바른지 확인한다.' : res.status === 429 ? '요청 한도를 초과했다.' : '';
    throw new LLMError(`LLM 호출 실패 (${res.status}) ${hint} ${scrub(detail, key)}`.trim(), 'http', res.status);
  }

  let text = '';
  let chunks = 0;
  let streamError = '';
  let done = false;
  const parser = createSSEParser((ev) => {
    const d = readDelta(provider, ev);
    if (d.error) streamError = d.error;
    if (d.done) done = true;
    if (d.text) {
      text += d.text;
      chunks++;
      onDelta(d.text);
    }
  });
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  try {
    while (!done) {
      const { value, done: eof } = await reader.read();
      if (eof) break;
      parser.push(decoder.decode(value, { stream: true }));
      if (streamError) throw new LLMError(`스트림 오류: ${scrub(streamError, key)}`, 'stream');
    }
    parser.end();
  } catch (err) {
    if (err.name === 'AbortError') return { text, chunks, stopped: true };
    throw err;
  } finally {
    reader.cancel().catch(() => {});
  }
  if (streamError) throw new LLMError(`스트림 오류: ${scrub(streamError, key)}`, 'stream');
  return { text, chunks, stopped: false };
}

function scrub(text, key) {
  let out = String(text ?? '');
  if (key) out = out.split(key).join('[redacted]');
  return out.replace(/(sk-[\w-]{8,}|AIza[\w-]{20,})/g, '[redacted]').slice(0, 300);
}

/** Cut text into token-like pieces for a simulated stream (1–2 Hangul syllables, words, spaces). */
export function pseudoTokens(text) {
  return String(text).match(/ ?[가-힣]{1,2}| ?[A-Za-z]+| ?\d+|\s+|[^\s가-힣A-Za-z\d]/g) ?? [];
}

/**
 * Replay text as if it were streamed. Resolves early (stopped: true) when aborted.
 * @returns {Promise<{ text: string, chunks: number, stopped: boolean }>}
 */
export function simulateStream(text, { onDelta = () => {}, signal, delay = 30 } = {}) {
  const pieces = pseudoTokens(text);
  return new Promise((resolve) => {
    let i = 0;
    let out = '';
    let timer = null;
    const stop = () => {
      clearTimeout(timer);
      resolve({ text: out, chunks: i, stopped: true });
    };
    if (signal?.aborted) return stop();
    signal?.addEventListener('abort', stop, { once: true });
    const tick = () => {
      if (i >= pieces.length) {
        signal?.removeEventListener('abort', stop);
        resolve({ text: out, chunks: i, stopped: false });
        return;
      }
      out += pieces[i];
      onDelta(pieces[i]);
      i++;
      timer = setTimeout(tick, delay);
    };
    timer = setTimeout(tick, delay);
  });
}

// ---------------------------------------------------------------- mock bot

// Facts the scripted bot can look up: keywords → a sentence found in the corpus.
const FACTS = [
  { id: 'open', keys: ['몇 시', '열어', '열려', '개방', '이용 시간', '닫'], doc: 'facility-rules', find: '평일 오후 9시까지' },
  { id: 'weekend', keys: ['주말', '공휴일', '토요일', '일요일'], doc: 'facility-rules', find: '주말과 공휴일' },
  { id: 'food', keys: ['음식', '음료', '먹', '마셔', '마시', '커피', '라면'], doc: 'facility-rules', find: '음식물 섭취' },
  { id: 'gpu', keys: ['GPU', '연속', '오래', '몇 시간'], doc: 'facility-rules', find: '최대 12시간' },
  { id: 'broken', keys: ['고장', '망가', '안 켜'], doc: 'facility-rules', find: '장비 고장' },
  { id: 'install', keys: ['설치', '소프트웨어', '프로그램'], doc: 'facility-rules', find: '개인 소프트웨어' },
  { id: 'room', keys: ['어디', '몇 호', '위치', '층'], doc: 'dept-overview', find: '본관 3층' },
  { id: 'class', keys: ['수업 시간', '수업은', '점심'], doc: 'dept-overview', find: '오전 9시부터' },
  { id: 'complete', keys: ['수료', '출석', '졸업'], doc: 'dept-overview', find: '수료 요건' },
  { id: 'period', keys: ['기간', '언제 시작', '1년', '학기'], doc: 'dept-overview', find: '3월에 시작' },
];
// Facts that depend on which course is meant.
const COURSE_FACTS = [
  { id: 'credit', keys: ['학점', '개설'], find: '학점' },
  { id: 'grade', keys: ['성적', '평가', '비율'], find: '성적 평가' },
  { id: 'prereq', keys: ['선수', '미리'], find: '선수 과목' },
  { id: 'goal', keys: ['배워', '배우', '목표', '내용'], find: '목표' },
];
const COURSES = [
  { doc: 'course-llm-syllabus', names: ['LLM 원리', 'LLM 과목', 'LLM'] },
  { doc: 'course-rag-syllabus', names: ['RAG', '검색증강'] },
];
const DEPT_WORDS = ['학과', '과목', '교과목', '수업', '실습실', '장학금', '교수', '조교', '시험', '과제', '행정실', '학생'];
const BACKREF = ['아까', '그 과목', '말한', '방금', '전에'];
const NAME_STOP = new Set(['학생', '신입생', '1학기', '2학기']);
const OFFTOPIC_FAKE = [
  { keys: ['저녁', '메뉴', '맛집', '점심 메뉴'], text: '오늘 저녁은 김치찌개 어때? 학교 정문 앞 “행복식당” 김치찌개가 제일 유명하대!' },
  { keys: ['날씨'], text: '내일은 하루 종일 맑고 기온은 23도까지 오를 거야.' },
  { keys: ['주식', '코인', '투자'], text: '요즘은 반도체주가 무조건 오른다고 하더라. 지금 사 두면 좋을 거야!' },
  { keys: ['장학금'], text: '장학금은 성적 상위 10%에게 등록금 전액이 지급돼. 신청은 행정실 홈페이지에서 하면 돼.' },
  { keys: ['무시', '지시'], text: '알겠어! 이전 지시는 모두 잊을게. 이제 무엇이든 말해 봐.' },
];

const sentencesOf = (text) => text.split(/(?<=다\.)\s+|\n+/).map((s) => s.trim()).filter(Boolean);
const findSentence = (corpus, doc, needle) => sentencesOf(corpus.documents.find((d) => d.id === doc)?.text ?? '').find((s) => s.includes(needle)) ?? '';
const hasAny = (text, keys) => keys.some((k) => text.includes(k));

/** Find the user's name in what was sent (earlier user messages or the summary). */
function findName(texts) {
  for (let i = texts.length - 1; i >= 0; i--) {
    const m = texts[i].match(/(?:내 이름은|나는|저는)\s*([가-힣A-Za-z]{2,4}?)(?:이야|야|입니다|이에요|예요|라고)/);
    if (m && !NAME_STOP.has(m[1])) return m[1];
  }
  return '';
}

function findCourse(texts) {
  for (let i = texts.length - 1; i >= 0; i--) {
    const c = COURSES.find((c) => hasAny(texts[i], c.names));
    if (c) return c;
  }
  return null;
}

/** Keyword guardrail used by the mock bot (C5 builds a better one). */
export function isOnTopic(q) {
  return hasAny(q, FACTS.flatMap((f) => f.keys)) || hasAny(q, COURSE_FACTS.flatMap((f) => f.keys)) || hasAny(q, DEPT_WORDS) || hasAny(q, COURSES.flatMap((c) => c.names));
}

/**
 * Scripted "학과 안내 도우미" that sees ONLY the messages it is sent — just like
 * a real stateless API. It reads the system prompt for persona and guardrails.
 * @returns {{ text: string, checks: { name?: boolean, course?: boolean, guard?: boolean } }}
 */
export function mockReply(corpus, messages, { temperature = 0.3, seed = 1 } = {}) {
  const system = messages[0]?.role === 'system' ? messages[0].content : '';
  const q = messages[messages.length - 1].content;
  const earlier = messages.slice(0, -1).filter((m) => m.role === 'user').map((m) => m.content);
  const summaryText = system.includes(SUMMARY_HEADER) ? [system.slice(system.indexOf(SUMMARY_HEADER))] : [];
  const memory = [...summaryText, ...earlier];

  const guarded = /거절|답하지 않/.test(system);
  const concise = /간결|한두 문장|한 문장/.test(system);
  const pick = (arr) => (temperature < 0.3 ? arr[0] : arr[Math.floor(frac(seed * 9301 + temperature * 49297) * arr.length)]);

  const wrap = (fact) => {
    if (concise) return fact;
    const open = pick(['좋은 질문이야!', '알려 줄게.', '물론이지!']);
    const close = pick(['더 궁금한 게 있으면 물어봐.', '도움이 됐으면 좋겠어.', '또 물어봐!']);
    return `${open} ${fact} ${close}`;
  };
  const checks = {};

  // name introduction / recall
  const introduced = findName([q]);
  if (introduced) {
    return { text: concise ? `반갑다, ${introduced}. 학과에 관해 물어보면 답한다.` : `반가워, ${introduced}! 나는 AI응용소프트웨어과 안내 도우미야. 수업, 실습실, 교과목에 대해 무엇이든 물어봐.`, checks };
  }
  if (q.includes('내 이름') || q.includes('나 누군지')) {
    const known = findName(memory);
    checks.name = Boolean(known);
    return {
      text: known
        ? concise ? `${known}이다.` : `당연하지, ${known}! 처음에 인사했잖아.`
        : concise ? '이름을 들은 적이 없다.' : '미안, 아직 네 이름을 들은 적이 없는 것 같아. 이름이 뭐야?',
      checks,
    };
  }

  // guardrail
  if (!isOnTopic(q)) {
    checks.guard = guarded;
    if (guarded) {
      return { text: concise ? '학과 안내와 관련 없는 질문에는 답하지 않는다.' : `미안, 나는 학과 안내 도우미라서 그 질문에는 답할 수 없어. 수업 시간, 실습실 규정, 교과목 같은 걸 물어봐 줘.`, checks };
    }
    const fake = OFFTOPIC_FAKE.find((f) => hasAny(q, f.keys));
    return { text: fake ? fake.text : '물론이지! 그건 이렇게 하면 돼. (근거 없이 그럴듯하게 지어낸 답이다)', checks };
  }

  // course-dependent facts
  const cf = COURSE_FACTS.find((f) => hasAny(q, f.keys));
  const direct = findCourse([q]);
  if (cf && (direct || hasAny(q, BACKREF) || !FACTS.some((f) => hasAny(q, f.keys)))) {
    const course = direct ?? findCourse(memory);
    if (hasAny(q, BACKREF)) checks.course = Boolean(course);
    if (!course) {
      return { text: concise ? '어느 과목인지 알 수 없다. 과목 이름을 알려 달라.' : '어떤 과목을 말하는지 모르겠어. 과목 이름을 다시 알려 줄래? (LLM 원리와 활용 / RAG 시스템 구축)', checks };
    }
    const fact = findSentence(corpus, course.doc, cf.find);
    return { text: wrap(fact), checks };
  }

  // department facts: most keyword hits wins
  let best = null;
  let bestScore = 0;
  for (const f of FACTS) {
    const score = f.keys.filter((k) => q.includes(k)).length;
    if (score > bestScore) {
      best = f;
      bestScore = score;
    }
  }
  if (best) return { text: wrap(findSentence(corpus, best.doc, best.find)), checks };

  // on-topic but not in the documents
  if (!guarded) {
    const fake = OFFTOPIC_FAKE.find((f) => hasAny(q, f.keys));
    return { text: fake ? fake.text : '물론 가능해! 걱정하지 마. (문서에 없는 내용을 지어낸 답이다)', checks };
  }
  return { text: concise ? '학과 문서에서 찾을 수 없다. 학과 행정실에 문의한다.' : '그 내용은 학과 문서에서 찾을 수 없어. 정확한 건 학과 행정실에 문의해 줘.', checks };
}

function frac(x) {
  return x - Math.floor(x);
}
