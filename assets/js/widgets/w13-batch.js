// w13 문서 일괄 처리기
// One concept: running an LLM over many documents is a pipeline, not a chat.
// Template → N calls (concurrency, retry, cache) → parse/validate → compare with
// a gold set. Works without an API key using hand-written example responses
// (assets/data/w13/*.json) that include typical failure modes.

import { registerOutput, unregisterOutput } from '../site/result.js';
import { loadCorpus } from '../core/text.js';
import { PROVIDERS, generate, setKey, hasKey, clearKey } from '../core/llm.js';
import { mulberry32 } from '../core/rng.js';

const DATA = new URL('../../data/w13/', import.meta.url);
const TASKS = {
  summarize: { label: '요약', file: 'summarize.json', maxTokens: 300 },
  classify: { label: '분류', file: 'classify.json', maxTokens: 20 },
  extract: { label: '정보 추출 (JSON)', file: 'extract.json', maxTokens: 400 },
};
const CHARS_PER_TOKEN = 1.5; // rough estimate for Korean text
const MAX_RETRIES = 3;
const SHORT = { 학과안내: '학과', 규정: '규정', 교과목: '교과', '이론-LLM': 'LLM', '이론-RAG': 'RAG' };

export const DEFAULT_TEMPLATES = {
  summarize: `다음 문서를 {{limit}}자 이내의 한국어 한두 문장으로 요약하라.
요약문만 출력하고 다른 말은 붙이지 않는다.
{{examples}}
문서:
"""
{{document}}
"""
요약:`,
  classify: `다음 문서의 종류를 아래 라벨 중 하나로만 답하라.
라벨: {{labels}}
라벨 이외의 말은 출력하지 않는다.
{{examples}}
문서:
"""
{{document}}
"""
라벨:`,
  extract: `다음 문서에서 교과목 정보를 추출해 JSON 객체 하나로만 답하라.
키: 과목명(문자열), 학점(정수), 주당_수업시간(정수), 주차수(정수), 성적비율(항목명→정수 퍼센트 객체)
문서에 없는 값은 null로 둔다. 추측하지 않는다.
{{examples}}
문서:
"""
{{document}}
"""
JSON:`,
};

export const EXAMPLES = {
  summarize: `예시)
문서: 도서관은 평일 오전 9시부터 오후 10시까지 운영한다. 주말에는 오후 5시에 닫는다. 대출은 1인 5권까지이며 기간은 2주다.
요약: 도서관은 평일 오후 10시, 주말 오후 5시까지 열고 1인 5권을 2주간 빌려준다.
`,
  classify: `예시)
문서: 도서관은 평일 오후 10시까지 운영하며 대출은 1인 5권까지다. → 규정
문서: 이 과목은 1학기 2학점 과목이며 성적은 중간고사 40%, 기말고사 60%다. → 교과목
문서: 검색 결과를 평가할 때는 정답 문서가 몇 번째에 나오는지 본다. → 이론-RAG
문서: 셀프 어텐션은 토큰마다 쿼리와 키를 내적해 가중치를 만든다. → 이론-LLM
문서: 우리 학과는 2년 과정이며 졸업하려면 60학점을 이수해야 한다. → 학과안내
`,
  extract: `예시)
문서: 데이터베이스 설계는 1학기 2학점 과목으로 주 1회 2시간씩 15주 진행한다. 성적은 중간고사 40%, 기말고사 60%다.
JSON: {"과목명": "데이터베이스 설계", "학점": 2, "주당_수업시간": 2, "주차수": 15, "성적비율": {"중간고사": 40, "기말고사": 60}}
문서: 실습실은 평일 오후 9시까지 개방된다.
JSON: {"과목명": null, "학점": null, "주당_수업시간": null, "주차수": null, "성적비율": null}
`,
};

// ------------------------------------------------------------ pure helpers (exported for tests)

/** 32-bit FNV-1a hash as 8 hex chars — the cache key. */
export function fnv1a(str) {
  let h = 0x811c9dc5;
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  return h.toString(16).padStart(8, '0');
}

export const estTokens = (s) => Math.ceil([...String(s ?? '')].length / CHARS_PER_TOKEN);

/** Replace {{name}} placeholders. Unknown names stay in the text and are reported. */
export function fillTemplate(tpl, vars) {
  const missing = new Set();
  const text = tpl
    .replace(/\{\{\s*(\w+)\s*\}\}/g, (m, k) => (k in vars ? vars[k] : (missing.add(k), m)))
    .replace(/\n{3,}/g, '\n\n');
  return { text, missing: [...missing] };
}

/** Label parsing: exact label after trimming quotes/punctuation; lenient = the only label mentioned. */
export function parseLabel(raw, labels, lenient) {
  const t = String(raw ?? '').trim().replace(/^["'“”‘’[(]+|["'“”‘’\]).。!]+$/g, '').trim();
  if (labels.includes(t)) return { label: t, recovered: false };
  if (lenient) {
    const hits = labels.filter((l) => t.includes(l));
    if (hits.length === 1) return { label: hits[0], recovered: true };
  }
  return { label: null, text: t };
}

/** Strict JSON.parse, or (lenient) strip code fences / surrounding prose and take the first balanced {...}. */
export function parseJSON(raw, lenient) {
  const s = String(raw ?? '');
  try {
    return { ok: true, value: JSON.parse(s), recovered: false };
  } catch (err) {
    if (!lenient) return { ok: false, error: err.message };
  }
  const body = s.replace(/```(?:json)?/gi, '');
  const start = body.indexOf('{');
  if (start < 0) return { ok: false, error: 'JSON 객체가 없다' };
  let depth = 0;
  let inStr = false;
  for (let i = start; i < body.length; i++) {
    const c = body[i];
    if (inStr) {
      if (c === '\\') i++;
      else if (c === '"') inStr = false;
    } else if (c === '"') inStr = true;
    else if (c === '{') depth++;
    else if (c === '}' && --depth === 0) {
      try {
        return { ok: true, value: JSON.parse(body.slice(start, i + 1)), recovered: true };
      } catch (err) {
        return { ok: false, error: err.message };
      }
    }
  }
  return { ok: false, error: '괄호가 닫히지 않았다' };
}

/** "3학점" → 3, "주 1회 3시간" → 3, "40%" → 40, null → null, anything else → NaN. */
export function toInt(v) {
  if (v === null || v === undefined) return null;
  if (typeof v === 'number') return v;
  if (typeof v !== 'string') return NaN;
  const m = v.match(/(\d+)\s*시간/) ?? v.match(/-?\d+(?:\.\d+)?/);
  return m ? Number(m[1] ?? m[0]) : NaN;
}

/** Validate against the extraction schema and normalize values. */
export function normalizeRecord(obj, fields) {
  const issues = [];
  let changed = 0;
  if (!obj || typeof obj !== 'object' || Array.isArray(obj)) return { record: null, issues: ['JSON 객체가 아니다'], schemaOk: false, changed };
  const extra = Object.keys(obj).filter((k) => !fields.includes(k));
  const missing = fields.filter((k) => !(k in obj));
  if (extra.length) issues.push(`스키마에 없는 필드: ${extra.join(', ')}`);
  if (missing.length) issues.push(`빠진 필드: ${missing.join(', ')}`);
  const record = {};
  let typeOk = true;
  for (const f of fields) {
    const v = obj[f] ?? null;
    if (f === '과목명') {
      if (v !== null && typeof v !== 'string') (typeOk = false), issues.push('과목명이 문자열이 아니다');
      record[f] = v === null ? null : String(v).trim();
    } else if (f === '성적비율') {
      if (v === null) record[f] = null;
      else if (typeof v !== 'object' || Array.isArray(v)) (typeOk = false), (record[f] = null), issues.push('성적비율이 객체가 아니다');
      else {
        const o = {};
        for (const [k, x] of Object.entries(v)) {
          const n = toInt(x);
          if (n !== x) changed++;
          if (Number.isNaN(n)) typeOk = false;
          o[k.trim()] = n;
        }
        const sum = Object.values(o).reduce((a, b) => a + b, 0);
        if (sum !== 100) issues.push(`성적비율 합이 ${sum}%다`);
        record[f] = o;
      }
    } else {
      const n = toInt(v);
      if (n !== v) changed++;
      if (Number.isNaN(n) || (n !== null && !Number.isInteger(n))) (typeOk = false), issues.push(`${f}를 정수로 바꿀 수 없다`);
      record[f] = n;
    }
  }
  return { record, issues, schemaOk: !extra.length && !missing.length && typeOk, changed };
}

const same = (a, b) => JSON.stringify(sortKeys(a)) === JSON.stringify(sortKeys(b));
function sortKeys(v) {
  if (!v || typeof v !== 'object') return v;
  return Object.fromEntries(Object.keys(v).sort().map((k) => [k, v[k]]));
}

/**
 * Judge one output. status: 'ok' ✓ · 'warn' ⚠ (valid but wrong/mismatch) · 'bad' ✗ (invalid format)
 */
export function judge(task, raw, docId, gold, { limit = 100, lenient = false } = {}) {
  const issues = [];
  if (task === 'summarize') {
    const text = String(raw ?? '').trim();
    const len = [...text].length;
    if (!len) return { status: 'bad', issues: ['빈 응답'], short: '(빈 응답)', len: 0, kwHit: 0, kwTotal: 0 };
    const kws = gold.docs[docId]?.keywords ?? [];
    const miss = kws.filter((k) => !text.includes(k));
    if (len > limit) issues.push(`길이 초과 ${len}/${limit}자`);
    if (/^(다음은|요약\s*[:：]|네[,.\s]|물론)/.test(text)) issues.push('서두 문장이 붙었다 (“요약문만” 지시 위반)');
    if (miss.length) issues.push(`핵심어 누락: ${miss.join(', ')}`);
    return { status: issues.length ? 'warn' : 'ok', issues, short: `${len}자 · 핵심어 ${kws.length - miss.length}/${kws.length}`, len, kwHit: kws.length - miss.length, kwTotal: kws.length, overLimit: len > limit };
  }
  if (task === 'classify') {
    const want = gold.docs[docId]?.category;
    const p = parseLabel(raw, gold.labels, lenient);
    if (!p.label) return { status: 'bad', issues: [`라벨 집합 밖의 답: “${p.text}”`], short: `✗ ${p.text.slice(0, 12)}`, pred: null, gold: want };
    if (p.recovered) issues.push('군더더기를 걷어 내고 라벨을 찾았다 (관대한 파싱)');
    if (p.label !== want) issues.push(`정답은 ${want}`);
    return { status: p.label === want ? 'ok' : 'warn', issues, short: p.label === want ? p.label : `${p.label} (정답 ${want})`, pred: p.label, gold: want, recovered: p.recovered };
  }
  // extract
  const ex = gold.extraction;
  const want = ex.items[docId] ?? ex.default;
  const parsed = parseJSON(raw, lenient);
  if (!parsed.ok) {
    return { status: 'bad', issues: [`JSON 파싱 실패: ${parsed.error}`], short: '✗ JSON 아님', fields: ex.fields.map((f) => ({ f, ok: false, kind: 'parse' })) };
  }
  if (parsed.recovered) issues.push('코드 펜스·설명문을 걷어 내고 JSON을 찾았다 (관대한 파싱)');
  const n = normalizeRecord(parsed.value, ex.fields);
  issues.push(...n.issues);
  if (n.changed) issues.push(`값 ${n.changed}개를 정규화했다 (“3학점” → 3 등)`);
  const fields = ex.fields.map((f) => {
    const got = n.record ? n.record[f] : undefined;
    const ok = n.record ? same(got, want[f]) : false;
    const kind = ok ? 'ok' : want[f] === null && got !== null ? 'halluc' : 'mismatch';
    if (!ok) issues.push(kind === 'halluc' ? `환각: 문서에 없는 ${f} = ${JSON.stringify(got)}` : `${f} 불일치: ${JSON.stringify(got)} ≠ ${JSON.stringify(want[f])}`);
    return { f, ok, kind, got, want: want[f] };
  });
  const hit = fields.filter((x) => x.ok).length;
  const status = !n.schemaOk ? 'bad' : hit === fields.length ? 'ok' : 'warn';
  return { status, issues, short: `${!n.schemaOk ? '✗ 스키마 · ' : ''}필드 ${hit}/${fields.length}`, fields };
}

/** Run `worker` over items with at most `limit` in flight. */
export async function runPool(items, limit, worker, signal) {
  let next = 0;
  let inFlight = 0;
  let maxInFlight = 0;
  const lane = async () => {
    while (next < items.length && !signal?.aborted) {
      const i = next++;
      inFlight++;
      maxInFlight = Math.max(maxInFlight, inFlight);
      try {
        await worker(items[i], i);
      } finally {
        inFlight--;
      }
    }
  };
  await Promise.all(Array.from({ length: Math.max(1, Math.min(limit, items.length)) }, lane));
  return { maxInFlight };
}

function sleep(ms, signal) {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) return reject(new DOMException('aborted', 'AbortError'));
    const t = setTimeout(resolve, ms);
    signal?.addEventListener('abort', () => (clearTimeout(t), reject(new DOMException('aborted', 'AbortError'))), { once: true });
  });
}

// ------------------------------------------------------------ templates

const template = document.createElement('template');
template.innerHTML = `
  <div class="widget w13-bp">
    <h3 class="widget__title">문서 일괄 처리기</h3>
    <div data-slot="status" class="widget__status" role="status" aria-live="polite">
      <span class="spinner" aria-hidden="true"></span> <span>문서셋과 예시 응답 불러오는 중…</span>
    </div>
    <div class="widget__controls">
      <label class="field">
        <span class="field__label">작업</span>
        <select data-in="task">
          <option value="summarize">요약</option>
          <option value="classify">분류 (라벨 5개)</option>
          <option value="extract">정보 추출 (JSON)</option>
        </select>
      </label>
      <label class="field">
        <span class="field__label">실행 방식</span>
        <select data-in="mode">
          <option value="mock">예시 응답 (키 없이 · 수업용 작성 예시)</option>
          <option value="real">실제 LLM (내 API 키)</option>
        </select>
      </label>
      <label class="field">
        <span class="field__label">동시 실행 수 <output data-out="conc"></output></span>
        <input type="range" data-in="conc" min="1" max="4" step="1">
      </label>
      <label class="field">
        <span class="field__label">온도(temperature) <output data-out="temp"></output></span>
        <input type="range" data-in="temp" min="0" max="1" step="0.1">
      </label>
      <label class="field" data-show="summarize">
        <span class="field__label">요약 길이 제한 {{limit}} <output data-out="limit"></output></span>
        <input type="range" data-in="limit" min="40" max="160" step="10">
      </label>
    </div>
    <div class="w13-checks">
      <label><input type="checkbox" data-in="few"> 퓨샷 예시 넣기 ({{examples}})</label>
      <label><input type="checkbox" data-in="lenient"> 관대한 파싱 (코드 펜스·군더더기 걷어 내기)</label>
      <label><input type="checkbox" data-in="cache"> 캐시 사용 (같은 프롬프트는 다시 호출하지 않음)</label>
    </div>

    <div class="w13-real" data-slot="real" hidden>
      <div class="widget__controls">
        <label class="field">
          <span class="field__label">공급자</span>
          <select data-in="provider"></select>
        </label>
        <label class="field">
          <span class="field__label">모델</span>
          <select data-in="model"></select>
        </label>
        <label class="field">
          <span class="field__label">API 키 (이 탭에서만 보관)</span>
          <input type="password" data-in="key" autocomplete="off" spellcheck="false" placeholder="키를 붙여 넣고 저장">
        </label>
      </div>
      <div class="btn-row">
        <button type="button" class="btn small" data-act="save-key">키 저장</button>
        <button type="button" class="btn small ghost" data-act="clear-key">키 지우기</button>
        <span class="w13-keystate" data-slot="keystate" role="status" aria-live="polite"></span>
      </div>
    </div>

    <label class="field w13-tpl">
      <span class="field__label">프롬프트 템플릿 <small class="w13-muted">{{document}} 자리에 문서가 들어간다</small></span>
      <textarea data-in="tpl" spellcheck="false" rows="9"></textarea>
    </label>
    <p class="w13-tplmsg" data-slot="tplmsg" aria-live="polite"></p>

    <details class="w13-docs" open>
      <summary>문서 선택 <span data-out="ndocs"></span></summary>
      <div class="btn-row">
        <button type="button" class="btn small ghost" data-act="all">전체</button>
        <button type="button" class="btn small ghost" data-act="none">해제</button>
        <button type="button" class="btn small ghost" data-act="guide">안내 문서 4개</button>
      </div>
      <div class="w13-doclist" data-slot="docs"></div>
    </details>

    <details class="w13-price">
      <summary>비용 설정 (예시 단가 · 직접 수정)</summary>
      <div class="widget__controls">
        <label class="field">
          <span class="field__label">입력 $ / 100만 토큰</span>
          <input type="text" inputmode="decimal" data-in="pin">
        </label>
        <label class="field">
          <span class="field__label">출력 $ / 100만 토큰</span>
          <input type="text" inputmode="decimal" data-in="pout">
        </label>
      </div>
      <p class="w13-muted">토큰 수는 “한국어 1.5자 ≈ 1토큰”으로 어림한다. 실제 LLM 모드에서 공급자가 사용량을 돌려주면 그 값을 쓴다. 단가는 공급자 요금표를 보고 고친다.</p>
    </details>

    <div class="btn-row w13-run">
      <button type="button" class="btn primary" data-act="run">▶ 일괄 실행</button>
      <button type="button" class="btn small ghost" data-act="stop">■ 중지</button>
      <button type="button" class="btn small ghost" data-act="clear-cache">캐시 비우기</button>
      <button type="button" class="btn small ghost" data-act="reset-tpl">템플릿 기본값</button>
    </div>
    <p class="w13-progress" data-slot="progress" role="status" aria-live="polite"></p>
    <div data-slot="out-inline"></div>
  </div>`;

const outTemplate = document.createElement('template');
outTemplate.innerHTML = `
  <div class="w13-bp-out">
    <div data-slot="verdict" aria-live="polite"></div>
    <div class="stat-row" data-slot="stats"></div>
    <h4>문서별 결과 <small class="w13-muted">— 행을 누르면 프롬프트와 원문 응답이 아래에 나온다</small></h4>
    <div class="w13-table-wrap"><table class="w13-table" data-slot="rows"></table></div>
    <div data-slot="extra"></div>
    <div class="w13-detail" data-slot="detail"></div>
  </div>`;

const state = new WeakMap();
let seq = 0;

/**
 * @param {HTMLElement} el
 * @param {{ task?: 'summarize'|'classify'|'extract', outputTitle?: string }} options
 */
export async function mount(el, options = {}) {
  const root = template.content.firstElementChild.cloneNode(true);
  const out = outTemplate.content.firstElementChild.cloneNode(true);
  el.replaceChildren(root);
  const $ = (sel) => root.querySelector(sel) ?? out.querySelector(sel);
  const ctrl = new AbortController();
  const outputId = `w13-batch:${++seq}`;
  state.set(el, { ctrl, outputId, get run() { return s.runCtrl; } });

  const s = {
    task: options.task ?? 'classify',
    mode: 'mock',
    few: false,
    lenient: false,
    cache: true,
    conc: 2,
    temp: 0,
    limit: 100,
    pin: '1.00',
    pout: '5.00',
    provider: 'anthropic',
    model: PROVIDERS.anthropic.defaultModel,
    tpl: { ...DEFAULT_TEMPLATES },
    sel: {},
    corpus: null,
    gold: null,
    canned: {},
    cacheMap: new Map(),
    runNo: 0,
    runCtrl: null,
    run: null, // { task, mode, rows, t0, elapsed, calls, hits, retries, maxInFlight, done, error }
    pick: -1,
  };

  // --- initial control values
  $('[data-in=task]').value = s.task;
  $('[data-in=conc]').value = String(s.conc);
  $('[data-in=temp]').value = String(s.temp);
  $('[data-in=limit]').value = String(s.limit);
  $('[data-in=cache]').checked = s.cache;
  $('[data-in=pin]').value = s.pin;
  $('[data-in=pout]').value = s.pout;
  $('[data-in=provider]').innerHTML = Object.entries(PROVIDERS).map(([id, p]) => `<option value="${id}">${esc(p.label)}</option>`).join('');
  $('[data-in=provider]').value = s.provider;

  registerOutput(outputId, { title: options.outputTitle ?? '문서 일괄 처리 결과', node: out, inlineHost: $('[data-slot=out-inline]') });

  const docs = () => s.corpus.documents;
  const guideIds = () => docs().filter((d) => d.type === 'guide').map((d) => d.id);
  const defaultSel = (task) => (task === 'extract' ? guideIds() : docs().map((d) => d.id));
  const selected = () => docs().filter((d) => s.sel[s.task].has(d.id));

  function syncControls() {
    $('[data-out=conc]').textContent = String(s.conc);
    $('[data-out=temp]').textContent = s.temp.toFixed(1);
    $('[data-out=limit]').textContent = `${s.limit}자`;
    root.querySelectorAll('[data-show]').forEach((n) => (n.hidden = n.dataset.show !== s.task));
    $('[data-slot=real]').hidden = s.mode !== 'real';
    $('[data-in=tpl]').value = s.tpl[s.task];
    $('[data-in=few]').checked = s.few;
    $('[data-in=lenient]').checked = s.lenient;
    const models = PROVIDERS[s.provider].models;
    $('[data-in=model]').innerHTML = models.map((m) => `<option>${esc(m)}</option>`).join('');
    $('[data-in=model]').value = models.includes(s.model) ? s.model : PROVIDERS[s.provider].defaultModel;
    s.model = $('[data-in=model]').value;
    syncKeyState();
    renderDocList();
  }

  function syncKeyState() {
    let has = false;
    try {
      has = hasKey(s.provider);
    } catch {
      has = false;
    }
    $('[data-slot=keystate]').textContent = has ? `🔑 ${PROVIDERS[s.provider].label} 키 저장됨 (이 탭에서만)` : '키 없음 — 실행하면 예시 응답 모드로 돌아간다';
  }

  function renderDocList() {
    if (!s.corpus) return;
    const sel = s.sel[s.task];
    $('[data-slot=docs]').innerHTML = docs()
      .map((d) => `<label><input type="checkbox" data-doc="${esc(d.id)}"${sel.has(d.id) ? ' checked' : ''}> ${esc(d.title)}</label>`)
      .join('');
    $('[data-out=ndocs]').textContent = `(${sel.size}/${docs().length})`;
  }

  // --- prompts
  function vars(doc) {
    return {
      document: doc.text,
      limit: String(s.limit),
      labels: s.gold.labels.join(', '),
      examples: s.few ? EXAMPLES[s.task] : '',
    };
  }
  function buildPrompt(doc) {
    let tpl = s.tpl[s.task];
    if (s.few && !/\{\{\s*examples\s*\}\}/.test(tpl)) tpl = `${EXAMPLES[s.task]}\n${tpl}`;
    return fillTemplate(tpl, vars(doc));
  }
  function templateCheck() {
    const tpl = s.tpl[s.task];
    if (!/\{\{\s*document\s*\}\}/.test(tpl)) return { error: '{{document}} 자리표시자가 없다. 문서가 프롬프트에 들어가지 않으므로 모든 문서에 같은 프롬프트가 나간다.' };
    const probe = fillTemplate(tpl, vars({ text: '' }));
    return { missing: probe.missing };
  }

  // --- LLM call (mock or real) with retry + backoff
  async function callOnce(R, job, attempt, signal) {
    if (R.mode === 'mock') {
      const c = s.canned[R.task];
      await sleep(70 + (parseInt(fnv1a(job.doc.id), 16) % 70), signal);
      if (attempt <= (c.transient?.[job.doc.id] ?? 0)) {
        const e = new Error('429 요청 한도 초과 (모의)');
        e.status = 429;
        throw e;
      }
      const variant = R.few ? 'few' : 'zero';
      let text = c[variant]?.[job.doc.id] ?? c.zero?.[job.doc.id] ?? c.default ?? '';
      const alt = c.alt?.[variant]?.[job.doc.id];
      if (alt && R.temp > 0) {
        const r = mulberry32(R.no * 1009 + job.i * 7 + 1)();
        if (r < Math.min(0.9, R.temp * 0.6)) text = alt;
      }
      return { text, usage: null };
    }
    const res = await generate({
      provider: R.provider,
      model: R.model,
      messages: [{ role: 'user', content: job.prompt }],
      maxTokens: TASKS[R.task].maxTokens,
      temperature: R.temp,
      signal,
    });
    return res;
  }

  async function callWithRetry(R, job, signal) {
    for (let attempt = 1; ; attempt++) {
      job.row.attempts = attempt;
      try {
        return await callOnce(R, job, attempt, signal);
      } catch (err) {
        if (err.name === 'AbortError') throw err;
        const retriable = err.status === 429 || err.status >= 500 || err.code === 'network';
        job.row.lastError = err.message;
        if (!retriable || attempt > MAX_RETRIES) throw err;
        R.retries++;
        const wait = (R.mode === 'mock' ? 150 : 1000) * 2 ** (attempt - 1);
        job.row.state = `재시도 대기 ${wait}ms`;
        schedule();
        await sleep(wait, signal);
        job.row.state = 'running';
      }
    }
  }

  async function run() {
    if (!s.corpus) return;
    s.runCtrl?.abort();
    const tc = templateCheck();
    const list = selected();
    if (tc.error || !list.length) {
      s.run = { task: s.task, mode: s.mode, rows: [], error: tc.error ?? '선택한 문서가 없다. 문서를 하나 이상 고른다.', done: true };
      renderOut();
      return;
    }
    let mode = s.mode;
    if (mode === 'real') {
      let ok = false;
      try {
        ok = hasKey(s.provider);
      } catch {
        ok = false;
      }
      if (!ok) {
        mode = 'mock';
        $('[data-slot=progress]').textContent = '키가 없어 예시 응답으로 실행했다.';
      }
    }
    const runCtrl = new AbortController();
    s.runCtrl = runCtrl;
    const signal = runCtrl.signal;
    ctrl.signal.addEventListener('abort', () => runCtrl.abort(), { once: true, signal });
    s.runNo++;
    const rows = list.map((doc) => ({ doc, state: 'pending', attempts: 0, cached: false }));
    const jobs = rows.map((row, i) => ({ row, doc: row.doc, i, prompt: buildPrompt(row.doc).text }));
    rows.forEach((r, i) => (r.prompt = jobs[i].prompt));
    const R = { no: s.runNo, task: s.task, mode, provider: s.provider, model: s.model, rows, t0: performance.now(), elapsed: 0, calls: 0, hits: 0, retries: 0, maxInFlight: 0, done: false, missing: tc.missing, temp: s.temp, cacheOn: s.cache, few: s.few };
    s.run = R;
    s.pick = -1;
    renderOut();

    const worker = async (job) => {
      const key = fnv1a(`${mode}|${R.provider}|${R.model}|T${R.temp}|${job.prompt}`);
      job.row.state = 'running';
      schedule();
      if (s.cache && s.cacheMap.has(key)) {
        const hit = s.cacheMap.get(key);
        Object.assign(job.row, { text: hit.text, usage: hit.usage, cached: true, state: 'done', attempts: 0 });
        R.hits++;
        schedule();
        return;
      }
      try {
        R.calls++;
        const res = await callWithRetry(R, job, signal);
        Object.assign(job.row, { text: res.text, usage: res.usage, state: 'done' });
        if (s.cache) s.cacheMap.set(key, { text: res.text, usage: res.usage });
      } catch (err) {
        if (err.name === 'AbortError') {
          job.row.state = 'aborted';
        } else {
          job.row.state = 'failed';
          job.row.lastError = err.message;
        }
      }
      schedule();
    };
    try {
      const { maxInFlight } = await runPool(jobs, s.conc, worker, signal);
      R.maxInFlight = maxInFlight;
    } finally {
      rows.forEach((r) => r.state === 'pending' && (r.state = 'aborted'));
      R.elapsed = performance.now() - R.t0;
      R.done = true;
      if (s.run === R) renderOut();
    }
  }

  // --- rendering
  let raf = 0;
  function schedule() {
    if (raf) return;
    raf = requestAnimationFrame(() => {
      raf = 0;
      renderOut();
    });
  }

  function judged(row) {
    if (row.state !== 'done') return null;
    return judge(s.run.task, row.text, row.doc.id, s.gold, { limit: s.limit, lenient: s.lenient });
  }

  function price() {
    const pin = Number.parseFloat(s.pin);
    const pout = Number.parseFloat(s.pout);
    return { pin: Number.isFinite(pin) ? pin : 0, pout: Number.isFinite(pout) ? pout : 0 };
  }

  function tokensOf(row) {
    const u = row.usage ?? {};
    const inT = u.input_tokens ?? u.prompt_tokens ?? u.promptTokenCount ?? estTokens(row.prompt);
    const outT = u.output_tokens ?? u.completion_tokens ?? u.candidatesTokenCount ?? estTokens(row.text);
    return { inT, outT, measured: Boolean(row.usage) };
  }

  function renderOut() {
    const r = s.run;
    if (!r) return;
    const task = r.task;
    if (r.error) {
      $('[data-slot=verdict]').innerHTML = `<div class="callout callout--danger"><span class="callout__title">⛔ 실행하지 않았다</span><p>${esc(r.error)}</p></div>`;
      $('[data-slot=stats]').innerHTML = '';
      $('[data-slot=rows]').innerHTML = '';
      $('[data-slot=extra]').innerHTML = '';
      $('[data-slot=detail]').innerHTML = '';
      return;
    }
    const J = r.rows.map(judged);
    const n = r.rows.length;
    const done = r.rows.filter((x) => x.state === 'done').length;
    const failed = r.rows.filter((x) => x.state === 'failed').length;
    const cnt = { ok: 0, warn: 0, bad: 0 };
    J.forEach((j) => j && cnt[j.status]++);

    // tokens and cost (cache hits cost nothing)
    const { pin, pout } = price();
    let inT = 0;
    let outT = 0;
    let measured = false;
    r.rows.forEach((row) => {
      if (row.state !== 'done' || row.cached) return;
      const t = tokensOf(row);
      inT += t.inT;
      outT += t.outT;
      measured ||= t.measured;
    });
    const cost = (inT * pin + outT * pout) / 1e6;
    const paid = r.rows.filter((x) => x.state === 'done' && !x.cached).length;
    const per1k = paid ? (cost / paid) * 1000 : 0;

    // task-specific quality
    let quality;
    if (task === 'classify') {
      const right = J.filter((j) => j && j.pred && j.pred === j.gold).length;
      quality = stat('정확도', `${right}/${n} · ${pct(right / n)}`);
    } else if (task === 'extract') {
      const fh = J.reduce((a, j) => a + (j ? j.fields.filter((f) => f.ok).length : 0), 0);
      const ft = n * s.gold.extraction.fields.length;
      const full = J.filter((j) => j && j.status === 'ok').length;
      quality = stat('필드 일치', `${fh}/${ft} · ${pct(fh / ft)}`) + stat('문서 완전 일치', `${full}/${n}`);
    } else {
      const within = J.filter((j) => j && j.len > 0 && !j.overLimit).length;
      const kh = J.reduce((a, j) => a + (j ? j.kwHit : 0), 0);
      const kt = J.reduce((a, j) => a + (j ? j.kwTotal : 0), 0);
      quality = stat('길이 준수', `${within}/${n}`) + stat('핵심어 포함', kt ? pct(kh / kt) : '—');
    }
    const elapsed = r.done ? r.elapsed : performance.now() - r.t0;
    $('[data-slot=stats]').innerHTML = [
      quality,
      stat('형식 유효', `${cnt.ok + cnt.warn}/${n}`),
      stat('LLM 호출', `${r.calls}회`),
      stat('캐시 적중', `${r.hits}회`),
      stat('재시도', `${r.retries}회`),
      stat(`토큰${measured ? '' : ' (추정)'}`, `${fmtInt(inT)} + ${fmtInt(outT)}`),
      stat('비용 (예시 단가)', `$${cost.toFixed(5)}`),
      stat('1,000문서 환산', `$${per1k.toFixed(3)}`),
      stat('소요 시간', `${(elapsed / 1000).toFixed(2)}초`),
    ].join('');

    // verdict
    const v = [];
    if (r.mode === 'mock') v.push(`<p class="w13-note">📝 <b>예시 응답</b> — 기본 템플릿에 대해 미리 작성한 수업용 응답이다. 템플릿을 고쳐도 응답 내용은 같다. 실제 변화는 “실제 LLM” 모드에서 확인한다.</p>`);
    if (r.missing?.length) v.push(callout('danger', '채워지지 않은 자리표시자', `${r.missing.map((m) => `{{${m}}}`).join(', ')}가 그대로 LLM에 전달된다.`));
    if (!r.done) v.push(`<p class="w13-note">⏳ 실행 중 · ${done + failed}/${n} 완료</p>`);
    else {
      const bits = [];
      if (failed) bits.push(`${failed}건은 재시도 ${MAX_RETRIES}번 뒤에도 실패했다`);
      if (task === 'classify') {
        const outside = J.filter((j) => j && !j.pred);
        const wrong = J.filter((j) => j && j.pred && j.pred !== j.gold);
        const rec = J.filter((j) => j && j.recovered);
        if (outside.length) bits.push(`라벨 집합 밖의 답 ${outside.length}건`);
        if (wrong.length) bits.push(`틀린 라벨 ${wrong.length}건`);
        if (rec.length) bits.push(`군더더기를 걷어 낸 답 ${rec.length}건`);
      } else if (task === 'extract') {
        const bad = J.filter((j) => j && j.status === 'bad').length;
        const hall = J.reduce((a, j) => a + (j ? j.fields.filter((f) => f.kind === 'halluc').length : 0), 0);
        if (bad) bits.push(`파싱·스키마 실패 ${bad}건`);
        if (hall) bits.push(`문서에 없는 값(환각) ${hall}개`);
      } else {
        const over = J.filter((j) => j && j.overLimit).length;
        const pre = J.filter((j) => j && j.issues.some((x) => x.startsWith('서두'))).length;
        const kw = J.filter((j) => j && j.issues.some((x) => x.startsWith('핵심어'))).length;
        if (over) bits.push(`길이 초과 ${over}건`);
        if (pre) bits.push(`서두 문장 ${pre}건`);
        if (kw) bits.push(`핵심어 누락 ${kw}건`);
      }
      if (bits.length) v.push(callout('danger', `실패 ${cnt.bad + cnt.warn + failed}건 / ${n}건`, `${bits.join(' · ')}. ${hintFor(task, r)}`));
      else v.push(callout('ok', `전부 통과 ${cnt.ok}/${n}`, '골드셋과 일치한다. 문서를 늘리거나 제한을 조이면 어디서 깨지는지 확인한다.'));
      if (r.temp > 0 && r.cacheOn && r.hits) v.push(callout('more', '온도 > 0 인데 캐시가 켜져 있다', '캐시는 첫 번째 무작위 결과를 고정한다. 재현성이 필요하면 온도 0을, 다양성이 필요하면 캐시를 끈다.'));
    }
    $('[data-slot=verdict]').innerHTML = v.join('');

    // rows
    const icon = { ok: '✓', warn: '⚠', bad: '✗' };
    const label = { ok: '통과', warn: task === 'summarize' ? '규칙 위반' : '골드와 다름', bad: task === 'classify' ? '라벨 무효' : task === 'extract' ? '형식 무효' : '무효' };
    if (s.pick < 0 && r.done) {
      const firstBad = J.findIndex((j, i) => (j && j.status !== 'ok') || r.rows[i].state === 'failed');
      s.pick = firstBad >= 0 ? firstBad : 0;
    }
    $('[data-slot=rows]').innerHTML =
      `<thead><tr><th>문서</th><th>상태</th><th>결과</th><th title="시도 횟수 · 캐시">시도</th></tr></thead><tbody>` +
      r.rows
        .map((row, i) => {
          const j = J[i];
          let st;
          if (row.state === 'done') st = `<span class="w13-st ${j.status}">${icon[j.status]} ${label[j.status]}</span>`;
          else if (row.state === 'failed') st = '<span class="w13-st bad">⛔ 호출 실패</span>';
          else if (row.state === 'aborted') st = '<span class="w13-st">■ 중지</span>';
          else if (row.state === 'pending') st = '<span class="w13-st">대기</span>';
          else st = `<span class="w13-st run">⋯ ${esc(row.state === 'running' ? '실행 중' : row.state)}</span>`;
          const res = j ? esc(j.short) : row.state === 'failed' ? esc(row.lastError ?? '') : '';
          const tries = row.cached ? '캐시' : row.attempts ? `${row.attempts}${row.attempts > 1 ? ' ↻' : ''}` : '';
          return `<tr class="${i === s.pick ? 'sel' : ''}" data-row="${i}" tabindex="0"><td>${esc(shortTitle(row.doc.title))}</td><td>${st}</td><td>${res}</td><td class="w13-num">${tries}</td></tr>`;
        })
        .join('') +
      '</tbody>';

    // task extra: confusion matrix / field table
    if (task === 'classify' && r.done) $('[data-slot=extra]').innerHTML = confusion(J, s.gold.labels);
    else if (task === 'extract' && r.done) $('[data-slot=extra]').innerHTML = fieldTable(J, s.gold.extraction.fields);
    else $('[data-slot=extra]').innerHTML = '';

    renderDetail(J);
  }

  function renderDetail(J) {
    const i = s.pick;
    const row = s.run.rows[i];
    if (!row) {
      $('[data-slot=detail]').innerHTML = '';
      return;
    }
    const j = J[i];
    const t = row.state === 'done' ? tokensOf(row) : null;
    let body = `<h4>${esc(row.doc.title)} <small class="w13-muted">${esc(row.doc.id)}</small></h4>`;
    body += `<details><summary>채운 프롬프트 (${[...row.prompt].length}자 · 약 ${estTokens(row.prompt)}토큰)</summary><pre class="w13-pre">${esc(row.prompt)}</pre></details>`;
    if (row.state === 'done') {
      body += `<p class="w13-muted">원문 응답${row.cached ? ' (캐시에서 가져옴 · 비용 0)' : ''} · 출력 약 ${t.outT}토큰</p><pre class="w13-pre w13-raw">${esc(row.text)}</pre>`;
      body += j.issues.length ? `<ul class="w13-issues">${j.issues.map((x) => `<li>${esc(x)}</li>`).join('')}</ul>` : '<p class="w13-ok">✓ 문제 없음</p>';
      if (j.fields) {
        body += `<table class="w13-mini"><thead><tr><th>필드</th><th>추출</th><th>정답</th></tr></thead><tbody>${j.fields
          .map((f) => `<tr class="${f.ok ? '' : f.kind}"><td>${esc(f.f)}</td><td>${esc(f.kind === 'parse' ? '—' : JSON.stringify(f.got))}</td><td>${esc(JSON.stringify(f.want ?? s.gold.extraction.default[f.f]))}</td></tr>`)
          .join('')}</tbody></table>`;
      }
    } else if (row.state === 'failed') {
      body += `<ul class="w13-issues"><li>${esc(row.lastError ?? '호출 실패')}</li></ul>`;
    }
    $('[data-slot=detail]').innerHTML = body;
  }

  // --- events
  const autoRun = () => s.mode === 'mock' && run();
  const on = (sel, type, fn) => $(sel).addEventListener(type, fn, { signal: ctrl.signal });
  on('[data-in=task]', 'change', (e) => {
    s.task = e.target.value;
    syncControls();
    tplMsg();
    autoRun();
  });
  on('[data-in=mode]', 'change', (e) => {
    s.mode = e.target.value;
    syncControls();
    if (s.mode === 'mock') run();
    else $('[data-slot=progress]').textContent = `실제 LLM 모드: 문서 ${selected().length}개 → 호출 최대 ${selected().length}회 (실행 전 추정 ${preEstimate()}). ▶ 일괄 실행을 눌러야 호출한다.`;
  });
  on('[data-in=conc]', 'input', (e) => {
    s.conc = Number(e.target.value);
    syncControls();
  });
  on('[data-in=conc]', 'change', autoRun);
  on('[data-in=temp]', 'input', (e) => {
    s.temp = Number(e.target.value);
    $('[data-out=temp]').textContent = s.temp.toFixed(1);
  });
  on('[data-in=temp]', 'change', autoRun);
  on('[data-in=limit]', 'input', (e) => {
    s.limit = Number(e.target.value);
    $('[data-out=limit]').textContent = `${s.limit}자`;
    if (s.run?.task === 'summarize') renderOut(); // re-judge immediately
  });
  on('[data-in=limit]', 'change', autoRun);
  on('[data-in=few]', 'change', (e) => {
    s.few = e.target.checked;
    autoRun();
  });
  on('[data-in=lenient]', 'change', (e) => {
    s.lenient = e.target.checked;
    renderOut(); // parsing only — no new calls
  });
  on('[data-in=cache]', 'change', (e) => {
    s.cache = e.target.checked;
  });
  on('[data-in=pin]', 'input', (e) => {
    s.pin = e.target.value;
    renderOut();
  });
  on('[data-in=pout]', 'input', (e) => {
    s.pout = e.target.value;
    renderOut();
  });
  on('[data-in=provider]', 'change', (e) => {
    s.provider = e.target.value;
    s.model = PROVIDERS[s.provider].defaultModel;
    syncControls();
  });
  on('[data-in=model]', 'change', (e) => {
    s.model = e.target.value;
  });
  let tplTimer = 0;
  on('[data-in=tpl]', 'input', (e) => {
    s.tpl[s.task] = e.target.value;
    tplMsg();
    clearTimeout(tplTimer);
    tplTimer = setTimeout(autoRun, 500);
  });
  $('[data-slot=docs]').addEventListener(
    'change',
    (e) => {
      const id = e.target.dataset.doc;
      if (!id) return;
      if (e.target.checked) s.sel[s.task].add(id);
      else s.sel[s.task].delete(id);
      $('[data-out=ndocs]').textContent = `(${s.sel[s.task].size}/${docs().length})`;
      tplMsg();
      autoRun();
    },
    { signal: ctrl.signal },
  );
  root.addEventListener(
    'click',
    (e) => {
      const act = e.target.closest('[data-act]')?.dataset.act;
      if (!act) return;
      if (act === 'run') run();
      if (act === 'stop') s.runCtrl?.abort();
      if (act === 'clear-cache') {
        const nCache = s.cacheMap.size;
        s.cacheMap.clear();
        $('[data-slot=progress]').textContent = `캐시 ${nCache}개 항목을 비웠다. 다시 실행하면 모두 새로 호출한다.`;
      }
      if (act === 'reset-tpl') {
        s.tpl[s.task] = DEFAULT_TEMPLATES[s.task];
        syncControls();
        tplMsg();
        autoRun();
      }
      if (act === 'all' || act === 'none' || act === 'guide') {
        s.sel[s.task] = new Set(act === 'all' ? docs().map((d) => d.id) : act === 'guide' ? guideIds() : []);
        renderDocList();
        tplMsg();
        autoRun();
      }
      if (act === 'save-key') {
        const input = $('[data-in=key]');
        const v = input.value.trim();
        input.value = '';
        if (v) {
          try {
            setKey(s.provider, v);
          } catch {
            /* storage unavailable */
          }
        }
        syncKeyState();
      }
      if (act === 'clear-key') {
        try {
          clearKey(s.provider);
        } catch {
          /* storage unavailable */
        }
        syncKeyState();
      }
    },
    { signal: ctrl.signal },
  );
  const pickRow = (e) => {
    const tr = e.target.closest('[data-row]');
    if (!tr || !s.run) return;
    if (e.type === 'keydown' && e.key !== 'Enter' && e.key !== ' ') return;
    if (e.type === 'keydown') e.preventDefault();
    s.pick = Number(tr.dataset.row);
    renderOut();
    out.querySelector(`[data-row="${s.pick}"]`)?.focus();
  };
  out.addEventListener('click', pickRow, { signal: ctrl.signal });
  out.addEventListener('keydown', pickRow, { signal: ctrl.signal });

  function tplMsg() {
    const tc = templateCheck();
    const m = $('[data-slot=tplmsg]');
    m.className = 'w13-tplmsg';
    if (tc.error) {
      m.classList.add('bad');
      m.textContent = `⛔ ${tc.error}`;
    } else if (tc.missing.length) {
      m.classList.add('warn');
      m.textContent = `⚠ 알 수 없는 자리표시자 ${tc.missing.map((x) => `{{${x}}}`).join(', ')} — 그대로 전달된다. 쓸 수 있는 것: {{document}} {{limit}} {{labels}} {{examples}}`;
    } else {
      const first = selected()[0];
      m.textContent = first ? `첫 문서 기준 프롬프트 ${[...buildPrompt(first).text].length}자 · 약 ${estTokens(buildPrompt(first).text)}토큰 · 예상 ${preEstimate()}` : '';
    }
  }

  function preEstimate() {
    const list = selected();
    const { pin, pout } = price();
    const inT = list.reduce((a, d) => a + estTokens(buildPrompt(d).text), 0);
    const outPer = s.task === 'summarize' ? Math.ceil(s.limit / CHARS_PER_TOKEN) : s.task === 'classify' ? 4 : 70;
    const cost = (inT * pin + outPer * list.length * pout) / 1e6;
    return `입력 약 ${fmtInt(inT)}토큰 · $${cost.toFixed(5)}`;
  }

  // --- load
  try {
    const [corpus, gold, ...canned] = await Promise.all([
      loadCorpus(),
      fetchJSON('gold.json'),
      ...Object.values(TASKS).map((t) => fetchJSON(t.file)),
    ]);
    if (ctrl.signal.aborted) return;
    s.corpus = corpus;
    s.gold = gold;
    Object.keys(TASKS).forEach((k, i) => (s.canned[k] = canned[i]));
    Object.keys(TASKS).forEach((k) => (s.sel[k] = new Set(defaultSel(k))));
    $('[data-slot=status]').hidden = true;
    syncControls();
    tplMsg();
    run();
  } catch (err) {
    $('[data-slot=status]').innerHTML = `<div class="widget__error">데이터를 불러오지 못했다 (${esc(err.message)}). <code>file://</code>로 열었다면 <code>python -m http.server</code>로 연다.</div>`;
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

// ------------------------------------------------------------ view helpers

async function fetchJSON(name) {
  const res = await fetch(new URL(name, DATA));
  if (!res.ok) throw new Error(`${name} (${res.status})`);
  return res.json();
}

function hintFor(task, r) {
  if (task === 'classify') return r.few ? '퓨샷으로도 남는 오류는 경계가 모호한 문서다. 라벨 정의를 템플릿에 적어 본다.' : '퓨샷 예시를 켜고 다시 본다. 라벨 밖의 답은 관대한 파싱으로도 복구되지 않는다.';
  if (task === 'extract') return r.few ? '남은 오류는 문서 속 비슷한 숫자(최대 12시간)를 필드로 착각한 것이다. 검증 규칙이 필요하다.' : '관대한 파싱을 켜면 형식 오류 일부가 복구된다. 환각은 파싱으로 고칠 수 없다 — 퓨샷을 켜 본다.';
  return r.few ? '길이 제한을 줄여 어디서 깨지는지 본다.' : '퓨샷 예시를 켜 본다. 길이 초과는 “N자 이내”라고 적어도 생긴다 — 코드로 검사해야 하는 이유다.';
}

function confusion(J, labels) {
  const cols = [...labels, null];
  const m = labels.map(() => cols.map(() => 0));
  J.forEach((j) => {
    if (!j) return;
    const gi = labels.indexOf(j.gold);
    const pi = j.pred ? labels.indexOf(j.pred) : cols.length - 1;
    if (gi >= 0) m[gi][pi]++;
  });
  const head = `<tr><th scope="col" class="corner">정답＼예측</th>${cols.map((c) => `<th scope="col">${c ? esc(SHORT[c] ?? c) : '✗'}</th>`).join('')}</tr>`;
  const body = labels
    .map((g, gi) => `<tr><th scope="row">${esc(SHORT[g] ?? g)}</th>${m[gi].map((v, pi) => `<td class="${v ? (pi === gi ? 'diag' : 'off') : ''}">${v || ''}</td>`).join('')}</tr>`)
    .join('');
  return `<h4>혼동 행렬 <small class="w13-muted">— 대각선이 정답, 나머지가 오류 · ✗ = 라벨 무효</small></h4><div class="w13-table-wrap"><table class="w13-cm">${head}${body}</table></div>`;
}

function fieldTable(J, fields) {
  const rows = fields.map((f) => {
    let ok = 0;
    let bad = 0;
    let hall = 0;
    J.forEach((j) => {
      if (!j) return;
      const x = j.fields.find((y) => y.f === f);
      if (x.ok) ok++;
      else if (x.kind === 'halluc') hall++;
      else bad++;
    });
    return `<tr><td>${esc(f)}</td><td class="w13-num">${ok}</td><td class="w13-num">${bad || ''}</td><td class="w13-num">${hall || ''}</td></tr>`;
  });
  return `<h4>필드별 정확 일치 <small class="w13-muted">— 불일치에는 파싱 실패 포함</small></h4><div class="w13-table-wrap"><table class="w13-mini"><thead><tr><th>필드</th><th>일치</th><th>불일치</th><th>환각</th></tr></thead><tbody>${rows.join('')}</tbody></table></div>`;
}

const callout = (kind, title, text) => `<div class="callout callout--${kind}"><span class="callout__title">${esc(title)}</span><p>${esc(text)}</p></div>`;
const stat = (label, value) => `<div class="stat"><span class="stat__label">${esc(label)}</span><span class="stat__value">${esc(value)}</span></div>`;
const pct = (x) => `${Math.round(x * 100)}%`;
const fmtInt = (x) => Math.round(x).toLocaleString('en-US');
const shortTitle = (t) => (t.length > 14 ? `${t.slice(0, 13)}…` : t);

function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
}
