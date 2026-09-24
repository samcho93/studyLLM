// Provider-neutral helpers for week 11 (LLM API · structured output · tool calling).
// No DOM — works in pages, workers and Node.
//
// One neutral conversation format is converted to each provider's wire format:
//   req = { system?, messages, temperature?, maxTokens?, stop?: string[], tools? }
//   message = { role: 'user', content }
//           | { role: 'assistant', content, toolCalls?: [{ id, name, input }] }
//           | { role: 'tool', results: [{ id, name, content, isError? }] }
//   tool = { name, description, parameters }   // parameters = JSON Schema
//
// Keys: read from sessionStorage 'llmlab:key:<provider>' (written by llm.js setKey)
// only inside send(). They are never returned, logged or put in error text.

import { PROVIDERS } from './llm.js';

export const KEY_PLACEHOLDER = '‹API 키 · sessionStorage에만 있음 · 표시하지 않음›';
const KEY_PREFIX = 'llmlab:key:';

/**
 * Example prices in USD per 1M tokens (input / output). They are EXAMPLES the
 * user can edit — always check the provider's current price table.
 */
export const EXAMPLE_PRICES = {
  'claude-haiku-4-5-20251001': { input: 1, output: 5 },
  'claude-sonnet-5': { input: 2, output: 10 },
  'gpt-4.1-mini': { input: 0.4, output: 1.6 },
  'gpt-4.1': { input: 2, output: 8 },
  'gemini-2.5-flash': { input: 0.3, output: 2.5 },
  'gemini-2.5-pro': { input: 1.25, output: 10 },
};

/** Providers whose tool calling this site drives live. Others show saved examples. */
export const LIVE_TOOL_PROVIDERS = ['anthropic', 'openai'];

// ---------------------------------------------------------------- requests

/**
 * Build the HTTP request for a provider. The key is ALWAYS the placeholder here;
 * send() swaps in the real key at the last moment.
 * @returns {{ url: string, headers: Record<string,string>, body: object }}
 */
export function buildRequest(provider, model, req) {
  const { system, messages, temperature = 0.2, maxTokens = 1024, stop, tools } = req;
  const stops = (stop ?? []).filter(Boolean);
  switch (provider) {
    case 'anthropic': {
      const body = { model, max_tokens: maxTokens };
      if (system) body.system = system;
      body.messages = toAnthropicMessages(messages);
      body.temperature = temperature;
      if (stops.length) body.stop_sequences = stops;
      if (tools?.length) {
        body.tools = tools.map((t) => ({ name: t.name, description: t.description, input_schema: t.parameters }));
      }
      return {
        url: 'https://api.anthropic.com/v1/messages',
        headers: {
          'content-type': 'application/json',
          'x-api-key': KEY_PLACEHOLDER,
          'anthropic-version': '2023-06-01',
          'anthropic-dangerous-direct-browser-access': 'true',
        },
        body,
      };
    }
    case 'openai': {
      const body = {
        model,
        messages: [...(system ? [{ role: 'system', content: system }] : []), ...toOpenAIMessages(messages)],
        max_tokens: maxTokens,
        temperature,
      };
      if (stops.length) body.stop = stops;
      if (tools?.length) {
        body.tools = tools.map((t) => ({
          type: 'function',
          function: { name: t.name, description: t.description, parameters: t.parameters },
        }));
      }
      return {
        url: 'https://api.openai.com/v1/chat/completions',
        headers: { 'content-type': 'application/json', authorization: `Bearer ${KEY_PLACEHOLDER}` },
        body,
      };
    }
    case 'gemini': {
      const body = {};
      if (system) body.systemInstruction = { parts: [{ text: system }] };
      body.contents = toGeminiContents(messages);
      body.generationConfig = { temperature, maxOutputTokens: maxTokens };
      if (stops.length) body.generationConfig.stopSequences = stops;
      if (tools?.length) {
        body.tools = [{ functionDeclarations: tools.map((t) => ({ name: t.name, description: t.description, parameters: t.parameters })) }];
      }
      return {
        url: `https://generativelanguage.googleapis.com/v1beta/models/${encodeURIComponent(model)}:generateContent`,
        headers: { 'content-type': 'application/json', 'x-goog-api-key': KEY_PLACEHOLDER },
        body,
      };
    }
    default:
      throw new Error(`지원하지 않는 공급자: ${provider}`);
  }
}

function toAnthropicMessages(messages) {
  return messages.map((m) => {
    if (m.role === 'tool') {
      return {
        role: 'user',
        content: m.results.map((r) => ({
          type: 'tool_result',
          tool_use_id: r.id,
          content: r.content,
          ...(r.isError ? { is_error: true } : {}),
        })),
      };
    }
    if (m.role === 'assistant' && m.toolCalls?.length) {
      return {
        role: 'assistant',
        content: [
          ...(m.content ? [{ type: 'text', text: m.content }] : []),
          ...m.toolCalls.map((c) => ({ type: 'tool_use', id: c.id, name: c.name, input: c.input })),
        ],
      };
    }
    return { role: m.role, content: m.content };
  });
}

function toOpenAIMessages(messages) {
  return messages.flatMap((m) => {
    if (m.role === 'tool') return m.results.map((r) => ({ role: 'tool', tool_call_id: r.id, content: r.content }));
    if (m.role === 'assistant' && m.toolCalls?.length) {
      return [{
        role: 'assistant',
        content: m.content || null,
        tool_calls: m.toolCalls.map((c) => ({
          id: c.id,
          type: 'function',
          function: { name: c.name, arguments: JSON.stringify(c.input) },
        })),
      }];
    }
    return [{ role: m.role, content: m.content }];
  });
}

function toGeminiContents(messages) {
  return messages.map((m) => {
    if (m.role === 'tool') {
      return {
        role: 'user',
        parts: m.results.map((r) => ({ functionResponse: { name: r.name, response: { content: r.content } } })),
      };
    }
    const parts = [];
    if (m.content) parts.push({ text: m.content });
    for (const c of m.toolCalls ?? []) parts.push({ functionCall: { name: c.name, args: c.input } });
    return { role: m.role === 'assistant' ? 'model' : 'user', parts };
  });
}

// ---------------------------------------------------------------- responses

/**
 * Normalize a provider response.
 * @returns {{ text: string, toolCalls: {id,name,input,argError?}[], stopReason: string, rawStop: string,
 *   usage: { input: number, output: number } }}
 */
export function parseResponse(provider, j) {
  switch (provider) {
    case 'anthropic': {
      const blocks = j.content ?? [];
      return {
        text: blocks.filter((b) => b.type === 'text').map((b) => b.text).join(''),
        toolCalls: blocks.filter((b) => b.type === 'tool_use').map((b) => ({ id: b.id, name: b.name, input: b.input ?? {} })),
        stopReason: normStop(j.stop_reason),
        rawStop: String(j.stop_reason ?? ''),
        usage: { input: j.usage?.input_tokens ?? 0, output: j.usage?.output_tokens ?? 0 },
      };
    }
    case 'openai': {
      const ch = j.choices?.[0] ?? {};
      const msg = ch.message ?? {};
      return {
        text: msg.content ?? '',
        toolCalls: (msg.tool_calls ?? []).map((c) => {
          // arguments is a JSON *string* — it can be malformed
          try {
            return { id: c.id, name: c.function?.name, input: JSON.parse(c.function?.arguments || '{}') };
          } catch {
            return { id: c.id, name: c.function?.name, input: {}, argError: String(c.function?.arguments) };
          }
        }),
        stopReason: normStop(ch.finish_reason),
        rawStop: String(ch.finish_reason ?? ''),
        usage: { input: j.usage?.prompt_tokens ?? 0, output: j.usage?.completion_tokens ?? 0 },
      };
    }
    case 'gemini': {
      const cand = j.candidates?.[0] ?? {};
      const parts = cand.content?.parts ?? [];
      return {
        text: parts.filter((p) => typeof p.text === 'string').map((p) => p.text).join(''),
        toolCalls: parts.filter((p) => p.functionCall).map((p, i) => ({ id: `call_${i}`, name: p.functionCall.name, input: p.functionCall.args ?? {} })),
        stopReason: parts.some((p) => p.functionCall) ? 'tool_use' : normStop(cand.finishReason),
        rawStop: String(cand.finishReason ?? ''),
        usage: { input: j.usageMetadata?.promptTokenCount ?? 0, output: j.usageMetadata?.candidatesTokenCount ?? 0 },
      };
    }
    default:
      throw new Error(`지원하지 않는 공급자: ${provider}`);
  }
}

function normStop(s) {
  const v = String(s ?? '').toLowerCase();
  if (v === 'max_tokens' || v === 'length') return 'max_tokens';
  if (v === 'stop_sequence') return 'stop_sequence';
  if (v === 'tool_use' || v === 'tool_calls') return 'tool_use';
  return 'end';
}

/**
 * A provider-shaped response built from a saved classroom example (no network).
 * Same field names as the real API so students can read the real shape.
 */
export function mockResponse(provider, model, { text = '', toolCalls = [], stopReason = 'end', stopSequence = null, usage }) {
  const u = usage ?? { input: 0, output: 0 };
  switch (provider) {
    case 'anthropic':
      return {
        id: 'msg_classroom_example',
        type: 'message',
        role: 'assistant',
        model,
        content: [
          ...(text ? [{ type: 'text', text }] : []),
          ...toolCalls.map((c) => ({ type: 'tool_use', id: c.id, name: c.name, input: c.input })),
        ],
        stop_reason: { end: 'end_turn', max_tokens: 'max_tokens', stop_sequence: 'stop_sequence', tool_use: 'tool_use' }[stopReason],
        stop_sequence: stopReason === 'stop_sequence' ? stopSequence : null,
        usage: { input_tokens: u.input, output_tokens: u.output },
      };
    case 'openai':
      return {
        id: 'chatcmpl-classroom-example',
        object: 'chat.completion',
        model,
        choices: [{
          index: 0,
          message: {
            role: 'assistant',
            content: text || null,
            ...(toolCalls.length
              ? { tool_calls: toolCalls.map((c) => ({ id: c.id, type: 'function', function: { name: c.name, arguments: JSON.stringify(c.input) } })) }
              : {}),
          },
          finish_reason: { end: 'stop', max_tokens: 'length', stop_sequence: 'stop', tool_use: 'tool_calls' }[stopReason],
        }],
        usage: { prompt_tokens: u.input, completion_tokens: u.output, total_tokens: u.input + u.output },
      };
    case 'gemini':
      return {
        candidates: [{
          content: {
            role: 'model',
            parts: [
              ...(text ? [{ text }] : []),
              ...toolCalls.map((c) => ({ functionCall: { name: c.name, args: c.input } })),
            ],
          },
          finishReason: stopReason === 'max_tokens' ? 'MAX_TOKENS' : 'STOP',
        }],
        usageMetadata: { promptTokenCount: u.input, candidatesTokenCount: u.output, totalTokenCount: u.input + u.output },
        modelVersion: model,
      };
    default:
      throw new Error(`지원하지 않는 공급자: ${provider}`);
  }
}

// ---------------------------------------------------------------- network

/** Send a request built by buildRequest(). Returns { raw, parsed }. */
export async function send(provider, request, { signal } = {}) {
  let key = null;
  try {
    key = sessionStorage.getItem(KEY_PREFIX + provider);
  } catch {
    /* storage blocked */
  }
  if (!key) throw new ToolsError('API 키가 입력되지 않았다. 키를 입력한 뒤 다시 시도한다.', 'no-key');
  const headers = {};
  for (const [k, v] of Object.entries(request.headers)) headers[k] = v.split(KEY_PLACEHOLDER).join(key);

  let res;
  try {
    res = await fetch(request.url, { method: 'POST', headers, body: JSON.stringify(request.body), signal });
  } catch (err) {
    if (err.name === 'AbortError') throw err;
    throw new ToolsError('네트워크 오류로 LLM에 연결하지 못했다.', 'network');
  }
  if (!res.ok) {
    let detail = '';
    try {
      const b = await res.json();
      detail = b?.error?.message ?? b?.message ?? '';
    } catch {
      /* non-JSON body */
    }
    const hint = res.status === 401 || res.status === 403
      ? 'API 키가 올바른지 확인한다.'
      : res.status === 429 ? '요청 한도를 초과했다. 잠시 뒤 다시 시도한다.' : '';
    throw new ToolsError(`LLM 호출 실패 (${res.status}) ${hint} ${scrub(detail, key)}`.trim(), 'http', res.status);
  }
  const raw = await res.json();
  return { raw, parsed: parseResponse(provider, raw) };
}

export class ToolsError extends Error {
  constructor(message, code, status) {
    super(message);
    this.name = 'ToolsError';
    this.code = code;
    this.status = status;
  }
}

function scrub(text, key) {
  let out = String(text ?? '');
  if (key) out = out.split(key).join('[redacted]');
  return out.replace(/(sk-[\w-]{8,}|AIza[\w-]{20,})/g, '[redacted]').slice(0, 400);
}

export function defaultModel(provider) {
  return PROVIDERS[provider]?.defaultModel ?? '';
}

// ---------------------------------------------------------------- tokens & cost

/**
 * Rough token estimate used ONLY for saved examples (real calls report usage):
 * Hangul syllable ≈ 0.8 token, other characters ≈ 4 per token.
 */
export function estimateTokens(text) {
  let hangul = 0;
  let other = 0;
  for (const ch of String(text ?? '')) {
    if (/[가-힣ㄱ-ㆎ]/.test(ch)) hangul++;
    else other++;
  }
  return Math.ceil(hangul * 0.8 + other / 4);
}

/** Estimate input tokens of a neutral request (system + every message + tool definitions). */
export function estimateInputTokens(req) {
  const parts = [req.system ?? ''];
  for (const m of req.messages) {
    if (m.role === 'tool') for (const r of m.results) parts.push(r.content);
    else {
      parts.push(m.content ?? '');
      for (const c of m.toolCalls ?? []) parts.push(c.name, JSON.stringify(c.input));
    }
  }
  if (req.tools?.length) parts.push(JSON.stringify(req.tools));
  // small fixed overhead per message for role markers (the chat template of week 10)
  return parts.reduce((a, t) => a + estimateTokens(t), 0) + 4 * req.messages.length;
}

/** Cost in USD for one call. price = { input, output } in USD per 1M tokens. */
export function costUSD(usage, price) {
  return (usage.input * price.input + usage.output * price.output) / 1e6;
}

// ---------------------------------------------------------------- structured output

/**
 * Pull the first JSON object out of model text: strips ``` fences and prose
 * around it, then JSON.parse. Never throws.
 * @returns {{ ok: boolean, value?: any, jsonText?: string, stage: string, error?: string }}
 */
export function extractJson(text) {
  const src = String(text ?? '');
  const fenced = src.match(/```(?:json)?\s*([\s\S]*?)```/i);
  const body = fenced ? fenced[1] : src;
  const start = body.indexOf('{');
  if (start < 0) return { ok: false, stage: 'find', error: '텍스트에 { 가 없다. JSON 객체를 찾지 못했다.' };
  // scan to the matching brace, respecting strings
  let depth = 0;
  let inStr = false;
  let escp = false;
  let end = -1;
  for (let i = start; i < body.length; i++) {
    const c = body[i];
    if (inStr) {
      if (escp) escp = false;
      else if (c === '\\') escp = true;
      else if (c === '"') inStr = false;
      continue;
    }
    if (c === '"') inStr = true;
    else if (c === '{') depth++;
    else if (c === '}' && --depth === 0) {
      end = i;
      break;
    }
  }
  if (end < 0) return { ok: false, stage: 'find', jsonText: body.slice(start), error: '여는 { 에 짝이 맞는 } 가 없다. 출력이 중간에 잘렸을 수 있다.' };
  const jsonText = body.slice(start, end + 1);
  try {
    return { ok: true, value: JSON.parse(jsonText), jsonText, stage: 'parse' };
  } catch (err) {
    return { ok: false, stage: 'parse', jsonText, error: `JSON.parse 실패: ${err.message}` };
  }
}

/**
 * Tiny JSON Schema subset validator: type, properties, required,
 * additionalProperties:false, enum, minimum, maximum, items, minLength.
 * @returns {string[]} error messages (empty = valid)
 */
export function validate(value, schema, path = '$') {
  const errs = [];
  const t = schema.type;
  if (t && !typeOk(value, t)) {
    errs.push(`${path}: ${t} 이어야 하는데 ${typeName(value)} (${JSON.stringify(value)}) 이다`);
    return errs;
  }
  if (schema.enum && !schema.enum.includes(value)) errs.push(`${path}: ${JSON.stringify(schema.enum)} 중 하나여야 한다`);
  if (typeof value === 'number') {
    if (schema.minimum !== undefined && value < schema.minimum) errs.push(`${path}: ${schema.minimum} 이상이어야 한다`);
    if (schema.maximum !== undefined && value > schema.maximum) errs.push(`${path}: ${schema.maximum} 이하여야 한다`);
  }
  if (typeof value === 'string' && schema.minLength !== undefined && value.length < schema.minLength) {
    errs.push(`${path}: 길이가 ${schema.minLength} 이상이어야 한다`);
  }
  if (t === 'object' && value && typeof value === 'object') {
    for (const k of schema.required ?? []) if (!(k in value)) errs.push(`${path}.${k}: 필수 필드가 없다`);
    for (const [k, v] of Object.entries(value)) {
      if (schema.properties?.[k]) errs.push(...validate(v, schema.properties[k], `${path}.${k}`));
      else if (schema.additionalProperties === false) errs.push(`${path}.${k}: 스키마에 없는 필드다`);
    }
  }
  if (t === 'array' && Array.isArray(value) && schema.items) {
    value.forEach((v, i) => errs.push(...validate(v, schema.items, `${path}[${i}]`)));
  }
  return errs;
}

function typeOk(v, t) {
  switch (t) {
    case 'object': return v !== null && typeof v === 'object' && !Array.isArray(v);
    case 'array': return Array.isArray(v);
    case 'string': return typeof v === 'string';
    case 'integer': return Number.isInteger(v);
    case 'number': return typeof v === 'number' && Number.isFinite(v);
    case 'boolean': return typeof v === 'boolean';
    case 'null': return v === null;
    default: return true;
  }
}

function typeName(v) {
  if (v === null) return 'null';
  if (Array.isArray(v)) return 'array';
  if (typeof v === 'number') return Number.isInteger(v) ? 'integer' : 'number';
  return typeof v;
}

/** The follow-up user message for a retry: tell the model exactly what failed. */
export function retryMessage(errors) {
  return `방금 출력은 검증을 통과하지 못했다.\n오류:\n${errors.map((e) => `- ${e}`).join('\n')}\n스키마에 맞는 JSON 객체 하나만 다시 출력한다. 설명 문장이나 코드 블록 표시는 쓰지 않는다.`;
}

// ---------------------------------------------------------------- untrusted text

/**
 * Wrap untrusted text (documents, tool results) in a tag so the model can tell
 * data from instructions. A closing tag hidden inside the text is neutralized
 * so the text cannot "break out" of the wrapper.
 */
export function wrapUntrusted(text, tag = 'document', attrs = '') {
  const safe = String(text).replace(new RegExp(`<\\s*/?\\s*${tag}\\b[^>]*>`, 'gi'), (m) => m.replace(/</g, '‹').replace(/>/g, '›'));
  return `<${tag}${attrs ? ` ${attrs}` : ''}>\n${safe}\n</${tag}>`;
}

// ---------------------------------------------------------------- classroom tools
// Two small local tools the model can ask for in week 11. Pure functions over
// the course corpus (corpus.documents), so they run the same in Node.

export const FACILITY_TOPICS = {
  '개방 시간': ['개방', '주말', '공휴일'],
  '음식물': ['음식물', '음료'],
  'GPU 서버': ['GPU'],
  '소프트웨어': ['소프트웨어'],
  '장비 고장': ['고장'],
};

export const TOOL_DEFS = [
  {
    name: 'search_corpus',
    description: '학과 문서 코퍼스(학과 소개, 교과목 안내, 실습실 규정, 강의 교안)에서 키워드로 문서를 찾는다. 점수가 높은 문서의 id, 제목, 관련 문단을 돌려준다.',
    parameters: {
      type: 'object',
      properties: {
        query: { type: 'string', description: '찾을 키워드들 (공백으로 구분)', minLength: 1 },
        k: { type: 'integer', description: '돌려줄 문서 수 (1~3)', minimum: 1, maximum: 3 },
      },
      required: ['query'],
      additionalProperties: false,
    },
  },
  {
    name: 'get_facility_rule',
    description: '실습실 이용 규정에서 주제 하나에 해당하는 문장을 돌려준다.',
    parameters: {
      type: 'object',
      properties: {
        topic: { type: 'string', enum: Object.keys(FACILITY_TOPICS), description: '규정 주제' },
      },
      required: ['topic'],
      additionalProperties: false,
    },
  },
];

const PARTICLES = /(은|는|이|가|을|를|의|에|에서|로|으로|와|과|도|만)$/;

/** Keyword search: score = number of times each query term appears in title + text. */
export function searchCorpus(documents, query, k = 2) {
  const terms = [...new Set(String(query).split(/[\s,.?!]+/).filter((t) => t.length >= 2))]
    .map((t) => (t.length >= 3 ? t.replace(PARTICLES, '') : t));
  const count = (hay, t) => hay.split(t).length - 1;
  return documents
    .map((d) => {
      const hay = `${d.title}\n${d.text}`;
      const score = terms.reduce((a, t) => a + count(hay, t), 0);
      const paras = d.text.split(/\n\n+/);
      const best = paras
        .map((p) => ({ p, s: terms.reduce((a, t) => a + count(p, t), 0) }))
        .sort((a, b) => b.s - a.s)[0];
      return { id: d.id, title: d.title, score, snippet: best?.p ?? '' };
    })
    .filter((r) => r.score > 0)
    .sort((a, b) => b.score - a.score || a.id.localeCompare(b.id))
    .slice(0, Math.max(1, Math.min(3, k)));
}

/** Sentences of the facility rules that match one topic. Throws on unknown topic. */
export function getFacilityRule(documents, topic) {
  const keys = FACILITY_TOPICS[topic];
  if (!keys) throw new Error(`알 수 없는 topic: "${topic}". 가능한 값: ${Object.keys(FACILITY_TOPICS).join(', ')}`);
  const doc = documents.find((d) => d.id === 'facility-rules');
  const sentences = doc.text.split(/(?<=다\.)\s+/);
  return { source: doc.id, topic, rules: sentences.filter((s) => keys.some((k) => s.includes(k))) };
}

/**
 * Run one tool call against the corpus. Validates arguments with validate()
 * first; any failure becomes { isError: true } so the model can recover.
 * @returns {{ content: string, isError: boolean }}
 */
export function runTool(documents, call) {
  const def = TOOL_DEFS.find((t) => t.name === call.name);
  if (!def) return { content: JSON.stringify({ error: `없는 도구: ${call.name}` }), isError: true };
  const errs = validate(call.input, def.parameters);
  if (errs.length) return { content: JSON.stringify({ error: '인자 검증 실패', details: errs }), isError: true };
  try {
    const out = call.name === 'search_corpus'
      ? { results: searchCorpus(documents, call.input.query, call.input.k ?? 2) }
      : getFacilityRule(documents, call.input.topic);
    return { content: JSON.stringify(out), isError: false };
  } catch (err) {
    return { content: JSON.stringify({ error: err.message }), isError: true };
  }
}
