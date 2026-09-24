// LLM calls with the user's own API key.
// Keys live in sessionStorage only (gone when the tab closes) and are never
// logged, rendered or included in error messages.

export const PROVIDERS = {
  anthropic: {
    label: 'Anthropic Claude',
    defaultModel: 'claude-haiku-4-5-20251001',
    models: ['claude-haiku-4-5-20251001', 'claude-sonnet-5'],
  },
  openai: {
    label: 'OpenAI',
    defaultModel: 'gpt-4.1-mini',
    models: ['gpt-4.1-mini', 'gpt-4.1'],
  },
  gemini: {
    label: 'Google Gemini',
    defaultModel: 'gemini-2.5-flash',
    models: ['gemini-2.5-flash', 'gemini-2.5-pro'],
  },
};

const KEY_PREFIX = 'llmlab:key:';

export function setKey(provider, key) {
  assertProvider(provider);
  sessionStorage.setItem(KEY_PREFIX + provider, key.trim());
}

export function hasKey(provider) {
  return Boolean(sessionStorage.getItem(KEY_PREFIX + provider));
}

export function clearKey(provider) {
  sessionStorage.removeItem(KEY_PREFIX + provider);
}

function getKey(provider) {
  const key = sessionStorage.getItem(KEY_PREFIX + provider);
  if (!key) throw new LLMError('API 키가 입력되지 않았다. 키를 입력한 뒤 다시 시도한다.', 'no-key');
  return key;
}

function assertProvider(provider) {
  if (!PROVIDERS[provider]) throw new LLMError(`지원하지 않는 공급자: ${provider}`, 'bad-provider');
}

export class LLMError extends Error {
  constructor(message, code, status) {
    super(message);
    this.name = 'LLMError';
    this.code = code;
    this.status = status;
  }
}

/** Remove anything that looks like a key from provider error text. */
function scrub(text, key) {
  let out = String(text ?? '');
  if (key) out = out.split(key).join('[redacted]');
  return out.replace(/(sk-[\w-]{8,}|AIza[\w-]{20,})/g, '[redacted]').slice(0, 400);
}

/**
 * Generate a completion.
 * @param {{ provider: keyof PROVIDERS, model?: string, system?: string,
 *   messages: { role: 'user'|'assistant', content: string }[],
 *   maxTokens?: number, temperature?: number, signal?: AbortSignal }} req
 * @returns {Promise<{ text: string, usage?: object }>}
 */
export async function generate(req) {
  const { provider } = req;
  assertProvider(provider);
  const key = getKey(provider);
  const model = req.model || PROVIDERS[provider].defaultModel;
  const { url, init, parse } = buildRequest(provider, model, key, req);

  let res;
  try {
    res = await fetch(url, { ...init, signal: req.signal });
  } catch (err) {
    if (err.name === 'AbortError') throw err;
    throw new LLMError('네트워크 오류로 LLM에 연결하지 못했다.', 'network');
  }

  if (!res.ok) {
    let detail = '';
    try {
      const body = await res.json();
      detail = body?.error?.message ?? body?.message ?? '';
    } catch {
      /* non-JSON error body */
    }
    const hint = res.status === 401 || res.status === 403
      ? 'API 키가 올바른지 확인한다.'
      : res.status === 429
        ? '요청 한도를 초과했다. 잠시 뒤 다시 시도한다.'
        : '';
    throw new LLMError(`LLM 호출 실패 (${res.status}) ${hint} ${scrub(detail, key)}`.trim(), 'http', res.status);
  }
  return parse(await res.json());
}

function buildRequest(provider, model, key, { system, messages, maxTokens = 1024, temperature = 0.2 }) {
  switch (provider) {
    case 'anthropic':
      return {
        url: 'https://api.anthropic.com/v1/messages',
        init: {
          method: 'POST',
          headers: {
            'content-type': 'application/json',
            'x-api-key': key,
            'anthropic-version': '2023-06-01',
            // required for direct browser (CORS) access
            'anthropic-dangerous-direct-browser-access': 'true',
          },
          body: JSON.stringify({ model, system, messages, max_tokens: maxTokens, temperature }),
        },
        parse: (j) => ({
          text: j.content?.filter((b) => b.type === 'text').map((b) => b.text).join('') ?? '',
          usage: j.usage,
        }),
      };
    case 'openai':
      return {
        url: 'https://api.openai.com/v1/chat/completions',
        init: {
          method: 'POST',
          headers: { 'content-type': 'application/json', authorization: `Bearer ${key}` },
          body: JSON.stringify({
            model,
            messages: system ? [{ role: 'system', content: system }, ...messages] : messages,
            max_tokens: maxTokens,
            temperature,
          }),
        },
        parse: (j) => ({ text: j.choices?.[0]?.message?.content ?? '', usage: j.usage }),
      };
    case 'gemini':
      return {
        url: `https://generativelanguage.googleapis.com/v1beta/models/${encodeURIComponent(model)}:generateContent`,
        init: {
          method: 'POST',
          headers: { 'content-type': 'application/json', 'x-goog-api-key': key },
          body: JSON.stringify({
            systemInstruction: system ? { parts: [{ text: system }] } : undefined,
            contents: messages.map((m) => ({
              role: m.role === 'assistant' ? 'model' : 'user',
              parts: [{ text: m.content }],
            })),
            generationConfig: { maxOutputTokens: maxTokens, temperature },
          }),
        },
        parse: (j) => ({
          text: j.candidates?.[0]?.content?.parts?.map((p) => p.text).join('') ?? '',
          usage: j.usageMetadata,
        }),
      };
    default:
      throw new LLMError(`지원하지 않는 공급자: ${provider}`, 'bad-provider');
  }
}

/** Build a grounded RAG prompt with numbered sources for citation. */
export function ragPrompt(question, contexts) {
  const sources = contexts.map((c, i) => `[${i + 1}] ${c.text}`).join('\n\n');
  return {
    system:
      '너는 주어진 근거 문서만 사용해 답하는 조교다. 근거에 없는 내용은 "문서에서 찾을 수 없다"고 답한다. ' +
      '문장마다 사용한 근거 번호를 [1]처럼 표기한다.',
    messages: [{ role: 'user', content: `근거 문서:\n${sources}\n\n질문: ${question}` }],
  };
}
