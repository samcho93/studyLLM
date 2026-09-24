// w17 LLM 비교표 탐색기
// One concept: "which LLM?" is a constraint problem, not a leaderboard. Weights you can
// download, the license, Korean support, size and context all rule models in or out.
// Failure modes surfaced: non-commercial licenses, "open" weights that are not open source,
// huge context windows on models no laptop can run.

import { registerOutput, unregisterOutput } from '../site/result.js';

const DATA_URL = new URL('../../data/w17/models.json', import.meta.url);

// ---------- pure helpers (also used by the challenges' logic; no DOM) ----------

export const SIZE_BANDS = [
  { id: 'all', label: '전체' },
  { id: 'tiny', label: '≤1B', test: (p) => p !== null && p <= 1 },
  { id: 'small', label: '1–4B', test: (p) => p !== null && p > 1 && p <= 4 },
  { id: 'mid', label: '4–15B', test: (p) => p !== null && p > 4 && p <= 15 },
  { id: 'large', label: '>15B', test: (p) => p !== null && p > 15 },
  { id: 'hidden', label: '비공개', test: (p) => p === null },
];

export const COMMERCIAL_LABEL = { yes: '상업 이용 가능', conditional: '조건부', no: '비상업 전용', api: 'API 약관' };

/** Weight bytes in GB for a parameter count (billions) at a given bits-per-weight. */
export const weightsGB = (paramsB, bits) => (paramsB * bits) / 8;

/** Rule of thumb used by the picker: 4-bit weights ≤ 4 GB → fits a 16 GB RAM laptop without a GPU. */
export const LAPTOP_RULE = { bits: 4, maxGB: 4 };
export const fitsLaptop = (m) => m.paramsB !== null && weightsGB(m.paramsB, LAPTOP_RULE.bits) <= LAPTOP_RULE.maxGB;

export function licenseInfo(m, licenses) {
  return licenses[m.license] ?? { label: m.license, weights: false, osi: false, commercial: 'api', note: '' };
}

/** 'oss' (OSI open source weights) · 'weights' (downloadable, custom terms) · 'closed' */
export function openness(m, licenses) {
  const l = licenseInfo(m, licenses);
  if (!l.weights) return 'closed';
  return l.osi ? 'oss' : 'weights';
}

export const KOREAN_OK = new Set(['특화', '명시']);

/** Warnings a student should notice for a row. */
export function flags(m, licenses) {
  const l = licenseInfo(m, licenses);
  const out = [];
  if (l.commercial === 'no') out.push({ kind: 'danger', text: '비상업 라이선스: 상업 서비스 불가' });
  if (l.weights && !l.osi) out.push({ kind: 'warn', text: '가중치는 공개, 그러나 OSI 오픈소스가 아님 (맞춤 약관)' });
  if (l.weights && m.paramsB !== null && m.paramsB > 100) out.push({ kind: 'warn', text: `오픈 웨이트지만 4비트로도 ${Math.round(weightsGB(m.paramsB, 4))}GB — 노트북에서 못 돈다` });
  if (!l.weights && m.ctx !== null && m.ctx >= 1_000_000) out.push({ kind: 'muted', text: '컨텍스트 1M이지만 내 기기에서는 못 돌린다 (API 전용)' });
  if (m.ctx !== null && m.ctx <= 8192) out.push({ kind: 'muted', text: `컨텍스트 ${fmtCtx(m.ctx)}: 긴 문서는 잘라 넣어야 한다` });
  if (m.korean === '영어 중심') out.push({ kind: 'muted', text: '공식 지원 언어에 한국어 없음' });
  return out;
}

export const CONSTRAINTS = [
  { id: 'private', label: '개인정보: 외부 전송 불가', short: '외부 전송 불가' },
  { id: 'laptop', label: 'GPU 없는 노트북 (16GB 램)', short: 'GPU 없음' },
  { id: 'korean', label: '한국어가 중요', short: '한국어' },
  { id: 'commercial', label: '상업 서비스에 쓴다', short: '상업 이용' },
  { id: 'longdoc', label: '긴 문서 (≥128K 토큰)', short: '긴 문서' },
];

/**
 * Apply hard constraints. Returns kept models with the reasons they passed,
 * and rejected models with the first constraint that removed them.
 */
export function pick(models, licenses, active) {
  const kept = [];
  const rejected = [];
  for (const m of models) {
    const l = licenseInfo(m, licenses);
    const why = [];
    let fail = '';
    if (active.private) {
      if (!l.weights) fail ||= '가중치가 비공개라 데이터가 공급자 서버로 간다';
      else why.push('가중치를 내려받아 내부에서 돌릴 수 있다');
    }
    if (active.laptop) {
      if (m.paramsB === null) fail ||= '파라미터 비공개 · 로컬 실행 불가';
      else if (!fitsLaptop(m)) fail ||= `4비트로도 약 ${weightsGB(m.paramsB, 4).toFixed(2)}GB — 노트북 어림(${LAPTOP_RULE.maxGB}GB) 초과`;
      else why.push(`4비트 가중치 약 ${weightsGB(m.paramsB, 4).toFixed(2)}GB`);
    }
    if (active.korean) {
      if (!KOREAN_OK.has(m.korean)) fail ||= `한국어: ${m.korean}`;
      else why.push(`한국어 ${m.korean}`);
    }
    if (active.commercial) {
      if (l.commercial === 'no') fail ||= `${l.label}: 상업 이용 금지`;
      else if (l.commercial === 'conditional') why.push(`조건부 상업 이용 (${l.label})`);
      else if (l.commercial === 'api') why.push('API 약관에 따른 유료 이용');
      else why.push(`${l.label}: 상업 이용 가능`);
    }
    if (active.longdoc) {
      if (m.ctx === null) fail ||= '컨텍스트 확인 필요';
      else if (m.ctx < 128000) fail ||= `컨텍스트 ${fmtCtx(m.ctx)} < 128K`;
      else why.push(`컨텍스트 ${fmtCtx(m.ctx)}`);
    }
    if (fail) rejected.push({ m, reason: fail });
    else kept.push({ m, why });
  }
  return { kept, rejected };
}

export function fmtParams(m) {
  if (m.paramsB === null) return '비공개';
  const p = m.paramsB;
  const s = p >= 100 ? `${Math.round(p)}B` : p >= 1 ? `${+p.toFixed(1)}B` : `${Math.round(p * 1000)}M`;
  return m.activeB ? `${s} (활성 ${m.activeB}B)` : s;
}

export function fmtCtx(n) {
  if (n === null || n === undefined) return '확인 필요';
  if (n >= 1_000_000) return `${+(n / 1_000_000).toFixed(2)}M`;
  return `${Math.round(n / 1024)}K`;
}

function sortModels(list, key) {
  const val = {
    params: (m) => m.paramsB ?? Infinity,
    ctx: (m) => -(m.ctx ?? -1),
    date: (m) => (m.date ? -Date.parse(m.date.length === 7 ? `${m.date}-01` : m.date) : Infinity),
    name: () => 0,
  }[key];
  return [...list].sort((a, b) => val(a) - val(b) || a.name.localeCompare(b.name));
}

// ---------- widget ----------

const template = document.createElement('template');
template.innerHTML = `
  <div class="widget w17-map">
    <h3 class="widget__title">LLM 비교표 탐색기</h3>
    <p class="w17-notice" data-slot="notice"></p>
    <div class="widget__controls w17-filters">
      <label class="field"><span class="field__label">구분</span>
        <select data-in="kind"><option value="all">전체</option><option value="api">상용 API</option><option value="open">오픈 웨이트</option></select></label>
      <label class="field"><span class="field__label">크기</span><select data-in="size"></select></label>
      <label class="field"><span class="field__label">라이선스 · 상업 이용</span>
        <select data-in="lic"><option value="all">전체</option><option value="yes">상업 이용 가능</option><option value="conditional">조건부</option><option value="no">비상업 전용</option><option value="api">API 약관</option><option value="oss">OSI 오픈소스만</option></select></label>
      <label class="field"><span class="field__label">한국어</span>
        <select data-in="ko"><option value="all">전체</option><option value="ok">특화 · 명시만</option><option value="special">특화만</option></select></label>
      <label class="field"><span class="field__label">입력</span>
        <select data-in="mod"><option value="all">전체</option><option value="text">텍스트 전용</option><option value="image">이미지 입력 가능</option><option value="audio">음성 입력 가능</option></select></label>
      <label class="field"><span class="field__label">정렬</span>
        <select data-in="sort"><option value="params">파라미터 작은 순</option><option value="ctx">컨텍스트 긴 순</option><option value="date">최근 등록 순</option></select></label>
    </div>
    <fieldset class="w17-picker">
      <legend>용도로 고르기 (조건을 켜면 결과 창에 후보와 탈락 이유가 나온다)</legend>
      <div class="w17-checks" data-slot="checks"></div>
    </fieldset>
    <div class="widget__status" data-slot="status" role="status" aria-live="polite"></div>
    <div class="w17-table-wrap" tabindex="0" aria-label="모델 비교표 (가로 스크롤)"><table class="w17-table" data-slot="table"></table></div>
    <div data-slot="out-inline"></div>
  </div>`;

const outTemplate = document.createElement('template');
outTemplate.innerHTML = `
  <div class="w17-out">
    <div class="stat-row" data-slot="stats"></div>
    <figure class="w17-chart-fig">
      <svg data-slot="chart" viewBox="0 0 360 250" role="img" aria-label="파라미터 수와 컨텍스트 길이 산점도 (로그 눈금)"></svg>
      <figcaption class="w17-legend">
        <span><i class="w17-dot yes"></i>상업 가능</span><span><i class="w17-dot conditional"></i>조건부</span><span><i class="w17-dot no"></i>비상업</span><span><i class="w17-dot api"></i>API</span>
      </figcaption>
    </figure>
    <div data-slot="pick"></div>
  </div>`;

const state = new WeakMap();
let seq = 0;

/**
 * @param {HTMLElement} el
 * @param {{ outputTitle?: string, preset?: string[] }} options
 */
export async function mount(el, options = {}) {
  const root = template.content.firstElementChild.cloneNode(true);
  const out = outTemplate.content.firstElementChild.cloneNode(true);
  el.replaceChildren(root);
  const $ = (sel) => root.querySelector(sel) ?? out.querySelector(sel);
  const ctrl = new AbortController();
  const outputId = `w17-model-map:${++seq}`;
  state.set(el, { ctrl, outputId });

  const s = { kind: 'all', size: 'all', lic: 'all', ko: 'all', mod: 'all', sort: 'params', active: {} };
  for (const id of options.preset ?? ['private', 'laptop', 'korean', 'commercial']) s.active[id] = true;
  let data = null;

  $('[data-in=size]').innerHTML = SIZE_BANDS.map((b) => `<option value="${b.id}">${esc(b.label)}</option>`).join('');
  $('[data-slot=checks]').innerHTML = CONSTRAINTS.map(
    (c) => `<label class="w17-check"><input type="checkbox" data-c="${c.id}"${s.active[c.id] ? ' checked' : ''}> ${esc(c.label)}</label>`,
  ).join('');
  registerOutput(outputId, { title: options.outputTitle ?? 'LLM 비교 결과', node: out, inlineHost: $('[data-slot=out-inline]') });
  $('[data-slot=status]').innerHTML = '<span class="spinner" aria-hidden="true"></span> models.json을 불러오는 중…';

  function filtered() {
    const band = SIZE_BANDS.find((b) => b.id === s.size);
    return data.models.filter((m) => {
      const l = licenseInfo(m, data.licenses);
      if (s.kind !== 'all' && m.kind !== s.kind) return false;
      if (band?.test && !band.test(m.paramsB)) return false;
      if (s.lic === 'oss' ? !l.osi : s.lic !== 'all' && l.commercial !== s.lic) return false;
      if (s.ko === 'ok' && !KOREAN_OK.has(m.korean)) return false;
      if (s.ko === 'special' && m.korean !== '특화') return false;
      if (s.mod === 'text' && m.modality.some((x) => x !== 'text')) return false;
      if ((s.mod === 'image' || s.mod === 'audio') && !m.modality.includes(s.mod)) return false;
      return true;
    });
  }

  function render() {
    const list = sortModels(filtered(), s.sort);
    const lic = data.licenses;
    const rows = list
      .map((m) => {
        const l = licenseInfo(m, lic);
        const fl = flags(m, lic)
          .map((f) => `<span class="w17-flag ${f.kind}">${esc(f.text)}</span>`)
          .join('');
        return `<tr>
          <th scope="row"><a href="${esc(m.source)}" target="_blank" rel="noopener">${esc(m.name)}</a><div class="w17-sub">${esc(m.provider)} · ${esc(m.tier)}</div>${fl ? `<div class="w17-flags">${fl}</div>` : ''}</th>
          <td>${m.kind === 'api' ? '<span class="chip">API</span>' : '<span class="chip ok">오픈 웨이트</span>'}</td>
          <td class="num" title="${esc(m.paramsNote ?? '')}">${esc(fmtParams(m))}</td>
          <td class="num" title="${esc(m.ctxNote ?? '')}">${esc(fmtCtx(m.ctx))}</td>
          <td><span class="w17-lic ${l.commercial}">${esc(l.label)}</span><div class="w17-sub">${esc(COMMERCIAL_LABEL[l.commercial])}${l.weights ? (l.osi ? ' · OSI 오픈소스' : ' · 오픈 웨이트(맞춤 약관)') : ''}</div></td>
          <td>${esc(m.korean)}</td>
          <td>${esc(m.modality.map((x) => ({ text: '텍스트', image: '이미지', audio: '음성', video: '영상' })[x]).join(' · '))}</td>
          <td class="num">${esc(m.date ?? '—')}</td>
          <td class="w17-use">${esc(m.use)}</td>
        </tr>`;
      })
      .join('');
    $('[data-slot=table]').innerHTML = `<thead><tr><th>모델</th><th>구분</th><th>파라미터</th><th>컨텍스트</th><th>라이선스</th><th>한국어</th><th>입력</th><th>등록일</th><th>주 용도</th></tr></thead><tbody>${
      rows || '<tr><td colspan="9">조건에 맞는 모델이 없다. 필터를 하나 풀어 본다.</td></tr>'
    }</tbody>`;

    const all = data.models;
    const openN = all.filter((m) => licenseInfo(m, lic).weights).length;
    const ossN = all.filter((m) => licenseInfo(m, lic).osi).length;
    $('[data-slot=stats]').innerHTML = [
      stat('표시', `${list.length} / ${all.length}`),
      stat('오픈 웨이트', openN),
      stat('그중 OSI 오픈소스', ossN),
      stat('비상업', all.filter((m) => licenseInfo(m, lic).commercial === 'no').length),
    ].join('');
    $('[data-slot=status]').textContent = `기준일 ${data.asOf} · ${list.length}개 표시 · 모델 이름을 누르면 출처(모델 카드 · 공식 문서)가 열린다.`;
    drawChart(new Set(list.map((m) => m.id)));
    renderPick();
  }

  function drawChart(visible) {
    const svg = $('[data-slot=chart]');
    const W = 360, H = 250, L = 64, R = 10, T = 12, B = 34;
    const x = (p) => L + ((Math.log10(p) + 1) / (Math.log10(2000) + 1)) * (W - L - R); // 0.1B … 2000B
    const y = (c) => H - B - ((Math.log2(c) - 12) / (21 - 12)) * (H - T - B); // 4K … 2M
    const parts = [];
    for (const p of [0.1, 1, 10, 100, 1000]) parts.push(`<line class="w17-grid" x1="${x(p)}" x2="${x(p)}" y1="${T}" y2="${H - B}"/><text class="w17-ax" x="${x(p)}" y="${H - B + 14}" text-anchor="middle">${p >= 1000 ? '1T' : p >= 1 ? `${p}B` : '100M'}</text>`);
    for (const [c, t] of [[4096, '4K'], [32768, '32K'], [131072, '128K'], [1048576, '1M']]) parts.push(`<line class="w17-grid" x1="${L}" x2="${W - R}" y1="${y(c)}" y2="${y(c)}"/><text class="w17-ax" x="${L - 4}" y="${y(c) + 4}" text-anchor="end">${t}</text>`);
    parts.push(`<text class="w17-ax" x="${(L + W) / 2}" y="${H - 4}" text-anchor="middle">전체 파라미터 (로그)</text>`);
    parts.push(`<text class="w17-ax" x="12" y="${T + 4}" text-anchor="start">ctx</text>`);
    // API models: parameter count not public → a strip left of the axis
    parts.push(`<text class="w17-ax" x="${L - 26}" y="${H - B + 26}" text-anchor="middle">비공개</text>`);
    parts.push(`<line class="w17-strip" x1="${L - 26}" x2="${L - 26}" y1="${T}" y2="${H - B}"/>`);
    parts.push(`<rect class="w17-laptop" x="${L}" y="${T}" width="${x(8) - L}" height="${H - T - B}"><title>4비트 가중치 ≤ 4GB (≈ 8B 이하): GPU 없는 노트북 어림</title></rect>`);
    parts.push(`<text class="w17-ax w17-laptop-t" x="${L + 4}" y="${T + 12}">노트북 영역 (≤8B)</text>`);
    const lic = data.licenses;
    let jitter = 0;
    for (const m of data.models) {
      if (m.ctx === null) continue;
      const l = licenseInfo(m, lic);
      const cx = m.paramsB === null ? L - 26 + (((jitter++ % 5) - 2) * 4) : x(m.paramsB);
      const cy = y(Math.min(m.ctx, 2_000_000));
      const on = visible.has(m.id);
      const shape = m.kind === 'api' ? `<rect x="${cx - 4}" y="${cy - 4}" width="8" height="8" rx="1.5"` : `<circle cx="${cx}" cy="${cy}" r="4.5"`;
      parts.push(`${shape} class="w17-pt ${l.commercial}${on ? '' : ' off'}"><title>${esc(m.name)} · ${esc(fmtParams(m))} · ${esc(fmtCtx(m.ctx))} · ${esc(l.label)}</title>${m.kind === 'api' ? '</rect>' : '</circle>'}`);
    }
    svg.innerHTML = parts.join('');
  }

  function renderPick() {
    const active = s.active;
    const on = CONSTRAINTS.filter((c) => active[c.id]);
    const { kept, rejected } = pick(data.models, data.licenses, active);
    const keptSorted = sortModels(kept.map((k) => k.m), 'params');
    const keptHtml = keptSorted
      .map((m) => {
        const k = kept.find((x) => x.m === m);
        return `<li><b>${esc(m.name)}</b> <span class="w17-sub">${esc(fmtParams(m))} · ${esc(fmtCtx(m.ctx))}</span><div class="w17-sub">${esc(k.why.join(' · ') || '조건 없음')}</div></li>`;
      })
      .join('');
    // Most instructive rejections: models that pass everything except exactly one constraint.
    // License and privacy misses come first — those are the ones students overlook.
    const PRIORITY = ['commercial', 'private', 'korean', 'longdoc', 'laptop'];
    const nearMiss = [];
    for (const r of rejected) {
      const failing = CONSTRAINTS.filter((c) => active[c.id] && pick([r.m], data.licenses, { [c.id]: true }).rejected.length);
      if (failing.length === 1) nearMiss.push({ ...r, by: failing[0].id });
    }
    nearMiss.sort((a, b) => PRIORITY.indexOf(a.by) - PRIORITY.indexOf(b.by) || (a.m.paramsB ?? 0) - (b.m.paramsB ?? 0));
    $('[data-slot=pick]').innerHTML = `
      <h4 class="w17-h">용도로 고르기 · ${on.length ? esc(on.map((c) => c.short).join(' + ')) : '조건 없음'}</h4>
      <p class="w17-sub">후보 ${kept.length}개 · 탈락 ${rejected.length}개</p>
      ${kept.length ? `<ol class="w17-cands">${keptHtml}</ol>` : '<p class="w17-empty">모든 조건을 만족하는 모델이 없다. 조건 하나를 풀거나 (예: 사내 GPU 서버를 두면 “GPU 없음”을 뺄 수 있다) 기준을 바꿔야 한다.</p>'}
      ${nearMiss.length ? `<details open><summary>아깝게 탈락 (조건 하나만 걸림)</summary><ul class="w17-miss">${nearMiss
        .slice(0, 8)
        .map((r) => `<li><b>${esc(r.m.name)}</b> — ${esc(r.reason)}</li>`)
        .join('')}${nearMiss.length > 8 ? `<li class="w17-sub">외 ${nearMiss.length - 8}개</li>` : ''}</ul></details>` : ''}`;
  }

  const on = (sel, type, fn) => $(sel).addEventListener(type, fn, { signal: ctrl.signal });
  for (const key of ['kind', 'size', 'lic', 'ko', 'mod', 'sort'])
    on(`[data-in=${key}]`, 'change', (e) => {
      s[key] = e.target.value;
      if (data) render();
    });
  on('[data-slot=checks]', 'change', (e) => {
    const id = e.target.dataset.c;
    if (!id) return;
    s.active[id] = e.target.checked;
    if (data) renderPick();
  });

  try {
    const res = await fetch(DATA_URL, { signal: ctrl.signal });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    data = await res.json();
  } catch (err) {
    if (ctrl.signal.aborted) return;
    $('[data-slot=status]').innerHTML = `<span class="widget__error">models.json을 불러오지 못했다: ${esc(err.message)} — 로컬에서는 python -m http.server로 연다.</span>`;
    return;
  }
  if (ctrl.signal.aborted) return;
  $('[data-slot=notice]').textContent = `📅 ${data.notice}`;
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

const stat = (label, value) => `<div class="stat"><span class="stat__label">${label}</span><span class="stat__value">${value}</span></div>`;

function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
}
