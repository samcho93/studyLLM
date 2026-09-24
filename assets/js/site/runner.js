// Runnable code blocks: <pre data-run="c1" data-title="C1 …"><code>…</code></pre>
// becomes an editable code block (MLBasic style) with ▶ 실행. Code runs as an
// ES module inside a Web Worker (so an infinite loop can be terminated) and
// console output goes to the 실행 결과 terminal. Root-relative paths
// ('/assets/…') are rewritten to the site root so the same snippet works
// locally and on GitHub Pages.

import { startRun, setRunState } from './result.js';

const ROOT = new URL('../../../', import.meta.url).href;
const TIMEOUT_MS = 10000;

const PRELUDE = `const __fmt=(v)=>{if(typeof v==='string')return v;if(v instanceof Error)return v.stack||v.message;try{return JSON.stringify(v,(k,x)=>ArrayBuffer.isView(x)?Array.from(x):x instanceof Map?Object.fromEntries(x):x instanceof Set?[...x]:x,2)??String(v)}catch{return String(v)}};const __send=(level,args)=>self.postMessage({type:'log',level,text:args.map(__fmt).join(' ')});const console={log:(...a)=>__send('o',a),info:(...a)=>__send('s',a),warn:(...a)=>__send('w',a),error:(...a)=>__send('e',a),table:(d)=>{try{self.postMessage({type:'table',rows:JSON.parse(JSON.stringify(d))})}catch{__send('o',[d])}},clear:()=>self.postMessage({type:'clear'})};self.addEventListener('unhandledrejection',(e)=>__send('e',[e.reason]));`;

function rewritePaths(code) {
  return code.replace(/(['"`])\/assets\//g, `$1${ROOT}assets/`);
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
}

function tableHtml(rows) {
  const list = Array.isArray(rows)
    ? rows
    : Object.entries(rows).map(([k, v]) => ({ '(index)': k, ...(typeof v === 'object' ? v : { value: v }) }));
  const cols = [...new Set(list.flatMap((r) => (r && typeof r === 'object' ? Object.keys(r) : ['value'])))];
  const cell = (v) =>
    escapeHtml(typeof v === 'number' ? +v.toFixed(4) : typeof v === 'object' && v !== null ? JSON.stringify(v) : v ?? '');
  return `<table><thead><tr>${cols.map((c) => `<th>${escapeHtml(c)}</th>`).join('')}</tr></thead><tbody>${list
    .map((r) => `<tr>${cols.map((c) => `<td>${cell(r && typeof r === 'object' ? r[c] : r)}</td>`).join('')}</tr>`)
    .join('')}</tbody></table>`;
}

/** Run a module source in a worker; stream console output into `out`. */
export function runModule(code, out) {
  out.innerHTML = '<div class="m">실행 중…</div>';
  const src = `${PRELUDE}\n${rewritePaths(code)}\n;self.postMessage({type:'done'});`;
  const blobUrl = URL.createObjectURL(new Blob([src], { type: 'text/javascript' }));
  const worker = new Worker(blobUrl, { type: 'module' });
  const started = performance.now();
  let first = true;

  const append = (html) => {
    if (first) {
      out.innerHTML = '';
      first = false;
    }
    out.insertAdjacentHTML('beforeend', html);
  };

  return new Promise((resolve) => {
    const finish = (html, state) => {
      clearTimeout(timer);
      worker.terminate();
      URL.revokeObjectURL(blobUrl);
      append(html);
      setRunState(state);
      resolve(state);
    };
    const timer = setTimeout(
      () => finish(`<div class="e">⏱ ${TIMEOUT_MS / 1000}초 안에 끝나지 않아 중단했다. 무한 루프가 없는지 확인한다 (예: overlap ≥ size).</div>`, 'error'),
      TIMEOUT_MS,
    );
    worker.onmessage = ({ data }) => {
      if (data.type === 'log') append(`<div class="${data.level}">${escapeHtml(data.text)}</div>`);
      else if (data.type === 'table') append(tableHtml(data.rows));
      else if (data.type === 'clear') out.innerHTML = '';
      else if (data.type === 'done') finish(`<div class="ok">✓ 완료 (${Math.round(performance.now() - started)}ms)</div>`, 'done');
    };
    worker.onerror = (e) => {
      e.preventDefault();
      const where = e.lineno ? ` (${Math.max(1, e.lineno - 1)}행)` : '';
      finish(`<div class="e">✗ ${escapeHtml(e.message || '실행 오류 — import 경로나 문법을 확인한다')}${where}</div>`, 'error');
    };
  });
}

export function initRunners(scope = document) {
  scope.querySelectorAll('pre[data-run]').forEach((pre) => {
    const title = pre.dataset.title ?? '코드 실행';
    const original = pre.textContent.replace(/\n$/, '');

    const box = document.createElement('div');
    box.className = 'code-block';
    box.innerHTML = `
      <div class="code-head">
        <span class="t"><span class="tag">JS</span>${escapeHtml(title)}</span>
        <button type="button" class="btn ghost small" data-act="reset" title="처음 코드로 되돌리기">↺ 초기화</button>
        <button type="button" class="btn ghost small" data-act="copy" title="코드 복사">⧉ 복사</button>
        <button type="button" class="btn primary small" data-act="run" title="실행 (Ctrl+Enter)">▶ 실행</button>
      </div>
      <textarea spellcheck="false" autocapitalize="off" autocomplete="off" aria-label="${escapeHtml(title)} 코드"></textarea>
      <div class="code-inline-out" aria-live="polite"></div>`;
    const ta = box.querySelector('textarea');
    ta.value = original;
    ta.rows = Math.min(28, original.split('\n').length + 1);
    pre.replaceWith(box);

    const run = () => runModule(ta.value, startRun(title, box.querySelector('.code-inline-out')));

    box.addEventListener('click', async (e) => {
      const act = e.target.closest('[data-act]')?.dataset.act;
      if (act === 'run') run();
      if (act === 'reset') ta.value = original;
      if (act === 'copy') {
        try {
          await navigator.clipboard.writeText(ta.value);
          e.target.textContent = '✓ 복사됨';
          setTimeout(() => (e.target.textContent = '⧉ 복사'), 1200);
        } catch {
          ta.select();
        }
      }
    });
    ta.addEventListener('keydown', (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        e.preventDefault();
        run();
      }
    });
  });
}
