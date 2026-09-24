// Site shell, matching the sibling course sites (studyMLBasic):
//   left nav (brand · 학생용/교사용 · 진도 · 검색 · PART/주차 트리)
//   middle (topbar + page content) · right dark "실행 결과" terminal.
// Each page only ships <div class="layout" data-layout …><main class="pane center">…</main></div>;
// everything around the content is generated here.

import { PARTS, WEEKS, PATHWAY, weekId } from './weeks.js';

const ROOT = new URL('../../../', import.meta.url);
const THEME_KEY = 'llmlab:theme';
const DONE_KEY = 'llmlab:done';
const WIDTH_KEY = 'llmlab:widths';

// ---------- storage helpers (private mode can throw) ----------
function read(key, fallback) {
  try {
    const v = localStorage.getItem(key);
    return v === null ? fallback : JSON.parse(v);
  } catch {
    return fallback;
  }
}

function write(key, value) {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch {
    /* ignore */
  }
}

const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
const url = (path) => new URL(path, ROOT).href;
const weekUrl = (no, role) => url(`weeks/${weekId(no)}/${role === 'teacher' ? 'teacher.html' : ''}`);
const homeUrl = (role) => url(role === 'teacher' ? 'teacher/' : '');

export function doneWeeks() {
  return new Set(read(DONE_KEY, []));
}

// ---------- theme ----------
const THEME_ICON = {
  system: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true"><circle cx="12" cy="12" r="9"/><path d="M12 3a9 9 0 0 0 0 18z" fill="currentColor"/></svg>',
  light: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg>',
  dark: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true"><path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z"/></svg>',
};
const THEME_LABEL = { system: '시스템 설정', light: '라이트', dark: '다크' };
const THEME_ORDER = ['system', 'light', 'dark'];

function applyTheme(mode) {
  if (mode === 'system') document.documentElement.removeAttribute('data-theme');
  else document.documentElement.setAttribute('data-theme', mode);
}

function bindThemeButton(btn) {
  let mode = read(THEME_KEY, 'system');
  const render = () => {
    btn.innerHTML = THEME_ICON[mode];
    btn.title = `테마: ${THEME_LABEL[mode]} (눌러서 변경)`;
    btn.setAttribute('aria-label', btn.title);
  };
  render();
  btn.addEventListener('click', () => {
    mode = THEME_ORDER[(THEME_ORDER.indexOf(mode) + 1) % THEME_ORDER.length];
    write(THEME_KEY, mode);
    applyTheme(mode);
    render();
  });
}

// ---------- left nav ----------
function sectionsOf(main) {
  return [...main.querySelectorAll('.doc section[id] > h2')].map((h) => {
    const num = h.querySelector('.section-num')?.textContent ?? '';
    const text = [...h.childNodes]
      .filter((n) => !n.classList?.contains('section-num'))
      .map((n) => n.textContent)
      .join('')
      .trim();
    return { id: h.parentElement.id, num, text };
  });
}

function buildNav(layout, { role, week, sections }) {
  const done = doneWeeks();
  const readyCount = WEEKS.filter((w) => w.ready).length;
  const isHome = !week;

  const chapter = (w) => {
    const no = String(w.no).padStart(2, '0');
    const isCur = w.no === week;
    const side = !w.ready
      ? '<span class="pending">준비 중</span>'
      : `${done.has(w.no) ? '<span class="done" aria-label="완료">✓</span>' : ''}<span class="caret" aria-hidden="true">${isCur ? '▾' : '▸'}</span>`;
    const inner = `<span class="no">${no}</span><span class="t">${w.icon} ${esc(w.title)}</span>${side}`;
    const head = w.ready
      ? `<a class="nav-ch-head" href="${weekUrl(w.no, role)}"${isCur ? ' aria-current="page"' : ''}>${inner}</a>`
      : `<div class="nav-ch-head" aria-disabled="true">${inner}</div>`;
    let secs = '';
    if (isCur) {
      secs =
        role === 'teacher'
          ? '<ol class="nav-secs" data-nav-secs></ol>'
          : `<ol class="nav-secs" data-nav-secs>${sections
              .map((s) => `<li><a class="nav-sec" href="#${s.id}" data-sec="${s.id}"><span class="chk">${esc(s.num)}</span><span>${esc(s.text)}</span></a></li>`)
              .join('')}</ol>`;
    }
    return `<div class="nav-ch${isCur ? ' is-current' : ''}${w.ready ? '' : ' is-disabled'}" data-search="${esc(`${w.no} ${w.title} ${w.widget}`)}">${head}${secs}</div>`;
  };

  const tree = PARTS.map(
    (p) => `<div class="nav-group"><div class="nav-part">PART ${p.no}. ${esc(p.title)}</div>${p.weeks
      .map((n) => chapter(WEEKS.find((w) => w.no === n)))
      .join('')}</div>`,
  ).join('');

  const studentHref = week ? weekUrl(week, 'student') : homeUrl('student');
  const teacherHref = week ? weekUrl(week, 'teacher') : homeUrl('teacher');
  const pct = Math.round((done.size / WEEKS.length) * 100);

  const nav = document.createElement('aside');
  nav.className = 'pane nav';
  nav.id = 'nav';
  nav.setAttribute('aria-label', '강좌 목차');
  nav.innerHTML = `
    <div class="brand">
      <a class="brand-logo" href="${homeUrl(role)}" title="처음 화면" aria-label="처음 화면">🧠</a>
      <div class="brand-text">
        <a class="brand-title" href="${homeUrl(role)}">LLM 원리와 활용</a>
        <div class="brand-sub">LLM Lab · 브라우저 실습 · 🤗</div>
      </div>
      <button type="button" class="icon-btn" data-theme-btn></button>
      <button type="button" class="icon-btn nav-close" data-nav-close title="메뉴 접기" aria-label="메뉴 접기">⟨</button>
    </div>
    <nav class="role-switch" aria-label="보기 대상">
      <a data-role="student" href="${studentHref}"${role === 'student' ? ' aria-current="page"' : ''} title="학생용: 문서 + 실습">🎓 학생용</a>
      <a data-role="teacher" href="${teacherHref}"${role === 'teacher' ? ' aria-current="page"' : ''} title="교사용: PPT 슬라이드 + 교사 노트 + 정답">🧑‍🏫 교사용</a>
    </nav>
    <div class="progress-wrap">
      <div class="progress-label"><span>학습 진도</span><span>${done.size} / ${WEEKS.length}주 · 공개 ${readyCount}주</span></div>
      <div class="progress" role="progressbar" aria-valuenow="${pct}" aria-valuemin="0" aria-valuemax="100" aria-label="학습 진도"><div class="progress-fill" style="width: ${pct}%"></div></div>
    </div>
    <input class="nav-search" type="search" placeholder="검색 (예: 토큰, 어텐션, 디코딩)" aria-label="목차 검색">
    <div class="nav-tree">
      <a class="nav-home" href="${homeUrl(role)}"${isHome ? ' aria-current="page"' : ''}>🏠 ${role === 'teacher' ? '교사용 안내' : '강좌 소개'}</a>
      ${tree}
      <div class="nav-group"><div class="nav-part">학습 경로 (연계 과정)</div>
        ${PATHWAY.map((c) => `<div class="nav-ch${c.current ? ' is-current' : ''}" data-search="${esc(`연계 과정 ${c.title} ${c.sub}`)}"><a class="nav-ch-head" href="${c.current ? homeUrl(role) : c.url}"${c.current ? '' : ' target="_blank" rel="noopener"'}><span class="no">${c.step}</span><span class="t">${c.icon} ${esc(c.title)}</span>${c.current ? '<span class="pending">지금</span>' : '<span class="caret" aria-hidden="true">↗</span>'}</a></div>`).join('')}
      </div>
      <p class="nav-empty" hidden>검색 결과가 없다.</p>
    </div>
    <div class="nav-foot">
      <div class="env-badge" data-env-badge><span class="dot"></span><span>실행 환경 확인 중…</span></div>
    </div>`;
  layout.prepend(nav);
  nav.insertAdjacentHTML('afterend', '<div class="gutter" data-resize="nav" title="드래그하여 너비 조절" aria-hidden="true"></div>');

  // search
  const search = nav.querySelector('.nav-search');
  search.addEventListener('input', () => {
    const q = search.value.trim().toLowerCase();
    let any = false;
    nav.querySelectorAll('.nav-group').forEach((g) => {
      let groupAny = false;
      g.querySelectorAll('.nav-ch').forEach((ch) => {
        const hit = !q || ch.dataset.search.toLowerCase().includes(q) || ch.textContent.toLowerCase().includes(q);
        ch.hidden = !hit;
        groupAny ||= hit;
      });
      g.hidden = !groupAny;
      any ||= groupAny;
    });
    nav.querySelector('.nav-empty').hidden = any;
  });

  // environment badge: Web Worker + module scripts are all the runner needs
  const env = nav.querySelector('[data-env-badge]');
  const ok = 'Worker' in window && 'noModule' in HTMLScriptElement.prototype;
  env.classList.toggle('ok', ok);
  env.lastElementChild.textContent = ok ? 'JavaScript 실행 환경 준비 완료' : '이 브라우저는 코드 실행을 지원하지 않는다';

  bindThemeButton(nav.querySelector('[data-theme-btn]'));
  return nav;
}

// ---------- topbar ----------
function buildTopbar(main, { role, week, crumb }) {
  const bar = document.createElement('div');
  bar.className = 'topbar';
  const ready = WEEKS.filter((w) => w.ready);
  const prev = week ? [...ready].reverse().find((w) => w.no < week) : null;
  const next = week ? ready.find((w) => w.no > week) : null;
  bar.innerHTML = `
    <button type="button" class="icon-btn nav-open" data-nav-open title="메뉴 펼치기" aria-label="메뉴 펼치기">☰</button>
    <div class="crumb">${crumb}</div>
    <span class="spacer"></span>
    ${week ? `<nav class="view-switch" aria-label="보기 방식">
      <a href="${weekUrl(week, 'student')}"${role === 'student' ? ' aria-current="page"' : ''} title="문서 보기">📄 문서</a>
      <a href="${weekUrl(week, 'teacher')}"${role === 'teacher' ? ' aria-current="page"' : ''} title="슬라이드 보기 (PPT)">🖼️ 슬라이드</a>
    </nav>
    ${prev ? `<a class="btn ghost small" href="${weekUrl(prev.no, role)}" title="이전 주차: ${esc(prev.title)}">◀</a>` : '<span class="btn ghost small" aria-disabled="true" style="opacity:.4">◀</span>'}
    ${next ? `<a class="btn ghost small" href="${weekUrl(next.no, role)}" title="다음 주차: ${esc(next.title)}">▶</a>` : '<span class="btn ghost small" aria-disabled="true" style="opacity:.4" title="다음 주차 준비 중">▶</span>'}` : ''}`;
  main.prepend(bar);
}

// ---------- right terminal ----------
function buildOutput(layout) {
  const out = document.createElement('aside');
  out.className = 'pane output';
  out.id = 'output';
  out.setAttribute('aria-label', '실행 결과');
  out.innerHTML = `
    <div class="console-panel">
      <div class="console-head">
        <span class="console-dots" aria-hidden="true"><i></i><i></i><i></i></span>
        <span class="console-title">실행 결과 <span class="muted">· JavaScript</span></span>
        <span class="spacer"></span>
        <span class="run-state" data-run-state>대기</span>
      </div>
      <div class="console-tools">
        <button type="button" class="btn small" data-console-clear title="코드 실행 기록 지우기">🧹 지우기</button>
        <button type="button" class="btn small" data-console-font="-1" title="글자 작게">A−</button>
        <button type="button" class="btn small" data-console-font="1" title="글자 크게">A+</button>
        <span class="spacer"></span><span><kbd>Ctrl</kbd>+<kbd>Enter</kbd> 실행</span>
      </div>
      <div class="console" data-console tabindex="0" aria-live="polite">
        <div class="console-welcome" data-console-welcome>
          <div class="cw-icon">&gt;_</div>
          <p><b>실습 결과가 여기에 나온다</b></p>
          <p>위젯을 조작하거나 코드 상자의 <b>▶ 실행</b>을 누른다.<br>코드는 페이지와 분리된 워커에서 돌아가며 10초가 넘으면 자동으로 멈춘다.</p>
        </div>
        <div data-console-pinned></div>
        <div data-console-log></div>
      </div>
      <div class="console-status"><span>● 브라우저 실행</span><span>Web Worker · 10초 제한</span></div>
    </div>`;
  layout.append(out);
  out.insertAdjacentHTML('beforebegin', '<div class="gutter" data-resize="out" title="드래그하여 너비 조절" aria-hidden="true"></div>');

  const panel = out.querySelector('.console-panel');
  let size = read('llmlab:console-font', 13.5);
  panel.style.setProperty('--c-font', `${size}px`);
  out.addEventListener('click', (e) => {
    const f = e.target.closest('[data-console-font]');
    if (f) {
      size = Math.min(20, Math.max(11, size + Number(f.dataset.consoleFont)));
      panel.style.setProperty('--c-font', `${size}px`);
      write('llmlab:console-font', size);
    }
  });
}

// ---------- resizable columns & nav collapse ----------
function bindLayout(layout) {
  const saved = read(WIDTH_KEY, {});
  if (saved.nav) layout.style.setProperty('--nav-w', `${saved.nav}px`);
  if (saved.out) layout.style.setProperty('--out-w', `${saved.out}px`);

  layout.querySelectorAll('.gutter').forEach((g) => {
    g.addEventListener('pointerdown', (e) => {
      e.preventDefault();
      g.setPointerCapture(e.pointerId);
      g.classList.add('drag');
      const which = g.dataset.resize;
      const move = (ev) => {
        const r = layout.getBoundingClientRect();
        const w =
          which === 'nav'
            ? Math.min(480, Math.max(200, ev.clientX - r.left))
            : Math.min(760, Math.max(280, r.right - ev.clientX));
        layout.style.setProperty(which === 'nav' ? '--nav-w' : '--out-w', `${w}px`);
        saved[which] = Math.round(w);
      };
      const up = () => {
        g.classList.remove('drag');
        g.removeEventListener('pointermove', move);
        write(WIDTH_KEY, saved);
        window.dispatchEvent(new Event('resize'));
      };
      g.addEventListener('pointermove', move);
      g.addEventListener('pointerup', up, { once: true });
    });
  });

  const mobile = matchMedia('(max-width: 860px)');
  layout.querySelector('[data-nav-close]')?.addEventListener('click', () => {
    if (mobile.matches) layout.classList.remove('nav-open');
    else layout.classList.add('nav-collapsed');
  });
  layout.querySelector('[data-nav-open]')?.addEventListener('click', () => {
    if (mobile.matches) layout.classList.toggle('nav-open');
    else layout.classList.remove('nav-collapsed');
  });
  layout.querySelector('.nav-tree')?.addEventListener('click', (e) => {
    if (mobile.matches && e.target.closest('a')) layout.classList.remove('nav-open');
  });
}

// ---------- student: highlight the section in view ----------
function bindSectionSpy(main) {
  const links = new Map([...document.querySelectorAll('[data-sec]')].map((a) => [a.dataset.sec, a]));
  if (!links.size) return;
  const root = matchMedia('(max-width: 860px)').matches ? null : main.querySelector('.content');
  const io = new IntersectionObserver(
    (entries) => {
      entries.forEach((en) => {
        if (!en.isIntersecting) return;
        links.forEach((a) => a.removeAttribute('aria-current'));
        links.get(en.target.id)?.setAttribute('aria-current', 'true');
      });
    },
    { root, rootMargin: '0px 0px -70% 0px' },
  );
  links.forEach((_, id) => {
    const s = document.getElementById(id);
    if (s) io.observe(s);
  });
}

// ---------- progress (done toggle, cards, prev/next) ----------
function bindProgress() {
  const box = document.querySelector('[data-done-toggle]');
  if (box) {
    const no = Number(box.dataset.doneToggle);
    box.checked = doneWeeks().has(no);
    box.addEventListener('change', () => {
      const set = doneWeeks();
      if (box.checked) set.add(no);
      else set.delete(no);
      write(DONE_KEY, [...set].sort((a, b) => a - b));
    });
  }
  const done = doneWeeks();
  document.querySelectorAll('[data-week-progress]').forEach((el) => {
    el.style.width = done.has(Number(el.dataset.weekProgress)) ? '100%' : '0';
  });
}

function bindWeekNav() {
  const nav = document.querySelector('[data-week-nav]');
  if (!nav) return;
  const no = Number(nav.dataset.weekNav);
  const role = nav.dataset.role ?? 'student';
  const ready = WEEKS.filter((w) => w.ready);
  const prev = [...ready].reverse().find((w) => w.no < no);
  const next = ready.find((w) => w.no > no);
  const link = (w, cls, label) =>
    w
      ? `<a class="${cls}" href="${weekUrl(w.no, role)}"><small>${label}</small>${w.no}주차 · ${esc(w.title)}</a>`
      : `<span class="${cls}"><small>${label}</small>${cls === 'prev' ? '처음 공개된 주차' : '다음 주차 준비 중'}</span>`;
  nav.innerHTML = link(prev, 'prev', '← 이전') + link(next, 'next', '다음 →');
}

// ---------- boot ----------
function init() {
  applyTheme(read(THEME_KEY, 'system'));
  const layout = document.querySelector('[data-layout]');
  if (!layout) return;
  const main = layout.querySelector('main');
  const role = layout.dataset.role ?? 'student';
  const week = layout.dataset.week ? Number(layout.dataset.week) : null;
  const crumb = layout.dataset.crumb ?? '';
  const sections = role === 'student' ? sectionsOf(main) : [];

  document.body.classList.add(`role-${role}`);
  buildNav(layout, { role, week, sections });
  buildTopbar(main, { role, week, crumb });
  buildOutput(layout);
  bindLayout(layout);
  bindSectionSpy(main);
  bindProgress();
  bindWeekNav();
}

init();
