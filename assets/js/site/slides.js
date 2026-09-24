// PPT-style slide deck for teacher pages (same controls as the sibling course sites).
// Slides are authored in HTML as <section class="slide" data-title="…"> with an
// optional <aside class="notes"> (teacher notes, shown in the notes pane only).
//
// Keys: ← → / Space / PageUp PageDown · Home End · F fullscreen · N notes
//       B blackout · T timer · P presenter window
//       판서: D pen · H highlighter · L laser · E eraser · Z undo · C clear · Esc stop
// A second window opened with P (?presenter) stays in sync via BroadcastChannel
// (slide position and ink strokes).

import { createInk } from './ink.js';

const W = 1280;
const H = 720;

export function initDeck(root = document.querySelector('[data-deck]')) {
  if (!root) return;
  const isPresenter = new URLSearchParams(location.search).has('presenter');
  document.body.classList.toggle('is-presenter', isPresenter);

  const wrap = root.querySelector('.deck-wrap');
  const viewport = root.querySelector('.deck-viewport');
  const stage = root.querySelector('.deck-stage');
  const slides = [...stage.querySelectorAll(':scope > .slide')];
  const notesPane = root.querySelector('.notes-pane');
  const notesBody = notesPane.querySelector('.notes-body');
  const notesNext = notesPane.querySelector('.notes-next');
  const progress = root.querySelector('.deck-progress i');
  const deckTitle = root.dataset.deckTitle ?? document.title;
  const channel = 'BroadcastChannel' in window ? new BroadcastChannel(`llmlab-deck:${location.pathname}`) : null;
  const label = (s) => s.dataset.title ?? s.querySelector('h1,h2')?.textContent ?? '';

  // ---------- toolbar ----------
  const bar = root.querySelector('[data-deck-controls]');
  bar.innerHTML = `
    <button type="button" class="dk-btn" data-act="first" title="처음 슬라이드 (Home)">⏮<span class="lbl">처음</span></button>
    <button type="button" class="dk-btn" data-act="prev" title="이전 슬라이드 (←)" aria-label="이전 슬라이드">◀</button>
    <span class="s-count" aria-live="polite"></span>
    <button type="button" class="dk-btn" data-act="next" title="다음 슬라이드 (→, Space)" aria-label="다음 슬라이드">▶</button>
    <input class="s-slider" type="range" min="1" max="${slides.length}" value="1" step="1" aria-label="슬라이드 이동">
    <span class="s-slide-title"></span>
    <span class="spacer"></span>
    <button type="button" class="timer-btn" data-act="timer" title="수업 타이머 시작/일시정지 (T)" aria-pressed="false">⏱ <span data-timer>00:00</span></button>
    <button type="button" class="dk-btn" data-act="notes" title="교사 노트 (N)">📝<span class="lbl">노트</span></button>
    <button type="button" class="dk-btn" data-act="blackout" title="화면 가리기 (B)">⬛<span class="lbl">가리기</span></button>
    <button type="button" class="dk-btn" data-act="presenter" title="발표자 창 (P)">🖥<span class="lbl">발표자 창</span></button>
    <button type="button" class="dk-btn" data-act="full" title="전체 화면 (F)">⛶<span class="lbl">전체 화면</span></button>`;
  const counter = bar.querySelector('.s-count');
  const slider = bar.querySelector('.s-slider');
  const titleEl = bar.querySelector('.s-slide-title');
  const timerEl = bar.querySelector('[data-timer]');

  viewport.insertAdjacentHTML('beforeend', `
    <button type="button" class="edge edge-prev" data-act="prev" aria-label="이전 슬라이드">‹</button>
    <button type="button" class="edge edge-next" data-act="next" aria-label="다음 슬라이드">›</button>
    <div class="deck-blackout" hidden></div>`);
  const blackout = viewport.querySelector('.deck-blackout');

  // ---------- whiteboard ink (second toolbar row) ----------
  const inkRow = document.createElement('div');
  inkRow.className = 'deck-row ink-row';
  root.querySelector('.deck-bar').append(inkRow);
  const ink = createInk({
    row: inkRow,
    stage,
    onStrokes: (i, strokes) => channel?.postMessage({ type: 'ink', index: i, strokes }),
  });

  slides.forEach((s, i) => {
    if (s.classList.contains('slide--title') || s.classList.contains('slide--demo')) return;
    s.insertAdjacentHTML('beforeend', `<div class="slide-footer" aria-hidden="true"><span>${deckTitle}</span><span class="pg">${i + 1} / ${slides.length}</span></div>`);
  });

  // ---------- slide list in the left nav ----------
  const navSecs = document.querySelector('[data-nav-secs]');
  if (navSecs) {
    navSecs.innerHTML = slides
      .map((s, i) => `<li><button type="button" class="nav-sec" data-go="${i}"><span class="chk">${i + 1}</span><span>${label(s)}</span></button></li>`)
      .join('');
    navSecs.addEventListener('click', (e) => {
      const b = e.target.closest('[data-go]');
      if (b) go(Number(b.dataset.go));
    });
  }

  // ---------- scaling ----------
  const PAD = 14;
  const fit = () => {
    const r = viewport.getBoundingClientRect();
    const pad = document.fullscreenElement ? 0 : PAD;
    const scale = Math.max(0.05, Math.min((r.width - pad * 2) / W, (r.height - pad * 2) / H));
    const x = (r.width - W * scale) / 2;
    const y = (r.height - H * scale) / 2;
    stage.style.transform = `translate(${x}px, ${y}px) scale(${scale})`;
  };
  new ResizeObserver(fit).observe(viewport);

  // ---------- state ----------
  let index = 0;
  const readHash = () => {
    const n = parseInt(location.hash.slice(1), 10);
    return Number.isFinite(n) ? Math.min(Math.max(n - 1, 0), slides.length - 1) : 0;
  };

  function go(i, { broadcast = true } = {}) {
    index = Math.min(Math.max(i, 0), slides.length - 1);
    slides.forEach((s, j) => {
      s.classList.toggle('is-active', j === index);
      s.setAttribute('aria-hidden', j === index ? 'false' : 'true');
      s.inert = j !== index;
    });
    counter.textContent = `${index + 1} / ${slides.length}`;
    slider.value = String(index + 1);
    titleEl.textContent = label(slides[index]);
    progress.style.width = `${((index + 1) / slides.length) * 100}%`;
    history.replaceState(null, '', `${location.search}#${index + 1}`);
    renderNotes();
    ink.setSlide(index);
    navSecs?.querySelectorAll('[data-go]').forEach((b) => b.setAttribute('aria-current', String(Number(b.dataset.go) === index)));
    if (broadcast) channel?.postMessage({ type: 'go', index });
  }

  function renderNotes() {
    const s = slides[index];
    const notes = s.querySelector(':scope > .notes');
    const next = slides[index + 1];
    notesBody.innerHTML = `<h5>${index + 1}. ${label(s)}</h5>${notes ? notes.innerHTML : '<p class="muted">이 슬라이드에는 노트가 없다.</p>'}`;
    notesNext.textContent = next ? `다음 → ${label(next)}` : '마지막 슬라이드';
  }

  // ---------- notes toggle ----------
  const NOTES_KEY = 'llmlab:deck-notes';
  let notesOn = isPresenter || readPref(NOTES_KEY, true);
  const setNotes = (on) => {
    notesOn = on;
    notesPane.classList.toggle('collapsed', !on);
    notesPane.querySelector('.notes-toggle').textContent = on ? '▾ 접기' : '▸ 펼치기';
    bar.querySelector('[data-act=notes]').setAttribute('aria-pressed', String(on));
    if (!isPresenter) writePref(NOTES_KEY, on);
    fit();
  };
  notesPane.querySelector('.notes-head').addEventListener('click', () => setNotes(!notesOn));

  // ---------- timer ----------
  let elapsed = 0;
  let startedAt = null;
  let tick = null;
  const renderTimer = () => {
    const total = Math.floor((elapsed + (startedAt ? Date.now() - startedAt : 0)) / 1000);
    timerEl.textContent = `${String(Math.floor(total / 60)).padStart(2, '0')}:${String(total % 60).padStart(2, '0')}`;
  };
  const toggleTimer = () => {
    if (startedAt) {
      elapsed += Date.now() - startedAt;
      startedAt = null;
      clearInterval(tick);
    } else {
      startedAt = Date.now();
      tick = setInterval(renderTimer, 500);
    }
    bar.querySelector('[data-act=timer]').setAttribute('aria-pressed', String(Boolean(startedAt)));
    renderTimer();
  };

  const toggleBlackout = () => {
    blackout.hidden = !blackout.hidden;
    bar.querySelector('[data-act=blackout]').setAttribute('aria-pressed', String(!blackout.hidden));
  };

  const toggleFull = () => {
    if (document.fullscreenElement) document.exitFullscreen();
    else wrap.requestFullscreen?.();
  };
  document.addEventListener('fullscreenchange', () => {
    bar.querySelector('[data-act=full]').setAttribute('aria-pressed', String(Boolean(document.fullscreenElement)));
    fit();
  });

  const openPresenter = () => {
    const u = new URL(location.href);
    u.searchParams.set('presenter', '');
    window.open(u, 'llmlab-presenter', 'width=1200,height=800');
  };

  const actions = {
    first: () => go(0),
    prev: () => go(index - 1),
    next: () => go(index + 1),
    notes: () => setNotes(!notesOn),
    timer: toggleTimer,
    blackout: toggleBlackout,
    full: toggleFull,
    presenter: openPresenter,
  };

  root.addEventListener('click', (e) => {
    const act = e.target.closest('[data-act]')?.dataset.act;
    // runner buttons inside slides share the data-act attribute; only handle deck actions
    if (Object.hasOwn(actions, act)) actions[act]();
  });
  slider.addEventListener('input', () => go(Number(slider.value) - 1));

  document.addEventListener('keydown', (e) => {
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    // widgets, code boxes and the nav search keep their own keyboard handling
    if (e.target.closest('input, select, textarea, [contenteditable], summary')) return;
    if (ink.handleKey(e)) {
      e.preventDefault();
      return;
    }
    const k = e.key;
    if (k === 'ArrowRight' || k === 'PageDown' || (k === ' ' && !e.target.closest('button, a'))) {
      e.preventDefault();
      go(index + 1);
    } else if (k === 'ArrowLeft' || k === 'PageUp') {
      e.preventDefault();
      go(index - 1);
    } else if (k === 'Home') go(0);
    else if (k === 'End') go(slides.length - 1);
    else if (k === 'f' || k === 'F') toggleFull();
    else if (k === 'n' || k === 'N') setNotes(!notesOn);
    else if (k === 'b' || k === 'B' || k === '.') toggleBlackout();
    else if (k === 't' || k === 'T') toggleTimer();
    else if (k === 'p' || k === 'P') openPresenter();
  });

  // swipe on touch screens
  let touchX = null;
  viewport.addEventListener('touchstart', (e) => { touchX = e.touches[0].clientX; }, { passive: true });
  viewport.addEventListener('touchend', (e) => {
    if (touchX === null || ink.tool !== 'pointer' || e.target.closest('.widget, .code-block')) return;
    const dx = e.changedTouches[0].clientX - touchX;
    if (Math.abs(dx) > 60) go(index + (dx < 0 ? 1 : -1));
    touchX = null;
  });

  channel?.addEventListener('message', (e) => {
    if (e.data?.type === 'go') go(e.data.index, { broadcast: false });
    if (e.data?.type === 'ink') ink.setStrokes(e.data.index, e.data.strokes);
    if (e.data?.type === 'hello' && !isPresenter) channel.postMessage({ type: 'go', index });
  });
  window.addEventListener('hashchange', () => go(readHash()));

  setNotes(notesOn);
  go(readHash(), { broadcast: false });
  if (isPresenter) channel?.postMessage({ type: 'hello' });
  fit();
}

function readPref(key, fallback) {
  try {
    const v = localStorage.getItem(key);
    return v === null ? fallback : JSON.parse(v);
  } catch {
    return fallback;
  }
}

function writePref(key, value) {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch {
    /* ignore */
  }
}
