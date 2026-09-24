// Right-hand "실행 결과" terminal (built by site.js).
//  - Live widget output is pinned at the top as a page-themed card.
//  - Code runs append to a chronological log with a separator line, like the
//    console in the sibling course sites.
// Below 860px the terminal is hidden: output renders inline next to its source.

const mq = matchMedia('(max-width: 860px)');
const pinned = new Map();

const $ = (sel) => document.querySelector(sel);

function syncWelcome() {
  const welcome = $('[data-console-welcome]');
  if (!welcome) return;
  const hasPinned = $('[data-console-pinned]')?.children.length > 0;
  const hasLog = $('[data-console-log]')?.children.length > 0;
  welcome.hidden = hasPinned || hasLog;
}

function placePinned(entry) {
  const host = $('[data-console-pinned]');
  if (!mq.matches && host) {
    entry.body.append(entry.node);
    if (!entry.card.isConnected) host.append(entry.card);
  } else {
    entry.inlineHost.append(entry.node);
    entry.card.remove();
  }
  syncWelcome();
}

/**
 * Pin a live output node (e.g. a widget's result view) to the terminal.
 * @param {string} id
 * @param {{ title: string, node: HTMLElement, inlineHost: HTMLElement }} opts
 */
export function registerOutput(id, { title, node, inlineHost }) {
  let entry = pinned.get(id);
  if (!entry) {
    const card = document.createElement('section');
    card.className = 'rich-widget';
    card.innerHTML = '<div class="rich-widget__head"></div><div class="rich-widget__body"></div>';
    entry = { card, body: card.querySelector('.rich-widget__body') };
    pinned.set(id, entry);
  }
  entry.card.querySelector('.rich-widget__head').textContent = title;
  entry.node = node;
  entry.inlineHost = inlineHost;
  entry.body.replaceChildren();
  placePinned(entry);
  return entry;
}

export function unregisterOutput(id) {
  const entry = pinned.get(id);
  if (!entry) return;
  entry.card.remove();
  entry.node.remove();
  pinned.delete(id);
  syncWelcome();
}

/** Scroll a pinned output into view and flash it. */
export function focusOutput(id) {
  const entry = pinned.get(id);
  if (!entry || mq.matches) return;
  entry.card.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
  entry.card.classList.add('is-flash');
  setTimeout(() => entry.card.classList.remove('is-flash'), 900);
}

const STATE_LABEL = { idle: '대기', running: '실행 중', done: '완료', error: '오류' };

export function setRunState(state) {
  const el = $('[data-run-state]');
  if (!el) return;
  el.className = `run-state ${state === 'idle' ? '' : state}`.trim();
  el.textContent = STATE_LABEL[state] ?? state;
}

/**
 * Start a new run block. Returns the element to write output lines into.
 * @param {string} title
 * @param {HTMLElement} inlineHost used on narrow screens (replaced on every run)
 */
export function startRun(title, inlineHost) {
  const out = document.createElement('div');
  const log = $('[data-console-log]');
  if (!mq.matches && log) {
    const time = new Date().toLocaleTimeString('en-GB', { hour12: false });
    log.insertAdjacentHTML('beforeend', `<span class="run-sep">── ${escapeHtml(title)} · ${time} ──</span>`);
    log.append(out);
    syncWelcome();
    const scroller = $('[data-console]');
    requestAnimationFrame(() => scroller?.scrollTo({ top: scroller.scrollHeight, behavior: 'smooth' }));
  } else {
    inlineHost.replaceChildren(out);
  }
  setRunState('running');
  return out;
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
}

mq.addEventListener('change', () => pinned.forEach(placePinned));

document.addEventListener('click', (e) => {
  if (!e.target.closest('[data-console-clear]')) return;
  $('[data-console-log]')?.replaceChildren();
  setRunState('idle');
  syncWelcome();
});
