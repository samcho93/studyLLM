// Whiteboard ink for teacher slides (판서): pen · highlighter · laser · eraser.
// A canvas sits on top of the 1280×720 stage; strokes are stored per slide in
// stage coordinates, so they survive resizing, fullscreen and slide changes.
//
// Keys (handled by slides.js): D pen · H highlighter · L laser · E eraser ·
//       Esc pointer (no tool) · Z undo · C clear this slide

const W = 1280;
const H = 720;
const RES = 2; // backing-store scale so strokes stay sharp when the stage is enlarged

const TOOLS = {
  pointer: { label: '가리키기', icon: '👆', key: 'Esc' },
  pen: { label: '펜', icon: '🖊', key: 'D' },
  marker: { label: '형광펜', icon: '🖍', key: 'H' },
  laser: { label: '레이저', icon: '🔴', key: 'L' },
  eraser: { label: '지우개', icon: '🧽', key: 'E' },
};
const COLORS = ['--ink-1', '--ink-2', '--ink-3', '--ink-4', '--ink-5'];
const COLOR_NAMES = ['빨강', '파랑', '초록', '노랑', '검정'];
const WIDTHS = [3, 6, 12];

/**
 * @param {{ row: HTMLElement, stage: HTMLElement, onStrokes?: (index: number, strokes: object[]) => void }} opts
 */
export function createInk({ row, stage, onStrokes = () => {} }) {
  const canvas = document.createElement('canvas');
  canvas.className = 'ink-canvas';
  canvas.width = W * RES;
  canvas.height = H * RES;
  canvas.setAttribute('aria-hidden', 'true');
  stage.append(canvas);
  const ctx = canvas.getContext('2d');
  ctx.scale(RES, RES);

  const pages = new Map(); // slide index -> strokes[]
  let index = 0;
  let tool = 'pointer';
  let color = 0;
  let width = 1;
  let drawing = null;
  let laserTrail = [];
  let laserFade = null;

  row.innerHTML = `
    <span class="ink-label">✏️<span class="lbl">판서</span></span>
    <span class="ink-group" role="group" aria-label="판서 도구">
      ${Object.entries(TOOLS)
        .map(([k, t]) => `<button type="button" class="dk-btn" data-ink-tool="${k}" title="${t.label} (${t.key})" aria-pressed="false">${t.icon}<span class="lbl">${t.label}</span></button>`)
        .join('')}
    </span>
    <span class="ink-group" role="group" aria-label="펜 색">
      ${COLORS.map((c, i) => `<button type="button" class="ink-color" data-ink-color="${i}" style="--c: var(${c})" title="${COLOR_NAMES[i]}" aria-label="펜 색 ${COLOR_NAMES[i]}" aria-pressed="false"></button>`).join('')}
    </span>
    <span class="ink-group" role="group" aria-label="펜 굵기">
      ${WIDTHS.map((w, i) => `<button type="button" class="ink-width" data-ink-width="${i}" title="굵기 ${i + 1}" aria-label="펜 굵기 ${i + 1}" aria-pressed="false"><i style="width:${w + 2}px;height:${w + 2}px"></i></button>`).join('')}
    </span>
    <span class="ink-group">
      <button type="button" class="dk-btn" data-ink-act="undo" title="되돌리기 (Z)">↶<span class="lbl">되돌리기</span></button>
      <button type="button" class="dk-btn" data-ink-act="clear" title="이 슬라이드 판서 지우기 (C)">🗑<span class="lbl">지우기</span></button>
    </span>`;

  const cssColor = (i) => getComputedStyle(stage).getPropertyValue(COLORS[i]).trim() || '#e53935';
  const laserColor = () => getComputedStyle(stage).getPropertyValue('--ink-laser').trim() || '#ff3b30';
  const strokes = () => {
    if (!pages.has(index)) pages.set(index, []);
    return pages.get(index);
  };

  function syncButtons() {
    row.querySelectorAll('[data-ink-tool]').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.inkTool === tool)));
    row.querySelectorAll('[data-ink-color]').forEach((b) => b.setAttribute('aria-pressed', String(Number(b.dataset.inkColor) === color)));
    row.querySelectorAll('[data-ink-width]').forEach((b) => b.setAttribute('aria-pressed', String(Number(b.dataset.inkWidth) === width)));
    stage.dataset.tool = tool;
    stage.classList.toggle('inking', tool !== 'pointer');
  }

  function drawStroke(s) {
    if (s.points.length === 0) return;
    ctx.save();
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.strokeStyle = s.color;
    ctx.lineWidth = s.width;
    if (s.tool === 'marker') {
      ctx.globalAlpha = 0.35;
      ctx.lineCap = 'butt';
    }
    ctx.beginPath();
    const [x0, y0] = s.points[0];
    ctx.moveTo(x0, y0);
    if (s.points.length === 1) ctx.lineTo(x0 + 0.1, y0 + 0.1);
    // smooth with quadratic curves through midpoints
    for (let i = 1; i < s.points.length - 1; i++) {
      const [x1, y1] = s.points[i];
      const [x2, y2] = s.points[i + 1];
      ctx.quadraticCurveTo(x1, y1, (x1 + x2) / 2, (y1 + y2) / 2);
    }
    if (s.points.length > 1) {
      const [xl, yl] = s.points[s.points.length - 1];
      ctx.lineTo(xl, yl);
    }
    ctx.stroke();
    ctx.restore();
  }

  function redraw() {
    ctx.clearRect(0, 0, W, H);
    strokes().forEach(drawStroke);
    if (drawing && drawing.tool !== 'eraser') drawStroke(drawing);
    if (laserTrail.length) {
      const now = performance.now();
      laserTrail = laserTrail.filter((p) => now - p.t < 600);
      ctx.save();
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';
      for (let i = 1; i < laserTrail.length; i++) {
        const a = laserTrail[i - 1];
        const b = laserTrail[i];
        ctx.globalAlpha = Math.max(0, 1 - (now - b.t) / 600);
        ctx.strokeStyle = laserColor();
        ctx.lineWidth = 6;
        ctx.beginPath();
        ctx.moveTo(a.x, a.y);
        ctx.lineTo(b.x, b.y);
        ctx.stroke();
      }
      const last = laserTrail[laserTrail.length - 1];
      if (last) {
        ctx.globalAlpha = 1;
        ctx.fillStyle = laserColor();
        ctx.shadowColor = laserColor();
        ctx.shadowBlur = 16;
        ctx.beginPath();
        ctx.arc(last.x, last.y, 7, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.restore();
    }
  }

  function loopLaser() {
    redraw();
    if (laserTrail.length) laserFade = requestAnimationFrame(loopLaser);
    else laserFade = null;
  }

  // pointer → stage coordinates (the stage is CSS-scaled, so divide by the scale)
  function toStage(e) {
    const r = canvas.getBoundingClientRect();
    return [((e.clientX - r.left) / r.width) * W, ((e.clientY - r.top) / r.height) * H];
  }

  // distance from (x, y) to the segment a–b
  function segDist(x, y, [ax, ay], [bx, by]) {
    const dx = bx - ax;
    const dy = by - ay;
    const len2 = dx * dx + dy * dy;
    const t = len2 ? Math.max(0, Math.min(1, ((x - ax) * dx + (y - ay) * dy) / len2)) : 0;
    return Math.hypot(x - (ax + t * dx), y - (ay + t * dy));
  }

  function hits(s, x, y, radius) {
    const r = radius + s.width / 2;
    if (s.points.length === 1) return segDist(x, y, s.points[0], s.points[0]) < r;
    for (let i = 1; i < s.points.length; i++) {
      if (segDist(x, y, s.points[i - 1], s.points[i]) < r) return true;
    }
    return false;
  }

  function eraseAt(x, y) {
    const list = strokes();
    const radius = 14;
    const keep = list.filter((s) => !hits(s, x, y, radius));
    if (keep.length !== list.length) {
      pages.set(index, keep);
      redraw();
      return true;
    }
    return false;
  }

  canvas.addEventListener('pointerdown', (e) => {
    if (tool === 'pointer') return;
    e.preventDefault();
    canvas.setPointerCapture(e.pointerId);
    const [x, y] = toStage(e);
    if (tool === 'laser') {
      laserTrail.push({ x, y, t: performance.now() });
      if (!laserFade) loopLaser();
      drawing = { tool: 'laser' };
      return;
    }
    if (tool === 'eraser') {
      drawing = { tool: 'eraser', changed: eraseAt(x, y) };
      return;
    }
    drawing = {
      tool,
      color: cssColor(color),
      width: tool === 'marker' ? WIDTHS[width] * 3 + 8 : WIDTHS[width],
      points: [[x, y]],
    };
    redraw();
  });

  canvas.addEventListener('pointermove', (e) => {
    if (tool === 'laser') {
      // the laser also follows the pointer without pressing, like a real pointer
      const [x, y] = toStage(e);
      laserTrail.push({ x, y, t: performance.now() });
      if (!laserFade) loopLaser();
      return;
    }
    if (!drawing) return;
    const [x, y] = toStage(e);
    if (drawing.tool === 'eraser') {
      drawing.changed = eraseAt(x, y) || drawing.changed;
      return;
    }
    const last = drawing.points[drawing.points.length - 1];
    if ((last[0] - x) ** 2 + (last[1] - y) ** 2 < 2) return;
    drawing.points.push([x, y]);
    redraw();
  });

  const endStroke = () => {
    if (!drawing) return;
    if (drawing.tool === 'pen' || drawing.tool === 'marker') {
      strokes().push(drawing);
      onStrokes(index, strokes());
    } else if (drawing.tool === 'eraser' && drawing.changed) {
      onStrokes(index, strokes());
    }
    drawing = null;
    redraw();
  };
  canvas.addEventListener('pointerup', endStroke);
  canvas.addEventListener('pointercancel', endStroke);
  canvas.addEventListener('pointerleave', () => {
    if (tool === 'laser') laserTrail = [];
  });

  function setTool(t) {
    tool = TOOLS[t] ? t : 'pointer';
    laserTrail = [];
    syncButtons();
    redraw();
  }

  function undo() {
    const list = strokes();
    if (list.pop()) {
      onStrokes(index, list);
      redraw();
    }
  }

  function clear() {
    if (!strokes().length) return;
    pages.set(index, []);
    onStrokes(index, []);
    redraw();
  }

  row.addEventListener('click', (e) => {
    const t = e.target.closest('[data-ink-tool]');
    const c = e.target.closest('[data-ink-color]');
    const w = e.target.closest('[data-ink-width]');
    const a = e.target.closest('[data-ink-act]');
    if (t) setTool(t.dataset.inkTool === tool && tool !== 'pointer' ? 'pointer' : t.dataset.inkTool);
    if (c) {
      color = Number(c.dataset.inkColor);
      if (tool === 'pointer' || tool === 'eraser' || tool === 'laser') setTool('pen');
      else syncButtons();
    }
    if (w) {
      width = Number(w.dataset.inkWidth);
      syncButtons();
    }
    if (a?.dataset.inkAct === 'undo') undo();
    if (a?.dataset.inkAct === 'clear') clear();
  });

  syncButtons();

  return {
    /** Show the strokes of another slide. */
    setSlide(i) {
      index = i;
      drawing = null;
      laserTrail = [];
      redraw();
    },
    /** Replace strokes for a slide (e.g. from the presenter window). */
    setStrokes(i, list) {
      pages.set(i, list);
      if (i === index) redraw();
    },
    get tool() {
      return tool;
    },
    /** Returns true when the key was handled. */
    handleKey(e) {
      const k = e.key.toLowerCase();
      if (k === 'd') setTool(tool === 'pen' ? 'pointer' : 'pen');
      else if (k === 'h') setTool(tool === 'marker' ? 'pointer' : 'marker');
      else if (k === 'l') setTool(tool === 'laser' ? 'pointer' : 'laser');
      else if (k === 'e') setTool(tool === 'eraser' ? 'pointer' : 'eraser');
      else if (k === 'z') undo();
      else if (k === 'c') clear();
      else if (e.key === 'Escape' && tool !== 'pointer') setTool('pointer');
      else return false;
      return true;
    },
  };
}
