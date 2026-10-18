// Visual effects. One "effects" slider (calm <-> degen, default "loud but readable") controls them all:
//   calm  (0-24)  : nothing animates, no flashes
//   loud  (25-70) : screen flash + glitch on TP / SL / kill only (the default, 50)
//   degen (71-100): + idle logo glitch, scanlines, animated ASCII spinners, tick flashes, typed log lines
// prefers-reduced-motion always forces "calm". The choice is kept in localStorage (try/catch).

const KEY = 'tb.fx';
const mq = typeof window !== 'undefined' && window.matchMedia ? window.matchMedia('(prefers-reduced-motion: reduce)') : null;
let level = 50;
try {
  const v = parseInt(localStorage.getItem(KEY), 10);
  if (Number.isFinite(v)) level = Math.max(0, Math.min(100, v));
} catch (_) {
  /* storage blocked */
}

export const reducedMotion = () => !!(mq && mq.matches);
export const getLevel = () => level;
export function mode() {
  if (reducedMotion() || level < 25) return 'calm';
  return level > 70 ? 'degen' : 'loud';
}
export const modeLabel = () => ({ calm: 'calm', loud: 'loud but readable', degen: 'full degen' })[mode()];

function apply() {
  if (typeof document === 'undefined') return;
  document.documentElement.dataset.fx = mode();
  document.documentElement.style.setProperty('--fx', String(level / 100));
}

export function setLevel(v, { persist = true } = {}) {
  level = Math.max(0, Math.min(100, Math.round(v)));
  if (persist) {
    try {
      localStorage.setItem(KEY, String(level));
    } catch (_) {
      /* ignore */
    }
  }
  apply();
}
if (mq && mq.addEventListener) mq.addEventListener('change', apply);
apply();

let flashEl = null;
/** kind: 'tp' | 'sl' | 'kill' | 'open' */
export function flash(kind) {
  if (mode() === 'calm') return;
  flashEl = flashEl || document.getElementById('fx-flash');
  if (!flashEl) return;
  flashEl.className = '';
  void flashEl.offsetWidth; // restart animation
  flashEl.className = `flash-${kind}`;
}

/** Add the .glitch class to an element for a short burst. */
export function glitch(el, ms = 700) {
  if (mode() === 'calm' || !el) return;
  el.classList.remove('glitch');
  void el.offsetWidth;
  el.classList.add('glitch');
  clearTimeout(el._glitchT);
  el._glitchT = setTimeout(() => el.classList.remove('glitch'), ms);
}

/** Retrigger a one-shot CSS animation class (e.g. a number "pop"). Degen+loud only. */
export function pulse(el, cls = 'pulse', ms = 600) {
  if (mode() === 'calm' || !el) return;
  el.classList.remove(cls);
  void el.offsetWidth;
  el.classList.add(cls);
  clearTimeout(el['_t_' + cls]);
  el['_t_' + cls] = setTimeout(() => el.classList.remove(cls), ms);
}

/** Idle glitch on the logo (degen only). */
export function startIdleGlitch(el) {
  if (!el) return;
  const loop = () => {
    if (mode() === 'degen') glitch(el, 450);
    setTimeout(loop, 6000 + Math.random() * 9000);
  };
  setTimeout(loop, 4000);
}

// ---------------------------------------------------------------- ASCII spinners
const FRAMES = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];
let spinN = 0;
if (typeof setInterval === 'function' && typeof document !== 'undefined') {
  setInterval(() => {
    const nodes = document.querySelectorAll('[data-spin]');
    if (!nodes.length) return;
    const f = mode() === 'degen' ? FRAMES[spinN++ % FRAMES.length] : '◆';
    for (const n of nodes) if (n.textContent !== f) n.textContent = f;
  }, 110);
}
/** HTML for a spinner glyph (decorative -> mono). */
export const spinner = () => '<span class="spin deco" data-spin>◆</span>';

// ---------------------------------------------------------------- typed text
/** Reveal `text` into el like a terminal. Instant unless degen. */
export function typeInto(el, text, cps = 90) {
  el.textContent = '';
  if (mode() !== 'degen' || text.length < 4) {
    el.textContent = text;
    return;
  }
  let i = 0;
  const step = Math.max(1, Math.round(cps / 30));
  const t = setInterval(() => {
    i += step;
    el.textContent = text.slice(0, i);
    if (i >= text.length || !el.isConnected) {
      el.textContent = text;
      clearInterval(t);
    }
  }, 33);
}

/** Toast notifications. */
export function toast(msg, kind = 'info', ms = 3500) {
  if (typeof document === 'undefined') return;
  const host = document.getElementById('toasts');
  if (!host) return;
  const d = document.createElement('div');
  d.className = `toast toast-${kind}`;
  d.textContent = msg; // textContent: never HTML
  host.appendChild(d);
  setTimeout(() => d.classList.add('out'), ms);
  setTimeout(() => d.remove(), ms + 400);
}
