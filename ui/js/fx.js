// Visual effects: full-screen flash on fills, glitch bursts. All no-ops under prefers-reduced-motion.

const mq = typeof window !== 'undefined' && window.matchMedia ? window.matchMedia('(prefers-reduced-motion: reduce)') : null;
export const reducedMotion = () => !!(mq && mq.matches);

let flashEl = null;

function ensureFlash() {
  if (flashEl || typeof document === 'undefined') return flashEl;
  flashEl = document.getElementById('fx-flash');
  return flashEl;
}

/** kind: 'tp' | 'sl' | 'kill' | 'open' */
export function flash(kind) {
  if (reducedMotion()) return;
  const el = ensureFlash();
  if (!el) return;
  el.className = '';
  void el.offsetWidth; // restart animation
  el.className = `flash-${kind}`;
}

/** Add the .glitch class to an element for a short burst. */
export function glitch(el, ms = 700) {
  if (reducedMotion() || !el) return;
  el.classList.remove('glitch');
  void el.offsetWidth;
  el.classList.add('glitch');
  clearTimeout(el._glitchT);
  el._glitchT = setTimeout(() => el.classList.remove('glitch'), ms);
}

/** Retrigger a one-shot CSS animation class (e.g. a number "pop"). */
export function pulse(el, cls = 'pulse', ms = 600) {
  if (reducedMotion() || !el) return;
  el.classList.remove(cls);
  void el.offsetWidth;
  el.classList.add(cls);
  clearTimeout(el['_t_' + cls]);
  el['_t_' + cls] = setTimeout(() => el.classList.remove(cls), ms);
}

/** Periodic idle glitch on the logo. */
export function startIdleGlitch(el) {
  if (!el) return;
  const loop = () => {
    glitch(el, 450);
    setTimeout(loop, 6000 + Math.random() * 9000);
  };
  setTimeout(loop, 4000);
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
