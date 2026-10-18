// WebAudio synthesized blips. Default OFF; preference kept in localStorage (try/catch everywhere).

const KEY = 'tb.sound';
let ctx = null;
let enabled = false;
let explicit = false; // user has made a choice (localStorage) -> beats the server `sound` setting

try {
  const v = localStorage.getItem(KEY);
  if (v === '1' || v === '0') {
    enabled = v === '1';
    explicit = true;
  }
} catch (_) {
  /* storage blocked */
}

function ensureCtx() {
  if (ctx) return ctx;
  const AC = typeof window !== 'undefined' && (window.AudioContext || window.webkitAudioContext);
  if (!AC) return null;
  try {
    ctx = new AC();
  } catch (_) {
    ctx = null;
  }
  return ctx;
}

export function isEnabled() {
  return enabled;
}

export function hasExplicitChoice() {
  return explicit;
}

export function setEnabled(on, { persist = true } = {}) {
  enabled = !!on;
  if (persist) {
    explicit = true;
    try {
      localStorage.setItem(KEY, enabled ? '1' : '0');
    } catch (_) {
      /* ignore */
    }
  }
  if (enabled) {
    const c = ensureCtx();
    if (c && c.state === 'suspended') c.resume().catch(() => {});
    blip([{ f: 660, d: 0.06 }, { f: 990, d: 0.08 }], 'square', 0.05);
  }
}

/** Adopt the server-side setting only if the user never chose locally. */
export function adoptServerSetting(on) {
  if (!explicit) enabled = !!on;
}

/** Resume the context on any user gesture (autoplay policy). */
export function unlockOnGesture() {
  if (typeof window === 'undefined') return;
  const go = () => {
    if (enabled) {
      const c = ensureCtx();
      if (c && c.state === 'suspended') c.resume().catch(() => {});
    }
  };
  window.addEventListener('pointerdown', go, { passive: true });
  window.addEventListener('keydown', go, { passive: true });
}

function blip(notes, type = 'square', gain = 0.06) {
  const c = ensureCtx();
  if (!c) return;
  if (c.state === 'suspended') c.resume().catch(() => {});
  let t = c.currentTime + 0.01;
  for (const n of notes) {
    const o = c.createOscillator();
    const g = c.createGain();
    o.type = n.type || type;
    o.frequency.setValueAtTime(n.f, t);
    if (n.slideTo) o.frequency.exponentialRampToValueAtTime(n.slideTo, t + n.d);
    g.gain.setValueAtTime(0.0001, t);
    g.gain.exponentialRampToValueAtTime(gain, t + 0.01);
    g.gain.exponentialRampToValueAtTime(0.0001, t + n.d);
    o.connect(g).connect(c.destination);
    o.start(t);
    o.stop(t + n.d + 0.02);
    t += n.d * 0.9;
  }
}

const SOUNDS = {
  open: () => blip([{ f: 520, d: 0.05 }, { f: 780, d: 0.07 }], 'square', 0.04),
  tp: () => blip([{ f: 523, d: 0.07 }, { f: 659, d: 0.07 }, { f: 784, d: 0.07 }, { f: 1047, d: 0.16 }], 'triangle', 0.09),
  sl: () => blip([{ f: 330, d: 0.12, slideTo: 140 }, { f: 160, d: 0.22, slideTo: 70 }], 'sawtooth', 0.07),
  kill: () => blip([{ f: 880, d: 0.15 }, { f: 440, d: 0.15 }, { f: 880, d: 0.15 }, { f: 440, d: 0.3 }], 'sawtooth', 0.09),
  click: () => blip([{ f: 900, d: 0.03 }], 'square', 0.03),
};

export function play(name) {
  if (!enabled) return;
  try {
    SOUNDS[name]?.();
  } catch (_) {
    /* never let audio break trading UI */
  }
}
