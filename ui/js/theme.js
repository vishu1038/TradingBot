// Colour helpers. Green / red are RESERVED for money (PnL, candles, win/loss); every other thing gets
// its own hue: panels have accents, each coin has one fixed colour, probabilities run cool -> hot.

export const ACCENT = {
  coral: '#D97757',
  violet: '#a78bfa',
  cyan: '#4fe3ff',
  amber: '#ffb020',
  pink: '#ff7ac6',
  sky: '#5aa9ff',
  teal: '#3fd8c8',
  yellow: '#ffd84f',
  magenta: '#ff4fd8',
};

// hues that avoid green (90-170) and red (340-15): orange..yellow, cyan..indigo, violet..magenta
const COIN_HUES = [28, 44, 56, 184, 200, 216, 236, 256, 274, 292, 306];
const FIXED = {
  BTCUSDT: '#ff9f2e',
  ETHUSDT: '#8f9bff',
  SOLUSDT: '#b57bff',
  BNBUSDT: '#ffd84f',
  DOGEUSDT: '#ffc36b',
  XRPUSDT: '#5aa9ff',
  SUIUSDT: '#4fe3ff',
  WIFUSDT: '#ff8fe0',
  '1000PEPEUSDT': '#a0e0ff',
  TAOUSDT: '#c8a2ff',
  ENAUSDT: '#7f8cff',
};

function hslToHex(h, s, l) {
  s /= 100;
  l /= 100;
  const k = (n) => (n + h / 30) % 12;
  const a = s * Math.min(l, 1 - l);
  const f = (n) => l - a * Math.max(-1, Math.min(k(n) - 3, Math.min(9 - k(n), 1)));
  const x = (v) => Math.round(v * 255).toString(16).padStart(2, '0');
  return `#${x(f(0))}${x(f(8))}${x(f(4))}`;
}

const coinCache = new Map();
/** Fixed colour per coin, identical everywhere (strip, tables, chart legend, brain bars). */
export function coinColor(sym) {
  const key = String(sym || '');
  if (coinCache.has(key)) return coinCache.get(key);
  let c = FIXED[key];
  if (!c) {
    let h = 2166136261;
    for (let i = 0; i < key.length; i++) {
      h ^= key.charCodeAt(i);
      h = Math.imul(h, 16777619);
    }
    c = hslToHex(COIN_HUES[(h >>> 0) % COIN_HUES.length], 92, 66);
  }
  coinCache.set(key, c);
  return c;
}

const PROB_STOPS = [
  [0.0, [79, 124, 255]], // cool blue
  [0.35, [155, 107, 255]], // violet
  [0.6, [255, 79, 216]], // magenta
  [0.8, [255, 154, 61]], // orange
  [1.0, [255, 225, 79]], // hot yellow
];

/** Probability 0..1 on a cool -> hot gradient, as #rrggbb. */
export function probColor(p) {
  const v = Math.max(0, Math.min(1, Number.isFinite(p) ? p : 0));
  for (let i = 1; i < PROB_STOPS.length; i++) {
    const [b, cb] = PROB_STOPS[i];
    if (v <= b) {
      const [a, ca] = PROB_STOPS[i - 1];
      const t = (v - a) / (b - a || 1);
      const m = (k) => Math.round(ca[k] + (cb[k] - ca[k]) * t);
      return `#${[m(0), m(1), m(2)].map((x) => x.toString(16).padStart(2, '0')).join('')}`;
    }
  }
  return '#ffe14f';
}

/** CSS gradient string for a probability bar. */
export const PROB_GRADIENT = `linear-gradient(90deg, ${PROB_STOPS.map(([p, c]) => `rgb(${c.join(',')}) ${p * 100}%`).join(', ')})`;
