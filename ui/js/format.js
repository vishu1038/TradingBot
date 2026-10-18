// Formatting + escaping helpers. Pure functions, no DOM access at import time.

const ESC = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' };

/** Escape any untrusted text before putting it in innerHTML. */
export function esc(v) {
  return String(v ?? '').replace(/[&<>"']/g, (c) => ESC[c]);
}

export const isNum = (v) => typeof v === 'number' && Number.isFinite(v);

export function clamp(v, lo, hi) {
  return Math.min(hi, Math.max(lo, v));
}

/** Short symbol: BTCUSDT -> BTC */
export function shortSym(sym) {
  return String(sym ?? '').replace(/USDT$/, '');
}

/** Decimals needed to show a price sensibly. */
export function pricePrecision(p) {
  const a = Math.abs(Number(p));
  if (!isNum(a) || a === 0) return 2;
  if (a >= 100) return 2;
  if (a >= 10) return 3;
  if (a >= 1) return 4;
  if (a >= 0.1) return 5;
  if (a >= 0.01) return 6;
  return 7;
}

export function fmtPrice(p) {
  if (!isNum(p)) return '–';
  const dp = pricePrecision(p);
  return p.toLocaleString('en-US', { minimumFractionDigits: dp, maximumFractionDigits: dp });
}

export function fmtUsd(v, dp = 2, sign = false) {
  if (!isNum(v)) return '–';
  const s = Math.abs(v).toLocaleString('en-US', { minimumFractionDigits: dp, maximumFractionDigits: dp });
  const pre = v < 0 ? '-' : sign && v > 0 ? '+' : '';
  return `${pre}$${s}`;
}

/** Compact USD: 1.23B / 456.7M / 12.3K */
export function fmtCompact(v) {
  if (!isNum(v)) return '–';
  const a = Math.abs(v);
  const sgn = v < 0 ? '-' : '';
  if (a >= 1e9) return `${sgn}$${(a / 1e9).toFixed(2)}B`;
  if (a >= 1e6) return `${sgn}$${(a / 1e6).toFixed(1)}M`;
  if (a >= 1e3) return `${sgn}$${(a / 1e3).toFixed(1)}K`;
  return `${sgn}$${a.toFixed(0)}`;
}

export function fmtPct(v, dp = 2, sign = true) {
  if (!isNum(v)) return '–';
  const pre = v > 0 && sign ? '+' : '';
  return `${pre}${v.toFixed(dp)}%`;
}

/** funding rate is a fraction (0.0001 = 0.01%) */
export function fmtFunding(v) {
  if (!isNum(v)) return '–';
  const pct = v * 100;
  return `${pct > 0 ? '+' : ''}${pct.toFixed(4)}%`;
}

export function fmtNum(v, dp = 2) {
  if (!isNum(v)) return '–';
  return v.toLocaleString('en-US', { minimumFractionDigits: dp, maximumFractionDigits: dp });
}

export function fmtQty(q) {
  if (!isNum(q)) return '–';
  if (q >= 1000) return q.toFixed(0);
  if (q >= 10) return q.toFixed(2);
  if (q >= 1) return q.toFixed(3);
  return q.toFixed(5);
}

const pad = (n) => String(n).padStart(2, '0');

/** unix seconds -> HH:MM:SS (local) */
export function fmtClock(t) {
  if (!isNum(t)) return '--:--:--';
  const d = new Date(t * 1000);
  return `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
}

/** unix seconds -> MM-DD HH:MM (local) */
export function fmtDateTime(t) {
  if (!isNum(t)) return '–';
  const d = new Date(t * 1000);
  return `${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`;
}

export function fmtDate(t) {
  if (!isNum(t)) return '–';
  const d = new Date(t * 1000);
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`;
}

/** seconds -> 1h 05m / 4m 12s */
export function fmtDuration(s) {
  if (!isNum(s) || s < 0) return '–';
  s = Math.round(s);
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = s % 60;
  if (h > 0) return `${h}h ${pad(m)}m`;
  if (m > 0) return `${m}m ${pad(sec)}s`;
  return `${sec}s`;
}

/** CSS class for sign colouring. */
export function signClass(v) {
  if (!isNum(v) || v === 0) return 'flat';
  return v > 0 ? 'pos' : 'neg';
}

/** Generic key/value formatter for lab report summaries. Returns ESCAPED text. */
export function fmtSummaryValue(key, v) {
  if (typeof v === 'boolean') return v ? 'yes' : 'no';
  if (!isNum(v)) return v == null ? '–' : esc(typeof v === 'object' ? JSON.stringify(v) : v);
  const k = key.toLowerCase();
  if (/(win_rate|winrate|hit_rate|accuracy|precision|recall)/.test(k)) return `${(v <= 1 ? v * 100 : v).toFixed(1)}%`;
  if (/(_pct|pct_|drawdown)/.test(k)) return `${v.toFixed(2)}%`;
  if (/fees/.test(k)) return fmtUsd(v, 2);
  if (/(pnl|equity|avg_win|avg_loss|profit|expectancy)/.test(k) && !/factor|ratio/.test(k)) return fmtUsd(v, 2, true);
  if (Number.isInteger(v)) return String(v);
  return v.toFixed(Math.abs(v) < 1 ? 3 : 2);
}

export function titleCase(k) {
  return String(k).replace(/_/g, ' ');
}

export function debounce(fn, ms) {
  let t = null;
  return (...a) => {
    clearTimeout(t);
    t = setTimeout(() => fn(...a), ms);
  };
}

/** Trailing throttle: runs at most once per `ms`, always with the latest call. */
export function throttle(fn, ms) {
  let timer = null;
  let pending = false;
  const run = () => {
    timer = null;
    if (pending) {
      pending = false;
      fn();
      timer = setTimeout(run, ms);
    }
  };
  const t = () => {
    if (timer) {
      pending = true;
      return;
    }
    fn();
    timer = setTimeout(run, ms);
  };
  t.cancel = () => {
    clearTimeout(timer);
    timer = null;
    pending = false;
  };
  return t;
}

/** Time ago like "4m", "2h" from unix seconds. */
export function fmtAgo(t, now = Date.now() / 1000) {
  if (!isNum(t)) return '–';
  const d = Math.max(0, now - t);
  if (d < 90) return `${Math.round(d)}s`;
  if (d < 5400) return `${Math.round(d / 60)}m`;
  if (d < 172800) return `${(d / 3600).toFixed(1)}h`;
  return `${Math.round(d / 86400)}d`;
}

/** win_rate may be a fraction (0.62) or percent (62): normalise to percent. */
export function winRatePct(v) {
  if (!isNum(v)) return null;
  return v <= 1 ? v * 100 : v;
}

/** Coalesce many calls per frame into one (falls back to setTimeout without rAF). */
export function rafThrottle(fn) {
  let queued = false;
  const raf = typeof requestAnimationFrame === 'function' ? requestAnimationFrame : (f) => setTimeout(f, 16);
  return (...a) => {
    if (queued) return;
    queued = true;
    raf(() => {
      queued = false;
      fn(...a);
    });
  };
}
