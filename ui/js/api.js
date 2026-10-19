// Data layer: ONE interface for REST + WS. Views never know whether it is the real API or the mock.
//
//   await api.init()            -> decides real vs mock (?mock=1, or /api/health unreachable)
//   api.status() ... api.control(...)   -> Promises (throw ApiError on failure)
//   api.connect(onMessage, onState)     -> {close()}  (real WS with backoff, or mock stream)
//
// Query params: ?mock=1 force mock, ?api=http://host:port override base, ?speed=N mock ticks per 5m candle.

import { startMock } from './mock.js';

const DEFAULT_BASE = 'http://127.0.0.1:8787';

export class ApiError extends Error {
  constructor(message, status = 0) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
  }
}

let base = DEFAULT_BASE;
let mock = null;
let mockReason = '';

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const qs = () => new URLSearchParams(typeof location !== 'undefined' ? location.search : '');

async function fetchJson(url, opts = {}, timeoutMs = 8000) {
  const ctl = new AbortController();
  const timer = setTimeout(() => ctl.abort(), timeoutMs);
  try {
    const res = await fetch(url, { ...opts, signal: ctl.signal });
    let body = null;
    try {
      body = await res.json();
    } catch (_) {
      /* non-JSON body */
    }
    if (!res.ok) {
      const msg = (body && (body.error || body.detail || body.message)) || `HTTP ${res.status}`;
      throw new ApiError(typeof msg === 'string' ? msg : JSON.stringify(msg), res.status);
    }
    return body;
  } catch (e) {
    if (e instanceof ApiError) throw e;
    throw new ApiError(e.name === 'AbortError' ? 'request timed out' : `network error: ${e.message}`, 0);
  } finally {
    clearTimeout(timer);
  }
}

function startMockBackend(reason) {
  const p = qs();
  const speed = parseInt(p.get('speed'), 10);
  const seed = parseInt(p.get('seed'), 10);
  mock = startMock({ ticksPerBar: Number.isFinite(speed) && speed > 1 ? speed : 30, seed: Number.isFinite(seed) ? seed : undefined });
  mockReason = reason;
}

export const api = {
  get base() {
    return base;
  },
  get isMock() {
    return !!mock;
  },
  get mockReason() {
    return mockReason;
  },

  /** Decide backend. Resolves to {mock, base, reason}. */
  async init() {
    const p = qs();
    const override = p.get('api');
    if (override) base = override.replace(/\/+$/, '');
    if (p.get('mock') === '1') {
      startMockBackend('forced by ?mock=1');
    } else {
      // candidates: ?api=, the origin serving this page (tb-api serves ui/ at /), then the default port
      const cands = [];
      if (override) cands.push(base);
      else {
        if (typeof location !== 'undefined' && /^https?:$/.test(location.protocol)) cands.push(location.origin);
        cands.push(DEFAULT_BASE);
      }
      let lastErr = null;
      let found = false;
      for (const c of [...new Set(cands)]) {
        try {
          const h = await fetchJson(`${c}/api/health`, {}, 1500);
          if (!h || h.ok === false) throw new ApiError('health not ok');
          base = c;
          found = true;
          break;
        } catch (e) {
          lastErr = e;
        }
      }
      if (!found) startMockBackend(`${base}/api/health unreachable (${lastErr ? lastErr.message : 'no answer'})`);
    }
    return { mock: !!mock, base, reason: mockReason };
  },

  async request(method, path, query = {}, body = null) {
    if (mock) {
      await sleep(8 + Math.random() * 25);
      const r = mock.request(method, path, query, body == null ? null : JSON.parse(JSON.stringify(body)));
      if (r.status >= 400) throw new ApiError(r.body?.error || `HTTP ${r.status}`, r.status);
      return JSON.parse(JSON.stringify(r.body));
    }
    const u = new URL(base + path);
    for (const [k, v] of Object.entries(query)) if (v != null) u.searchParams.set(k, String(v));
    const opts = { method };
    if (body != null) {
      opts.headers = { 'Content-Type': 'application/json' };
      opts.body = JSON.stringify(body);
    }
    return fetchJson(u.toString(), opts);
  },

  health: () => api.request('GET', '/api/health'),
  status: () => api.request('GET', '/api/status'),
  market: () => api.request('GET', '/api/market'),
  universe: () => api.request('GET', '/api/universe'),
  orderbook: (symbol) => api.request('GET', '/api/orderbook', { symbol }),
  candles: (symbol, limit = 500, tf = '5m') => api.request('GET', '/api/candles', { symbol, tf, limit }),
  signals: (symbol, limit = 500) => api.request('GET', '/api/signals', { symbol, limit }),
  positions: () => api.request('GET', '/api/positions'),
  trades: (limit = 500) => api.request('GET', '/api/trades', { limit }),
  equity: (limit = 2000) => api.request('GET', '/api/equity', { limit }),
  logs: (limit = 200) => api.request('GET', '/api/logs', { limit }),
  settings: () => api.request('GET', '/api/settings'),
  saveSettings: (patch) => api.request('PUT', '/api/settings', {}, patch),
  shadow: () => api.request('GET', '/api/shadow'),
  /** cumulative net PnL per model (champion + every challenger): [{model,status,kind,points:[{t,pnl}]}] */
  shadowEquity: (limit = 5000) => api.request('GET', '/api/shadow/equity', { limit }),
  labReports:() => api.request('GET', '/api/lab/reports'),
  mirror: () => api.request('GET', '/api/mirror'),
  mirrorRetry: () => api.request('POST', '/api/mirror/retry', {}, {}),
  /** control actions: start|stop|kill|reset_kill|close_position|close_all|promote_model|rollback_model */
  async control(action, extra = {}) {
    const r = await api.request('POST', '/api/control', {}, { action, ...extra });
    if (r && r.ok === false) throw new ApiError(r.error || `${action} failed`);
    return r;
  },

  /**
   * Subscribe to the live stream. onMessage({type,data}); onState('connecting'|'open'|'closed'|'mock').
   * Real WS reconnects with exponential backoff + jitter (0.5 s -> 15 s).
   */
  connect(onMessage, onState = () => {}) {
    if (mock) {
      onState('mock');
      const off = mock.subscribe(onMessage);
      return { close: off };
    }
    let ws = null;
    let closed = false;
    let attempt = 0;
    let timer = null;
    const url = base.replace(/^http/i, 'ws') + '/ws';
    const open = () => {
      if (closed) return;
      onState('connecting');
      try {
        ws = new WebSocket(url);
      } catch (e) {
        schedule();
        return;
      }
      ws.onopen = () => {
        attempt = 0;
        onState('open');
      };
      ws.onmessage = (ev) => {
        try {
          const m = JSON.parse(ev.data);
          if (m && m.type) onMessage(m);
        } catch (e) {
          console.warn('[ws] bad message', e);
        }
      };
      ws.onclose = () => {
        onState('closed');
        schedule();
      };
      ws.onerror = () => {
        try {
          ws.close();
        } catch (_) {
          /* already closing */
        }
      };
    };
    const schedule = () => {
      if (closed || timer) return;
      const delay = Math.min(15000, 500 * 2 ** Math.min(attempt, 6)) * (0.75 + Math.random() * 0.5);
      attempt++;
      timer = setTimeout(() => {
        timer = null;
        open();
      }, delay);
    };
    open();
    return {
      close() {
        closed = true;
        clearTimeout(timer);
        try {
          ws && ws.close();
        } catch (_) {
          /* ignore */
        }
      },
    };
  },
};
