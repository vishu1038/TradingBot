// Tiny store + pub/sub. Views never talk to the network; they read the store and listen to topics.
//
// store.get(key) / store.set(key, value)  -> emits "change:<key>" and "change"
// store.on(topic, fn) -> unsubscribe fn       store.emit(topic, payload)
//
// Event topics emitted by sync.js:  candle {symbol, candle}, signal {signal}, tick {symbol,t,price},
// trade_closed {trade, position}, position_opened {position}, log {entry}, toast {kind, msg}

const data = {
  view: 'live',
  symbol: null, // selected symbol on the Live view
  conn: { mock: false, ws: 'idle', apiBase: '', reason: '' },
  health: null,
  status: null,
  market: null,
  universe: [],
  positions: [],
  trades: [],
  equity: [],
  logs: [],
  settings: null,
  labReports: [],
  shadow: [],
  orderbook: {}, // symbol -> {symbol,t,mid,spread,bands,bids,asks}
  orderbookState: {}, // symbol -> 'ok' | 'unavailable'
  brain: {}, // symbol -> latest signal (any symbol in the universe)
  pausedLocal: [], // symbols paused through this UI (merged with status.paused_symbols)
  candles: {}, // symbol -> [{t,o,h,l,c,v}]
  signals: {}, // symbol -> [{t,symbol,prob_long,...}]
  ticks: {}, // symbol -> {price, prev, t}
  selectedPosition: null,
};

const listeners = new Map();

function on(topic, fn) {
  if (!listeners.has(topic)) listeners.set(topic, new Set());
  listeners.get(topic).add(fn);
  return () => listeners.get(topic)?.delete(fn);
}

function emit(topic, payload) {
  const set = listeners.get(topic);
  if (!set) return;
  for (const fn of [...set]) {
    try {
      fn(payload);
    } catch (e) {
      console.error(`[state] listener for "${topic}" threw`, e);
    }
  }
}

let serverT = 0;

export const store = {
  get: (k) => data[k],
  /** Latest server timestamp (unix s) seen on the stream; the mock runs a virtual clock ahead of wall time. */
  noteServerTime(t) {
    if (typeof t === 'number' && t > serverT) serverT = t;
  },
  serverNow: () => serverT || Date.now() / 1000,
  set(k, v) {
    data[k] = v;
    emit(`change:${k}`, v);
    emit('change', k);
  },
  /** Shallow-merge into an object-valued key. */
  patch(k, p) {
    data[k] = { ...(data[k] || {}), ...p };
    emit(`change:${k}`, data[k]);
    emit('change', k);
  },
  on,
  emit,
  /** Subscribe to several topics at once; returns one unsubscribe. */
  onAny(topics, fn) {
    const offs = topics.map((t) => on(t, fn));
    return () => offs.forEach((o) => o());
  },
};
