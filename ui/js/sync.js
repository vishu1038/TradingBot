// Keeps the store in sync with the backend: initial REST load, periodic polling as a safety net,
// and WS messages applied incrementally. Also fires the sound/flash effects for fills.

import { api } from './api.js';
import { store } from './state.js';
import * as sound from './sound.js';
import * as fx from './fx.js';
import { debounce, fmtUsd, shortSym } from './format.js';

const MAX_CANDLES = 1500;
const MAX_LOGS = 300;
let wsHandle = null;
let timers = [];
let started = false;

const safe = (p, label) =>
  p.catch((e) => {
    // keep going: one failing endpoint must not blank the UI
    console.warn(`[sync] ${label}: ${e.message}`);
    return undefined;
  });

const byClosedDesc = (a, b) => (b.closed_at || 0) - (a.closed_at || 0);

async function loadStatus() {
  const s = await safe(api.status(), 'status');
  if (s) store.set('status', s);
}

async function loadSymbol(sym) {
  if (!sym) return;
  const [c, s] = await Promise.all([safe(api.candles(sym, 500), `candles ${sym}`), safe(api.signals(sym, 500), `signals ${sym}`)]);
  if (c) {
    store.get('candles')[sym] = c;
    store.emit('candles_loaded', { symbol: sym });
  }
  if (s) {
    store.get('signals')[sym] = s;
    store.emit('signals_loaded', { symbol: sym });
  }
}

export function selectSymbol(sym) {
  if (!sym) return;
  store.set('symbol', sym);
  store.set('selectedPosition', null);
  loadSymbol(sym);
}

const refetchTrades = debounce(async () => {
  const [pos, tr, st] = await Promise.all([safe(api.positions(), 'positions'), safe(api.trades(500), 'trades'), safe(api.status(), 'status')]);
  if (pos) store.set('positions', pos);
  if (tr) store.set('trades', [...tr].sort(byClosedDesc));
  if (st) store.set('status', st);
}, 700);

const refetchShadow = debounce(async () => {
  const r = await safe(api.shadow(), 'shadow');
  if (r) store.set('shadow', r);
}, 1000);

export const refreshAll = refetchTrades;

function upsertCandle(sym, c) {
  const arr = store.get('candles')[sym];
  if (!arr || !arr.length) return;
  const last = arr[arr.length - 1];
  if (c.t === last.t) arr[arr.length - 1] = c;
  else if (c.t > last.t) {
    arr.push(c);
    if (arr.length > MAX_CANDLES) arr.shift();
  } else {
    const i = arr.findIndex((x) => x.t === c.t);
    if (i >= 0) arr[i] = c;
  }
  store.emit('candle', { symbol: sym, candle: c });
}

function upsertSignal(s) {
  const arr = store.get('signals')[s.symbol];
  if (!arr) return;
  const last = arr[arr.length - 1];
  if (last && last.t === s.t) arr[arr.length - 1] = s;
  else if (!last || s.t > last.t) arr.push(s);
  else {
    const i = arr.findIndex((x) => x.t === s.t);
    if (i >= 0) arr[i] = s;
  }
  if (arr.length > 1500) arr.shift();
  store.emit('signal', { signal: s });
}

function fillEffects(trade) {
  const r = trade.exit_reason;
  const win = (trade.pnl || 0) > 0;
  if (r === 'kill') {
    fx.flash('kill');
    sound.play('kill');
  } else if (r === 'tp' || win) {
    fx.flash('tp');
    sound.play('tp');
  } else {
    fx.flash('sl');
    sound.play('sl');
  }
  const kind = win ? 'win' : 'loss';
  fx.toast(`${shortSym(trade.symbol)} ${trade.side.toUpperCase()} closed ${String(r).toUpperCase()}  ${fmtUsd(trade.pnl, 2, true)}`, kind);
}

function handle(m) {
  const d = m.data;
  switch (m.type) {
    case 'status':
      store.set('status', d);
      break;
    case 'safety': {
      const st = store.get('status');
      if (st) store.set('status', { ...st, safety: d });
      break;
    }
    case 'tick': {
      store.noteServerTime(d.t);
      const prev = store.get('ticks')[d.symbol];
      store.get('ticks')[d.symbol] = { price: d.price, prev: prev ? prev.price : d.price, t: d.t };
      store.emit('tick', d);
      break;
    }
    case 'candle':
      upsertCandle(d.symbol, d.candle);
      break;
    case 'signal':
      upsertSignal(d);
      break;
    case 'position_opened': {
      const list = store.get('positions').filter((p) => p.id !== d.id);
      list.push(d);
      store.set('positions', list);
      store.emit('position_opened', { position: d });
      sound.play('open');
      fx.toast(`${shortSym(d.symbol)} ${d.side.toUpperCase()} opened @ ${d.entry_price}`, 'open');
      break;
    }
    case 'position_updated': {
      const list = store.get('positions');
      const i = list.findIndex((p) => p.id === d.id);
      if (i >= 0) list[i] = d;
      else list.push(d);
      store.emit('position_updated', { position: d });
      break;
    }
    case 'position_closed': {
      const trade = d.trade || d;
      const id = d.position_id ?? trade.id ?? d.id;
      const pos = store.get('positions').find((p) => p.id === id);
      store.set('positions', store.get('positions').filter((p) => p.id !== id));
      if (store.get('selectedPosition') === id) store.set('selectedPosition', null);
      if (trade && trade.symbol) {
        const trades = [trade, ...store.get('trades').filter((t) => t.id !== trade.id)];
        store.set('trades', trades);
        store.emit('trade_closed', { trade, position: pos });
        fillEffects(trade);
      }
      refetchTrades();
      break;
    }
    case 'universe':
      store.set('universe', Array.isArray(d) ? d : store.get('universe'));
      break;
    case 'equity': {
      const eq = store.get('equity');
      const last = eq[eq.length - 1];
      if (last && last.t === d.t) eq[eq.length - 1] = d;
      else eq.push(d);
      if (eq.length > 6000) eq.shift();
      store.emit('equity_point', d);
      break;
    }
    case 'log': {
      const logs = store.get('logs');
      logs.push(d);
      if (logs.length > MAX_LOGS) logs.shift();
      store.emit('log', { entry: d });
      break;
    }
    case 'kill': {
      const st = store.get('status');
      if (st) store.set('status', { ...st, kill: d });
      if (d && d.tripped) {
        fx.flash('kill');
        sound.play('kill');
        fx.toast(`KILL SWITCH: ${d.reason || 'tripped'}`, 'loss', 6000);
      }
      break;
    }
    case 'shadow_trade':
      store.emit('shadow_trade', d);
      refetchShadow();
      break;
    default:
      break;
  }
}

function poll(fn, ms) {
  timers.push(setInterval(fn, ms));
}

export async function startSync() {
  if (started) return;
  started = true;

  store.set('conn', { mock: api.isMock, ws: 'idle', apiBase: api.base, reason: api.mockReason });

  const [health, status, market, universe, positions, trades, equity, logs, settings, shadow] = await Promise.all([
    safe(api.health(), 'health'),
    safe(api.status(), 'status'),
    safe(api.market(), 'market'),
    safe(api.universe(), 'universe'),
    safe(api.positions(), 'positions'),
    safe(api.trades(500), 'trades'),
    safe(api.equity(2000), 'equity'),
    safe(api.logs(200), 'logs'),
    safe(api.settings(), 'settings'),
    safe(api.shadow(), 'shadow'),
  ]);
  if (health) store.set('health', health);
  if (status) store.set('status', status);
  if (market) {
    store.set('market', market);
    store.noteServerTime(market.updated_at);
  }
  if (universe) store.set('universe', universe);
  if (positions) store.set('positions', positions);
  if (trades) store.set('trades', [...trades].sort(byClosedDesc));
  if (equity) store.set('equity', equity);
  if (logs) store.set('logs', logs);
  if (settings) {
    store.set('settings', settings);
    sound.adoptServerSetting(settings.sound);
  }
  if (shadow) store.set('shadow', shadow);

  const ticks = store.get('ticks');
  for (const u of universe || []) ticks[u.symbol] = { price: u.last_price, prev: u.last_price, t: Date.now() / 1000 };

  // default symbol: the first open position's symbol, else the top-ranked universe symbol
  if (!store.get('symbol')) {
    const first = (positions && positions[0] && positions[0].symbol) || (universe && universe[0] && universe[0].symbol) || 'BTCUSDT';
    store.set('symbol', first);
  }
  await loadSymbol(store.get('symbol'));

  wsHandle = api.connect(handle, (s) => store.patch('conn', { ws: s }));

  // polling = safety net when WS is down (and keeps slow-moving things fresh)
  poll(async () => {
    const h = await safe(api.health(), 'health');
    if (h) store.set('health', h);
    else store.set('health', { ...(store.get('health') || {}), ok: false, engine_alive: false, unreachable: true });
  }, 2000);
  poll(loadStatus, 5000);
  poll(async () => {
    const m = await safe(api.market(), 'market');
    if (m) store.set('market', m);
  }, 8000);
  poll(async () => {
    const u = await safe(api.universe(), 'universe');
    if (u) store.set('universe', u);
  }, 30000);
  poll(async () => {
    const p = await safe(api.positions(), 'positions');
    if (p) store.set('positions', p);
  }, 6000);
  poll(refetchTrades, 30000);
  poll(refetchShadow, 15000);
  poll(async () => {
    const e = await safe(api.equity(2000), 'equity');
    if (e) store.set('equity', e);
  }, 60000);
  if (!api.isMock) poll(() => loadSymbol(store.get("symbol")), 60000);
}

export function stopSync() {
  if (wsHandle) wsHandle.close();
  timers.forEach(clearInterval);
  timers = [];
  started = false;
}
