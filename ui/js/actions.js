// User actions shared by the top bar, widgets, keyboard shortcuts and the `/` command palette.
// Every state-changing call goes through here so confirmations and error toasts behave the same everywhere.
import { api } from './api.js';
import { store } from './state.js';
import { confirmDialog } from './dialog.js';
import * as fx from './fx.js';
import * as sound from './sound.js';
import { selectSymbol } from './sync.js';
import { shortSym } from './format.js';

/** Run a control action; toasts the backend's error instead of throwing. Resolves to true on success. */
export async function ctl(action, extra = {}, errPrefix = `${action} failed`) {
  try {
    await api.control(action, extra);
    return true;
  } catch (e) {
    fx.toast(`${errPrefix}: ${e.message}`, 'loss', 6000);
    return false;
  }
}

/** Paused coins: the backend's status.paused_symbols when present, else what this UI paused. */
export function pausedSet() {
  const st = store.get('status');
  if (st && Array.isArray(st.paused_symbols)) return new Set(st.paused_symbols);
  return new Set(store.get('pausedLocal'));
}

export const isRunning = () => !!store.get('status')?.running;

export async function toggleRun() {
  sound.play('click');
  return ctl(isRunning() ? 'stop' : 'start');
}

export async function killAll() {
  const ok = await confirmDialog({
    title: 'KILL SWITCH',
    message: 'Close ALL positions at market, stop opening trades and halt the bot until a manual reset. Continue?',
    confirmText: 'KILL EVERYTHING',
    danger: true,
  });
  return ok ? ctl('kill', {}, 'kill failed') : false;
}

export async function resetKill() {
  const ok = await confirmDialog({ title: 'RESET KILL SWITCH', message: 'Re-arm the bot after a kill? It stays stopped until you press START.', confirmText: 'RESET' });
  return ok ? ctl('reset_kill', {}, 'reset failed') : false;
}

export async function closePosition(p) {
  if (!p) return false;
  const ok = await confirmDialog({
    title: 'CLOSE POSITION',
    message: `Close ${shortSym(p.symbol)} ${String(p.side).toUpperCase()} at market now?`,
    confirmText: 'CLOSE NOW',
    danger: true,
  });
  return ok ? ctl('close_position', { position_id: p.id }, 'close failed') : false;
}

export function positionOf(sym) {
  return store.get('positions').find((p) => p.symbol === sym) || null;
}

export async function closeSymbol(sym) {
  const p = positionOf(sym);
  if (!p) {
    fx.toast(`no open position on ${shortSym(sym)}`, 'info');
    return false;
  }
  return closePosition(p);
}

export async function closeAll() {
  const ok = await confirmDialog({ title: 'CLOSE ALL', message: 'Close every open position at market?', confirmText: 'CLOSE ALL', danger: true });
  return ok ? ctl('close_all', {}, 'close all failed') : false;
}

export async function togglePause(sym) {
  if (!sym) return false;
  const paused = pausedSet().has(sym);
  const okk = await ctl(paused ? 'unpause_symbol' : 'pause_symbol', { symbol: sym }, `${paused ? 'unpause' : 'pause'} failed`);
  if (okk) {
    const loc = new Set(store.get('pausedLocal'));
    if (paused) loc.delete(sym);
    else loc.add(sym);
    store.set('pausedLocal', [...loc]);
    fx.toast(`${shortSym(sym)} ${paused ? 'unpaused' : 'paused: no new entries'}`, 'info');
  }
  return okk;
}

export async function refreshUniverse() {
  const okk = await ctl('refresh_universe', {}, 'refresh failed');
  if (okk) {
    fx.toast('universe refresh requested', 'info');
    try {
      const u = await api.universe();
      if (u) store.set('universe', u);
    } catch (_) {
      /* polling will catch up */
    }
  }
  return okk;
}

/** Symbols shown in the symbol bar, in order. */
export function barSymbols() {
  const uni = store.get('universe').map((u) => u.symbol);
  const extra = store.get('positions').map((p) => p.symbol).filter((s) => !uni.includes(s));
  return [...uni, ...new Set(extra)];
}

export function cycleSymbol(dir) {
  const list = barSymbols();
  if (!list.length) return;
  const i = list.indexOf(store.get('symbol'));
  selectSymbol(list[(i + dir + list.length) % list.length]);
}

/** Select a coin by (partial) name, e.g. "sol" or "SOLUSDT". Returns the symbol or null. */
export function selectCoin(q) {
  const t = String(q || '').trim().toUpperCase();
  if (!t) return null;
  const list = barSymbols();
  const hit =
    list.find((s) => s === t || s === `${t}USDT`) || list.find((s) => shortSym(s).startsWith(t)) || list.find((s) => s.includes(t));
  if (!hit) return null;
  selectSymbol(hit);
  store.set('view', 'live');
  return hit;
}

export const goPage = (id) => store.set('view', id);
