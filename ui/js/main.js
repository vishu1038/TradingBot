// Entry point: boot the data layer, mount top bar + nav, route between views.
import { api } from './api.js';
import { store } from './state.js';
import { startSync } from './sync.js';
import { mountTopbar } from './topbar.js';
import * as sound from './sound.js';
import * as fx from './fx.js';
import * as live from './views/live.js';
import * as markets from './views/markets.js';
import * as bots from './views/bots.js';
import * as trades from './views/trades.js';
import * as lab from './views/lab.js';
import * as settings from './views/settings.js';

const ICON = {
  live: '<path d="M2 12h5l3-8 4 16 3-8h5"/>',
  markets: '<path d="M3 3h8v8H3zM13 3h8v5h-8zM13 10h8v11h-8zM3 13h8v8H3z"/>',
  bots: '<path d="M5 8h14v11H5zM12 4v4M9 13h.01M15 13h.01M9 16h6"/>',
  trades: '<path d="M4 6h16M4 12h16M4 18h10"/>',
  lab: '<path d="M9 3h6M10 3v6L4 20h16L14 9V3M7 15h10"/>',
  settings: '<path d="M4 7h8M16 7h4M4 17h4M12 17h8"/><circle cx="14" cy="7" r="2"/><circle cx="10" cy="17" r="2"/>',
};

const VIEWS = [
  { id: 'live', label: 'LIVE', mod: live },
  { id: 'markets', label: 'MARKETS', mod: markets },
  { id: 'bots', label: 'BOTS', mod: bots },
  { id: 'trades', label: 'TRADES', mod: trades },
  { id: 'lab', label: 'LAB', mod: lab },
  { id: 'settings', label: 'SETTINGS', mod: settings },
];

let unmount = null;
let current = null;

function renderNav() {
  const nav = document.getElementById('nav');
  nav.innerHTML =
    VIEWS.map(
      (v, i) => `<button type="button" class="nav-item" data-view="${v.id}" title="${v.label} (${i + 1})">
        <svg viewBox="0 0 24 24" aria-hidden="true">${ICON[v.id]}</svg><span class="nav-l">${v.label}</span><span class="nav-key">${i + 1}</span></button>`
    ).join('') + `<div class="nav-foot">═══════════<br><b>5m</b> candles<br>keys <b>1-6</b></div>`;
  nav.addEventListener('click', (e) => {
    const b = e.target.closest('[data-view]');
    if (b) store.set('view', b.dataset.view);
  });
}

function show(id) {
  const v = VIEWS.find((x) => x.id === id) || VIEWS[0];
  if (current === v.id) return;
  if (unmount) {
    try {
      unmount();
    } catch (e) {
      console.error('[main] unmount failed', e);
    }
    unmount = null;
  }
  current = v.id;
  const root = document.getElementById('view');
  root.innerHTML = '';
  root.className = `view-${v.id}`;
  root.scrollTop = 0;
  document.querySelectorAll('.nav-item').forEach((n) => n.classList.toggle('active', n.dataset.view === v.id));
  try {
    unmount = v.mod.mount(root);
  } catch (e) {
    console.error(`[main] view ${v.id} failed to mount`, e);
    root.innerHTML = `<div class="empty neg">view "${v.id}" crashed: see console</div>`;
  }
  try {
    history.replaceState(null, '', `#/${v.id}`);
  } catch (_) {
    /* file:// or sandboxed */
  }
  document.title = `${v.label} · TRADINGBOT//v2`;
}

function initialView() {
  const h = (location.hash || '').replace(/^#\/?/, '');
  return VIEWS.some((v) => v.id === h) ? h : 'live';
}

function bindKeys() {
  window.addEventListener('keydown', (e) => {
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    const t = e.target;
    if (t && (t.tagName === 'INPUT' || t.tagName === 'SELECT' || t.tagName === 'TEXTAREA' || t.isContentEditable)) return;
    if (!document.getElementById('modal-host').hidden) return;
    const n = parseInt(e.key, 10);
    if (n >= 1 && n <= VIEWS.length) {
      store.set('view', VIEWS[n - 1].id);
      e.preventDefault();
    }
  });
}

async function boot() {
  const msg = document.getElementById('boot-msg');
  renderNav();
  sound.unlockOnGesture();
  mountTopbar(document.getElementById('topbar'));
  bindKeys();
  store.set('view', initialView());
  store.on('change:view', (id) => show(id));
  try {
    msg.textContent = 'probing API…';
    const r = await api.init();
    msg.textContent = r.mock ? 'MOCK MODE: spinning up fake market…' : `connected to ${r.base}`;
    await startSync();
  } catch (e) {
    console.error('[main] boot failed', e);
    msg.textContent = `boot failed: ${e.message}`;
    fx.toast(`boot failed: ${e.message}`, 'loss', 8000);
  }
  show(store.get('view'));
  const b = document.getElementById('boot');
  b.classList.add('gone');
  setTimeout(() => b.remove(), 600);
}

window.addEventListener('unhandledrejection', (e) => console.error('[unhandled rejection]', e.reason));
boot();
