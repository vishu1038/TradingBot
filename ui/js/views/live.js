// PAGE 1 (LIVE): symbols bar, hero (rotating), fixed main chart, live trades, depth / bot / market panels.
//   desktop  (>=1280 px): hero + log | chart | live trades on the left; depth, bot, market on the right
//   handheld (<1280 px) : hero + chart on top, then a swipeable carousel Depth -> Bot -> Market -> Trades
import { store } from '../state.js';
import { selectSymbol } from '../sync.js';
import { createPriceChart } from '../charts.js';
import * as fx from '../fx.js';
import { esc, fmtPrice, fmtPct, fmtClock, shortSym, signClass, throttle, isNum } from '../format.js';
import { coinColor } from '../theme.js';
import { probPill } from '../components.js';
import { barSymbols, refreshUniverse, pausedSet } from '../actions.js';
import { createHero } from '../widgets/hero.js';
import { createDepth } from '../widgets/depth.js';
import { createBotPanel, createMarketPanel } from '../widgets/panels.js';
import { createLiveTrades } from '../widgets/livetrades.js';

export const HANDHELD_MQ = '(max-width: 1279px)';
const PANELS = [
  ['depth', 'DEPTH'],
  ['bot', 'BOT'],
  ['market', 'MARKET'],
  ['trades', 'TRADES'],
];

export function mount(root) {
  const mq = window.matchMedia(HANDHELD_MQ);
  const handheld = mq.matches;
  const offs = [];
  const parts = []; // widgets to destroy

  const hero = createHero();
  const depth = createDepth();
  const bot = createBotPanel();
  const market = createMarketPanel();
  const trades = createLiveTrades();
  parts.push(hero, depth, bot, market, trades);

  // ------------------------------------------------------------ chart card
  const chartCard = document.createElement('div');
  chartCard.className = 'card chartcard ac-teal';
  chartCard.innerHTML = `
    <h3>CHART <span class="why-sub" id="c-sub"></span></h3>
    <div class="chart-stack">
      <div class="chart-price">
        <div class="chart-legend" id="legend"></div>
        <div class="chart-el" id="price-el"></div>
        <button class="chart-reset" id="chart-reset" title="Reset zoom">fit</button>
      </div>
      <div class="chart-prob">
        <div class="chart-tag">MODEL P(long) <span class="dim">dashed = entry threshold</span></div>
        <div class="chart-el" id="prob-el"></div>
      </div>
    </div>`;

  // ------------------------------------------------------------ log card
  const logCard = document.createElement('div');
  logCard.className = 'card logcard ac-sky';
  logCard.innerHTML = '<h3>LOG <span class="why-sub"><span class="deco" data-spin>◆</span> engine</span></h3><div class="log-scroll" id="log-scroll"></div>';

  // ------------------------------------------------------------ symbol bar
  const symbar = document.createElement('div');
  symbar.className = 'symbar';

  // ------------------------------------------------------------ layout
  const wrap = document.createElement('div');
  wrap.className = `live ${handheld ? 'live-hand' : 'live-desk'}`;
  if (!handheld) {
    const left = document.createElement('div');
    left.className = 'live-left';
    // the chart gets the whole upper left; live trades and the engine log share the row under it (user 2026-10-19)
    const row1 = document.createElement('div');
    row1.className = 'live-row1';
    row1.append(trades.el, logCard);
    left.append(chartCard, row1);
    const right = document.createElement('div');
    right.className = 'live-right';
    right.append(hero.el, depth.el, bot.el, market.el);
    wrap.append(symbar, left, right);
  } else {
    const top = document.createElement('div');
    top.className = 'live-top';
    top.append(hero.el, chartCard);
    const panels = document.createElement('div');
    panels.className = 'live-panels';
    const tabs = document.createElement('div');
    tabs.className = 'car-tabs';
    tabs.setAttribute('role', 'tablist');
    tabs.innerHTML = PANELS.map(([id, label], i) => `<button type="button" role="tab" class="car-tab ${i === 0 ? 'on' : ''}" data-p="${i}">${label}</button>`).join('');
    const track = document.createElement('div');
    track.className = 'car-track';
    const els = { depth: depth.el, bot: bot.el, market: market.el, trades: trades.el };
    for (const [id] of PANELS) {
      const slot = document.createElement('div');
      slot.className = 'car-panel';
      slot.dataset.id = id;
      slot.append(els[id]);
      track.append(slot);
    }
    panels.append(tabs, track);
    wrap.append(symbar, top, panels);

    const setTab = (i) => tabs.querySelectorAll('.car-tab').forEach((t, k) => t.classList.toggle('on', k === i));
    tabs.addEventListener('click', (e) => {
      const b = e.target.closest('[data-p]');
      if (!b) return;
      const i = Number(b.dataset.p);
      const slot = track.children[i];
      track.scrollTo({ left: slot.offsetLeft - track.offsetLeft, behavior: 'smooth' });
      setTab(i);
    });
    let raf = 0;
    track.addEventListener('scroll', () => {
      cancelAnimationFrame(raf);
      raf = requestAnimationFrame(() => {
        const w = track.children[0].getBoundingClientRect().width + 10;
        setTab(Math.round(track.scrollLeft / w));
      });
    }, { passive: true });
  }
  root.append(wrap);
  const $ = (sel) => wrap.querySelector(sel);

  // ------------------------------------------------------------ chart wiring
  let chart = null;
  let loadedSymbol = null;
  function ensureChart() {
    if (chart || typeof LightweightCharts === 'undefined') return;
    chart = createPriceChart($('#price-el'), $('#prob-el'), $('#legend'));
  }
  const symPositions = () => store.get('positions').filter((p) => p.symbol === store.get('symbol'));
  function refreshOverlays() {
    if (!chart || loadedSymbol !== store.get('symbol')) return;
    const sym = store.get('symbol');
    const pos = symPositions();
    chart.setPositionLines(pos);
    chart.setMarkers(store.get('trades').filter((t) => t.symbol === sym), pos);
    const thr = store.get('status')?.model?.threshold_long;
    if (isNum(thr)) chart.setThreshold(thr);
  }
  function renderChartTitle() {
    const sym = store.get('symbol');
    $('#c-sub').innerHTML = sym ? `· <b style="color:${coinColor(sym)}">${esc(sym)}</b> <span class="dim">5m</span>` : '';
  }
  function loadChart() {
    ensureChart();
    const sym = store.get('symbol');
    const cs = store.get('candles')[sym];
    renderChartTitle();
    if (!chart || !sym || !cs) return;
    chart.setData(sym, cs, store.get('signals')[sym] || []);
    loadedSymbol = sym;
    refreshOverlays();
  }
  offs.push(store.on('candles_loaded', ({ symbol }) => symbol === store.get('symbol') && loadChart()));
  offs.push(store.on('signals_loaded', ({ symbol }) => symbol === store.get('symbol') && loadChart()));
  offs.push(store.on('candle', ({ symbol, candle }) => chart && symbol === loadedSymbol && symbol === store.get('symbol') && chart.updateCandle(candle)));
  offs.push(store.on('signal', ({ signal }) => chart && signal.symbol === loadedSymbol && chart.updateSignal(signal)));
  offs.push(store.on('change:symbol', () => {
    renderStrip();
    renderChartTitle();
    if (store.get('candles')[store.get('symbol')]) loadChart();
  }));
  const overlayT = throttle(refreshOverlays, 400);
  for (const t of ['change:positions', 'position_updated', 'change:trades', 'change:status']) offs.push(store.on(t, overlayT));
  $('#chart-reset').addEventListener('click', () => chart && chart.resetView());

  // ------------------------------------------------------------ symbol bar
  function renderStrip() {
    const list = barSymbols();
    const rows = new Map(store.get('universe').map((u) => [u.symbol, u]));
    const ticks = store.get('ticks');
    const brain = store.get('brain');
    const sel = store.get('symbol');
    const held = new Set(store.get('positions').map((p) => p.symbol));
    const paused = pausedSet();
    const inUni = new Set(rows.keys());
    const chip = (sym) => {
      const u = rows.get(sym);
      const tk = ticks[sym];
      const price = tk ? tk.price : u?.last_price;
      let chg = u?.change_24h_pct;
      if (u && tk && isNum(u.last_price) && u.last_price > 0 && isNum(chg)) chg = (tk.price / (u.last_price / (1 + chg / 100)) - 1) * 100;
      const dir = tk && tk.price > tk.prev ? 'up' : tk && tk.price < tk.prev ? 'dn' : '';
      const p = brain[sym]?.prob_long;
      return `<button type="button" role="tab" aria-selected="${sym === sel}" class="chip ${sym === sel ? 'sel' : ''} ${inUni.has(sym) ? '' : 'off'} ${paused.has(sym) ? 'paused' : ''}" data-sym="${esc(sym)}" style="--cc:${coinColor(sym)}">
        <span class="chip-top"><i class="coin-dot"></i><b>${esc(shortSym(sym))}</b>${held.has(sym) ? '<em class="held" title="bot holds this coin">●</em>' : ''}${paused.has(sym) ? '<em class="pz" title="paused">⏸</em>' : ''}</span>
        <span class="chip-px ${dir}">${fmtPrice(price)}</span>
        <span class="chip-row"><span class="chip-ch ${signClass(chg)}">${fmtPct(chg, 1)}</span>${probPill(p)}</span>
      </button>`;
    };
    symbar.innerHTML = `<div class="symbar-scroll" role="tablist" aria-label="Hot coins">${list.map(chip).join('')}</div>
      <button type="button" class="refresh-btn" id="sb-refresh" title="Force universe refresh (r)" aria-label="Refresh universe"><span class="deco">⟳</span></button>`;
    // keep the selected chip in view (handhelds scroll horizontally)
    const selEl = symbar.querySelector('.chip.sel');
    if (selEl && symbar.__lastSel !== sel) {
      symbar.__lastSel = sel;
      const sc = symbar.querySelector('.symbar-scroll');
      sc.scrollLeft = Math.max(0, selEl.offsetLeft - sc.clientWidth / 2 + selEl.clientWidth / 2);
    }
  }
  symbar.addEventListener('click', async (e) => {
    const b = e.target.closest('[data-sym]');
    if (b) return selectSymbol(b.dataset.sym);
    const r = e.target.closest('#sb-refresh');
    if (r) {
      r.classList.add('spinning');
      await refreshUniverse();
      setTimeout(() => r.classList.remove('spinning'), 600);
    }
    return undefined;
  });
  const stripT = throttle(renderStrip, 500);
  for (const t of ['tick', 'change:universe', 'change:positions', 'brain', 'change:pausedLocal', 'change:status']) offs.push(store.on(t, stripT));

  // ------------------------------------------------------------ log (typed lines)
  const lvl = (l) => String(l || 'INFO').toLowerCase().replace(/[^a-z]/g, '').replace('warning', 'warn');
  const logHtml = (e) => `<div class="log lv-${lvl(e.level)}"><span class="log-t">${fmtClock(e.t)}</span><span class="log-l">${esc(String(e.level || '').replace('WARNING', 'WARN').slice(0, 5))}</span><span class="log-m"></span></div>`;
  const logEl = logCard.querySelector('#log-scroll');
  store.get('logs').slice(-60).forEach((e) => {
    logEl.insertAdjacentHTML('beforeend', logHtml(e));
    logEl.lastElementChild.querySelector('.log-m').textContent = e.msg;
  });
  logEl.scrollTop = logEl.scrollHeight;
  offs.push(store.on('log', ({ entry }) => {
    const atBottom = logEl.scrollHeight - logEl.scrollTop - logEl.clientHeight < 40;
    logEl.insertAdjacentHTML('beforeend', logHtml(entry));
    fx.typeInto(logEl.lastElementChild.querySelector('.log-m'), String(entry.msg));
    while (logEl.childElementCount > 80) logEl.firstElementChild.remove();
    if (atBottom) logEl.scrollTop = logEl.scrollHeight;
  }));

  // ------------------------------------------------------------ init
  renderStrip();
  renderChartTitle();
  if (store.get('symbol') && store.get('candles')[store.get('symbol')]) loadChart();
  else ensureChart();

  window.__tbChart = () => chart; // test hook (WKWebView regression checks)

    // crossing the breakpoint rebuilds the page in the other layout
  const onMq = () => store.emit('relayout', null);
  mq.addEventListener('change', onMq);

  return () => {
    mq.removeEventListener('change', onMq);
    offs.forEach((o) => o());
    overlayT.cancel();
    stripT.cancel();
    parts.forEach((p) => p.destroy());
    if (chart) chart.destroy();
    chart = null;
  };
}
