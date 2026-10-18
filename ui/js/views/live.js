// LIVE view: symbol strip, candle chart (+volume, EMAs, probability pane, position lines), right-hand
// stats panel (market, bot, positions, WHY) and a log tail.
import { store } from '../state.js';
import { api } from '../api.js';
import { selectSymbol } from '../sync.js';
import { createPriceChart } from '../charts.js';
import { confirmDialog } from '../dialog.js';
import * as fx from '../fx.js';
import {
  esc, fmtPrice, fmtUsd, fmtPct, fmtFunding, fmtClock, fmtNum, shortSym, signClass, throttle, winRatePct, isNum, fmtDuration,
} from '../format.js';
import { tile, kv, sidePill, reasonBars, slTpBar, symLabel, safetyOf, derateGauge } from '../components.js';

const TREND = { up: ['▲ UP', 'pos'], down: ['▼ DOWN', 'neg'], flat: ['▬ FLAT', 'flat'] };

export function mount(root) {
  root.innerHTML = `
    <div class="live">
      <section class="live-main">
        <div class="sym-strip" id="sym-strip" role="tablist" aria-label="Universe"></div>
        <div class="chart-stack">
          <div class="chart-price">
            <div class="chart-legend" id="legend"></div>
            <div class="chart-el" id="price-el"></div>
            <button class="chart-reset" id="chart-reset" title="Reset zoom">[ fit ]</button>
          </div>
          <div class="chart-prob">
            <div class="chart-tag">MODEL P(long) <span class="dim">dashed = entry threshold</span></div>
            <div class="chart-el" id="prob-el"></div>
          </div>
        </div>
        <div class="logstrip card" id="logstrip" aria-live="off"></div>
      </section>
      <aside class="live-side">
        <div class="card"><h3>MARKET</h3><div id="card-market"></div></div>
        <div class="card"><h3>BOT</h3><div id="card-bot"></div></div>
        <div class="card"><h3>OPEN POSITIONS <span class="count" id="pos-count"></span></h3><div id="card-pos"></div></div>
        <div class="card"><h3>WHY <span class="why-sub" id="why-sub"></span></h3><div id="card-why"></div></div>
      </aside>
    </div>`;

  const $ = (id) => root.querySelector(`#${id}`);
  const strip = $('sym-strip');
  const logEl = $('logstrip');
  let chart = null;
  let loadedSymbol = null;
  const offs = [];

  // ------------------------------------------------------------------ chart
  function ensureChart() {
    if (chart || typeof LightweightCharts === 'undefined') return;
    chart = createPriceChart($('price-el'), $('prob-el'), $('legend'));
  }

  function symPositions() {
    const sym = store.get('symbol');
    return store.get('positions').filter((p) => p.symbol === sym);
  }

  function refreshOverlays() {
    if (!chart) return;
    const sym = store.get('symbol');
    const pos = symPositions();
    chart.setPositionLines(pos);
    chart.setMarkers(store.get('trades').filter((t) => t.symbol === sym), pos);
    const thr = store.get('status')?.model?.threshold_long;
    if (isNum(thr)) chart.setThreshold(thr);
  }

  function loadChart() {
    ensureChart();
    const sym = store.get('symbol');
    const cs = store.get('candles')[sym];
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
    renderSide();
    if (store.get('candles')[store.get('symbol')]) loadChart();
  }));
  const overlayThrottled = throttle(refreshOverlays, 400);
  offs.push(store.on('change:positions', overlayThrottled));
  offs.push(store.on('position_updated', overlayThrottled));
  offs.push(store.on('change:trades', overlayThrottled));
  offs.push(store.on('change:status', overlayThrottled));
  root.querySelector('#chart-reset').addEventListener('click', () => chart && chart.resetView());

  // ------------------------------------------------------------- symbol strip
  function stripSymbols() {
    const uni = store.get('universe').map((u) => u.symbol);
    const extra = store.get('positions').map((p) => p.symbol).filter((s) => !uni.includes(s));
    const sel = store.get('symbol');
    if (sel && !uni.includes(sel) && !extra.includes(sel)) extra.push(sel);
    return { uni, extra: [...new Set(extra)] };
  }

  function renderStrip() {
    const { uni, extra } = stripSymbols();
    const rows = new Map(store.get('universe').map((u) => [u.symbol, u]));
    const ticks = store.get('ticks');
    const sel = store.get('symbol');
    const open = new Set(store.get('positions').map((p) => p.symbol));
    const chip = (sym, off) => {
      const u = rows.get(sym);
      const tk = ticks[sym];
      const price = tk ? tk.price : u?.last_price;
      let chg = u?.change_24h_pct;
      if (u && tk && isNum(u.last_price) && u.last_price > 0 && isNum(chg)) {
        const ref = u.last_price / (1 + chg / 100);
        chg = (tk.price / ref - 1) * 100;
      }
      const dir = tk && tk.price > tk.prev ? 'up' : tk && tk.price < tk.prev ? 'dn' : '';
      return `<button type="button" role="tab" aria-selected="${sym === sel}" class="chip ${sym === sel ? 'sel' : ''} ${off ? 'off' : ''}" data-sym="${esc(sym)}">
        <span class="chip-top">${open.has(sym) ? '<i class="chip-pos"></i>' : ''}<b>${esc(shortSym(sym))}</b></span>
        <span class="chip-px ${dir}">${fmtPrice(price)}</span>
        <span class="chip-ch ${signClass(chg)}">${fmtPct(chg, 2)}</span>
      </button>`;
    };
    strip.innerHTML = uni.map((s) => chip(s, false)).join('') + extra.map((s) => chip(s, true)).join('');
  }
  strip.addEventListener('click', (e) => {
    const b = e.target.closest('[data-sym]');
    if (b) selectSymbol(b.dataset.sym);
  });

  // ----------------------------------------------------------------- side cards
  function renderMarket() {
    const m = store.get('market');
    if (!m) return ($('card-market').innerHTML = '<div class="empty">waiting for /api/market…</div>');
    const [tl, tc] = TREND[m.btc_trend] || TREND.flat;
    $('card-market').innerHTML = `
      <div class="mk-btc"><span class="mk-btc-l">BTC</span><span class="mk-btc-p">${fmtPrice(m.btc_price)}</span>
        <span class="${signClass(m.btc_change_24h_pct)}">${fmtPct(m.btc_change_24h_pct)}</span><span class="${tc} mk-trend">${tl}</span></div>
      <div class="kv-grid">
        ${kv('volatility (ATR%)', fmtPct(m.market_volatility_pct, 2, false))}
        ${kv('avg funding', `<span class="${signClass(m.avg_funding_rate)}">${fmtFunding(m.avg_funding_rate)}</span>`)}
        ${kv('long/short ratio', fmtNum(m.long_short_ratio, 2))}
      </div>`;
  }

  function renderBot() {
    const s = store.get('status');
    if (!s) return ($('card-bot').innerHTML = '<div class="empty">waiting for /api/status…</div>');
    const wr = winRatePct(s.win_rate);
    const sf = safetyOf(s);
    const m = s.model || {};
    const drift = sf.drift;
    $('card-bot').innerHTML = `
      <div class="tiles">
        ${tile('EQUITY', fmtUsd(s.equity), signClass(s.equity - s.start_equity))}
        ${tile('WIN RATE', wr == null ? '–' : `${wr.toFixed(1)}%`, wr != null && wr >= 50 ? 'pos' : 'neg')}
        ${tile('TRADES TODAY', esc(s.trades_today))}
        ${tile('OPEN / MAX', `${esc(s.open_positions)} / ${esc(s.max_positions)}`)}
      </div>
      <div class="kv-grid">
        ${kv('realized', `<span class="${signClass(s.realized_pnl)}">${fmtUsd(s.realized_pnl, 2, true)}</span>`)}
        ${kv('unrealized', `<span class="${signClass(s.unrealized_pnl)}">${fmtUsd(s.unrealized_pnl, 2, true)}</span>`)}
        ${kv('size multiplier', `<span class="${sf.derate >= 0.99 ? 'pos' : sf.derate >= 0.5 ? 'warn' : 'neg'}">x${sf.derate.toFixed(2)}</span>`)}
        ${
          drift
            ? kv('drift z (n=' + esc(drift.n) + ')', `<span class="${drift.z < -3 ? 'neg' : drift.z < -2 ? 'warn' : 'pos'}">${isNum(drift.z) ? drift.z.toFixed(2) : '–'}</span> <span class="dim">size x${isNum(drift.mult) ? drift.mult.toFixed(2) : '–'}</span>`)
            : kv('drift', '<span class="dim">n/a</span>')
        }
        ${kv('model', `${esc(m.name || '–')} <span class="dim">v${esc(m.version || '–')}</span>`)}
        ${kv('thresholds', `L ${isNum(m.threshold_long) ? m.threshold_long.toFixed(2) : '–'} / S ${isNum(m.threshold_short) ? m.threshold_short.toFixed(2) : '–'}`)}
      </div>`;
  }

  function renderPositions() {
    const ps = store.get('positions');
    $('pos-count').textContent = ps.length ? `(${ps.length})` : '';
    const selId = store.get('selectedPosition');
    if (!ps.length) return ($('card-pos').innerHTML = '<div class="empty">flat. waiting for a signal ≥ threshold</div>');
    const now = store.serverNow();
    $('card-pos').innerHTML = ps
      .map(
        (p) => `<div class="pos-row ${p.id === selId ? 'sel' : ''} ${p.symbol === store.get('symbol') ? 'cur' : ''}" data-pid="${esc(p.id)}" data-sym="${esc(p.symbol)}">
        <div class="pos-top">${symLabel(p.symbol)} ${sidePill(p.side)}<span class="dim pos-age">${fmtDuration(now - p.opened_at)}</span>
          <span class="pos-pnl ${signClass(p.upnl)}">${fmtUsd(p.upnl, 2, true)}</span>
          <button type="button" class="btn btn-mini btn-kill" data-close="${esc(p.id)}" title="Close at market">CLOSE</button></div>
        <div class="pos-mid dim">${fmtPrice(p.entry_price)} <span class="arrow">→</span> <span class="${signClass(p.upnl)}">${fmtPrice(p.mark_price)}</span>
          <span class="pos-lv">${esc(p.leverage)}x · $${fmtNum(p.notional, 0)}</span>${p.trail_active ? '<span class="trail-tag">TRAIL</span>' : ''}</div>
        ${slTpBar(p)}
      </div>`
      )
      .join('');
  }

  function renderWhy() {
    const ps = store.get('positions');
    const selId = store.get('selectedPosition');
    const sym = store.get('symbol');
    let reasons = null;
    let sub = '';
    const p = ps.find((x) => x.id === selId) || ps.find((x) => x.symbol === sym);
    if (p && p.reasons && p.reasons.length) {
      reasons = p.reasons;
      sub = `${shortSym(p.symbol)} ${p.side.toUpperCase()} · prob ${isNum(p.prob) ? p.prob.toFixed(2) : '–'}`;
    } else {
      const sigs = store.get('signals')[sym] || [];
      const s = [...sigs].reverse().find((x) => x.reasons && x.reasons.length);
      if (s) {
        reasons = s.reasons;
        sub = `${shortSym(sym)} signal · p_long ${s.prob_long.toFixed(2)}`;
      }
    }
    $('why-sub').textContent = sub ? `· ${sub}` : '';
    $('card-why').innerHTML = reasonBars(reasons);
  }

  $('card-pos').addEventListener('click', async (e) => {
    const closeBtn = e.target.closest('[data-close]');
    if (closeBtn) {
      e.stopPropagation();
      const id = closeBtn.dataset.close;
      const p = store.get('positions').find((x) => String(x.id) === id);
      const ok = await confirmDialog({
        title: 'CLOSE POSITION',
        message: `Close ${p ? `${shortSym(p.symbol)} ${p.side.toUpperCase()}` : `#${id}`} at market now?`,
        confirmText: 'CLOSE NOW',
        danger: true,
      });
      if (!ok) return;
      try {
        await api.control('close_position', { position_id: p ? p.id : Number(id) });
      } catch (err) {
        fx.toast(`close failed: ${err.message}`, 'loss', 5000);
      }
      return;
    }
    const row = e.target.closest('[data-pid]');
    if (row) {
      const p = store.get('positions').find((x) => String(x.id) === row.dataset.pid);
      if (p) {
        store.set('selectedPosition', p.id);
        if (p.symbol !== store.get('symbol')) selectSymbol(p.symbol);
        renderSide();
      }
    }
  });

  const renderSide = () => {
    renderMarket();
    renderBot();
    renderPositions();
    renderWhy();
  };
  const renderSideT = throttle(renderSide, 250);
  const renderStripT = throttle(renderStrip, 500);
  for (const k of ['status', 'market', 'positions', 'signals_loaded']) offs.push(store.on(k === 'signals_loaded' ? k : `change:${k}`, renderSideT));
  offs.push(store.on('position_updated', renderSideT));
  offs.push(store.on('signal', renderSideT));
  offs.push(store.on('tick', renderStripT));
  offs.push(store.on('change:universe', renderStripT));
  offs.push(store.on('change:positions', renderStripT));

  // ------------------------------------------------------------------- log tail
  const lvl = (l) => String(l || 'INFO').toLowerCase().replace(/[^a-z]/g, '');
  const logLine = (e) => `<div class="log lv-${lvl(e.level)}"><span class="log-t">${fmtClock(e.t)}</span><span class="log-l">${esc(String(e.level || '').slice(0, 5))}</span><span class="log-m">${esc(e.msg)}</span></div>`;
  function renderLogs() {
    logEl.innerHTML = `<h3>LOG</h3><div class="log-scroll" id="log-scroll">${store.get('logs').slice(-60).map(logLine).join('')}</div>`;
    const sc = logEl.querySelector('#log-scroll');
    sc.scrollTop = sc.scrollHeight;
  }
  offs.push(store.on('log', ({ entry }) => {
    const sc = logEl.querySelector('#log-scroll');
    if (!sc) return;
    const atBottom = sc.scrollHeight - sc.scrollTop - sc.clientHeight < 40;
    sc.insertAdjacentHTML('beforeend', logLine(entry));
    while (sc.childElementCount > 80) sc.firstElementChild.remove();
    if (atBottom) sc.scrollTop = sc.scrollHeight;
  }));

  // ----------------------------------------------------------------------- init
  renderStrip();
  renderSide();
  renderLogs();
  if (store.get('symbol') && store.get('candles')[store.get('symbol')]) loadChart();
  else ensureChart();

  // keep the "age" fields ticking without WS traffic
  const ageTimer = setInterval(renderPositions, 5000);

  return () => {
    clearInterval(ageTimer);
    offs.forEach((o) => o());
    overlayThrottled.cancel();
    renderSideT.cancel();
    renderStripT.cancel();
    if (chart) chart.destroy();
    chart = null;
  };
}
