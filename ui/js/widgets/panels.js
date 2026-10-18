// Right-column / carousel panels: BOT (stats + controls) and MARKET (BTC trend, volatility, funding, L/S).
import { store } from '../state.js';
import {
  esc, fmtUsd, fmtPct, fmtPrice, fmtFunding, fmtNum, signClass, throttle, winRatePct, isNum, shortSym, clamp, fmtAgo,
} from '../format.js';
import { tile, kv, safetyOf, symLabel } from '../components.js';
import { spinner } from '../fx.js';
import { toggleRun, closeSymbol, togglePause, refreshUniverse, pausedSet, positionOf, resetKill, closeAll } from '../actions.js';

const TREND = { up: ['▲ UP', 'pos'], down: ['▼ DOWN', 'neg'], flat: ['▬ FLAT', 'flat'] };

export function createBotPanel() {
  const el = document.createElement('div');
  el.className = 'card botpanel ac-violet';
  el.innerHTML = '<h3>BOT <span class="why-sub" id="b-sub"></span></h3><div id="b-body"></div>';
  const body = el.querySelector('#b-body');
  const offs = [];

  function render() {
    const s = store.get('status');
    if (!s) {
      body.innerHTML = `<div class="empty">${spinner()} waiting for /api/status…</div>`;
      return;
    }
    const wr = winRatePct(s.win_rate);
    const sf = safetyOf(s);
    const m = s.model || {};
    const d = sf.drift;
    const sym = store.get('symbol');
    const held = positionOf(sym);
    const paused = pausedSet().has(sym);
    const tripped = !!s.kill?.tripped;
    el.querySelector('#b-sub').innerHTML = `· ${esc(String(s.mode || '').toUpperCase())}`;
    body.innerHTML = `
      <div class="tiles">
        ${tile('WIN', wr == null ? '–' : `${wr.toFixed(1)}%`, wr == null ? '' : wr >= 50 ? 'pos' : 'neg')}
        ${tile('OPEN', `${esc(s.open_positions)}/${esc(s.max_positions)}`, 'c-sky')}
        ${tile('TODAY', esc(s.trades_today), 'c-pink')}
        ${tile('SIZE', `x${fmtNum(sf.derate, 2)}`, sf.derate >= 0.99 ? 'c-amber' : sf.derate >= 0.5 ? 'warn' : 'neg')}
      </div>
      <div class="kv-grid">
        ${kv('model', `${esc(m.name || '–')} <span class="dim">v${esc(m.version || '–')}</span>`)}
        ${kv('thresholds', `L ${isNum(m.threshold_long) ? m.threshold_long.toFixed(2) : '–'} <span class="dim">/</span> S ${isNum(m.threshold_short) ? m.threshold_short.toFixed(2) : '–'}`)}
        ${d && isNum(d.z) ? kv(`drift z (n=${esc(d.n)})`, `<span class="${d.z < -3 ? 'neg' : d.z < -2 ? 'warn' : 'c-teal'}">${d.z.toFixed(2)}</span> <span class="dim">size x${isNum(d.mult) ? d.mult.toFixed(2) : '–'}</span>`) : kv('drift', `<span class="dim">warming up${d ? ` (n=${esc(d.n)})` : ''}</span>`)}
        ${kv('realized', `<span class="${signClass(s.realized_pnl)}">${fmtUsd(s.realized_pnl, 2, true)}</span>`)}
        ${kv('unrealized', `<span class="${signClass(s.unrealized_pnl)}">${fmtUsd(s.unrealized_pnl, 2, true)}</span>`)}
      </div>
      <div class="ctl-label">CONTROLS</div>
      <div class="ctl-row">
        ${
          tripped
            ? '<button type="button" class="pillbtn cy" data-do="reset">RESET KILL</button>'
            : `<button type="button" class="pillbtn ${s.running ? 'amb' : 'on'}" data-do="run"><span class="deco">${s.running ? '■' : '▶'}</span> ${s.running ? 'STOP ENTRIES' : 'START'}</button>`
        }
        <button type="button" class="pillbtn vio" data-do="refresh" title="Force a universe re-rank"><span class="deco spin-now">⟳</span> REFRESH UNIVERSE</button>
      </div>
      <div class="ctl-row">
        <span class="ctl-coin">${symLabel(sym)}</span>
        <button type="button" class="pillbtn ${held ? 'bad' : ''}" data-do="close" ${held ? '' : 'disabled'} title="Close this coin's open position">CLOSE ×</button>
        <button type="button" class="pillbtn ${paused ? 'pnk' : ''}" data-do="pause" title="Pause = no new entries on this coin; an open trade keeps running">${paused ? '▶ RESUME' : '⏸ PAUSE'}</button>
      </div>
      ${s.open_positions > 1 ? '<div class="ctl-row"><button type="button" class="pillbtn bad ghost" data-do="closeall">CLOSE ALL</button></div>' : ''}`;
  }

  body.addEventListener('click', (e) => {
    const b = e.target.closest('[data-do]');
    if (!b || b.disabled) return;
    const sym = store.get('symbol');
    switch (b.dataset.do) {
      case 'run': return toggleRun();
      case 'reset': return resetKill();
      case 'refresh': return refreshUniverse();
      case 'close': return closeSymbol(sym);
      case 'pause': return togglePause(sym);
      case 'closeall': return closeAll();
      default: return undefined;
    }
  });

  const r = throttle(render, 300);
  for (const t of ['change:status', 'change:symbol', 'change:positions', 'change:pausedLocal']) offs.push(store.on(t, r));
  render();
  return { el, destroy: () => { r.cancel(); offs.forEach((o) => o()); } };
}

export function createMarketPanel() {
  const el = document.createElement('div');
  el.className = 'card marketpanel ac-amber';
  el.innerHTML = '<h3>MARKET <span class="why-sub" id="m-sub"></span></h3><div id="m-body"></div>';
  const body = el.querySelector('#m-body');
  const offs = [];
  function render() {
    const m = store.get('market');
    if (!m) {
      body.innerHTML = `<div class="empty">${spinner()} waiting for /api/market…</div>`;
      return;
    }
    const [tl, tc] = TREND[m.btc_trend] || TREND.flat;
    const lsr = isNum(m.long_short_ratio) ? m.long_short_ratio : null;
    const longShare = lsr == null ? 50 : (lsr / (1 + lsr)) * 100;
    el.querySelector('#m-sub').textContent = m.updated_at ? `· ${fmtAgo(m.updated_at, store.serverNow())} ago` : '';
    body.innerHTML = `
      <div class="mk-btc"><span class="mk-btc-l">BTC</span><span class="mk-btc-p">${fmtPrice(m.btc_price)}</span>
        <span class="${signClass(m.btc_change_24h_pct)}">${fmtPct(m.btc_change_24h_pct)}</span><span class="${tc} mk-trend">${tl}</span></div>
      <div class="kv-grid">
        ${kv('volatility (ATR%)', fmtPct(m.market_volatility_pct, 2, false))}
        ${kv('avg funding', `<span class="${signClass(m.avg_funding_rate)}">${fmtFunding(m.avg_funding_rate)}</span>`)}
        ${kv('long / short ratio', lsr == null ? '–' : fmtNum(lsr, 2))}
      </div>
      <div class="ls-bar" title="long vs short accounts"><i style="width:${longShare.toFixed(1)}%"></i></div>
      <div class="ls-lbl"><span>LONG ${longShare.toFixed(0)}%</span><span>SHORT ${(100 - longShare).toFixed(0)}%</span></div>`;
  }
  const r = throttle(render, 500);
  offs.push(store.on('change:market', r));
  render();
  return { el, destroy: () => { r.cancel(); offs.forEach((o) => o()); } };
}
