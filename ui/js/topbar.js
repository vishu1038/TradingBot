// Top bar (visible on every page): logo, mode pill, heartbeat, safety lamp, equity + giant PnL,
// drawdown meter + derate gauge, START/STOP, KILL, `/` palette, MOCK badge. Built once, updated in place.
import { store } from './state.js';
import { api } from './api.js';
import * as fx from './fx.js';
import { fmtUsd, fmtPct, clamp, isNum, throttle } from './format.js';
import { derateGauge, safetyOf, SAFETY_CLASS } from './components.js';
import { esc } from './format.js';
import { toggleRun, killAll, resetKill } from './actions.js';

const MODE = {
  paper_local: ['PAPER_LOCAL', 'mode-paper'],
  binance_demo: ['BINANCE_DEMO', 'mode-demo'],
  live: ['LIVE', 'mode-live'],
};

export function mountTopbar(el) {
  el.innerHTML = `
    <div class="logo deco" id="logo" data-text="TRADINGBOT//v2" aria-label="TradingBot v2">TRADINGBOT//v2</div>
    <span class="pill mode-paper" id="mode-pill" title="Trading mode">PAPER_LOCAL</span>
    <div class="hb" id="hb" title="Engine heartbeat"><i class="dot" id="hb-dot"></i><span id="hb-txt">ENGINE</span></div>
    <div class="safety ok" id="safety" tabindex="0" role="button" aria-haspopup="true" aria-label="Safety state">
      <span class="lamp" id="lamp"></span><span class="safety-l"><small>SAFETY</small><b id="safety-state">NORMAL</b></span>
      <div class="safety-pop" id="safety-pop" role="dialog"></div>
    </div>
    <div class="tb-metric tb-equity"><label>EQUITY</label><b id="tb-equity">–</b></div>
    <div class="pnl-wrap"><div class="pnl-top"><label>PNL</label><span class="pnl-pct" id="tb-pnl-pct"></span></div><div class="pnl flat" id="tb-pnl">$0.00</div></div>
    <div class="dd-wrap" id="dd-wrap" title="Drawdown from peak vs kill limit. Size derating starts at 40% of the limit, x0.5 at 80%, x0 at 100%.">
      <div class="dd-top"><label>DRAWDOWN</label><span id="dd-txt">0.00% / 50%</span></div>
      <div class="dd-bar"><i id="dd-fill"></i><b class="dd-tick" style="left:40%"></b><b class="dd-tick" style="left:80%"></b></div>
    </div>
    <div class="derate" id="derate" title="Position-size multiplier (drawdown derating x drift)"></div>
    <div class="tb-actions">
      <button type="button" class="pillbtn on" id="btn-run">START</button>
      <button type="button" class="pillbtn cy" id="btn-reset" hidden>RESET KILL</button>
      <button type="button" class="btn btn-kill big" id="btn-kill">KILL</button>
      <button type="button" class="pillbtn pal-btn" id="btn-pal" title="Command palette ( / )" aria-label="Command palette"><span class="deco">/</span><span class="pal-btn-t">commands</span></button>
    </div>
    <i class="tb-break"></i><i class="tb-break b2"></i>
    <span class="mock-badge" id="mock-badge" hidden>MOCK DATA</span>
    <span class="conn" id="conn" title="WebSocket state"></span>`;

  const $ = (id) => el.querySelector(`#${id}`);
  const logo = $('logo');
  fx.startIdleGlitch(logo);
  logo.addEventListener('click', () => fx.glitch(logo, 900));

  let lastPnl = null;
  let lastSafetyKey = '';
  let lastState = 'NORMAL';

  function renderStatus() {
    const s = store.get('status');
    if (!s) return;
    const [mlabel, mcls] = MODE[s.mode] || [String(s.mode || '?').toUpperCase(), 'mode-paper'];
    const mp = $('mode-pill');
    mp.textContent = mlabel;
    mp.className = `pill ${mcls}`;

    $('tb-equity').textContent = fmtUsd(s.equity);
    const pnl = isNum(s.equity) && isNum(s.start_equity) ? s.equity - s.start_equity : (s.realized_pnl || 0) + (s.unrealized_pnl || 0);
    const pnlEl = $('tb-pnl');
    pnlEl.textContent = fmtUsd(pnl, 2, true);
    pnlEl.className = `pnl ${pnl > 0.005 ? 'pos' : pnl < -0.005 ? 'neg' : 'flat'}`;
    if (lastPnl !== null && Math.abs(pnl - lastPnl) > 0.004) fx.pulse(pnlEl, pnl > lastPnl ? 'bump-up' : 'bump-dn', 450);
    lastPnl = pnl;
    $('tb-pnl-pct').textContent = isNum(s.start_equity) && s.start_equity > 0 ? fmtPct((pnl / s.start_equity) * 100, 2) : '';
    $('tb-pnl-pct').className = `pnl-pct ${pnl >= 0 ? 'pos' : 'neg'}`;

    const limit = s.kill?.dd_limit_pct || 50;
    const dd = s.drawdown_pct || 0;
    const frac = clamp(dd / limit, 0, 1);
    const fill = $('dd-fill');
    fill.style.width = `${(frac * 100).toFixed(1)}%`;
    fill.className = frac < 0.4 ? 'z-ok' : frac < 0.8 ? 'z-warn' : 'z-crit';
    $('dd-txt').textContent = `${dd.toFixed(2)}% / ${limit}%`;

    const sf = safetyOf(s);
    const dr = $('derate');
    const key = sf.derate.toFixed(2);
    if (dr.dataset.k !== key) {
      dr.innerHTML = derateGauge(sf.derate);
      dr.dataset.k = key;
    }

    const tripped = !!s.kill?.tripped;
    const run = $('btn-run');
    run.hidden = tripped;
    run.textContent = s.running ? 'STOP' : 'START';
    run.className = `pillbtn ${s.running ? 'amb' : 'on'}`;
    $('btn-reset').hidden = !tripped;
    renderSafety(sf);
  }

  function renderSafety(sf) {
    const key = JSON.stringify([sf.state, sf.reasons, sf.derate, sf.drift, sf.faults]);
    const box = $('safety');
    if (key === lastSafetyKey) return;
    lastSafetyKey = key;
    const cls = SAFETY_CLASS[sf.state] || 'warn';
    box.className = `safety ${cls}${box.classList.contains('open') ? ' open' : ''}`;
    $('safety-state').textContent = sf.state;
    if (sf.state !== lastState) {
      if (sf.state === 'FLATTEN' || sf.state === 'HALTED') fx.glitch(logo, 900);
      lastState = sf.state;
    }
    const drift = sf.drift;
    $('safety-pop').innerHTML = `
      <div class="sp-h ${cls}">${esc(sf.state)}</div>
      ${sf.reasons.length ? `<ul class="sp-reasons">${sf.reasons.map((r) => `<li>${esc(r)}</li>`).join('')}</ul>` : '<div class="dim sp-ok">all guards green. entries allowed.</div>'}
      <div class="sp-row"><span>size multiplier</span><b>x${sf.derate.toFixed(2)}</b></div>
      ${drift && isNum(drift.z) ? `<div class="sp-row"><span>drift z (n=${esc(drift.n)})</span><b>${drift.z.toFixed(2)} <span class="dim">x${isNum(drift.mult) ? drift.mult.toFixed(2) : '–'}</span></b></div>` : ''}
      <div class="sp-row"><span>data faults</span><b>${sf.faults.length}</b></div>
      ${sf.faults.length ? `<ul class="sp-reasons">${sf.faults.map((f) => `<li>${esc(f.symbol)} ${esc(f.kind)} x${esc(f.count)}${f.count >= 3 ? ' (excluded)' : ''}</li>`).join('')}</ul>` : ''}
      <div class="sp-note dim">NORMAL → DEGRADED (no entries) → FLATTEN (close all) → HALTED (manual reset)</div>`;
    box.setAttribute('aria-label', `Safety state ${sf.state}${sf.reasons.length ? ': ' + sf.reasons.join('; ') : ''}`);
    box.title = sf.reasons.length ? `${sf.state}: ${sf.reasons.join(' | ')}` : `${sf.state}: all guards green`;
  }

  function renderHealth() {
    const h = store.get('health');
    if (!h) return;
    const age = h.engine_heartbeat_age_s;
    let cls = 'ok';
    let label = 'ENGINE';
    if (h.unreachable || !h.ok) {
      cls = 'crit';
      label = 'API DOWN';
    } else if (!h.engine_alive || (isNum(age) && age > 15)) {
      cls = 'crit';
      label = `ENGINE ${isNum(age) ? Math.round(age) + 's' : 'DEAD'}`;
    } else if (isNum(age) && age > 5) {
      cls = 'warn';
      label = `ENGINE ${Math.round(age)}s`;
    } else label = isNum(age) ? `ENGINE ${age.toFixed(1)}s` : 'ENGINE';
    $('hb-dot').className = `dot ${cls}`;
    fx.pulse($('hb-dot'), 'beat', 700);
    $('hb-txt').textContent = label;
    $('hb').title = `Engine heartbeat age: ${isNum(age) ? age + ' s' : 'unknown'}  ·  API ${api.base}${api.isMock ? ' (mock)' : ''}`;
  }

  function renderConn() {
    const c = store.get('conn');
    $('mock-badge').hidden = !c.mock;
    $('mock-badge').title = c.reason || 'in-browser mock backend';
    const ws = c.mock ? 'mock stream' : c.ws === 'open' ? 'WS live' : c.ws === 'connecting' ? 'WS connecting…' : c.ws === 'closed' ? 'WS reconnecting…' : 'WS idle';
    const conn = $('conn');
    conn.textContent = ws;
    conn.className = `conn ${c.mock || c.ws === 'open' ? 'ok' : 'warn'}`;
  }

  $('btn-run').addEventListener('click', toggleRun);
  $('btn-reset').addEventListener('click', resetKill);
  $('btn-kill').addEventListener('click', killAll);
  $('btn-pal').addEventListener('click', () => store.emit('palette', '/'));

  const safety = $('safety');
  safety.addEventListener('click', (e) => {
    e.stopPropagation();
    safety.classList.toggle('open');
  });
  safety.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      safety.classList.toggle('open');
    }
  });
  document.addEventListener('click', () => safety.classList.remove('open'));

  store.on('change:status', throttle(renderStatus, 200));
  store.on('change:health', renderHealth);
  store.on('change:conn', renderConn);
  renderStatus();
  renderHealth();
  renderConn();
}
