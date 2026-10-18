// Top bar: logo, mode pill, heartbeat, safety lamp, equity + giant PnL, drawdown meter + derate gauge,
// START/STOP, KILL, MOCK badge. Built once; updated in place so animations survive.
import { store } from './state.js';
import { api } from './api.js';
import { confirmDialog } from './dialog.js';
import * as fx from './fx.js';
import * as sound from './sound.js';
import { esc, fmtUsd, fmtPct, clamp, isNum, fmtAgo, throttle } from './format.js';
import { derateGauge, safetyOf, SAFETY_CLASS, faultsTable } from './components.js';

const MODE = {
  paper_local: ['PAPER_LOCAL', 'mode-paper'],
  binance_demo: ['BINANCE_DEMO', 'mode-demo'],
  live: ['LIVE', 'mode-live'],
};

export function mountTopbar(el) {
  el.innerHTML = `
    <div class="logo" id="logo" data-text="TRADINGBOT//v2" aria-label="TradingBot v2">TRADINGBOT//v2</div>
    <span class="pill mode-paper" id="mode-pill" title="Trading mode">PAPER_LOCAL</span>
    <div class="hb" id="hb" title="Engine heartbeat"><i class="dot" id="hb-dot"></i><span id="hb-txt">ENGINE</span></div>
    <div class="safety ok" id="safety" tabindex="0" role="button" aria-haspopup="true" aria-label="Safety state">
      <span class="lamp" id="lamp"></span><span class="safety-l"><small>SAFETY</small><b id="safety-state">NORMAL</b></span>
      <div class="safety-pop" id="safety-pop" role="dialog"></div>
    </div>
    <div class="tb-metric"><label>EQUITY</label><b id="tb-equity">–</b></div>
    <div class="pnl-wrap"><label>PNL</label><div class="pnl flat" id="tb-pnl">$0.00</div><span class="pnl-pct" id="tb-pnl-pct"></span></div>
    <div class="dd-wrap" id="dd-wrap" title="Drawdown from peak vs kill limit. Size derating starts at 40% of the limit, x0.5 at 80%, x0 at 100%.">
      <div class="dd-top"><label>DRAWDOWN</label><span id="dd-txt">0.00% / 50%</span></div>
      <div class="dd-bar"><i id="dd-fill"></i><b class="dd-tick" style="left:40%"></b><b class="dd-tick" style="left:80%"></b></div>
    </div>
    <div class="derate" id="derate" title="Position-size multiplier (drawdown derating x drift)"></div>
    <div class="tb-actions">
      <button type="button" class="btn btn-run" id="btn-run">START</button>
      <button type="button" class="btn btn-reset" id="btn-reset" hidden>RESET KILL</button>
      <button type="button" class="btn btn-kill big" id="btn-kill">KILL</button>
    </div>
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
    const pnl = (isNum(s.equity) && isNum(s.start_equity) ? s.equity - s.start_equity : (s.realized_pnl || 0) + (s.unrealized_pnl || 0));
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
    $('derate').innerHTML = derateGauge(sf.derate);

    // buttons
    const tripped = !!s.kill?.tripped;
    const run = $('btn-run');
    run.hidden = tripped;
    run.textContent = s.running ? 'STOP' : 'START';
    run.classList.toggle('is-running', !!s.running);
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
      ${drift ? `<div class="sp-row"><span>drift z (n=${esc(drift.n)})</span><b>${isNum(drift.z) ? drift.z.toFixed(2) : '–'} <span class="dim">x${isNum(drift.mult) ? drift.mult.toFixed(2) : '–'}</span></b></div>` : ''}
      <div class="sp-row"><span>data faults</span><b>${sf.faults.length}</b></div>
      ${sf.faults.length ? `<ul class="sp-reasons">${sf.faults.map((f) => `<li>${esc(f.symbol)} ${esc(f.kind)} x${esc(f.count)}${f.count >= 3 ? ' (excluded)' : ''}</li>`).join('')}</ul>` : ''}
      <div class="sp-note dim">NORMAL → DEGRADED (no entries) → FLATTEN (close all) → HALTED (manual reset)</div>`;
    box.setAttribute('aria-label', `Safety state ${sf.state}${sf.reasons.length ? ': ' + sf.reasons.join('; ') : ''}`);
    box.title = sf.reasons.length ? `${sf.state}: ${sf.reasons.join(' | ')}` : `${sf.state}: all guards green`;
  }

  function renderHealth() {
    const h = store.get('health');
    const dot = $('hb-dot');
    const txt = $('hb-txt');
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
    } else {
      label = isNum(age) ? `ENGINE ${age.toFixed(1)}s` : 'ENGINE';
    }
    dot.className = `dot ${cls}`;
    fx.pulse(dot, 'beat', 700);
    txt.textContent = label;
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

  // ------------------------------------------------------------------ actions
  async function doControl(action, extra, errPrefix) {
    try {
      await api.control(action, extra);
    } catch (e) {
      fx.toast(`${errPrefix}: ${e.message}`, 'loss', 5000);
    }
  }

  $('btn-run').addEventListener('click', () => {
    const running = store.get('status')?.running;
    sound.play('click');
    doControl(running ? 'stop' : 'start', {}, running ? 'stop failed' : 'start failed');
  });
  $('btn-reset').addEventListener('click', async () => {
    const ok = await confirmDialog({ title: 'RESET KILL SWITCH', message: 'Re-arm the bot after a kill? It will stay stopped until you press START.', confirmText: 'RESET', danger: false });
    if (ok) doControl('reset_kill', {}, 'reset failed');
  });
  $('btn-kill').addEventListener('click', async () => {
    const ok = await confirmDialog({
      title: 'KILL SWITCH',
      message: 'Close ALL positions at market, stop opening trades and halt the bot until a manual reset. Continue?',
      confirmText: 'KILL EVERYTHING',
      danger: true,
    });
    if (ok) doControl('kill', {}, 'kill failed');
  });

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

  const rs = throttle(renderStatus, 200);
  store.on('change:status', rs);
  store.on('change:health', renderHealth);
  store.on('change:conn', renderConn);
  renderStatus();
  renderHealth();
  renderConn();
}
