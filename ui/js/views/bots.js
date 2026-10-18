// BOTS view: equity + drawdown charts, stat tiles, model card, risk governor, shadow models, controls.
import { store } from '../state.js';
import { api } from '../api.js';
import { createLineChart } from '../charts.js';
import { confirmDialog } from '../dialog.js';
import * as fx from '../fx.js';
import { esc, fmtUsd, fmtPct, fmtDateTime, fmtNum, signClass, throttle, winRatePct, isNum, clamp, fmtAgo } from '../format.js';
import { tile, kv, safetyOf, derateGauge, pnlSpan, SAFETY_CLASS } from '../components.js';

/** Same curve as backend risk.derate_multiplier (used only to DRAW the curve). */
export function derateCurve(ddPct, killPct) {
  const f = killPct > 0 ? ddPct / killPct : 0;
  if (f <= 0.4) return 1;
  if (f <= 0.8) return 1 - ((f - 0.4) / 0.4) * 0.5;
  if (f < 1) return 0.5 * (1 - (f - 0.8) / 0.2);
  return 0;
}

function curveSvg(dd, limit, mult) {
  const W = 260;
  const H = 96;
  const px = (f) => 8 + f * (W - 16);
  const py = (m) => 8 + (1 - m) * (H - 22);
  const pts = [0, 0.4, 0.8, 1].map((f) => `${px(f).toFixed(1)},${py(derateCurve(f * limit, limit)).toFixed(1)}`).join(' ');
  const f = clamp(limit > 0 ? dd / limit : 0, 0, 1);
  return `<svg class="curve" viewBox="0 0 ${W} ${H}" width="100%" height="${H}" role="img" aria-label="Derating curve">
    <line x1="8" y1="${py(0)}" x2="${W - 8}" y2="${py(0)}" stroke="rgba(255,255,255,0.15)"/>
    <line x1="${px(0.4)}" y1="8" x2="${px(0.4)}" y2="${py(0)}" stroke="rgba(255,255,255,0.1)" stroke-dasharray="2 3"/>
    <line x1="${px(0.8)}" y1="8" x2="${px(0.8)}" y2="${py(0)}" stroke="rgba(255,255,255,0.1)" stroke-dasharray="2 3"/>
    <polyline points="${pts}" fill="none" stroke="var(--coral)" stroke-width="2"/>
    <circle cx="${px(f).toFixed(1)}" cy="${py(mult).toFixed(1)}" r="5" fill="var(--cyan)" style="filter:drop-shadow(0 0 4px var(--cyan))"/>
    <text x="${px(0.4)}" y="${H - 3}" text-anchor="middle" class="curve-t">40%</text>
    <text x="${px(0.8)}" y="${H - 3}" text-anchor="middle" class="curve-t">80%</text>
    <text x="${px(1)}" y="${H - 3}" text-anchor="end" class="curve-t">kill ${limit}%</text>
    <text x="10" y="18" class="curve-t">x1.0</text>
  </svg>`;
}

export function mount(root) {
  root.innerHTML = `
    <div class="page bots">
      <div class="card span2"><h3>EQUITY <span class="why-sub" id="eq-sub"></span></h3><div class="line-chart tall" id="eq-chart"></div></div>
      <div class="card"><h3>STATS</h3><div id="bot-tiles"></div></div>
      <div class="card"><h3>DRAWDOWN</h3><div class="line-chart" id="dd-chart"></div></div>
      <div class="card"><h3>MODEL</h3><div id="model-card"></div></div>
      <div class="card"><h3>RISK GOVERNOR <span class="why-sub">drawdown derating + drift</span></h3><div id="risk-card"></div></div>
      <div class="card span2"><h3>SHADOW MODELS <span class="why-sub">challengers score every bar; promote needs ≥ 50 shadow trades + higher expectancy + no drift alarm</span></h3><div id="shadow-card"></div></div>
      <div class="card"><h3>CONTROLS</h3><div id="ctl-card"></div></div>
    </div>`;
  const $ = (id) => root.querySelector(`#${id}`);
  const offs = [];
  let eqChart = null;
  let ddChart = null;

  function drawdownSeries(eq) {
    let peak = -Infinity;
    return eq.map((p) => {
      peak = Math.max(peak, p.equity);
      return { t: p.t, v: peak > 0 ? -((peak - p.equity) / peak) * 100 : 0 };
    });
  }

  function loadCharts(fit = true) {
    const eq = store.get('equity');
    const start = store.get('status')?.start_equity;
    if (!eqChart && typeof LightweightCharts !== 'undefined') {
      eqChart = createLineChart($('eq-chart'), { kind: 'baseline', baseValue: isNum(start) ? start : eq[0]?.equity ?? 0, format: (v) => fmtUsd(v) });
      ddChart = createLineChart($('dd-chart'), { kind: 'baseline', baseValue: 0, format: (v) => `${v.toFixed(2)}%` });
    }
    if (!eqChart) return;
    eqChart.setData(eq.map((p) => ({ t: p.t, v: p.equity })), fit);
    eqChart.setBase(start);
    ddChart.setData(drawdownSeries(eq), fit);
    $('eq-sub').textContent = eq.length ? `· ${eq.length} pts · since ${fmtDateTime(eq[0].t)}` : '';
  }

  let peakSoFar = -Infinity;
  offs.push(store.on('equity_point', (p) => {
    if (!eqChart) return;
    eqChart.update({ t: p.t, v: p.equity });
    peakSoFar = Math.max(peakSoFar, p.equity, store.get('status')?.peak_equity ?? 0);
    ddChart.update({ t: p.t, v: peakSoFar > 0 ? -((peakSoFar - p.equity) / peakSoFar) * 100 : 0 });
  }));
  offs.push(store.on('change:equity', () => loadCharts(false)));

  function renderTiles() {
    const s = store.get('status');
    if (!s) return ($('bot-tiles').innerHTML = '<div class="empty">no status yet</div>');
    const wr = winRatePct(s.win_rate);
    $('bot-tiles').innerHTML = `<div class="tiles t3">
      ${tile('EQUITY', fmtUsd(s.equity), signClass(s.equity - s.start_equity))}
      ${tile('START', fmtUsd(s.start_equity))}
      ${tile('PEAK', fmtUsd(s.peak_equity))}
      ${tile('CASH', fmtUsd(s.cash))}
      ${tile('REALIZED', fmtUsd(s.realized_pnl, 2, true), signClass(s.realized_pnl))}
      ${tile('UNREALIZED', fmtUsd(s.unrealized_pnl, 2, true), signClass(s.unrealized_pnl))}
      ${tile('WIN RATE', wr == null ? '–' : `${wr.toFixed(1)}%`, wr >= 50 ? 'pos' : 'neg')}
      ${tile('TRADES TODAY', esc(s.trades_today))}
      ${tile('DRAWDOWN', fmtPct(s.drawdown_pct, 2, false), s.drawdown_pct > 10 ? 'neg' : '')}
    </div>`;
    const m = s.model || {};
    $('model-card').innerHTML = `<div class="kv-grid">
      ${kv('name', esc(m.name || '–'))}
      ${kv('version', esc(m.version || '–'))}
      ${kv('trained', m.trained_at ? `${fmtDateTime(m.trained_at)} <span class="dim">(${fmtAgo(m.trained_at, store.serverNow())} ago)</span>` : '–')}
      ${kv('long threshold', isNum(m.threshold_long) ? m.threshold_long.toFixed(2) : '–')}
      ${kv('short threshold', isNum(m.threshold_short) ? m.threshold_short.toFixed(2) : '–')}
      ${kv('mode', esc(s.mode))}
      ${kv('universe updated', s.universe_updated_at ? `${fmtAgo(s.universe_updated_at, store.serverNow())} ago` : '–')}
    </div>`;
    const sf = safetyOf(s);
    const limit = s.kill?.dd_limit_pct || 50;
    $('risk-card').innerHTML = `
      <div class="risk-top"><span class="lamp-s ${SAFETY_CLASS[sf.state] || 'warn'}">${esc(sf.state)}</span>${derateGauge(sf.derate)}<span class="dim">size multiplier</span></div>
      ${curveSvg(s.drawdown_pct || 0, limit, sf.derate)}
      <div class="kv-grid">
        ${kv('drawdown', `${(s.drawdown_pct || 0).toFixed(2)}% of ${limit}% kill limit`)}
        ${sf.drift ? kv('drift z / n', `${isNum(sf.drift.z) ? sf.drift.z.toFixed(2) : '–'} / ${esc(sf.drift.n)} <span class="dim">size x${isNum(sf.drift.mult) ? sf.drift.mult.toFixed(2) : '–'}</span>`) : ''}
        ${sf.reasons.length ? kv('reasons', esc(sf.reasons.join('; ')), 'warn') : ''}
      </div>`;
    const tripped = s.kill?.tripped;
    $('ctl-card').innerHTML = `
      <div class="ctl">
        ${tripped ? '<button class="btn btn-reset" data-ctl="reset_kill">RESET KILL</button>' : `<button class="btn ${s.running ? 'btn-run is-running' : 'btn-run'}" data-ctl="${s.running ? 'stop' : 'start'}">${s.running ? 'STOP ENTRIES' : 'START'}</button>`}
        <button class="btn" data-ctl="close_all">CLOSE ALL</button>
        <button class="btn btn-kill" data-ctl="kill">KILL</button>
      </div>
      <div class="dim ctl-note">${tripped ? `kill tripped: ${esc(s.kill.reason || '')}` : s.running ? 'running: entries enabled' : 'stopped: open trades keep running to TP/SL'}</div>`;
  }

  function renderShadow() {
    const rows = store.get('shadow');
    if (!rows || !rows.length) return ($('shadow-card').innerHTML = '<div class="empty">no models reported by /api/shadow</div>');
    const champ = rows.find((r) => r.status === 'champion');
    const sf = safetyOf(store.get('status'));
    $('shadow-card').innerHTML = `<div class="tbl-wrap"><table class="tbl"><thead><tr><th>model</th><th>status</th><th class="r">trades</th><th>progress</th><th class="r">win rate</th><th class="r">expectancy</th><th class="r">PnL</th><th class="r">vs champion</th><th></th></tr></thead><tbody>
      ${rows
        .map((r) => {
          const wr = winRatePct(r.win_rate);
          const prog = r.status === 'challenger' ? `<span class="prog"><i style="width:${clamp(((r.n_trades || 0) / 50) * 100, 0, 100)}%"></i></span> <span class="dim">${esc(r.n_trades)}/50</span>` : '';
          let act = '';
          if (r.status === 'challenger') {
            const ready = (r.n_trades || 0) >= 50;
            act = `<button class="btn btn-mini btn-primary" data-promote="${esc(r.model)}" ${ready ? '' : 'title="needs >= 50 shadow trades (the backend re-checks all gates)"'}>PROMOTE</button>`;
          } else if (r.status === 'champion') {
            act = `<button class="btn btn-mini" data-rollback="${esc(r.model)}" title="restore the previous champion">ROLLBACK</button>`;
          }
          return `<tr><td>${esc(r.model)}</td><td><span class="mstat mstat-${esc(r.status)}">${esc(String(r.status).toUpperCase())}</span></td>
            <td class="r">${esc(r.n_trades)}</td><td>${prog}</td><td class="r">${wr == null ? '–' : wr.toFixed(1) + '%'}</td>
            <td class="r">${pnlSpan(r.expectancy, 3)}</td><td class="r">${pnlSpan(r.pnl)}</td>
            <td class="r">${r === champ ? '<span class="dim">baseline</span>' : pnlSpan(r.vs_champion, 3)}</td><td class="r">${act}</td></tr>`;
        })
        .join('')}
    </tbody></table></div>${sf.state === 'DEGRADED' || (sf.drift && sf.drift.z < -3) ? '<div class="warn shadow-note">drift alarm active: promotion is blocked</div>' : ''}`;
  }

  async function control(action, extra, errPrefix) {
    try {
      await api.control(action, extra);
      return true;
    } catch (e) {
      fx.toast(`${errPrefix}: ${e.message}`, 'loss', 6000);
      return false;
    }
  }

  root.addEventListener('click', async (e) => {
    const pr = e.target.closest('[data-promote]');
    if (pr) {
      const model = pr.dataset.promote;
      const ok = await confirmDialog({
        title: 'PROMOTE MODEL',
        message: `Promote ${model} to CHAMPION? It will take over live entries immediately. You can roll back at any time.`,
        confirmText: 'PROMOTE',
      });
      if (ok && (await control('promote_model', { model }, 'promote refused'))) {
        fx.toast(`${model} promoted to champion`, 'win');
        const r = await api.shadow().catch(() => null);
        if (r) store.set('shadow', r);
      }
      return;
    }
    const rb = e.target.closest('[data-rollback]');
    if (rb) {
      const model = rb.dataset.rollback;
      const ok = await confirmDialog({
        title: 'ROLLBACK MODEL',
        message: `Roll back from ${model} to the previous champion instantly?`,
        confirmText: 'ROLLBACK',
        danger: true,
      });
      if (ok && (await control('rollback_model', { model }, 'rollback failed'))) {
        fx.toast('previous champion restored', 'info');
        const r = await api.shadow().catch(() => null);
        if (r) store.set('shadow', r);
      }
      return;
    }
    const c = e.target.closest('[data-ctl]');
    if (!c) return;
    const a = c.dataset.ctl;
    if (a === 'kill' || a === 'close_all') {
      const ok = await confirmDialog({
        title: a === 'kill' ? 'KILL SWITCH' : 'CLOSE ALL',
        message: a === 'kill' ? 'Close ALL positions, stop trading and halt until a manual reset?' : 'Close every open position at market?',
        confirmText: a === 'kill' ? 'KILL EVERYTHING' : 'CLOSE ALL',
        danger: true,
      });
      if (!ok) return;
    }
    control(a, {}, `${a} failed`);
  });

  const renderT = throttle(() => {
    renderTiles();
  }, 500);
  offs.push(store.on('change:status', renderT));
  offs.push(store.on('change:shadow', renderShadow));

  loadCharts(true);
  renderTiles();
  renderShadow();
  api.shadow().then((r) => r && store.set('shadow', r)).catch(() => {});

  return () => {
    offs.forEach((o) => o());
    renderT.cancel();
    eqChart && eqChart.destroy();
    ddChart && ddChart.destroy();
  };
}
