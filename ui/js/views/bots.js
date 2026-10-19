// BOTS view: equity + drawdown charts, stat tiles, model card, risk governor, shadow models, controls.
import { store } from '../state.js';
import { api } from '../api.js';
import { createLineChart, createMultiLineChart } from '../charts.js';
import { modelColor } from '../theme.js';
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

const HIDE_KEY = 'tb.bots.hidden';
const ACCOUNT = 'account';
const REFRESH_MS = 60000;

function loadHidden() {
  try {
    return new Set(JSON.parse(localStorage.getItem(HIDE_KEY) || '[]'));
  } catch (_) {
    return new Set();
  }
}
function saveHidden(set) {
  try {
    localStorage.setItem(HIDE_KEY, JSON.stringify([...set]));
  } catch (_) {
    /* storage blocked: the choice just does not persist */
  }
}

/** 'cnn_atr32_20261018_1716_papertest' -> 'cnn_atr32 papertest 1018-1716' (the long timestamp is what makes names unreadable). */
export function shortModel(name) {
  const m = String(name).match(/^(.*?)_\d{4}(\d{2})(\d{2})_(\d{2})(\d{2})(?:_(.+))?$/);
  if (!m) return String(name);
  return `${m[1]}${m[6] ? ' ' + m[6] : ''} ${m[2]}${m[3]}-${m[4]}${m[5]}`;
}
const kindLabel = (k) => String(k || 'per_symbol_5m').replace(/_5m$/, '');

export function mount(root) {
  root.innerHTML = `
    <div class="page bots">
      <div class="card span2 ac-magenta" id="ab-card">
        <h3>ALL BOTS <span class="why-sub" id="ab-sub"></span></h3>
        <div class="ab-bar"><button class="btn btn-mini" data-ab="all">SHOW ALL</button><button class="btn btn-mini" data-ab="none">HIDE ALL</button><button class="btn btn-mini" data-ab="fit">FIT</button><span class="dim ab-hint">click a row to hide / show its line</span></div>
        <div class="line-chart tall ab-chart" id="ab-chart"></div>
        <div class="tbl-wrap ab-legend" id="ab-legend"></div>
      </div>
      <div class="card span2 ac-coral"><h3>EQUITY <span class="why-sub" id="eq-sub"></span></h3><div class="line-chart tall" id="eq-chart"></div></div>
      <div class="card ac-violet"><h3>STATS</h3><div id="bot-tiles"></div></div>
      <div class="card ac-cyan"><h3>DRAWDOWN</h3><div class="line-chart" id="dd-chart"></div></div>
      <div class="card ac-pink"><h3>MODEL</h3><div id="model-card"></div></div>
      <div class="card ac-amber"><h3>RISK GOVERNOR <span class="why-sub">drawdown derating + drift</span></h3><div id="risk-card"></div></div>
      <div class="card span2 ac-sky"><h3>SHADOW MODELS <span class="why-sub">challengers score every bar; promote needs ≥ 50 shadow trades + higher expectancy + no drift alarm</span></h3><div id="shadow-card"></div></div>
      <div class="card ac-teal"><h3>CONTROLS</h3><div id="ctl-card"></div></div>
    </div>`;
  const $ = (id) => root.querySelector(`#${id}`);
  const offs = [];
  let eqChart = null;
  let ddChart = null;
  let abChart = null;
  let shadowEq = []; // GET /api/shadow/equity
  let abError = '';
  let hoverT = null;
  let lastLegendHtml = '';
  let lastStart = null;
  const hidden = loadHidden();
  let destroyed = false;

  const startEq = () => store.get('status')?.start_equity;

  function legendRows() {
    const stat = new Map((store.get('shadow') || []).map((r) => [r.model, r]));
    const rank = { champion: 0, challenger: 1, retired: 2 };
    const models = [...shadowEq].sort((a, b) => (rank[a.status] ?? 3) - (rank[b.status] ?? 3) || a.model.localeCompare(b.model));
    models.forEach((m) => modelColor(m.model)); // assign colours in a stable order
    const rows = [];
    const eq = store.get('equity');
    const st = store.get('status');
    if (eq.length && isNum(startEq())) {
      const trades = store.get('trades') || [];
      const total = (st && isNum(st.equity) ? st.equity : eq[eq.length - 1].equity) - startEq();
      const n = trades.length;
      rows.push({ id: ACCOUNT, name: 'ACCOUNT (real)', title: 'the real bot: equity - start equity', color: '#ffffff', kind: 'real', status: 'live', n: n >= 500 ? '500+' : n, exp: n && n < 500 ? total / n : null, total, account: true });
    }
    for (const m of models) {
      const s = stat.get(m.model);
      const last = m.points.length ? m.points[m.points.length - 1].pnl : 0;
      const n = s ? s.n_trades : Math.max(0, m.points.length - 1);
      rows.push({ id: m.model, name: shortModel(m.model), title: m.model, color: modelColor(m.model), kind: kindLabel(m.kind), status: m.status, n, exp: n ? last / n : null, total: last, champion: m.status === 'champion' });
    }
    return rows;
  }

  function buildItems() {
    const items = [];
    const eq = store.get('equity');
    if (eq.length && isNum(startEq())) {
      items.push({ id: ACCOUNT, color: '#ffffff', width: 3, step: false, label: true, visible: !hidden.has(ACCOUNT), points: eq.map((p) => ({ t: p.t, v: p.equity - startEq() })) });
    }
    for (const m of shadowEq) {
      const champ = m.status === 'champion';
      items.push({ id: m.model, color: modelColor(m.model), width: champ ? 4 : 2, step: true, label: champ, visible: !hidden.has(m.model), points: m.points.map((p) => ({ t: p.t, v: p.pnl })) });
    }
    return items;
  }

  function renderLegend() {
    if (destroyed) return;
    const rows = legendRows();
    if (!rows.length) {
      lastLegendHtml = '';
      return ($('ab-legend').innerHTML = `<div class="empty">${esc(abError || 'no models reported by /api/shadow/equity yet')}</div>`);
    }
    const html = `<table class="tbl compact"><thead><tr><th>bot</th><th class="hide-m">kind</th><th class="hide-m">status</th><th class="r">trades</th><th class="r">$/trade</th><th class="r">total PnL</th><th class="r ab-cur-h" id="ab-cur-h">at cursor</th></tr></thead><tbody>
      ${rows
        .map(
          (r) => `<tr class="ab-row${hidden.has(r.id) ? ' is-off' : ''}${r.account ? ' ab-acct' : ''}${r.champion ? ' ab-champ' : ''}" data-id="${esc(r.id)}" title="${esc(r.title)}">
            <td class="ab-name"><i class="ab-sw" style="background:${r.color};${r.account || r.champion ? 'height:5px' : ''}"></i><span>${esc(r.name)}</span></td>
            <td class="hide-m dim">${esc(r.kind)}</td><td class="hide-m"><span class="mstat mstat-${esc(r.status)}">${esc(String(r.status).toUpperCase())}</span></td>
            <td class="r">${esc(r.n)}</td><td class="r">${r.exp == null ? '<span class="dim">–</span>' : pnlSpan(r.exp, 3)}</td><td class="r">${pnlSpan(r.total)}</td>
            <td class="r ab-cur" data-cur="${esc(r.id)}"><span class="dim">–</span></td></tr>`
        )
        .join('')}
    </tbody></table>`;
    if (html === lastLegendHtml) return;
    lastLegendHtml = html;
    $('ab-legend').innerHTML = html;
    showHover(hoverT);
  }

  function showHover(t) {
    hoverT = t;
    const h = $('ab-cur-h');
    if (h) h.textContent = t == null ? 'at cursor' : `at ${fmtDateTime(t)}`;
    for (const cell of root.querySelectorAll('#ab-legend [data-cur]')) {
      const v = t == null || !abChart ? null : abChart.valueAt(cell.dataset.cur, t);
      cell.innerHTML = v == null ? '<span class="dim">–</span>' : pnlSpan(v);
    }
  }

  function drawBots() {
    if (destroyed) return;
    if (!abChart && typeof LightweightCharts !== 'undefined') {
      abChart = createMultiLineChart($('ab-chart'), { format: (v) => fmtUsd(v) });
      abChart.onHover(showHover);
    }
    if (abChart) {
      abChart.setSeries(buildItems(), !abChart.userMoved());
      lastStart = startEq();
    }
    renderLegend();
    const n = shadowEq.length;
    $('ab-sub').textContent = abError ? `· ${abError}` : n ? `· ${n} models + the real account · cumulative net PnL (USDT), refreshed every ${REFRESH_MS / 1000}s and on every shadow trade` : '';
  }

  async function refreshBots() {
    try {
      const r = await api.shadowEquity(5000);
      // champion first = drawn underneath, so a thin challenger line that overlaps it stays visible
      const rank = { champion: 0, challenger: 1, retired: 2 };
      shadowEq = (Array.isArray(r) ? r : []).sort((x, y) => (rank[x.status] ?? 3) - (rank[y.status] ?? 3) || x.model.localeCompare(y.model));
      abError = '';
    } catch (e) {
      abError = `/api/shadow/equity unavailable (${e.message}); restart tb.api`;
    }
    drawBots();
  }
  const refreshBotsT = throttle(refreshBots, 3000);

  $('ab-card').addEventListener('click', (e) => {
    const btn = e.target.closest('[data-ab]');
    if (btn) {
      if (btn.dataset.ab === 'fit') return abChart && abChart.fit();
      hidden.clear();
      if (btn.dataset.ab === 'none') for (const r of legendRows()) hidden.add(r.id);
      saveHidden(hidden);
      lastLegendHtml = '';
      return drawBots();
    }
    const row = e.target.closest('.ab-row');
    if (!row) return;
    const id = row.dataset.id;
    if (hidden.has(id)) hidden.delete(id);
    else hidden.add(id);
    saveHidden(hidden);
    row.classList.toggle('is-off', hidden.has(id));
    lastLegendHtml = '';
    abChart && abChart.setVisible(id, !hidden.has(id));
  });

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
    if (abChart && isNum(startEq())) abChart.update(ACCOUNT, { t: p.t, v: p.equity - startEq() });
    if (!eqChart) return;
    eqChart.update({ t: p.t, v: p.equity });
    peakSoFar = Math.max(peakSoFar, p.equity, store.get('status')?.peak_equity ?? 0);
    ddChart.update({ t: p.t, v: peakSoFar > 0 ? -((peakSoFar - p.equity) / peakSoFar) * 100 : 0 });
  }));
  offs.push(store.on('change:equity', () => {
    loadCharts(false);
    drawBots();
  }));

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
  const renderLegendT = throttle(renderLegend, 1000);
  offs.push(store.on('change:status', () => {
    renderT();
    if (startEq() !== lastStart) drawBots();
    else renderLegendT();
  }));
  offs.push(store.on('change:shadow', () => {
    renderShadow();
    renderLegendT();
  }));
  offs.push(store.on('shadow_trade', refreshBotsT));
  const refreshTimer = setInterval(refreshBots, REFRESH_MS);

  loadCharts(true);
  renderTiles();
  renderShadow();
  api.shadow().then((r) => r && store.set('shadow', r)).catch(() => {});
  refreshBots();

  return () => {
    destroyed = true;
    clearInterval(refreshTimer);
    refreshBotsT.cancel();
    renderLegendT.cancel();
    abChart && abChart.destroy();
    offs.forEach((o) => o());
    renderT.cancel();
    eqChart && eqChart.destroy();
    ddChart && ddChart.destroy();
  };
}
