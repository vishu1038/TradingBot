// LAB view: backtest / training reports from /api/lab/reports with an equity chart per report.
import { store } from '../state.js';
import { api } from '../api.js';
import { createLineChart } from '../charts.js';
import { esc, fmtDateTime, fmtSummaryValue, titleCase, fmtUsd, isNum } from '../format.js';

export function mount(root) {
  root.innerHTML = `<div class="page lab"><div class="card ac-pink"><h3>LAB REPORTS</h3><div id="lab-body" class="empty">loading /api/lab/reports…</div></div></div>`;
  const body = root.querySelector('#lab-body');
  let charts = [];
  let dead = false;

  function render(reports) {
    charts.forEach((c) => c.destroy());
    charts = [];
    if (!reports.length) {
      body.className = 'empty';
      body.textContent = 'no reports yet. run a backtest or training from the ml/ tools';
      return;
    }
    body.className = 'lab-list';
    body.innerHTML = reports
      .map((r, i) => {
        const s = r.summary || {};
        const sum = Object.keys(s)
          .map((k) => `<div class="tile mini"><div class="tile-l">${esc(titleCase(k))}</div><div class="tile-v">${fmtSummaryValue(k, s[k])}</div></div>`)
          .join('');
        return `<article class="lab-report">
          <header><span class="kind kind-${esc(r.kind)}">${esc(String(r.kind).toUpperCase())}</span><b class="lab-title">${esc(r.title)}</b><span class="dim">${fmtDateTime(r.created_at)} · id ${esc(r.id)}</span></header>
          <div class="lab-grid"><div class="tiles mini-grid">${sum}</div><div class="line-chart" id="lab-ch-${i}"></div></div>
        </article>`;
      })
      .join('');
    reports.forEach((r, i) => {
      const el = root.querySelector(`#lab-ch-${i}`);
      if (!el || typeof LightweightCharts === 'undefined') return;
      const eq = r.equity || [];
      const start = eq.length ? eq[0].equity : 0;
      const c = createLineChart(el, { kind: 'baseline', baseValue: start, format: (v) => fmtUsd(v) });
      c.setData(eq.map((p) => ({ t: p.t, v: p.equity })));
      c.setBase(isNum(start) ? start : null);
      charts.push(c);
    });
  }

  const cached = store.get('labReports');
  if (cached.length) render(cached);
  api
    .labReports()
    .then((r) => {
      if (dead) return;
      store.set('labReports', r);
      render(r);
    })
    .catch((e) => {
      if (dead) return;
      body.className = 'empty neg';
      body.textContent = `failed to load reports: ${e.message}`;
    });

  return () => {
    dead = true;
    charts.forEach((c) => c.destroy());
  };
}
