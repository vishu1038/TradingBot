// EXECUTION view: the testnet mirror (/api/mirror). The paper book is the scoreboard; this shows how the REAL order path
// (testnet, min size) behaves: account state, success rate, latency, error classes / codes, divergences, open mirror legs, events.
import { api } from '../api.js';
import { esc, fmtAgo, fmtNum } from '../format.js';
import * as fx from '../fx.js';

const CLS = {
  ok: ['OK', 'var(--up)'],
  our_bug: ['OUR BUG', 'var(--down)'],
  venue_fault: ['VENUE FAULT', 'var(--amber)'],
  account_unusable: ['ACCOUNT', 'var(--magenta)'],
  skipped: ['SKIPPED', 'var(--text-dim)'],
};
const STATE = { ok: 'var(--up)', unusable: 'var(--down)', unknown: 'var(--text-dim)', disabled: 'var(--text-dim)' };

const pct = (v) => (v == null ? '—' : `${(v * 100).toFixed(1)}%`);
const ms = (v) => (v == null ? '—' : `${Math.round(v)} ms`);
const chip = (cls) => {
  const [l, c] = CLS[cls] || [String(cls || '?').toUpperCase(), 'var(--text-dim)'];
  return `<span class="exec-chip" style="--chip:${c}">${esc(l)}</span>`;
};

function tile(label, value, sub = '') {
  return `<div class="tile"><div class="tile-l">${esc(label)}</div><div class="tile-v">${value}</div>${sub ? `<div class="tile-s">${sub}</div>` : ''}</div>`;
}

function render(root, d) {
  const st = d.account_state || 'unknown';
  const a = d.account || {};
  const reasons = (a.reasons || []).map(esc).join(' · ');
  if (!d.enabled) {
    root.innerHTML = `<div class="page exec"><div class="card ac-sky"><h3>EXECUTION</h3>
      <div class="empty">Mirror off. Enable <b>mirror_enabled</b> in Settings to send every paper order to the testnet at minimum size.</div></div></div>`;
    return;
  }
  const byCls = (d.counts && d.counts.by_cls) || {};
  const byCode = (d.counts && d.counts.by_error_code) || {};
  const div = d.divergences || {};
  const legs = d.open_positions || [];
  const evs = d.events || [];
  root.innerHTML = `<div class="page exec">
    <div class="card ac-sky"><h3>EXECUTION <span class="count">testnet mirror · ${esc(d.venue || '')} · size ${esc(d.size || 'min')}</span></h3>
      <div class="exec-head">
        <span class="exec-state" style="--chip:${STATE[st] || 'var(--text-dim)'}">● ACCOUNT ${esc(st.toUpperCase())}</span>
        <span class="dim">${a.margin_balance != null ? `margin $${fmtNum(a.margin_balance, 2)} · available $${fmtNum(a.available, 2)}` : 'not checked yet'}
          ${a.checked_at ? ` · checked ${fmtAgo(a.checked_at)}` : ''}</span>
        ${reasons ? `<span class="neg">${reasons}</span>` : ''}
        <button type="button" class="btn btn-mini${st === 'unusable' ? ' exec-hot' : ''}" id="exec-retry">↻ Retry mirror</button>
      </div>
      <div class="tiles t7">
        ${tile('SUCCESS 1H', pct(d.success_rate_1h), `${d.calls_1h || 0} calls`)}
        ${tile('SUCCESS 24H', pct(d.success_rate_24h), `${d.calls_24h || 0} calls`)}
        ${tile('LATENCY P50', ms(d.latency_p50_ms))}
        ${tile('LATENCY P95', ms(d.latency_p95_ms))}
        ${tile('OUR BUGS 24H', `<span class="${byCls.our_bug ? 'neg' : ''}">${byCls.our_bug || 0}</span>`)}
        ${tile('VENUE FAULTS 24H', `${byCls.venue_fault || 0}`)}
        ${tile('DIVERGENCES', `${div['24h'] || 0}`, `1h: ${div['1h'] || 0}`)}
      </div>
    </div>
    <div class="exec-grid">
      <div class="card ac-magenta"><h3>CLASSES · 24H</h3>
        <div class="exec-bars">${Object.keys(CLS).map((k) => {
          const n = byCls[k] || 0;
          const tot = Object.values(byCls).reduce((s, x) => s + x, 0) || 1;
          return `<div class="exec-bar">${chip(k)}<i style="--w:${(100 * n) / tot}%;--chip:${CLS[k][1]}"></i><b>${n}</b></div>`;
        }).join('')}</div>
        <div class="tbl-wrap"><table class="tbl"><thead><tr><th>error code</th><th>24h</th></tr></thead><tbody>
          ${Object.entries(byCode).sort((x, y) => y[1] - x[1]).map(([c, n]) => `<tr><td>${esc(c)}</td><td>${n}</td></tr>`).join('') ||
            '<tr><td colspan="2" class="dim">no errors</td></tr>'}
        </tbody></table></div>
        ${Object.keys(div.by_kind_24h || {}).length ? `<div class="dim exec-div">divergences: ${Object.entries(div.by_kind_24h).map(([k, n]) => `${esc(k)} ×${n}`).join(' · ')}</div>` : ''}
      </div>
      <div class="card ac-teal"><h3>OPEN MIRROR LEGS <span class="count">${legs.length}</span></h3>
        <div class="tbl-wrap"><table class="tbl"><thead><tr><th>coin</th><th>side</th><th>qty</th><th>state</th><th>stop</th><th>tp</th><th>paper</th></tr></thead><tbody>
          ${legs.map((p) => `<tr class="${!p.paper_open || !p.has_stop ? 'exec-warn' : ''}"><td>${esc(p.symbol)}</td>
            <td class="${p.side === 'long' ? 'pos' : 'neg'}">${esc(p.side)}</td><td>${fmtNum(p.qty, 4)}</td><td>${esc(p.state)}</td>
            <td>${p.has_stop ? '✓' : '<b class="neg">none</b>'}</td><td>${p.has_tp ? '✓' : '—'}</td>
            <td>${p.paper_open ? '✓' : '<b class="neg">closed</b>'}</td></tr>`).join('') || '<tr><td colspan="7" class="dim">no open mirror positions</td></tr>'}
        </tbody></table></div>
      </div>
    </div>
    <div class="card ac-pink"><h3>EVENTS <span class="count">last ${evs.length}</span></h3>
      <div class="tbl-wrap tall"><table class="tbl exec-log"><thead><tr><th>when</th><th>coin</th><th>action</th><th>class</th><th>code</th><th>ms</th><th>detail</th></tr></thead><tbody>
        ${evs.map((e) => `<tr><td class="dim">${fmtAgo(e.t)}</td><td>${esc(e.symbol || '—')}</td><td>${esc(e.action)}</td><td>${chip(e.cls)}</td>
          <td>${esc(e.error_code || '')}</td><td>${e.latency_ms != null ? e.latency_ms : ''}</td><td class="exec-msg">${esc(e.error_msg || '')}</td></tr>`).join('') ||
          '<tr><td colspan="7" class="dim">no mirror events yet</td></tr>'}
      </tbody></table></div>
    </div>
  </div>`;
  root.querySelector('#exec-retry').addEventListener('click', async () => {
    try {
      await api.mirrorRetry();
      fx.toast('mirror retry requested', 'info', 2000);
    } catch (e) {
      fx.toast(`retry failed: ${e.message}`, 'loss', 4000);
    }
  });
}

export function mount(root) {
  root.innerHTML = '<div class="page exec"><div class="card ac-sky"><h3>EXECUTION</h3><div class="empty">loading /api/mirror…</div></div></div>';
  let dead = false;
  let timer = null;
  async function tick() {
    if (dead) return;
    if (!document.hidden) {
      try {
        render(root, await api.mirror());
      } catch (e) {
        root.innerHTML = `<div class="page exec"><div class="card ac-sky"><h3>EXECUTION</h3><div class="empty neg">failed to load /api/mirror: ${esc(e.message)}</div></div></div>`;
      }
    }
    timer = setTimeout(tick, 5000);
  }
  tick();
  return () => {
    dead = true;
    clearTimeout(timer);
  };
}
