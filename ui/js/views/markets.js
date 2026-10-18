// MARKETS view: sortable universe table, volume-sized heatmap, data-fault list.
import { store } from '../state.js';
import { selectSymbol } from '../sync.js';
import { observeResize } from '../charts.js';
import {
  esc, fmtPrice, fmtPct, fmtFunding, fmtCompact, fmtNum, shortSym, signClass, throttle, fmtAgo, clamp, isNum,
} from '../format.js';
import { symLabel, faultsTable, safetyOf } from '../components.js';

const COLS = [
  ['rank', '#', null],
  ['symbol', 'symbol', (u) => symLabel(u.symbol)],
  ['score', 'score', (u) => fmtNum(u.score, 2)],
  ['atr_pct', 'ATR%', (u) => fmtPct(u.atr_pct, 2, false)],
  ['quote_volume_24h', 'volume 24h', (u) => fmtCompact(u.quote_volume_24h)],
  ['change_24h_pct', '24h %', (u) => `<span class="${signClass(u.change_24h_pct)}">${fmtPct(u.change_24h_pct)}</span>`],
  ['funding_rate', 'funding', (u) => `<span class="${signClass(u.funding_rate)}">${fmtFunding(u.funding_rate)}</span>`],
  ['open_interest_usd', 'open interest', (u) => fmtCompact(u.open_interest_usd)],
  ['oi_change_1h_pct', 'OI Δ1h', (u) => `<span class="${signClass(u.oi_change_1h_pct)}">${fmtPct(u.oi_change_1h_pct)}</span>`],
  ['last_price', 'price', (u) => fmtPrice(u.last_price)],
];

let sortKey = 'score';
let sortDir = -1;

/** Squarified treemap. items: [{value}], returns rects [{x,y,w,h}] in item order. */
export function squarify(values, W, H) {
  const total = values.reduce((a, b) => a + b, 0) || 1;
  const areas = values.map((v) => (v / total) * W * H);
  const out = [];
  const rect = { x: 0, y: 0, w: W, h: H };
  let row = [];
  const sum = (r) => r.reduce((a, b) => a + b, 0);
  const worst = (r, side) => {
    const s = sum(r);
    return Math.max((side * side * Math.max(...r)) / (s * s), (s * s) / (side * side * Math.min(...r)));
  };
  const flush = () => {
    if (!row.length) return;
    const s = sum(row);
    if (rect.w >= rect.h) {
      const cw = s / rect.h;
      let yy = rect.y;
      for (const a of row) {
        const hh = a / cw;
        out.push({ x: rect.x, y: yy, w: cw, h: hh });
        yy += hh;
      }
      rect.x += cw;
      rect.w -= cw;
    } else {
      const rh = s / rect.w;
      let xx = rect.x;
      for (const a of row) {
        const ww = a / rh;
        out.push({ x: xx, y: rect.y, w: ww, h: rh });
        xx += ww;
      }
      rect.y += rh;
      rect.h -= rh;
    }
    row = [];
  };
  for (const a of areas) {
    const side = Math.min(rect.w, rect.h);
    if (row.length && worst([...row, a], side) > worst(row, side)) flush();
    row.push(a);
  }
  flush();
  return out;
}

function tileColor(chg) {
  const t = clamp(Math.abs(chg || 0) / 6, 0, 1);
  const a = 0.14 + 0.62 * t;
  return chg >= 0 ? `rgba(52,240,160,${a.toFixed(2)})` : `rgba(255,77,125,${a.toFixed(2)})`;
}

export function mount(root) {
  root.innerHTML = `
    <div class="page">
      <div class="card span2 ac-cyan"><h3>UNIVERSE <span class="why-sub" id="uni-sub"></span></h3><div class="tbl-wrap"><table class="tbl sortable" id="uni-tbl"></table></div></div>
      <div class="card ac-pink"><h3>HEATMAP <span class="why-sub">size = volume (sqrt), colour = 24h %</span></h3><div class="heatmap" id="heatmap"></div></div>
      <div class="card ac-amber"><h3>DATA FAULTS <span class="why-sub">debounced: 3 to mature, 5 good cycles to clear</span></h3><div id="faults"></div></div>
    </div>`;
  const $ = (id) => root.querySelector(`#${id}`);
  const offs = [];

  function rows() {
    const ticks = store.get('ticks');
    const list = store.get('universe').map((u, i) => {
      const tk = ticks[u.symbol];
      let chg = u.change_24h_pct;
      let price = u.last_price;
      if (tk && isNum(u.last_price) && u.last_price > 0 && isNum(chg)) {
        const ref = u.last_price / (1 + chg / 100);
        price = tk.price;
        chg = (tk.price / ref - 1) * 100;
      }
      return { ...u, last_price: price, change_24h_pct: chg, rank: i + 1 };
    });
    list.sort((a, b) => {
      const va = sortKey === 'symbol' ? a.symbol : a[sortKey];
      const vb = sortKey === 'symbol' ? b.symbol : b[sortKey];
      if (typeof va === 'string') return sortDir * va.localeCompare(vb);
      return sortDir * ((va ?? 0) - (vb ?? 0));
    });
    return list;
  }

  function renderTable() {
    const list = rows();
    const upd = store.get('status')?.universe_updated_at;
    $('uni-sub').textContent = upd ? `· re-ranked ${fmtAgo(upd, store.serverNow())} ago · ${list.length} symbols` : '';
    const head = COLS.map(([k, label]) => {
      const arrow = k === sortKey ? (sortDir < 0 ? ' ▼' : ' ▲') : '';
      const num = k !== 'symbol' ? ' class="r"' : '';
      return k === 'rank' ? `<th class="r">${label}</th>` : `<th${num} data-sort="${k}">${esc(label)}${arrow}</th>`;
    }).join('');
    const body = list
      .map(
        (u, i) =>
          `<tr data-sym="${esc(u.symbol)}">${COLS.map(([k, , f]) => (k === 'rank' ? `<td class="r dim">${i + 1}</td>` : `<td class="${k === 'symbol' ? '' : 'r'}">${f(u)}</td>`)).join('')}</tr>`
      )
      .join('');
    $('uni-tbl').innerHTML = `<thead><tr>${head}</tr></thead><tbody>${body || '<tr><td colspan="10" class="empty">no universe yet</td></tr>'}</tbody>`;
  }

  function renderHeat() {
    const el = $('heatmap');
    const w = el.clientWidth;
    const h = el.clientHeight;
    const list = rows().sort((a, b) => b.quote_volume_24h - a.quote_volume_24h);
    if (!w || !h || !list.length) return (el.innerHTML = '');
    const rects = squarify(list.map((u) => Math.sqrt(u.quote_volume_24h)), w, h);
    el.innerHTML = list
      .map((u, i) => {
        const r = rects[i];
        const big = r.w > 110 && r.h > 70;
        return `<div class="hm-tile" data-sym="${esc(u.symbol)}" style="left:${r.x.toFixed(1)}px;top:${r.y.toFixed(1)}px;width:${(r.w - 3).toFixed(1)}px;height:${(r.h - 3).toFixed(1)}px;background:${tileColor(u.change_24h_pct)}">
          <b class="hm-s">${esc(shortSym(u.symbol))}</b><span class="hm-c ${signClass(u.change_24h_pct)}">${fmtPct(u.change_24h_pct)}</span>${big ? `<span class="hm-v">${fmtCompact(u.quote_volume_24h)}</span>` : ''}
        </div>`;
      })
      .join('');
  }

  function renderFaults() {
    if (!$('faults')) return;
    const sf = safetyOf(store.get('status'));
    $('faults').innerHTML = sf.present ? faultsTable(sf.faults, (t) => fmtAgo(t, store.serverNow())) : '<div class="empty">no safety data from /api/status yet</div>';
  }

  const goLive = (sym) => {
    selectSymbol(sym);
    store.set('view', 'live');
  };
  root.addEventListener('click', (e) => {
    const th = e.target.closest('th[data-sort]');
    if (th) {
      const k = th.dataset.sort;
      if (sortKey === k) sortDir = -sortDir;
      else {
        sortKey = k;
        sortDir = k === 'symbol' ? 1 : -1;
      }
      renderTable();
      return;
    }
    const t = e.target.closest('[data-sym]');
    if (t) goLive(t.dataset.sym);
  });

  const renderAll = () => {
    renderTable();
    renderHeat();
  };
  const tAll = throttle(renderAll, 1000);
  offs.push(store.on('change:universe', tAll));
  offs.push(store.on('tick', tAll));
  const tFaults = throttle(renderFaults, 1000);
  offs.push(store.on('change:status', tFaults));
  const offRes = observeResize($('heatmap'), () => renderHeat());
  renderAll();
  renderFaults();

  return () => {
    offs.forEach((o) => o());
    tAll.cancel();
    tFaults.cancel();
    offRes();
  };
}
