// DEPTH card: cumulative bid/ask depth at +-0.2 / 1 / 2 / 3 / 4 / 5 % of mid (the bands the model reads)
// as a filled depth chart, plus a band table and a small top-20 ladder tab.
import { store } from '../state.js';
import { esc, fmtCompact, fmtPrice, fmtNum, throttle, isNum, shortSym, clamp } from '../format.js';
import { symLabel, tile } from '../components.js';
import { spinner } from '../fx.js';

const W = 400;
const H = 150;
const PAD_L = 8;
const PAD_R = 8;
const PAD_T = 10;
const PAD_B = 22;
const MAXPCT = 5;
let tab = 'depth'; // survives remounts

const BID = '#4fe3ff';
const ASK = '#ffb020';

export function createDepth() {
  const el = document.createElement('div');
  el.className = 'card depth ac-cyan';
  el.innerHTML = `
    <h3>DEPTH <span class="why-sub" id="d-sub"></span></h3>
    <div class="seg" role="tablist"><button type="button" role="tab" data-tab="depth">DEPTH</button><button type="button" role="tab" data-tab="ladder">LADDER</button></div>
    <div id="d-body"></div>`;
  const $ = (id) => el.querySelector(`#${id}`);
  const body = $('d-body');
  let hoverIdx = null;
  const offs = [];

  const X = (pct) => PAD_L + ((pct + MAXPCT) / (2 * MAXPCT)) * (W - PAD_L - PAD_R);

  function chartSvg(ob) {
    const bands = ob.bands;
    const max = Math.max(1, ...bands.map((b) => Math.max(b.bid_notional, b.ask_notional))) * 1.1;
    const Y = (v) => PAD_T + (1 - v / max) * (H - PAD_T - PAD_B);
    const y0 = Y(0);
    const bidPts = [[0, 0], ...bands.map((b) => [-b.pct, b.bid_notional])];
    const askPts = [[0, 0], ...bands.map((b) => [b.pct, b.ask_notional])];
    const poly = (pts) => pts.map(([p, v]) => `${X(p).toFixed(1)},${Y(v).toFixed(1)}`).join(' ');
    // extend to the plot edge so the area reaches +-5 %
    const bidArea = `${X(0)},${y0} ${poly(bidPts)} ${X(-MAXPCT)},${y0}`;
    const askArea = `${X(0)},${y0} ${poly(askPts)} ${X(MAXPCT)},${y0}`;
    const ticks = [-5, -4, -3, -2, -1, -0.2, 0.2, 1, 2, 3, 4, 5];
    const grid = ticks
      .map((t) => {
        const major = [-5, -2, -1, 1, 2, 5].includes(t);
        return `<line x1="${X(t)}" y1="${PAD_T}" x2="${X(t)}" y2="${y0}" stroke="rgba(255,255,255,0.07)" stroke-dasharray="2 4"/>${
          major ? `<text x="${X(t)}" y="${H - 7}" text-anchor="middle" class="d-ax">${t > 0 ? '+' : '−'}${Math.abs(t)}%</text>` : ''
        }`;
      })
      .join('');
    const dots = bands
      .map((b, i) => {
        const on = hoverIdx === i;
        return `<circle cx="${X(-b.pct)}" cy="${Y(b.bid_notional)}" r="${on ? 4.5 : 2.6}" fill="${BID}"/><circle cx="${X(b.pct)}" cy="${Y(b.ask_notional)}" r="${on ? 4.5 : 2.6}" fill="${ASK}"/>`;
      })
      .join('');
    const hover = hoverIdx != null ? `<line x1="${X(-bands[hoverIdx].pct)}" y1="${PAD_T}" x2="${X(-bands[hoverIdx].pct)}" y2="${y0}" stroke="${BID}" opacity=".6"/><line x1="${X(bands[hoverIdx].pct)}" y1="${PAD_T}" x2="${X(bands[hoverIdx].pct)}" y2="${y0}" stroke="${ASK}" opacity=".6"/>` : '';
    return `<svg class="depth-svg" viewBox="0 0 ${W} ${H}" width="100%" role="img" aria-label="Cumulative depth">
      <defs>
        <linearGradient id="gB" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="${BID}" stop-opacity=".55"/><stop offset="1" stop-color="${BID}" stop-opacity=".04"/></linearGradient>
        <linearGradient id="gA" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="${ASK}" stop-opacity=".55"/><stop offset="1" stop-color="${ASK}" stop-opacity=".04"/></linearGradient>
      </defs>
      ${grid}
      <polygon points="${bidArea}" fill="url(#gB)"/><polygon points="${askArea}" fill="url(#gA)"/>
      <polyline points="${poly(bidPts)}" fill="none" stroke="${BID}" stroke-width="2" stroke-linejoin="round"/>
      <polyline points="${poly(askPts)}" fill="none" stroke="${ASK}" stroke-width="2" stroke-linejoin="round"/>
      ${hover}${dots}
      <line x1="${X(0)}" y1="${PAD_T}" x2="${X(0)}" y2="${y0}" stroke="rgba(255,255,255,.35)"/>
      <text x="${X(0)}" y="${H - 7}" text-anchor="middle" class="d-ax">mid</text>
      <text x="${PAD_L}" y="${PAD_T + 8}" class="d-ax" fill="${BID}">BIDS</text>
      <text x="${W - PAD_R}" y="${PAD_T + 8}" text-anchor="end" class="d-ax" fill="${ASK}">ASKS</text>
    </svg>`;
  }

  function readout(ob) {
    const i = hoverIdx;
    const b = i != null ? ob.bands[i] : ob.bands[1] || ob.bands[0];
    const tot = b.bid_notional + b.ask_notional;
    const share = tot > 0 ? (b.bid_notional / tot) * 100 : 50;
    return `<div class="d-read"><span class="dim">±${b.pct}%</span> <b style="color:${BID}">${fmtCompact(b.bid_notional)}</b> <span class="dim">bid</span> · <b style="color:${ASK}">${fmtCompact(b.ask_notional)}</b> <span class="dim">ask</span>
      <span class="d-imb" title="bid share of depth within ±${b.pct}%"><i style="width:${share.toFixed(1)}%"></i></span><b>${share.toFixed(0)}%</b></div>`;
  }

  function bandTable(ob) {
    const max = Math.max(1, ...ob.bands.map((b) => Math.max(b.bid_notional, b.ask_notional)));
    return `<div class="d-bands">${ob.bands
      .map(
        (b, i) => `<div class="d-band ${hoverIdx === i ? 'on' : ''}" data-i="${i}">
          <span class="d-pct">±${b.pct}%</span>
          <span class="d-bar bid"><i style="width:${((b.bid_notional / max) * 100).toFixed(1)}%"></i><em>${fmtCompact(b.bid_notional)}</em></span>
          <span class="d-bar ask"><i style="width:${((b.ask_notional / max) * 100).toFixed(1)}%"></i><em>${fmtCompact(b.ask_notional)}</em></span>
        </div>`
      )
      .join('')}</div>`;
  }

  function ladder(ob) {
    const n = 10;
    const asks = ob.asks.slice(0, n);
    const bids = ob.bids.slice(0, n);
    const maxQ = Math.max(1e-12, ...asks.map((r) => r[1]), ...bids.map((r) => r[1]));
    const row = (r, cls) => `<div class="l-row ${cls}"><i style="width:${((r[1] / maxQ) * 100).toFixed(1)}%"></i><span>${fmtPrice(r[0])}</span><span>${fmtNum(r[1], r[1] >= 100 ? 0 : r[1] >= 1 ? 2 : 4)}</span><span class="dim">${fmtCompact(r[0] * r[1])}</span></div>`;
    return `<div class="ladder"><div class="l-head"><span>price</span><span>size</span><span>value</span></div>
      ${[...asks].reverse().map((r) => row(r, 'ask')).join('')}
      <div class="l-mid"><b>${fmtPrice(ob.mid)}</b><span class="dim">spread ${isNum(ob.spread) ? fmtNum((ob.spread / ob.mid) * 1e4, 1) + ' bps' : '–'}</span></div>
      ${bids.map((r) => row(r, 'bid')).join('')}</div>`;
  }

  function render() {
    const sym = store.get('symbol');
    const ob = store.get('orderbook')[sym];
    const state = store.get('orderbookState')[sym];
    el.querySelectorAll('[data-tab]').forEach((b) => b.classList.toggle('on', b.dataset.tab === tab));
    $('d-sub').innerHTML = sym ? `· ${symLabel(sym)}${ob ? ` <span class="dim">mid</span> ${fmtPrice(ob.mid)}` : ''}` : '';
    if (!ob) {
      body.innerHTML =
        state === 'unavailable'
          ? '<div class="empty">order book not published yet for this coin.<br><span class="dim">the engine fills /api/orderbook every ~5-10 s once running</span></div>'
          : `<div class="empty">${spinner()} loading order book…</div>`;
      return;
    }
    hoverIdx = hoverIdx != null && hoverIdx < ob.bands.length ? hoverIdx : null;
    body.innerHTML = tab === 'depth' ? `${chartSvg(ob)}${readout(ob)}${bandTable(ob)}` : ladder(ob);
  }

  el.addEventListener('click', (e) => {
    const t = e.target.closest('[data-tab]');
    if (t) {
      tab = t.dataset.tab;
      render();
    }
  });
  const setHover = (i) => {
    if (i === hoverIdx) return;
    hoverIdx = i;
    render();
  };
  body.addEventListener('mousemove', (e) => {
    const svg = e.target.closest('.depth-svg');
    const band = e.target.closest('.d-band');
    if (band) return setHover(Number(band.dataset.i));
    if (!svg) return;
    const ob = store.get('orderbook')[store.get('symbol')];
    if (!ob) return;
    const r = svg.getBoundingClientRect();
    const pct = Math.abs(((e.clientX - r.left) / r.width) * W - W / 2) / ((W - PAD_L - PAD_R) / 2) * MAXPCT;
    let best = 0;
    ob.bands.forEach((b, i) => {
      if (Math.abs(b.pct - pct) < Math.abs(ob.bands[best].pct - pct)) best = i;
    });
    setHover(best);
  });
  body.addEventListener('mouseleave', () => setHover(null));

  const r = throttle(render, 400);
  offs.push(store.on('orderbook', ({ symbol }) => symbol === store.get('symbol') && r()));
  offs.push(store.on('change:symbol', render));
  render();
  return {
    el,
    destroy() {
      r.cancel();
      offs.forEach((o) => o());
    },
  };
}
