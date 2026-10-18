// LIVE TRADES: the bot's open positions (live uPnL, TP/SL progress, close [x]) + recently closed trades.
import { store } from '../state.js';
import { esc, fmtPrice, fmtUsd, fmtDuration, fmtDateTime, signClass, throttle, shortSym } from '../format.js';
import { symLabel, sidePill, exitPill, slTpBar } from '../components.js';
import { closePosition } from '../actions.js';
import { selectSymbol } from '../sync.js';

export function createLiveTrades() {
  const el = document.createElement('div');
  el.className = 'card livetrades ac-coral';
  el.innerHTML = '<h3>LIVE TRADES <span class="why-sub" id="lt-sub"></span></h3><div class="tbl-wrap lt-wrap" id="lt-body"></div>';
  const body = el.querySelector('#lt-body');
  const offs = [];

  function render() {
    const ps = store.get('positions');
    const closed = store.get('trades').slice(0, 10);
    const now = store.serverNow();
    const sel = store.get('symbol');
    el.querySelector('#lt-sub').textContent = `· ${ps.length} open · ${closed.length} recent`;
    const open = ps
      .map(
        (p) => `<tr class="lt-open ${p.symbol === sel ? 'cur' : ''}" data-sym="${esc(p.symbol)}">
          <td>${symLabel(p.symbol)} ${sidePill(p.side)}</td>
          <td class="r hide-s">${fmtPrice(p.entry_price)}</td>
          <td class="r">${fmtPrice(p.mark_price)}</td>
          <td class="lt-bar hide-s">${slTpBar(p)}</td>
          <td class="r"><b class="${signClass(p.upnl)}">${fmtUsd(p.upnl, 2, true)}</b></td>
          <td class="r hide-s dim">${fmtDuration(now - p.opened_at)}</td>
          <td class="r">${p.trail_active ? '<span class="trail-tag">TRAIL</span> ' : ''}<button type="button" class="xbtn" data-close="${esc(p.id)}" title="Close at market" aria-label="Close position">×</button></td>
        </tr>`
      )
      .join('');
    const rec = closed
      .map(
        (t) => `<tr class="lt-closed ${t.pnl > 0 ? 'row-win' : 'row-loss'}" data-sym="${esc(t.symbol)}">
          <td>${symLabel(t.symbol)} ${sidePill(t.side)}</td>
          <td class="r hide-s">${fmtPrice(t.entry_price)}</td>
          <td class="r">${fmtPrice(t.exit_price)}</td>
          <td class="hide-s">${exitPill(t.exit_reason)}</td>
          <td class="r"><b class="${signClass(t.pnl)}">${fmtUsd(t.pnl, 2, true)}</b></td>
          <td class="r hide-s dim">${fmtDateTime(t.closed_at)}</td>
          <td class="r"></td>
        </tr>`
      )
      .join('');
    body.innerHTML = `<table class="tbl compact"><thead><tr><th>position</th><th class="r hide-s">entry</th><th class="r">mark / exit</th><th class="hide-s">TP ◂ SL</th><th class="r">PnL</th><th class="r hide-s">age</th><th class="r"></th></tr></thead><tbody>
      ${open || '<tr><td colspan="7" class="empty">no open trades</td></tr>'}
      ${rec ? `<tr class="lt-sep"><td colspan="7">recently closed</td></tr>${rec}` : ''}
    </tbody></table>`;
  }

  body.addEventListener('click', (e) => {
    const c = e.target.closest('[data-close]');
    if (c) {
      e.stopPropagation();
      const p = store.get('positions').find((x) => String(x.id) === c.dataset.close);
      closePosition(p);
      return;
    }
    const row = e.target.closest('[data-sym]');
    if (row) selectSymbol(row.dataset.sym);
  });

  const r = throttle(render, 400);
  for (const t of ['change:positions', 'position_updated', 'change:trades', 'change:symbol']) offs.push(store.on(t, r));
  const age = setInterval(r, 5000);
  render();
  return { el, destroy: () => { clearInterval(age); r.cancel(); offs.forEach((o) => o()); } };
}
