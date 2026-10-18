// TRADES view: history table with filters, sortable columns and a summary row.
// `pnl` is treated as the trade's realized PnL (net headline figure); fees/funding are shown separately.
import { store } from '../state.js';
import {
  esc, fmtPrice, fmtUsd, fmtQty, fmtDateTime, fmtDuration, signClass, throttle, isNum, fmtNum,
} from '../format.js';
import { symLabel, sidePill, exitPill, tile } from '../components.js';

const filters = { symbol: '', side: '', exit_reason: '' };
let sortKey = 'closed_at';
let sortDir = -1;
const MAX_ROWS = 400;

const COLS = [
  ['closed_at', 'closed', (t) => fmtDateTime(t.closed_at)],
  ['symbol', 'symbol', (t) => symLabel(t.symbol)],
  ['side', 'side', (t) => sidePill(t.side)],
  ['entry_price', 'entry', (t) => fmtPrice(t.entry_price)],
  ['exit_price', 'exit', (t) => fmtPrice(t.exit_price)],
  ['qty', 'qty', (t) => fmtQty(t.qty)],
  ['pnl', 'PnL', (t) => `<span class="${signClass(t.pnl)} strong">${fmtUsd(t.pnl, 2, true)}</span>`],
  ['fees', 'fees (cost)', (t) => `<span class="dim">${fmtUsd(t.fees, 3)}</span>`],
  ['funding', 'funding (cost)', (t) => `<span class="dim">${fmtUsd(t.funding, 3, true)}</span>`],
  ['exit_reason', 'reason', (t) => exitPill(t.exit_reason)],
  ['prob', 'prob', (t) => (isNum(t.prob) ? t.prob.toFixed(2) : '–')],
  ['dur', 'held', (t) => fmtDuration(t.closed_at - t.opened_at)],
];

export function summarize(trades) {
  const n = trades.length;
  const wins = trades.filter((t) => t.pnl > 0);
  const losses = trades.filter((t) => t.pnl <= 0);
  const total = trades.reduce((a, t) => a + (t.pnl || 0), 0);
  const fees = trades.reduce((a, t) => a + (t.fees || 0), 0);
  const gw = wins.reduce((a, t) => a + t.pnl, 0);
  const gl = -losses.reduce((a, t) => a + t.pnl, 0);
  return {
    n,
    winRate: n ? (wins.length / n) * 100 : null,
    total,
    fees,
    avgWin: wins.length ? gw / wins.length : null,
    avgLoss: losses.length ? -gl / losses.length : null,
    pf: gl > 0 ? gw / gl : null,
  };
}

export function mount(root) {
  root.innerHTML = `
    <div class="page trades">
      <div class="card ac-violet"><h3>FILTERS</h3>
        <div class="filters">
          <label>symbol <select id="f-symbol"></select></label>
          <label>side <select id="f-side"><option value="">all</option><option value="long">long</option><option value="short">short</option></select></label>
          <label>exit <select id="f-exit"><option value="">all</option></select></label>
          <button type="button" class="btn btn-mini" id="f-clear">CLEAR</button>
        </div>
      </div>
      <div class="card ac-amber"><h3>SUMMARY <span class="why-sub" id="sum-sub"></span></h3><div id="summary"></div></div>
      <div class="card ac-coral"><h3>HISTORY</h3><div class="tbl-wrap tall"><table class="tbl sortable" id="tr-tbl"></table></div></div>
    </div>`;
  const $ = (id) => root.querySelector(`#${id}`);
  const offs = [];

  function fillSelects() {
    const trades = store.get('trades');
    const syms = [...new Set(trades.map((t) => t.symbol))].sort();
    const reasons = [...new Set(trades.map((t) => t.exit_reason))].sort();
    const sSym = $('f-symbol');
    sSym.innerHTML = `<option value="">all</option>${syms.map((s) => `<option value="${esc(s)}">${esc(s.replace('USDT', ''))}</option>`).join('')}`;
    sSym.value = syms.includes(filters.symbol) ? filters.symbol : '';
    const sEx = $('f-exit');
    sEx.innerHTML = `<option value="">all</option>${reasons.map((s) => `<option value="${esc(s)}">${esc(s)}</option>`).join('')}`;
    sEx.value = reasons.includes(filters.exit_reason) ? filters.exit_reason : '';
    $('f-side').value = filters.side;
  }

  function filtered() {
    return store
      .get('trades')
      .filter((t) => (!filters.symbol || t.symbol === filters.symbol) && (!filters.side || t.side === filters.side) && (!filters.exit_reason || t.exit_reason === filters.exit_reason));
  }

  function render() {
    const list = filtered();
    const s = summarize(list);
    $('sum-sub').textContent = `· ${list.length} of ${store.get('trades').length} trades`;
    $('summary').innerHTML = `<div class="tiles t7">
      ${tile('TRADES', esc(s.n))}
      ${tile('WIN RATE', s.winRate == null ? '–' : `${s.winRate.toFixed(1)}%`, s.winRate >= 50 ? 'pos' : 'neg')}
      ${tile('NET PNL', fmtUsd(s.total, 2, true), signClass(s.total))}
      ${tile('FEES PAID', fmtUsd(s.fees, 2), 'dim')}
      ${tile('AVG WIN', s.avgWin == null ? '–' : fmtUsd(s.avgWin, 2, true), 'pos')}
      ${tile('AVG LOSS', s.avgLoss == null ? '–' : fmtUsd(s.avgLoss, 2, true), 'neg')}
      ${tile('PROFIT FACTOR', s.pf == null ? '–' : fmtNum(s.pf, 2), s.pf >= 1 ? 'pos' : 'neg')}
    </div>`;
    const key = sortKey;
    const val = (t) => (key === 'dur' ? t.closed_at - t.opened_at : t[key]);
    list.sort((a, b) => {
      const va = val(a);
      const vb = val(b);
      if (typeof va === 'string') return sortDir * va.localeCompare(vb);
      return sortDir * ((va ?? 0) - (vb ?? 0));
    });
    const head = COLS.map(([k, l]) => `<th data-sort="${k}" class="${['symbol', 'side', 'exit_reason'].includes(k) ? '' : 'r'}">${esc(l)}${k === sortKey ? (sortDir < 0 ? ' ▼' : ' ▲') : ''}</th>`).join('');
    const body = list
      .slice(0, MAX_ROWS)
      .map((t) => `<tr class="${t.pnl > 0 ? 'row-win' : 'row-loss'}">${COLS.map(([k, , f]) => `<td class="${['symbol', 'side', 'exit_reason'].includes(k) ? '' : 'r'}">${f(t)}</td>`).join('')}</tr>`)
      .join('');
    $('tr-tbl').innerHTML = `<thead><tr>${head}</tr></thead><tbody>${body || `<tr><td colspan="${COLS.length}" class="empty">no trades match</td></tr>`}</tbody>`;
  }

  root.addEventListener('change', (e) => {
    if (e.target.id === 'f-symbol') filters.symbol = e.target.value;
    else if (e.target.id === 'f-side') filters.side = e.target.value;
    else if (e.target.id === 'f-exit') filters.exit_reason = e.target.value;
    else return;
    render();
  });
  root.addEventListener('click', (e) => {
    if (e.target.id === 'f-clear') {
      filters.symbol = filters.side = filters.exit_reason = '';
      fillSelects();
      render();
      return;
    }
    const th = e.target.closest('th[data-sort]');
    if (th) {
      const k = th.dataset.sort;
      if (sortKey === k) sortDir = -sortDir;
      else {
        sortKey = k;
        sortDir = ['symbol', 'side', 'exit_reason'].includes(k) ? 1 : -1;
      }
      render();
    }
  });

  const onTrades = throttle(() => {
    fillSelects();
    render();
  }, 800);
  offs.push(store.on('change:trades', onTrades));
  fillSelects();
  render();
  return () => {
    offs.forEach((o) => o());
    onTrades.cancel();
  };
}
