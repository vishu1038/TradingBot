// Headless self-test for js/mock.js. Run with macOS's built-in JavaScriptCore shell:
//   /System/Library/Frameworks/JavaScriptCore.framework/Versions/A/Helpers/jsc -m ui/tools/mock-selftest.js
import { MockServer } from '../js/mock.js';

const out = typeof print === 'function' ? print : console.log;
let fails = 0;
const check = (name, cond, extra = '') => {
  if (!cond) fails++;
  out(`${cond ? 'ok  ' : 'FAIL'} ${name} ${extra}`);
};

const m = new MockServer({ seed: 7, ticksPerBar: 30 });
const get = (p, q) => m.request('GET', p, q || {}).body;

// ---- shape checks against the API contract
const need = (obj, keys, what) => {
  const miss = keys.filter((k) => !(k in obj));
  check(`${what} has keys`, miss.length === 0, miss.join(','));
};
need(get('/api/health'), ['ok', 'engine_alive', 'engine_heartbeat_age_s', 'mode', 'version'], 'health');
need(get('/api/status'), ['mode', 'running', 'equity', 'start_equity', 'peak_equity', 'cash', 'realized_pnl', 'unrealized_pnl', 'drawdown_pct', 'open_positions', 'max_positions', 'trades_today', 'win_rate', 'kill', 'model', 'universe_updated_at'], 'status');
need(get('/api/status').kill, ['tripped', 'reason', 'dd_limit_pct'], 'status.kill');
need(get('/api/status').model, ['name', 'version', 'trained_at', 'threshold_long', 'threshold_short'], 'status.model');
need(get('/api/market'), ['btc_price', 'btc_change_24h_pct', 'btc_trend', 'market_volatility_pct', 'avg_funding_rate', 'long_short_ratio', 'updated_at'], 'market');
const uni = get('/api/universe');
check('universe has 8 rows', uni.length === 8, String(uni.length));
need(uni[0], ['symbol', 'score', 'atr_pct', 'quote_volume_24h', 'change_24h_pct', 'funding_rate', 'open_interest_usd', 'oi_change_1h_pct', 'last_price', 'in_universe_since'], 'universe row');
const sym = uni[0].symbol;
const cs = get('/api/candles', { symbol: sym, tf: '5m', limit: '500' });
check('candles 500', cs.length === 500, String(cs.length));
need(cs[0], ['t', 'o', 'h', 'l', 'c', 'v'], 'candle');
check('candles ascending & contiguous', cs.every((c, i) => i === 0 || c.t - cs[i - 1].t === 300));
check('candles sane OHLC', cs.every((c, i) =>c.h >= Math.max(c.o, c.c) - 1e-9 && c.l <= Math.min(c.o, c.c) + 1e-9 && (c.v > 0 || i === cs.length - 1)));
const sg = get('/api/signals', { symbol: sym, limit: '200' });
need(sg[0], ['t', 'symbol', 'prob_long', 'prob_short', 'action', 'reasons'], 'signal');
need(sg[0].reasons[0], ['feature', 'label', 'value', 'contribution'], 'reason');
const pos0 = get('/api/positions');
check('seeded positions', pos0.length >= 1, String(pos0.length));
need(pos0[0], ['id', 'symbol', 'side', 'entry_price', 'qty', 'notional', 'leverage', 'margin', 'sl_price', 'tp_price', 'trail_active', 'trail_price', 'mark_price', 'upnl', 'opened_at', 'prob', 'reasons'], 'position');
const tr = get('/api/trades', { limit: '200' });
check('seeded trades', tr.length > 30, String(tr.length));
need(tr[0], ['id', 'symbol', 'side', 'entry_price', 'exit_price', 'qty', 'opened_at', 'closed_at', 'pnl', 'fees', 'funding', 'exit_reason', 'prob'], 'trade');
const eq = get('/api/equity', { limit: '2000' });
check('equity history', eq.length > 100, String(eq.length));
need(get('/api/settings'), ['slice_margin_usd', 'max_leverage', 'max_total_leverage', 'max_margin_ratio', 'max_open_risk_pct', 'max_positions', 'exit_mode', 'tp_usd', 'sl_usd', 'tp_atr', 'sl_atr', 'trail', 'short_enabled', 'prob_threshold_long', 'prob_threshold_short', 'corr_cap', 'universe_size', 'universe_refresh_s', 'universe_whitelist', 'kill_dd_pct', 'fee_taker', 'sound', 'breakeven_r', 'sl_cooldown_bars', 'max_side_risk_pct', 'partial_tp_r', 'partial_tp_frac'], 'settings');
const labs = get('/api/lab/reports');
need(labs[0], ['id', 'created_at', 'kind', 'title', 'summary', 'equity'], 'lab report');
check('settings PUT merges', m.request('PUT', '/api/settings', {}, { trail: { enabled: false } }).body.trail.distance_atr === 1.0);
check('settings PUT validates', m.request('PUT', '/api/settings', {}, { max_leverage: 999 }).status === 422);
m.request('PUT', '/api/settings', {}, { trail: { enabled: true } });
check('whitelist defaults to null', get('/api/settings').universe_whitelist === null);
check('whitelist PUT validates', m.request('PUT', '/api/settings', {}, { universe_whitelist: ['not a symbol'] }).status === 422);
check('whitelist PUT rejects empty list', m.request('PUT', '/api/settings', {}, { universe_whitelist: [] }).status === 422);
const wlSym = get('/api/universe')[0].symbol;
check('whitelist PUT accepts', m.request('PUT', '/api/settings', {}, { universe_whitelist: [wlSym] }).body.universe_whitelist[0] === wlSym);
m.request('PUT', '/api/settings', {}, { universe_whitelist: null });

// ---- simulate 1800 ticks (1 virtual hour at 10x) and gather stats
const types = {};
let opened = 0;
let closed = 0;
const reasons = {};
let maxOpen = 0;
let nan = false;
for (let i = 0; i < 1800; i++) {
  const msgs = m.tick();
  for (const x of msgs) {
    types[x.type] = (types[x.type] || 0) + 1;
    if (x.type === 'position_opened') opened++;
    if (x.type === 'position_closed') {
      closed++;
      reasons[x.data.exit_reason] = (reasons[x.data.exit_reason] || 0) + 1;
    }
    if (x.type === 'tick' && !Number.isFinite(x.data.price)) nan = true;
  }
  maxOpen = Math.max(maxOpen, m.positions.length);
}
out('message types: ' + JSON.stringify(types));
out(`opened=${opened} closed=${closed} maxOpen=${maxOpen} reasons=${JSON.stringify(reasons)}`);
check('no NaN prices', !nan);
check('positions opened during run', opened >= 5, String(opened));
check('positions closed during run', closed >= 3, String(closed));
check('universe msg emitted', (types.universe || 0) >= 1);
const st = get('/api/status');
out(`status: equity=${st.equity} realized=${st.realized_pnl} unreal=${st.unrealized_pnl} dd=${st.drawdown_pct}% win=${st.win_rate} open=${st.open_positions}/${st.max_positions}`);
const cs2 = get('/api/candles', { symbol: sym, limit: '600' });
check('candles contiguous after run', cs2.every((c, i) => i === 0 || c.t - cs2[i - 1].t === 300));
const lastSigs = get('/api/signals', { symbol: sym, limit: '600' });
const pAbove = lastSigs.filter((s) => s.prob_long >= 0.7).length / lastSigs.length;
out(`P(prob_long>=0.7) on ${sym} = ${(pAbove * 100).toFixed(1)}%  (n=${lastSigs.length})`);
const atrs = get('/api/universe').map((u) => `${u.symbol}:${u.atr_pct}`);
out('atr% ' + atrs.join(' '));
need(get('/api/status').safety, ['state', 'reasons', 'derate_mult', 'drift', 'faults'], 'status.safety');
need(get('/api/status').safety.drift, ['z', 'n', 'mult'], 'safety.drift');
const sh = get('/api/shadow');
need(sh[0], ['model', 'status', 'n_trades', 'win_rate', 'expectancy', 'pnl', 'vs_champion'], 'shadow row');
check('shadow has champion + challenger', sh.some((r) => r.status === 'champion') && sh.some((r) => r.status === 'challenger'));
check('safety/shadow_trade messages emitted', (types.safety || 0) >= 1 && (types.shadow_trade || 0) >= 10, JSON.stringify([types.safety, types.shadow_trade]));
const chal = sh.find((r) => r.status === 'challenger');
const pr = m.request('POST', '/api/control', {}, { action: 'promote_model', model: chal.model }).body;
out(`promote_model -> ${JSON.stringify(pr)}`);
check('promote returns ok or a reason', pr.ok === true || typeof pr.error === 'string');
if (pr.ok) check('rollback restores champion', m.request('POST', '/api/control', {}, { action: 'rollback_model', model: chal.model }).body.ok === true);
check('promote of unknown model refused', m.request('POST', '/api/control', {}, { action: 'promote_model', model: 'nope' }).body.ok === false);
// ---- order book contract: bands at 0.2/1/2/3/4/5 %, cumulative (monotonic), top-20 ladder
const ob = get('/api/orderbook', { symbol: sym });
need(ob, ['symbol', 't', 'mid', 'spread', 'bands', 'bids', 'asks'], 'orderbook');
need(ob.bands[0], ['pct', 'bid_depth', 'bid_notional', 'ask_depth', 'ask_notional'], 'orderbook band');
check('orderbook bands pct', JSON.stringify(ob.bands.map((b) => b.pct)) === '[0.2,1,2,3,4,5]');
check('orderbook bands cumulative', ob.bands.every((b, i) => i === 0 || (b.bid_notional >= ob.bands[i - 1].bid_notional && b.ask_notional >= ob.bands[i - 1].ask_notional)));
check('orderbook ladder 20/20 sorted', ob.bids.length === 20 && ob.asks.length === 20 && ob.bids.every((r, i) => i === 0 || r[0] < ob.bids[i - 1][0]) && ob.asks.every((r, i) => i === 0 || r[0] > ob.asks[i - 1][0]) && ob.bids[0][0] < ob.mid && ob.asks[0][0] > ob.mid);
check('orderbook unknown symbol 404', m.request('GET', '/api/orderbook', { symbol: 'NOPEUSDT' }).status === 404);
check('orderbook WS messages emitted', (types.orderbook || 0) >= 100, String(types.orderbook));
// ---- pause / unpause / refresh / live arm controls
check('pause_symbol', m.request('POST', '/api/control', {}, { action: 'pause_symbol', symbol: sym }).body.ok === true && get('/api/status').paused_symbols.includes(sym));
check('pause needs a known symbol', m.request('POST', '/api/control', {}, { action: 'pause_symbol', symbol: 'NOPEUSDT' }).body.ok === false);
check('unpause_symbol', m.request('POST', '/api/control', {}, { action: 'unpause_symbol', symbol: sym }).body.ok === true && !get('/api/status').paused_symbols.includes(sym));
check('refresh_universe', m.request('POST', '/api/control', {}, { action: 'refresh_universe' }).body.ok === true);
check('arm_live / disarm_live', m.request('POST', '/api/control', {}, { action: 'arm_live' }).body.ok === true && get('/api/status').mode === 'live' && m.request('POST', '/api/control', {}, { action: 'disarm_live' }).body.ok === true && get('/api/status').mode === 'paper_local');
// ---- PnL convention: net = gross - fees - funding; fees >= 0; win_rate is a 0-1 fraction
const allT = get('/api/trades', { limit: '500' });
check('trades.pnl is net of fees+funding', allT.every((t) => { const g = (t.exit_price - t.entry_price) * t.qty * (t.side === 'long' ? 1 : -1); return Math.abs(g - t.fees - t.funding - t.pnl) < 0.02 + 1e-4 * Math.abs(g); }));
check('fees >= 0', allT.every((t) => t.fees >= 0));
check('win_rate is a 0-1 fraction', get('/api/status').win_rate >= 0 && get('/api/status').win_rate <= 1);
check('control close_all', m.request('POST', '/api/control', {}, { action: 'close_all' }).body.ok === true && m.positions.length === 0);
check('control kill', m.request('POST', '/api/control', {}, { action: 'kill' }).body.ok === true && m.kill.tripped);
check('start refused while tripped', m.request('POST', '/api/control', {}, { action: 'start' }).body.ok === false);
check('reset_kill', m.request('POST', '/api/control', {}, { action: 'reset_kill' }).body.ok === true && !m.kill.tripped);
out(fails ? `\n${fails} FAILED` : '\nall passed');
