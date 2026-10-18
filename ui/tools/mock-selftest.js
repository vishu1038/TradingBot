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
need(get('/api/settings'), ['slice_margin_usd', 'leverage', 'max_positions', 'exit_mode', 'tp_usd', 'sl_usd', 'tp_atr', 'sl_atr', 'trail', 'short_enabled', 'prob_threshold_long', 'prob_threshold_short', 'corr_cap', 'universe_size', 'universe_refresh_s', 'kill_dd_pct', 'fee_taker', 'sound'], 'settings');
const labs = get('/api/lab/reports');
need(labs[0], ['id', 'created_at', 'kind', 'title', 'summary', 'equity'], 'lab report');
check('settings PUT merges', m.request('PUT', '/api/settings', {}, { trail: { enabled: false } }).body.trail.distance_atr === 0.8);
check('settings PUT validates', m.request('PUT', '/api/settings', {}, { leverage: 999 }).status === 422);
m.request('PUT', '/api/settings', {}, { trail: { enabled: true } });

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
check('control close_all', m.request('POST', '/api/control', {}, { action: 'close_all' }).body.ok === true && m.positions.length === 0);
check('control kill', m.request('POST', '/api/control', {}, { action: 'kill' }).body.ok === true && m.kill.tripped);
check('start refused while tripped', m.request('POST', '/api/control', {}, { action: 'start' }).body.ok === false);
check('reset_kill', m.request('POST', '/api/control', {}, { action: 'reset_kill' }).body.ok === true && !m.kill.tripped);
out(fails ? `\n${fails} FAILED` : '\nall passed');
