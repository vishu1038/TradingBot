// In-browser mock backend. Implements every REST endpoint and the WS stream from the API contract
// (docs/ARCHITECTURE.md) with a live random-walk market, a fake model, positions that open/close
// on TP/SL/trail, an equity curve, logs and universe re-ranks.
//
// No DOM / window access here: the module is driven by tick() (1 call = 1 simulated second), so it can
// also be exercised from a plain JS shell (see ui/tools/mock-selftest.js).
//
// Time: the mock runs on a *virtual clock*. One tick advances 300/ticksPerBar virtual seconds so a new
// 5m candle appears every `ticksPerBar` ticks (default 30 -> a candle every 30 real seconds).

const BAR = 300;
const bucket = (t) => Math.floor(t / BAR) * BAR;
const HIST_BARS = 500;
const sigmoid = (x) => 1 / (1 + Math.exp(-x));
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

function mulberry32(a) {
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// sigma = typical 5m-bar return stdev (inflated a little so the demo is lively)
const COINS = [
  { s: 'BTCUSDT', p: 67450, sigma: 0.0016, qv: 18.4e9, oi: 9.1e9, fund: 0.00008 },
  { s: 'ETHUSDT', p: 3412, sigma: 0.0021, qv: 9.6e9, oi: 5.2e9, fund: 0.00006 },
  { s: 'SOLUSDT', p: 168.4, sigma: 0.0032, qv: 3.4e9, oi: 2.1e9, fund: 0.0001 },
  { s: 'WIFUSDT', p: 2.142, sigma: 0.0058, qv: 0.92e9, oi: 4.4e8, fund: 0.00021 },
  { s: '1000PEPEUSDT', p: 0.01243, sigma: 0.0054, qv: 1.15e9, oi: 6.1e8, fund: 0.00012 },
  { s: 'TAOUSDT', p: 436.2, sigma: 0.0052, qv: 0.46e9, oi: 3.0e8, fund: 0.00015 },
  { s: 'SUIUSDT', p: 1.924, sigma: 0.0046, qv: 0.81e9, oi: 5.2e8, fund: 0.00009 },
  { s: 'ENAUSDT', p: 0.6312, sigma: 0.0062, qv: 0.71e9, oi: 3.4e8, fund: 0.00018 },
  // bench: can rotate into the hot list on a re-rank
  { s: 'DOGEUSDT', p: 0.1832, sigma: 0.0036, qv: 1.9e9, oi: 9.0e8, fund: 0.00007 },
  { s: 'AVAXUSDT', p: 38.6, sigma: 0.0038, qv: 0.62e9, oi: 4.1e8, fund: 0.00005 },
  { s: 'WLDUSDT', p: 2.74, sigma: 0.0056, qv: 0.58e9, oi: 2.6e8, fund: 0.00011 },
  { s: 'NEARUSDT', p: 5.18, sigma: 0.0044, qv: 0.52e9, oi: 2.2e8, fund: 0.00004 },
];

const FEATURES = [
  ['ret_6', '6-bar momentum'],
  ['ema_gap', 'EMA20/50 gap'],
  ['vol_ratio', 'volume surge vs 20-bar'],
  ['rsi_14', 'RSI(14) stretch'],
  ['btc_ret_12', 'BTC 1h lead'],
  ['oi_delta', 'open-interest build-up'],
  ['funding', 'funding skew'],
  ['cnn_pattern', 'CNN latent pattern'],
];

export const DEFAULT_SETTINGS = () => ({
  slice_margin_usd: 20,
  leverage: 8,
  max_positions: 5,
  exit_mode: 'atr',
  tp_usd: 5,
  sl_usd: 10,
  tp_atr: 1.5,
  sl_atr: 1.0,
  trail: { enabled: true, activate_r: 1.0, distance_atr: 0.8 },
  short_enabled: false,
  prob_threshold_long: 0.7,
  prob_threshold_short: 0.8,
  corr_cap: { max_corr: 0.8, max_correlated: 2 },
  universe_size: 8,
  universe_refresh_s: 3600,
  kill_dd_pct: 50,
  fee_taker: 0.0005,
  sound: false,
});

// [path, min, max, integer?]
const SETTINGS_RULES = [
  ['slice_margin_usd', 1, 10000],
  ['leverage', 1, 50, true],
  ['max_positions', 1, 20, true],
  ['tp_usd', 0.01, 10000],
  ['sl_usd', 0.01, 10000],
  ['tp_atr', 0.1, 20],
  ['sl_atr', 0.1, 20],
  ['trail.activate_r', 0.1, 10],
  ['trail.distance_atr', 0.1, 10],
  ['prob_threshold_long', 0.5, 0.99],
  ['prob_threshold_short', 0.5, 0.99],
  ['corr_cap.max_corr', 0, 1],
  ['corr_cap.max_correlated', 1, 20, true],
  ['universe_size', 1, 30, true],
  ['universe_refresh_s', 60, 86400, true],
  ['kill_dd_pct', 1, 100],
  ['fee_taker', 0, 0.01],
];

export class MockServer {
  constructor({ seed = Date.now() & 0xffffffff, ticksPerBar = 30 } = {}) {
    this.rand = mulberry32(seed);
    this.N = Math.max(2, ticksPerBar | 0);
    this.now = Math.floor(Date.now() / 1000);
    this.tickN = 0;
    this.settings = DEFAULT_SETTINGS();
    this.running = true;
    this.kill = { tripped: false, reason: '', dd_limit_pct: this.settings.kill_dd_pct };
    this.startEquity = 100;
    this.realized = 0;
    this.positions = [];
    this.trades = [];
    this.equityHist = [];
    this.logs = [];
    this.nextId = 1;
    this.out = []; // messages produced during the current tick()
    this.cooldown = {};
    this.mode = 'paper_local';
    this.lastUniverseAt = this.now;
    this.btcTrend = 'flat';
    this.lsr = 1.18;
    // safety / shadow-mode state (see "Safety & robustness layer" in docs/ARCHITECTURE.md)
    this.champion = { model: 'cnn1d-5m@0.3.1', name: 'cnn1d-5m (mock)', version: '0.3.1' };
    this.prevChampion = { model: 'cnn1d-5m@0.2.9', name: 'cnn1d-5m (mock)', version: '0.2.9' };
    this.challenger = { model: 'cnn1d-5m@0.4.0-rc', name: 'cnn1d-5m (mock)', version: '0.4.0-rc', trades: [] };
    this.retired = [{ model: 'cnn1d-5m@0.2.9', n_trades: 212, win_rate: 0.47, expectancy: -0.03, pnl: -6.4 }];
    this.faults = [];
    this.paused = new Set();
    this.flattenUntil = 0;
    this.lastSafetyKey = '';

    this.syms = {};
    for (const c of COINS) this._initSym(c);
    this.order = COINS.map((c) => c.s);
    this._buildHistory();
    this.universeSyms = this._rankUniverse(null);
    for (const s of this.universeSyms) this.syms[s].since = this.now - 6 * 3600 - Math.floor(this.rand() * 40000);
    this._seedTrades();
    for (let i = 0; i < 34; i++) this._genShadowTrade(false, this.now - (34 - i) * 420);
    this.faults.push({ symbol: this.universeSyms[3], kind: 'stale_bar', count: 2, since: this.now - 40, age: 0 });
    this._seedLogs();
    this._seedPositions();
    this.labReports = this._makeLab();
    this.peak = Math.max(this.startEquity, ...this.equityHist.map((e) => e.equity), this._equity());
  }

  // ------------------------------------------------------------------ helpers
  gauss() {
    let u = 0;
    let v = 0;
    while (u === 0) u = this.rand();
    while (v === 0) v = this.rand();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  _initSym(c) {
    this.syms[c.s] = {
      sym: c.s,
      price: c.p,
      sigma: c.sigma,
      lv: 0,
      drift: 0,
      candles: [],
      ema20: [],
      ema50: [],
      noise: 0,
      qv: c.qv,
      oi: c.oi,
      oiDelta: 0,
      fund: c.fund,
      sigs: [],
      anchor: c.p,
      preview: null,
      tickInBar: 0,
      since: 0,
      oiHist: [],
    };
  }

  _stepPrice(S, gBtc, inBarTick) {
    const sg = (S.sigma * Math.exp(S.lv)) / Math.sqrt(this.N);
    S.drift = 0.985 * S.drift + 0.17 * this.gauss();
    const g = S.sym === 'BTCUSDT' ? gBtc : 0.62 * this.gauss() + 0.45 * gBtc;
    const r = sg * (0.7 * g + 0.15 * S.drift) - (0.12 / this.N / 30) * Math.log(S.price / S.anchor);
    S.price = Math.max(S.price * (1 + r), 1e-9);
    const c = S.candles[S.candles.length - 1];
    c.c = S.price;
    if (S.price > c.h) c.h = S.price;
    if (S.price < c.l) c.l = S.price;
    const volPerTick = S.qv / 288 / S.price / this.N;
    c.v += volPerTick * (0.45 + Math.abs(r) / sg) * (0.7 + 0.6 * this.rand());
    const n = S.candles.length;
    if (n >= 2) {
      S.ema20[n - 1] = S.ema20[n - 2] + (2 / 21) * (c.c - S.ema20[n - 2]);
      S.ema50[n - 1] = S.ema50[n - 2] + (2 / 51) * (c.c - S.ema50[n - 2]);
    }
    S.tickInBar = inBarTick;
  }

  _newBar(S, t) {
    const prev = S.candles[S.candles.length - 1];
    const o = prev ? prev.c : S.price;
    S.candles.push({ t, o, h: o, l: o, c: o, v: 0 });
    const n = S.candles.length;
    if (n === 1) {
      S.ema20.push(o);
      S.ema50.push(o);
    } else {
      S.ema20.push(S.ema20[n - 2]);
      S.ema50.push(S.ema50[n - 2]);
    }
    if (S.candles.length > 1500) {
      S.candles.shift();
      S.ema20.shift();
      S.ema50.shift();
    }
  }

  _closeBarHousekeeping(S) {
    S.lv = clamp(0.96 * S.lv + 0.14 * this.gauss(), -0.8, 0.95);
    S.noise = 0.8 * S.noise + 0.5 * this.gauss();
    S.oiDelta = 0.9 * S.oiDelta + 0.4 * this.gauss();
    S.oi *= 1 + S.oiDelta * 0.0006;
    S.oiHist.push(S.oi);
    if (S.oiHist.length > 40) S.oiHist.shift();
    S.fund = clamp(S.fund + this.gauss() * 0.000004, -0.0006, 0.0009);
  }

  _buildHistory() {
    const t0 = bucket(this.now) - HIST_BARS * BAR;
    for (const sym of this.order) {
      const S = this.syms[sym];
      // walk the price backwards-ish: start from a perturbed price and simulate forward, then rescale
      // so the final price equals the target (keeps the table of prices readable).
      const target = S.price;
      S.price = target * (1 + this.gauss() * 0.04);
      S.drift = 0;
    }
    const gb = () => this.gauss();
    for (let i = 0; i < HIST_BARS; i++) {
      const t = t0 + i * BAR;
      for (const sym of this.order) this._newBar(this.syms[sym], t);
      for (let k = 0; k < this.N; k++) {
        const g = gb();
        for (const sym of this.order) this._stepPrice(this.syms[sym], g, k);
      }
      for (const sym of this.order) {
        const S = this.syms[sym];
        this._closeBarHousekeeping(S);
        const idx = S.candles.length - 1;
        if (idx >= 60) S.sigs.push(this._evalSignal(S, idx, false));
      }
    }
    // rescale each history so the last close == target price, then the live bar opens there
    for (const c of COINS) {
      const S = this.syms[c.s];
      const last = S.candles[S.candles.length - 1].c;
      const k = c.p / last;
      for (let i = 0; i < S.candles.length; i++) {
        const x = S.candles[i];
        x.o *= k;
        x.h *= k;
        x.l *= k;
        x.c *= k;
        x.v /= k;
        S.ema20[i] *= k;
        S.ema50[i] *= k;
      }
      S.price = c.p;
      S.anchor = c.p;
      this._newBar(S, bucket(this.now));
      S.tickInBar = 0;
    }
    // the final history bar (idx HIST_BARS-1) belongs to history; the new one is live.
    for (const sym of this.order) {
      const S = this.syms[sym];
      S.preview = null;
    }
  }

  // -------------------------------------------------------------- indicators
  _atr(S, n = 14) {
    const cs = S.candles;
    const end = cs.length - 2; // last closed bar
    let sum = 0;
    let cnt = 0;
    for (let i = Math.max(1, end - n + 1); i <= end; i++) {
      const tr = Math.max(cs[i].h - cs[i].l, Math.abs(cs[i].h - cs[i - 1].c), Math.abs(cs[i].l - cs[i - 1].c));
      sum += tr;
      cnt++;
    }
    return cnt ? sum / cnt : S.price * S.sigma;
  }

  _atrPct(S) {
    const cs = S.candles;
    const end = cs.length - 2;
    let sum = 0;
    let cnt = 0;
    for (let i = Math.max(1, end - 71); i <= end; i++) {
      const tr = Math.max(cs[i].h - cs[i].l, Math.abs(cs[i].h - cs[i - 1].c), Math.abs(cs[i].l - cs[i - 1].c));
      sum += tr / cs[i].c;
      cnt++;
    }
    return cnt ? (sum / cnt) * 100 : S.sigma * 100;
  }

  _rsi(S, i, n = 14) {
    const cs = S.candles;
    let up = 0;
    let dn = 0;
    for (let k = i - n + 1; k <= i; k++) {
      const d = cs[k].c - cs[k - 1].c;
      if (d > 0) up += d;
      else dn -= d;
    }
    if (up + dn === 0) return 50;
    return (100 * up) / (up + dn);
  }

  /** Fake model. Features -> weighted log-odds -> probabilities, with per-feature contributions. */
  _evalSignal(S, i, preview) {
    const cs = S.candles;
    const btc = this.syms.BTCUSDT;
    const c = cs[i].c;
    const sg = S.sigma;
    const ret6 = Math.log(c / cs[i - 6].c) / (sg * Math.sqrt(6));
    const emaGap = (S.ema20[i] - S.ema50[i]) / c / (sg * 2.2);
    let vsum = 0;
    for (let k = i - 20; k < i; k++) vsum += cs[k].v;
    let volRatio = cs[i].v / Math.max(vsum / 20, 1e-12);
    if (preview) volRatio = (volRatio * this.N) / Math.max(1, S.tickInBar);
    volRatio = clamp(volRatio, 0.2, 4);
    const rsi = this._rsi(S, i);
    const bi = Math.min(i, btc.candles.length - 1);
    const btc12 = Math.log(btc.candles[bi].c / btc.candles[bi - 12].c) / (btc.sigma * Math.sqrt(12));
    const feats = {
      ret_6: [ret6, 0.55 * ret6],
      ema_gap: [emaGap, 0.32 * emaGap],
      vol_ratio: [volRatio, 0.2 * (volRatio - 1)],
      rsi_14: [rsi, -0.011 * (rsi - 50)],
      btc_ret_12: [btc12, 0.22 * btc12],
      oi_delta: [S.oiDelta, 0.2 * S.oiDelta],
      funding: [S.fund, -400 * S.fund + 0.02],
      cnn_pattern: [S.noise, 0.75 * S.noise],
    };
    let sum = 0;
    for (const k in feats) sum += feats[k][1];
    const B0L = -0.7;
    const B0S = -1.75;
    const probLong = sigmoid(B0L + sum);
    const probShort = sigmoid(B0S - sum);
    const thL = this.settings.prob_threshold_long;
    const thS = this.settings.prob_threshold_short;
    let action = 'none';
    if (probLong >= thL) action = 'long';
    else if (this.settings.short_enabled && probShort >= thS) action = 'short';
    const sign = action === 'short' ? -1 : 1;
    const reasons = FEATURES.map(([f, label]) => ({
      feature: f,
      label,
      value: +feats[f][0].toFixed(f === 'funding' ? 6 : 3),
      contribution: +(sign * feats[f][1]).toFixed(3),
    })).sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution));
    return {
      t: cs[i].t,
      symbol: S.sym,
      prob_long: +probLong.toFixed(4),
      prob_short: +probShort.toFixed(4),
      action,
      reasons,
    };
  }

  // --------------------------------------------------------------- universe
  _score(S) {
    return this._atrPct(S) * Math.sqrt(S.qv / 1e6);
  }

  _rankUniverse(current) {
    const size = this.settings.universe_size;
    const ranked = this.order.slice().sort((a, b) => this._score(this.syms[b]) - this._score(this.syms[a]));
    if (!current) return ranked.slice(0, size);
    const keep = Math.ceil(1.25 * size); // retention buffer (mock; the engine uses ceil(1.5*size))
    const next = [];
    for (const s of current) if (ranked.indexOf(s) < keep) next.push(s);
    for (const s of ranked) {
      if (next.length >= size) break;
      if (!next.includes(s)) next.push(s);
    }
    // positions never get force-closed by dropping out; nothing to do for them
    return next.slice(0, Math.max(size, 1));
  }

  _universeRows() {
    const rows = this.universeSyms.map((sym) => {
      const S = this.syms[sym];
      const cs = S.candles;
      const ref = cs[Math.max(0, cs.length - 289)].c;
      const oiRef = S.oiHist.length >= 12 ? S.oiHist[S.oiHist.length - 12] : S.oi;
      return {
        symbol: sym,
        score: +this._score(S).toFixed(3),
        atr_pct: +this._atrPct(S).toFixed(3),
        quote_volume_24h: Math.round(S.qv * (1 + 0.03 * Math.sin(this.tickN / 40))),
        change_24h_pct: +(((S.price / ref) - 1) * 100).toFixed(3),
        funding_rate: +S.fund.toFixed(6),
        open_interest_usd: Math.round(S.oi),
        oi_change_1h_pct: +((S.oi / oiRef - 1) * 100).toFixed(3),
        last_price: S.price,
        in_universe_since: S.since,
      };
    });
    return rows.sort((a, b) => b.score - a.score);
  }

  _market() {
    const b = this.syms.BTCUSDT;
    const cs = b.candles;
    const ref = cs[Math.max(0, cs.length - 289)].c;
    const n = cs.length - 1;
    const gap = (b.ema20[n] - b.ema50[n]) / b.price;
    this.btcTrend = gap > 0.0004 ? 'up' : gap < -0.0004 ? 'down' : 'flat';
    const rows = this._universeRows();
    const vol = rows.reduce((a, r) => a + r.atr_pct, 0) / Math.max(1, rows.length);
    const fund = rows.reduce((a, r) => a + r.funding_rate, 0) / Math.max(1, rows.length);
    this.lsr = clamp(this.lsr + this.gauss() * 0.004, 0.7, 2.2);
    return {
      btc_price: b.price,
      btc_change_24h_pct: +(((b.price / ref) - 1) * 100).toFixed(3),
      btc_trend: this.btcTrend,
      market_volatility_pct: +vol.toFixed(3),
      avg_funding_rate: +fund.toFixed(6),
      long_short_ratio: +this.lsr.toFixed(3),
      updated_at: this.now,
    };
  }

  // ------------------------------------------------------------- positions
  _sizePosition(entry, atr, side, mult = 1) {
    const st = this.settings;
    const notionalCap = st.slice_margin_usd * st.leverage;
    const dir = side === 'long' ? 1 : -1;
    let tpDist;
    let slDist;
    let qty;
    if (st.exit_mode === 'atr') {
      tpDist = st.tp_atr * atr;
      slDist = st.sl_atr * atr;
      qty = Math.min(st.sl_usd / slDist, notionalCap / entry);
    } else {
      qty = notionalCap / entry;
      tpDist = st.tp_usd / qty;
      slDist = st.sl_usd / qty;
    }
    qty *= mult; // drawdown derating x drift multiplier (risk per trade = sl_usd * mult)
    return {
      qty,
      notional: qty * entry,
      margin: (qty * entry) / st.leverage,
      tp_price: entry + dir * tpDist,
      sl_price: entry - dir * slDist,
      risk: slDist,
    };
  }

  _openPosition(S, side, sig, { price, openedAt } = {}) {
    const entry = price ?? S.price * (1 + (side === 'long' ? 1 : -1) * 0.0002);
    const sz = this._sizePosition(entry, this._atr(S), side, this._sizeMult());
    const p = {
      id: this.nextId++,
      symbol: S.sym,
      side,
      entry_price: entry,
      qty: sz.qty,
      notional: sz.notional,
      leverage: this.settings.leverage,
      margin: sz.margin,
      sl_price: sz.sl_price,
      tp_price: sz.tp_price,
      trail_active: false,
      trail_price: null,
      mark_price: S.price,
      upnl: 0,
      opened_at: openedAt ?? this.now,
      prob: side === 'long' ? sig.prob_long : sig.prob_short,
      reasons: sig.reasons.slice(0, 8),
      _risk: sz.risk,
      _sl0: sz.sl_price,
      _best: entry,
      _atr: this._atr(S),
    };
    this.positions.push(p);
    this._markPosition(p, S.price);
    this._log('INFO', `[engine] OPEN ${side.toUpperCase()} ${S.sym} @ ${this._fp(entry)} qty=${p.qty.toPrecision(4)} prob=${p.prob.toFixed(2)} SL=${this._fp(p.sl_price)} TP=${this._fp(p.tp_price)}`);
    this.out.push({ type: 'position_opened', data: this._pub(p) });
    return p;
  }

  _fp(x) {
    return x >= 100 ? x.toFixed(2) : x >= 1 ? x.toFixed(4) : x.toPrecision(4);
  }

  _markPosition(p, price) {
    const dir = p.side === 'long' ? 1 : -1;
    p.mark_price = price;
    p.upnl = (price - p.entry_price) * p.qty * dir;
  }

  _pub(p) {
    const o = {};
    for (const k in p) if (k[0] !== '_') o[k] = p[k];
    return o;
  }

  _closePosition(p, price, reason) {
    const dir = p.side === 'long' ? 1 : -1;
    const gross = (price - p.entry_price) * p.qty * dir;
    const fees = this.settings.fee_taker * p.qty * (p.entry_price + price);
    // funding is a COST (positive = paid): longs pay when the rate is positive
    const funding = +(p.notional * this.syms[p.symbol].fund * ((this.now - p.opened_at) / 28800) * dir).toFixed(4);
    const trade = {
      id: p.id,
      symbol: p.symbol,
      side: p.side,
      entry_price: p.entry_price,
      exit_price: price,
      qty: p.qty,
      opened_at: p.opened_at,
      closed_at: this.now,
      pnl: +(gross - fees - funding).toFixed(4), // NET realised: gross - fees - funding (contract)
      fees: +fees.toFixed(4),
      funding,
      exit_reason: reason,
      prob: p.prob,
    };
    this.positions = this.positions.filter((x) => x !== p);
    this.trades.push(trade);
    this.realized += trade.pnl;
    this.cooldown[p.symbol] = this.tickN + this.N * 2;
    const tag = reason.toUpperCase();
    this._log(trade.pnl >= 0 ? 'INFO' : 'WARN', `[engine] CLOSE ${p.side.toUpperCase()} ${p.symbol} ${tag} @ ${this._fp(price)} pnl=${trade.pnl >= 0 ? '+' : ''}${trade.pnl.toFixed(2)} fees=${fees.toFixed(3)}`);
    this.out.push({ type: 'position_closed', data: trade });
    return trade;
  }

  _manage(p) {
    const S = this.syms[p.symbol];
    const px = S.price;
    const long = p.side === 'long';
    const dir = long ? 1 : -1;
    this._markPosition(p, px);
    let changed = false;
    if (long ? px > p._best : px < p._best) p._best = px;
    const tr = this.settings.trail;
    if (tr.enabled) {
      const moved = (p._best - p.entry_price) * dir;
      if (!p.trail_active && moved >= tr.activate_r * p._risk) {
        p.trail_active = true;
        changed = true;
        this._log('INFO', `[engine] trailing stop ARMED ${p.symbol} (moved ${(moved / p._risk).toFixed(2)}R)`);
      }
      if (p.trail_active) {
        const cand = p._best - dir * tr.distance_atr * p._atr;
        if (p.trail_price == null || (long ? cand > p.trail_price : cand < p.trail_price)) {
          p.trail_price = cand; // never loosens
          changed = true;
        }
      }
    }
    // effective stop = tighter of SL and trailing
    let stop = p.sl_price;
    let stopKind = 'sl';
    if (p.trail_active && p.trail_price != null && (long ? p.trail_price > stop : p.trail_price < stop)) {
      stop = p.trail_price;
      stopKind = 'trail';
    }
    const hitStop = long ? px <= stop : px >= stop;
    const hitTp = long ? px >= p.tp_price : px <= p.tp_price;
    if (hitStop) {
      const slip = 0.0002 * (long ? -1 : 1);
      return this._closePosition(p, stop * (1 + slip), stopKind === 'trail' && p.trail_active ? 'trail' : 'sl');
    }
    if (hitTp) return this._closePosition(p, p.tp_price, 'tp');
    this.out.push({ type: 'position_updated', data: this._pub(p) });
    return changed;
  }

  _unrealized() {
    return this.positions.reduce((a, p) => a + p.upnl, 0);
  }

  _equity() {
    return this.startEquity + this.realized + this._unrealized();
  }

  _status() {
    const eq = this._equity();
    if (eq > this.peak) this.peak = eq;
    const dd = this.peak > 0 ? ((this.peak - eq) / this.peak) * 100 : 0;
    const dayStart = Math.floor(this.now / 86400) * 86400;
    const today = this.trades.filter((t) => t.closed_at >= dayStart);
    const wins = this.trades.filter((t) => t.pnl > 0).length;
    const st = this.settings;
    const margin = this.positions.reduce((a, p) => a + p.margin, 0);
    this.kill.dd_limit_pct = st.kill_dd_pct;
    return {
      mode: this.mode,
      running: this.running,
      equity: +eq.toFixed(4),
      start_equity: this.startEquity,
      peak_equity: +this.peak.toFixed(4),
      cash: +(this.startEquity + this.realized - margin).toFixed(4),
      realized_pnl: +this.realized.toFixed(4),
      unrealized_pnl: +this._unrealized().toFixed(4),
      drawdown_pct: +dd.toFixed(3),
      open_positions: this.positions.length,
      max_positions: this._maxPositions(),
      trades_today: today.length,
      win_rate: this.trades.length ? +(wins / this.trades.length).toFixed(4) : 0,
      kill: { ...this.kill },
      model: {
        name: this.champion.name,
        version: this.champion.version,
        trained_at: this.now - 2 * 86400 - 3600,
        threshold_long: st.prob_threshold_long,
        threshold_short: st.prob_threshold_short,
      },
      universe_updated_at: this.lastUniverseAt,
      paused_symbols: [...this.paused],
      live_armed: this.mode === 'live',
      safety: this._safety(),
    };
  }

  // ------------------------------------------------------- safety layer
  _hbStale() {
    const ph = this.tickN % 900;
    return ph >= 420 && ph < 510;
  }

  /** risk.derate_multiplier: 1.0 until 40% of the kill limit, 0.5 at 80%, 0 at 100%. */
  _derate() {
    const eq = this._equity();
    const dd = this.peak > 0 ? Math.max(0, ((this.peak - eq) / this.peak) * 100) : 0;
    const f = dd / this.settings.kill_dd_pct;
    if (f <= 0.4) return 1;
    if (f <= 0.8) return 1 - ((f - 0.4) / 0.4) * 0.5;
    if (f < 1) return 0.5 * (1 - (f - 0.8) / 0.2);
    return 0;
  }

  /** Drift: realised win rate of the last 30 trades vs the model's (mock-calibrated) prediction. */
  _drift() {
    const last = this.trades.slice(-30);
    const n = last.length;
    if (n < 10) return { z: 0, n, mult: 1 };
    const p = 0.55; // calibrated prediction the mock model is "expected" to hit
    const win = last.filter((t) => t.pnl > 0).length / n;
    const z = (win - p) / Math.sqrt((p * (1 - p)) / n);
    if (this._driftAlarm) {
      if (z > -1) this._driftAlarm = false;
    } else if (z < -3) this._driftAlarm = true;
    const mult = z < -2 || this._driftAlarm ? 0.5 : 1;
    return { z: +z.toFixed(2), n, mult };
  }

  _sizeMult() {
    return +(this._derate() * this._drift().mult).toFixed(3);
  }

  _safety() {
    const reasons = [];
    let state = 'NORMAL';
    const drift = this._drift();
    if (this.kill.tripped) {
      state = this.tickN < this.flattenUntil ? 'FLATTEN' : 'HALTED';
      reasons.push(`kill switch: ${this.kill.reason}`);
      reasons.push(state === 'FLATTEN' ? 'closing all positions at market' : 'manual reset_kill required');
    } else {
      if (this._hbStale()) {
        state = 'DEGRADED';
        reasons.push('engine heartbeat stale 34 s: no new entries, exchange stops confirmed by guard');
      }
      if (this._driftAlarm) {
        state = 'DEGRADED';
        reasons.push(`drift alarm: z=${drift.z} over last ${drift.n} trades (entries paused until z > -1)`);
      }
    }
    return {
      state,
      reasons,
      derate_mult: +this._derate().toFixed(3),
      drift,
      faults: this.faults.map((f) => ({ symbol: f.symbol, kind: f.kind, count: f.count, since: f.since })),
    };
  }

  _symbolExcluded(sym) {
    return this.faults.some((f) => f.symbol === sym && f.count >= 3);
  }

  _stepFaults() {
    // advance existing faults (one "cycle" = 5 ticks); clear after ~8 good cycles
    if (this.tickN % 5 === 0) {
      for (const f of this.faults) {
        f.age++;
        if (f.age <= 6) f.count++;
      }
      const before = this.faults.length;
      this.faults = this.faults.filter((f) => f.age < 14);
      if (this.faults.length < before) this._log('INFO', '[guard] data fault cleared (5 good cycles)');
    }
    if (this.tickN % 210 === 90 && this.faults.length < 3) {
      const sym = this.universeSyms[Math.floor(this.rand() * this.universeSyms.length)];
      if (!this.faults.some((f) => f.symbol === sym)) {
        const kinds = ['bar_return_spike', 'stale_bar', 'mark_last_divergence', 'zero_volume'];
        const kind = kinds[Math.floor(this.rand() * kinds.length)];
        this.faults.push({ symbol: sym, kind, count: 1, since: Math.floor(this.now), age: 0 });
        this._log('WARN', `[guard] data fault ${sym} ${kind} (debouncing, 1/3)`);
      }
    }
  }

  _emitSafetyIfChanged() {
    const s = this._safety();
    const key = JSON.stringify([s.state, s.reasons, s.derate_mult, s.drift]);
    if (key === this.lastSafetyKey) return;
    if (this.lastSafetyKey && JSON.parse(this.lastSafetyKey)[0] !== s.state) this._log(s.state === 'NORMAL' ? 'INFO' : 'WARN', `[guard] safety state -> ${s.state}`);
    this.lastSafetyKey = key;
    this.out.push({ type: 'safety', data: s });
  }

  // ------------------------------------------------------ shadow models
  _stat(trades) {
    const n = trades.length;
    const pnl = trades.reduce((a, t) => a + t.pnl, 0);
    const w = trades.filter((t) => t.pnl > 0).length;
    return { n_trades: n, win_rate: n ? +(w / n).toFixed(4) : 0, expectancy: n ? +(pnl / n).toFixed(4) : 0, pnl: +pnl.toFixed(4) };
  }

  _shadowRows() {
    const champ = { model: this.champion.model, status: 'champion', ...this._stat(this.trades) };
    const chal = this.challenger ? { model: this.challenger.model, status: 'challenger', ...this._stat(this.challenger.trades) } : null;
    const rows = [champ];
    if (chal) rows.push(chal);
    for (const r of this.retired) rows.push({ status: 'retired', ...r });
    for (const r of rows) r.vs_champion = +(r.expectancy - champ.expectancy).toFixed(4);
    return rows;
  }

  _genShadowTrade(emit, t) {
    const sym = this.universeSyms[Math.floor(this.rand() * this.universeSyms.length)];
    const S = this.syms[sym];
    const win = this.rand() < 0.6;
    const pnl = win ? 0.8 + this.rand() * 1.2 : -(0.7 + this.rand() * 0.45);
    const trade = {
      id: `s${this.challenger.trades.length + 1}`,
      symbol: sym,
      side: 'long',
      entry_price: S.price,
      exit_price: S.price * (1 + (win ? 0.008 : -0.006)),
      pnl: +pnl.toFixed(4),
      closed_at: Math.floor(t ?? this.now),
      exit_reason: win ? 'tp' : 'sl',
    };
    this.challenger.trades.push(trade);
    if (emit) this.out.push({ type: 'shadow_trade', data: { model: this.challenger.model, trade } });
  }

  _shadowTick() {
    // on every bar close the challenger "would have" opened 0..2 trades
    if (!this.challenger) return;
    for (let k = 0; k < 2; k++) if (this.rand() < 0.7) this._genShadowTrade(true);
  }

  _promote(model) {
    const c = this.challenger;
    if (!c || c.model !== model) return { ok: false, error: `${model} is not the current challenger` };
    const st = this._stat(c.trades);
    const champ = this._stat(this.trades);
    if (st.n_trades < 50) return { ok: false, error: `needs >= 50 shadow trades (has ${st.n_trades})` };
    if (st.expectancy <= champ.expectancy) return { ok: false, error: `expectancy ${st.expectancy.toFixed(3)} not above champion ${champ.expectancy.toFixed(3)}` };
    if (this._driftAlarm) return { ok: false, error: 'drift alarm active' };
    this.prevChampion = this.champion;
    const old = this.champion;
    this.retired.unshift({ model: old.model, ...champ });
    this.champion = { model: c.model, name: c.name, version: c.version };
    this.challenger = null;
    this._log('WARN', `[engine] PROMOTED ${c.model} to champion (previous: ${old.model})`);
    return { ok: true };
  }

  _rollback() {
    if (!this.prevChampion) return { ok: false, error: 'no previous champion to restore' };
    const cur = this.champion;
    this.champion = this.prevChampion;
    this.retired = this.retired.filter((r) => r.model !== this.champion.model);
    this.retired.unshift({ model: cur.model, ...this._stat(this.trades) });
    this.prevChampion = null;
    this._log('WARN', `[engine] ROLLBACK: champion restored to ${this.champion.model}`);
    return { ok: true };
  }

  _maxPositions() {
    const st = this.settings;
    return Math.max(0, Math.min(st.max_positions, Math.floor(this._equity() / st.slice_margin_usd)));
  }

  // ---------------------------------------------------------------- seeding
  _seedTrades() {
    const rnd = () => this.rand();
    const t1 = bucket(this.now) - 12 * BAR;
    const t0 = bucket(this.now) - 40 * 3600;
    const n = 54;
    const times = [];
    for (let i = 0; i < n; i++) times.push(t0 + Math.floor(rnd() * ((t1 - t0) / BAR)) * BAR);
    times.sort((a, b) => a - b);
    const specs = [];
    for (const ot of times) {
      const sym = this.universeSyms[Math.floor(rnd() * this.universeSyms.length)];
      const S = this.syms[sym];
      const idx = S.candles.findIndex((c) => c.t === ot);
      if (idx < 30) continue;
      const r = rnd();
      specs.push({
        ot,
        sym,
        idx,
        side: rnd() < 0.93 ? 'long' : 'short',
        reason: r < 0.45 ? 'tp' : r < 0.82 ? 'sl' : r < 0.94 ? 'trail' : 'manual',
        u: rnd(),
        dur: (2 + Math.floor(rnd() * 26)) * BAR,
        prob: +(0.7 + rnd() * 0.2).toFixed(3),
      });
    }
    const build = (sp) => {
      const S = this.syms[sp.sym];
      const dir = sp.side === 'long' ? 1 : -1;
      const entry = S.candles[sp.idx].o;
      const atr = S.price * (S.sigma * 1.3);
      const sz = this._sizePosition(entry, atr, sp.side);
      let exit;
      if (sp.reason === 'tp') exit = sz.tp_price;
      else if (sp.reason === 'sl') exit = sz.sl_price * (1 - dir * 0.0002);
      else if (sp.reason === 'trail') exit = entry + dir * sz.risk * (0.2 + sp.u * 1.1);
      else exit = entry + dir * sz.risk * (sp.u * 1.2 - 0.6);
      const gross = (exit - entry) * sz.qty * dir;
      const fees = this.settings.fee_taker * sz.qty * (entry + exit);
      const funding = +(sz.notional * S.fund * (((sp.dur) / 28800)) * dir).toFixed(4);
      return {
        symbol: sp.sym,
        side: sp.side,
        entry_price: entry,
        exit_price: exit,
        qty: sz.qty,
        opened_at: sp.ot + 2,
        closed_at: Math.min(sp.ot + sp.dur, bucket(this.now) - BAR),
        pnl: +(gross - fees - funding).toFixed(4),
        fees: +fees.toFixed(4),
        funding,
        exit_reason: sp.reason,
        prob: sp.prob,
      };
    };
    let built = specs.map(build);
    // keep the demo account in profit (+5..+16 USD) so the PnL starts green; flip some stop-outs to wins
    const target = 5 + rnd() * 11;
    for (let guard = 0; guard < 200 && built.reduce((a, t) => a + t.pnl, 0) < target; guard++) {
      const losers = specs.filter((sp) => sp.reason === 'sl');
      if (!losers.length) break;
      losers[Math.floor(rnd() * losers.length)].reason = 'tp';
      built = specs.map(build);
    }
    for (const t of built) this.trades.push({ id: this.nextId++, ...t });
    this.trades.sort((a, b) => a.closed_at - b.closed_at);
    // equity history (5m steps): step function over closed trades + a little unrealized wobble
    let eq = this.startEquity;
    let ti = 0;
    const first = this.trades.length ? this.trades[0].opened_at - 3600 : t0;
    const startT = bucket(first);
    this.equityHist.push({ t: startT - BAR, equity: eq });
    for (let t = startT; t <= bucket(this.now); t += BAR) {
      while (ti < this.trades.length && this.trades[ti].closed_at <= t) eq += this.trades[ti++].pnl;
      const open = this.trades.some((x) => x.opened_at <= t && x.closed_at > t);
      const wob = open ? Math.sin(t / 900) * 0.25 + (rnd() - 0.5) * 0.3 : 0;
      this.equityHist.push({ t, equity: +(eq + wob).toFixed(4) });
    }
    this.realized = this.trades.reduce((a, t) => a + t.pnl, 0);
  }

  _seedPositions() {
    const sigs = this.universeSyms.map((s) => {
      const S = this.syms[s];
      return { S, sig: this._evalSignal(S, S.candles.length - 2, false) };
    });
    sigs.sort((a, b) => b.sig.prob_long - a.sig.prob_long);
    for (const { S, sig } of sigs.slice(0, 2)) {
      const forced = { ...sig, prob_long: Math.max(sig.prob_long, 0.71 + this.rand() * 0.1) };
      this._openPosition(S, 'long', forced, { price: S.price * 0.9996, openedAt: this.now - 120 });
    }
    this.out = [];
  }

  _seedLogs() {
    const lines = [
      ['INFO', '[engine] tb-engine started (mock) mode=paper_local'],
      ['INFO', '[exchange] loaded exchangeInfo: 12 USDT perps in play'],
      ['INFO', `[universe] hot list: ${this.universeSyms.map((s) => s.replace('USDT', '')).join(', ')}`],
      ['INFO', '[scorer] model cnn1d-5m v0.3.1 loaded (numpy), 40 features x 64 bars'],
      ['WARN', '[exchange] depth stream lag 410 ms (recovered)'],
      ['INFO', '[risk] kill switch armed at drawdown 50% (paper)'],
    ];
    let t = this.now - 1800;
    for (const [level, msg] of lines) {
      this.logs.push({ t: (t += 37), level, msg });
    }
  }

  _log(level, msg) {
    const e = { t: this.now, level, msg };
    this.logs.push(e);
    if (this.logs.length > 600) this.logs.shift();
    this.out.push({ type: 'log', data: e });
  }

  _makeLab() {
    const rnd = () => this.rand();
    const mk = (n, drift, vol, start = 100) => {
      const t0 = this.now - 30 * 86400;
      const pts = [];
      let e = start;
      for (let i = 0; i < n; i++) {
        e += drift + (rnd() - 0.5) * vol;
        pts.push({ t: Math.floor(t0 + (i * 30 * 86400) / n), equity: +e.toFixed(3) });
      }
      return pts;
    };
    const wf = mk(240, 0.18, 2.4);
    const base = mk(240, -0.06, 2.4);
    const th = mk(240, 0.11, 1.8);
    const last = (a) => a[a.length - 1].equity;
    return [
      {
        id: 'wf-cnn-v1',
        created_at: this.now - 2 * 86400,
        kind: 'train',
        title: 'walk-forward CNN v1',
        summary: {
          folds: 6,
          purge_gap_bars: 288,
          n_trades: 412,
          win_rate: 0.617,
          total_pnl: +(last(wf) - 100).toFixed(2),
          profit_factor: 1.48,
          sharpe: 2.31,
          max_drawdown_pct: 9.4,
          fees: 66.2,
          avg_win: 1.52,
          avg_loss: -0.94,
          breakeven_threshold: 0.58,
          calibration_error: 0.034,
        },
        equity: wf,
      },
      {
        id: 'baseline-random',
        created_at: this.now - 2 * 86400 + 600,
        kind: 'backtest',
        title: 'baseline random entries (same exits)',
        summary: {
          n_trades: 418,
          win_rate: 0.452,
          total_pnl: +(last(base) - 100).toFixed(2),
          profit_factor: 0.88,
          sharpe: -0.94,
          max_drawdown_pct: 17.1,
          fees: 67.0,
          avg_win: 1.5,
          avg_loss: -1.0,
          breakeven_threshold: 0.6,
        },
        equity: base,
      },
      {
        id: 'bt-cnn-thr75',
        created_at: this.now - 86400,
        kind: 'backtest',
        title: 'CNN v1 @ thr 0.75, atr exits, trail on',
        summary: {
          n_trades: 188,
          win_rate: 0.649,
          total_pnl: +(last(th) - 100).toFixed(2),
          profit_factor: 1.71,
          sharpe: 2.02,
          max_drawdown_pct: 6.8,
          fees: 30.1,
          avg_win: 1.61,
          avg_loss: -0.9,
          breakeven_threshold: 0.58,
        },
        equity: th,
      },
    ];
  }

  // ------------------------------------------------------------------ tick
  /** Advance one simulated second. Returns the WS messages produced. */
  tick() {
    this.out = [];
    this.now += BAR / this.N;
    this.tickN++;
    const nowB = bucket(this.now);
    const gBtc = this.gauss();
    let barClosed = false;

    for (const sym of this.order) {
      const S = this.syms[sym];
      let cur = S.candles[S.candles.length - 1];
      if (nowB > cur.t) {
        // close the live bar, evaluate signal on it, open the next
        this._closeBarHousekeeping(S);
        const idx = S.candles.length - 1;
        const sig = this._evalSignal(S, idx, false);
        S.sigs.push(sig);
        if (S.sigs.length > 700) S.sigs.shift();
        S.preview = null;
        S.finalSig = sig;
        this.out.push({ type: 'candle', data: { symbol: sym, tf: '5m', candle: { ...cur } } });
        this._newBar(S, nowB);
        barClosed = true;
        S.tickInBar = 0;
        cur = S.candles[S.candles.length - 1];
      }
      this._stepPrice(S, gBtc, S.tickInBar + 1);
      this.out.push({ type: 'tick', data: { symbol: sym, t: this.now, price: S.price } });
      this.out.push({ type: 'candle', data: { symbol: sym, tf: '5m', candle: { ...S.candles[S.candles.length - 1] } } });
    }

    if (barClosed) {
      this._onBarClose();
      this._shadowTick();
    }

    // intrabar model preview for the hot list (staggered)
    this.universeSyms.forEach((sym, i) => {
      if ((this.tickN + i) % 4 !== 0) return;
      const S = this.syms[sym];
      const sig = this._evalSignal(S, S.candles.length - 1, true);
      S.preview = sig;
      this.out.push({ type: 'signal', data: sig });
    });

    for (const p of this.positions.slice()) this._manage(p);
    this._stepFaults();

    // kill switch
    const eq = this._equity();
    if (eq > this.peak) this.peak = eq;
    const dd = ((this.peak - eq) / this.peak) * 100;
    if (!this.kill.tripped && dd >= this.settings.kill_dd_pct) this._tripKill(`drawdown ${dd.toFixed(1)}% >= ${this.settings.kill_dd_pct}%`);

    if (this.tickN % 3 === 0) {
      const pt = { t: Math.floor(this.now), equity: +eq.toFixed(4) };
      this.equityHist.push(pt);
      if (this.equityHist.length > 4000) this.equityHist.shift();
      this.out.push({ type: 'equity', data: pt });
    }
    if (this.tickN % 2 === 0) this.out.push({ type: 'status', data: this._status() });
    this._emitSafetyIfChanged();
    if (this.tickN % 40 === 0) this._reRank();
    // order books: one universe symbol per tick, round robin (each refreshed every ~8 ticks)
    if (this.universeSyms.length) {
      const sym = this.universeSyms[this.tickN % this.universeSyms.length];
      this.out.push({ type: 'orderbook', data: this._orderbook(sym) });
    }
    if (this.tickN % 17 === 0) this._ambientLog();
    return this.out;
  }

  _onBarClose() {
    const n = this.universeSyms.length;
    this._log('DEBUG', `[scorer] scored ${n} symbols in ${(4 + this.rand() * 9).toFixed(0)} ms`);
    const cfgMax = this._maxPositions();
    for (const sym of this.universeSyms) {
      const S = this.syms[sym];
      const sig = S.finalSig;
      if (!sig) continue;
      this.out.push({ type: 'signal', data: sig });
      if (sig.action === 'none') continue;
      this._log('INFO', `[scorer] SIGNAL ${sig.action.toUpperCase()} ${sym} p_long=${sig.prob_long.toFixed(2)} p_short=${sig.prob_short.toFixed(2)}`);
      if (!this.running || this.kill.tripped) continue;
      const safety = this._safety();
      if (safety.state !== 'NORMAL') {
        this._log('INFO', `[guard] entry ${sym} blocked: safety ${safety.state}`);
        continue;
      }
      if (this.paused.has(sym)) {
        this._log('INFO', `[engine] entry ${sym} skipped: coin paused`);
        continue;
      }
      if (this._symbolExcluded(sym)) {
        this._log('INFO', `[guard] entry ${sym} blocked: data fault matured`);
        continue;
      }
      if (this.positions.some((p) => p.symbol === sym)) continue;
      if ((this.cooldown[sym] || 0) > this.tickN) continue;
      if (this.positions.length >= cfgMax) {
        this._log('INFO', `[risk] skip ${sym}: ${this.positions.length}/${cfgMax} slots used`);
        continue;
      }
      this._openPosition(S, sig.action, sig);
    }
  }

  _reRank() {
    const prev = this.universeSyms;
    const next = this._rankUniverse(prev);
    const added = next.filter((s) => !prev.includes(s));
    const removed = prev.filter((s) => !next.includes(s));
    this.universeSyms = next;
    this.lastUniverseAt = this.now;
    for (const s of added) this.syms[s].since = this.now;
    const sh = (s) => s.replace('USDT', '');
    this._log('INFO', `[universe] re-rank: ${added.length || removed.length ? `+${added.map(sh).join(',') || '-'} -${removed.map(sh).join(',') || '-'}` : 'no change'}`);
    this.out.push({ type: 'universe', data: this._universeRows() });
  }

  _ambientLog() {
    const r = this.rand();
    const sym = this.universeSyms[Math.floor(this.rand() * this.universeSyms.length)];
    if (r < 0.45) this._log('DEBUG', `[engine] loop ok: ${this.positions.length} open, stops verified on exchange`);
    else if (r < 0.7) this._log('INFO', `[exchange] funding ${sym.replace('USDT', '')} ${(this.syms[sym].fund * 100).toFixed(4)}%`);
    else if (r < 0.88) this._log('WARN', `[exchange] ${sym} spread widened to ${(2 + this.rand() * 6).toFixed(1)} bps`);
    else if (r < 0.95) this._log('INFO', '[risk] correlation check ok (max corr 0.62)');
    else this._log('ERROR', '[exchange] ws kline stream reconnect (1006), resubscribed in 0.8 s');
  }

  _tripKill(reason) {
    this.kill.tripped = true;
    this.kill.reason = reason;
    this.running = false;
    this.flattenUntil = this.tickN + 4;
    this._log('ERROR', `[risk] KILL SWITCH: ${reason} -> closing all positions`);
    for (const p of this.positions.slice()) this._closePosition(p, this.syms[p.symbol].price, 'kill');
    this.out.push({ type: 'kill', data: { ...this.kill } });
    this.out.push({ type: 'status', data: this._status() });
  }

  // ----------------------------------------------------------- order book
  /** Realistic book: cumulative notional per band grows ~ pct^0.85, imbalance random-walks per symbol. */
  _orderbook(sym) {
    const S = this.syms[sym];
    const B = (S.book = S.book || { imb: 0, liq: S.qv * 0.0016 * (0.7 + this.rand() * 0.6), seed: this.rand() });
    B.imb = clamp(0.92 * B.imb + 0.09 * this.gauss(), -0.55, 0.55);
    const mid = S.price;
    const spread = mid * (0.00004 + 0.00006 * this.rand());
    const mk = (side) => {
      const k = side === 'bid' ? 1 + B.imb : 1 - B.imb;
      let prev = 0;
      return [0.2, 1, 2, 3, 4, 5].map((pct) => {
        const base = B.liq * Math.pow(pct / 5, 0.85) * k * (0.9 + 0.2 * this.rand());
        const notional = Math.max(prev * 1.02, base);
        prev = notional;
        return { notional, depth: notional / mid };
      });
    };
    const bid = mk('bid');
    const ask = mk('ask');
    const bands = [0.2, 1, 2, 3, 4, 5].map((pct, i) => ({
      pct,
      bid_depth: +bid[i].depth.toPrecision(6),
      bid_notional: Math.round(bid[i].notional),
      ask_depth: +ask[i].depth.toPrecision(6),
      ask_notional: Math.round(ask[i].notional),
    }));
    const tick = Math.max(mid * 0.00002, 1e-9);
    const ladder = (dir, k) => {
      const out = [];
      let px = mid + dir * spread / 2;
      for (let i = 0; i < 20; i++) {
        const qty = (B.liq / mid) * 0.004 * k * (0.3 + this.rand() * 1.4) * (1 + (i % 5 === 0 ? 1.2 : 0));
        out.push([+px.toPrecision(7), +qty.toPrecision(5)]);
        px += dir * tick * (1 + Math.floor(this.rand() * 4));
      }
      return out;
    };
    return { symbol: sym, t: Math.floor(this.now), mid, spread, bands, bids: ladder(-1, 1 + B.imb), asks: ladder(1, 1 - B.imb) };
  }

  // ------------------------------------------------------------------ REST
  _validateSettings(patch) {
    const merged = this._deepMerge(JSON.parse(JSON.stringify(this.settings)), patch);
    for (const [path, lo, hi, int] of SETTINGS_RULES) {
      const v = path.split('.').reduce((o, k) => (o == null ? o : o[k]), merged);
      if (typeof v !== 'number' || !Number.isFinite(v)) return `${path}: must be a number`;
      if (v < lo || v > hi) return `${path}: must be between ${lo} and ${hi}`;
      if (int && !Number.isInteger(v)) return `${path}: must be an integer`;
    }
    if (!['fixed', 'atr'].includes(merged.exit_mode)) return 'exit_mode: must be "fixed" or "atr"';
    for (const k of ['short_enabled', 'sound']) if (typeof merged[k] !== 'boolean') return `${k}: must be a boolean`;
    if (typeof merged.trail.enabled !== 'boolean') return 'trail.enabled: must be a boolean';
    return null;
  }

  _deepMerge(a, b) {
    for (const k of Object.keys(b || {})) {
      if (b[k] && typeof b[k] === 'object' && !Array.isArray(b[k]) && a[k] && typeof a[k] === 'object') this._deepMerge(a[k], b[k]);
      else a[k] = b[k];
    }
    return a;
  }

  /** @returns {{status:number, body:any}} */
  request(method, path, query = {}, body = null) {
    const ok = (b) => ({ status: 200, body: b });
    const limit = (d) => Math.max(1, Math.min(5000, parseInt(query.limit, 10) || d));
    const route = `${method} ${path}`;
    switch (route) {
      case 'GET /api/health':
        return ok({ ok: true, engine_alive: true, engine_heartbeat_age_s: this._hbStale() ? +(31 + this.rand() * 5).toFixed(1) : +(this.rand() * 1.4).toFixed(2), mode: this.mode, version: '0.2.0-mock' });
      case 'GET /api/status':
        return ok(this._status());
      case 'GET /api/market':
        return ok(this._market());
      case 'GET /api/universe':
        return ok(this._universeRows());
      case 'GET /api/candles': {
        const S = this.syms[query.symbol];
        if (!S) return { status: 404, body: { error: `unknown symbol ${query.symbol}` } };
        return ok(S.candles.slice(-limit(500)).map((c) => ({ ...c })));
      }
      case 'GET /api/signals': {
        const S = this.syms[query.symbol];
        if (!S) return { status: 404, body: { error: `unknown symbol ${query.symbol}` } };
        const arr = S.sigs.slice(-limit(200));
        return ok(S.preview ? [...arr, S.preview] : arr);
      }
      case 'GET /api/positions':
        return ok(this.positions.map((p) => this._pub(p)));
      case 'GET /api/trades':
        return ok(this.trades.slice(-limit(200)).reverse());
      case 'GET /api/equity':
        return ok(this.equityHist.slice(-limit(2000)));
      case 'GET /api/logs':
        return ok(this.logs.slice(-limit(200)));
      case 'GET /api/settings':
        return ok(JSON.parse(JSON.stringify(this.settings)));
      case 'PUT /api/settings': {
        if (!body || typeof body !== 'object') return { status: 422, body: { error: 'body must be an object' } };
        const err = this._validateSettings(body);
        if (err) return { status: 422, body: { error: err } };
        this._deepMerge(this.settings, body);
        this.kill.dd_limit_pct = this.settings.kill_dd_pct;
        this._log('INFO', `[api] settings updated: ${Object.keys(body).join(', ')}`);
        return ok(JSON.parse(JSON.stringify(this.settings)));
      }
      case 'POST /api/control':
        return ok(this._control(body || {}));
      case 'GET /api/orderbook': {
        const S = this.syms[query.symbol];
        if (!S) return { status: 404, body: { error: `unknown symbol ${query.symbol}` } };
        return ok(this._orderbook(query.symbol));
      }
      case 'GET /api/shadow':
        return ok(this._shadowRows());
      case 'GET /api/lab/reports':
        return ok(this.labReports);
      default:
        return { status: 404, body: { error: `no route ${route}` } };
    }
  }

  _control({ action, position_id, model, symbol }) {
    this.out = [];
    switch (action) {
      case 'start':
        if (this.kill.tripped) return { ok: false, error: 'kill switch tripped: reset_kill first' };
        this.running = true;
        this._log('INFO', '[engine] START: entries enabled');
        break;
      case 'stop':
        this.running = false;
        this._log('WARN', '[engine] STOP: no new entries (open trades keep running to TP/SL)');
        break;
      case 'kill':
        this._tripKill('manual kill from UI');
        break;
      case 'reset_kill':
        this.kill.tripped = false;
        this.kill.reason = '';
        this.peak = this._equity();
        this._log('INFO', '[risk] kill switch reset; peak equity re-based');
        break;
      case 'close_position': {
        const p = this.positions.find((x) => x.id === position_id);
        if (!p) return { ok: false, error: `no open position ${position_id}` };
        this._closePosition(p, this.syms[p.symbol].price, 'manual');
        break;
      }
      case 'promote_model': {
        const r = this._promote(model);
        if (!r.ok) return r;
        break;
      }
      case 'rollback_model': {
        const r = this._rollback(model);
        if (!r.ok) return r;
        break;
      }
      case 'pause_symbol':
      case 'unpause_symbol': {
        if (!symbol || !this.syms[symbol]) return { ok: false, error: `unknown symbol ${symbol}` };
        if (action === 'pause_symbol') this.paused.add(symbol);
        else this.paused.delete(symbol);
        this._log('INFO', `[engine] ${symbol} ${action === 'pause_symbol' ? 'PAUSED: no new entries (open trade keeps running)' : 'unpaused'}`);
        break;
      }
      case 'refresh_universe':
        this._reRank();
        this._log('INFO', '[engine] universe refresh forced from UI');
        break;
      case 'arm_live': {
        // mock-only demo of the LIVE-mode card; the real backend has no such action yet
        this.modeBefore = this.mode;
        this.mode = 'live';
        this.killBefore = this.settings.kill_dd_pct;
        this.settings.kill_dd_pct = 25;
        this._log('WARN', '[engine] LIVE MODE ARMED (mock): kill limit 25%, first 20 trades at size x0.5');
        break;
      }
      case 'disarm_live':
        this.mode = this.modeBefore || 'paper_local';
        if (this.killBefore) this.settings.kill_dd_pct = this.killBefore;
        this._log('INFO', '[engine] live mode disarmed');
        break;
      case 'close_all':
        for (const p of this.positions.slice()) this._closePosition(p, this.syms[p.symbol].price, 'manual');
        break;
      default:
        return { ok: false, error: `unknown action ${action}` };
    }
    this.out.push({ type: 'status', data: this._status() });
    return { ok: true };
  }
}

/** Live runner: wraps a MockServer with a 1 s timer and a message callback. */
export function startMock({ ticksPerBar = 30, seed } = {}) {
  const server = new MockServer({ ticksPerBar, seed });
  const listeners = new Set();
  const emit = (msgs) => {
    for (const m of msgs) for (const fn of listeners) fn(m);
  };
  let timer = null;
  return {
    server,
    request: (method, path, query, body) => {
      server.out = [];
      const r = server.request(method, path, query, body);
      const extra = server.out; // messages caused by this request (logs, closes, status)
      server.out = [];
      if (extra.length) setTimeout(() => emit(extra), 0);
      return r;
    },
    subscribe(fn) {
      listeners.add(fn);
      if (!timer && typeof setInterval === 'function') {
        timer = setInterval(() => emit(server.tick()), 1000);
      }
      return () => {
        listeners.delete(fn);
        if (!listeners.size && timer) {
          clearInterval(timer);
          timer = null;
        }
      };
    },
  };
}
