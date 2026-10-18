// Chart wrappers around TradingView Lightweight Charts v4 (global `LightweightCharts`, vendored).
//  - createPriceChart: candles + volume + EMA20/50 + synced model-probability pane + position lines/markers
//  - createLineChart:  equity / baseline / drawdown line charts
// Everything here resizes with its container through ResizeObserver.

import { esc, fmtPrice, pricePrecision, isNum, fmtNum } from './format.js';
import { probColor } from './theme.js';

const LW = () => globalThis.LightweightCharts;
const BAR = 300;

const cssVar = (name, fallback) => {
  try {
    const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    return v || fallback;
  } catch (_) {
    return fallback;
  }
};

export function palette() {
  return {
    bg: '#000000',
    text: cssVar('--text-dim', '#8b8f9a'),
    grid: 'rgba(255,255,255,0.045)',
    border: 'rgba(255,255,255,0.12)',
    up: cssVar('--up', '#34f0a0'),
    down: cssVar('--down', '#ff4d7d'),
    coral: cssVar('--coral', '#D97757'),
    magenta: cssVar('--magenta', '#ff4fd8'),
    violet: cssVar('--violet', '#a78bfa'),
    cyan: cssVar('--cyan', '#4fe3ff'),
    lime: cssVar('--lime', '#9dff5a'),
    yellow: cssVar('--yellow', '#ffd84f'),
  };
}

const pad = (n) => String(n).padStart(2, '0');
function localTick(time, type) {
  const d = new Date(time * 1000);
  if (type === 0) return String(d.getFullYear());
  if (type === 1) return `${pad(d.getMonth() + 1)}/${pad(d.getDate())}`;
  if (type === 2) return `${pad(d.getMonth() + 1)}/${pad(d.getDate())}`;
  return `${pad(d.getHours())}:${pad(d.getMinutes())}`;
}
function localTime(time) {
  const d = new Date(time * 1000);
  return `${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`;
}

function baseOptions(P, { timeVisible = true, showTime = true } = {}) {
  return {
    layout: {
      background: { type: 'solid', color: P.bg },
      textColor: P.text,
      fontFamily: '-apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif',
      fontSize: 11,
    },
    grid: { vertLines: { color: P.grid }, horzLines: { color: P.grid } },
    rightPriceScale: { borderColor: P.border, minimumWidth: 94 },
    timeScale: {
      borderColor: P.border,
      timeVisible,
      secondsVisible: false,
      rightOffset: 6,
      visible: showTime,
      tickMarkFormatter: localTick,
    },
    localization: { timeFormatter: localTime },
    crosshair: {
      mode: 0,
      vertLine: { color: 'rgba(217,119,87,0.55)', width: 1, style: 3, labelBackgroundColor: '#3a2a22' },
      horzLine: { color: 'rgba(217,119,87,0.55)', width: 1, style: 3, labelBackgroundColor: '#3a2a22' },
    },
    handleScale: { axisPressedMouseMove: true },
  };
}

/** Run cb(width,height) whenever el resizes. Returns a disconnect fn. */
export function observeResize(el, cb) {
  if (typeof ResizeObserver === 'undefined') {
    const f = () => cb(el.clientWidth, el.clientHeight);
    window.addEventListener('resize', f);
    f();
    return () => window.removeEventListener('resize', f);
  }
  const ro = new ResizeObserver((entries) => {
    const r = entries[0].contentRect;
    cb(Math.floor(r.width), Math.floor(r.height));
  });
  ro.observe(el);
  return () => ro.disconnect();
}

function emaArray(candles, n) {
  const k = 2 / (n + 1);
  const out = new Array(candles.length);
  let e = null;
  for (let i = 0; i < candles.length; i++) {
    e = e == null ? candles[i].c : e + k * (candles[i].c - e);
    out[i] = e;
  }
  return out;
}

const snap = (t) => Math.floor(t / BAR) * BAR;

// ---------------------------------------------------------------------------------------------
export function createPriceChart(priceEl, probEl, legendEl) {
  const P = palette();
  const lw = LW();
  const priceChart = lw.createChart(priceEl, { ...baseOptions(P, { showTime: false }), width: priceEl.clientWidth || 600, height: priceEl.clientHeight || 400 });
  const probChart = lw.createChart(probEl, { ...baseOptions(P, { showTime: true }), width: probEl.clientWidth || 600, height: probEl.clientHeight || 120 });

  const candleSeries = priceChart.addCandlestickSeries({
    upColor: P.up,
    downColor: P.down,
    borderVisible: false,
    wickUpColor: P.up,
    wickDownColor: P.down,
    priceLineVisible: true,
    lastValueVisible: true,
  });
  priceChart.priceScale('right').applyOptions({ scaleMargins: { top: 0.06, bottom: 0.2 } });
  const volSeries = priceChart.addHistogramSeries({
    priceFormat: { type: 'volume' },
    priceScaleId: 'vol',
    lastValueVisible: false,
    priceLineVisible: false,
  });
  priceChart.priceScale('vol').applyOptions({ scaleMargins: { top: 0.84, bottom: 0 } });
  const lineOpts = { lineWidth: 1, crosshairMarkerVisible: false, lastValueVisible: false, priceLineVisible: false };
  const ema20S = priceChart.addLineSeries({ ...lineOpts, color: P.yellow });
  const ema50S = priceChart.addLineSeries({ ...lineOpts, color: P.violet });

  const fixed01 = () => ({ priceRange: { minValue: 0, maxValue: 1 } });
  const probLong = probChart.addLineSeries({
    color: P.lime,
    lineWidth: 2,
    autoscaleInfoProvider: fixed01,
    priceFormat: { type: 'custom', minMove: 0.01, formatter: (v) => v.toFixed(2) },
    lastValueVisible: true,
    priceLineVisible: false,
    crosshairMarkerVisible: true,
  });
  const probShort = probChart.addLineSeries({
    color: 'rgba(255,79,216,0.55)',
    lineWidth: 1,
    autoscaleInfoProvider: fixed01,
    priceFormat: { type: 'custom', minMove: 0.01, formatter: (v) => v.toFixed(2) },
    lastValueVisible: false,
    priceLineVisible: false,
    crosshairMarkerVisible: false,
  });
  probChart.priceScale('right').applyOptions({ scaleMargins: { top: 0.08, bottom: 0.08 } });

  let candles = [];
  let idxByT = new Map();
  let ema20 = [];
  let ema50 = [];
  let sigByT = new Map();
  let symbol = '';
  let thrLine = null;
  let thrValue = null;
  let lines = [];
  let lineKey = '';
  let lastLegend = null;
  let syncing = false;
  let destroyed = false;

  // --- time-scale sync (the prob series has whitespace for every candle time so indices line up) ---
  const syncRange = (from, to) => (range) => {
    if (syncing || !range) return;
    syncing = true;
    try {
      to.timeScale().setVisibleLogicalRange(range);
    } finally {
      syncing = false;
    }
  };
  const onPriceRange = syncRange(priceChart, probChart);
  const onProbRange = syncRange(probChart, priceChart);
  priceChart.timeScale().subscribeVisibleLogicalRangeChange(onPriceRange);
  probChart.timeScale().subscribeVisibleLogicalRangeChange(onProbRange);

  // --- crosshair sync + legend ---
  const setLegend = (c) => {
    if (!legendEl) return;
    const cc = c || candles[candles.length - 1];
    if (!cc) {
      legendEl.innerHTML = '';
      return;
    }
    const i = idxByT.get(cc.t);
    const prev = i > 0 ? candles[i - 1] : null;
    const chg = prev ? ((cc.c / prev.c - 1) * 100) : 0;
    const cls = cc.c >= cc.o ? 'pos' : 'neg';
    const e20 = i != null ? ema20[i] : null;
    const e50 = i != null ? ema50[i] : null;
    const sg = sigByT.get(cc.t);
    legendEl.innerHTML =
      `<span class="lg-sym">${esc(symbol)}</span><span class="lg-dim"> 5m</span> ` +
      `<span class="lg-dim">O</span><span class="${cls}">${fmtPrice(cc.o)}</span> ` +
      `<span class="lg-dim">H</span><span class="${cls}">${fmtPrice(cc.h)}</span> ` +
      `<span class="lg-dim">L</span><span class="${cls}">${fmtPrice(cc.l)}</span> ` +
      `<span class="lg-dim">C</span><span class="${cls}">${fmtPrice(cc.c)}</span> ` +
      `<span class="${cls}">${chg >= 0 ? '+' : ''}${chg.toFixed(2)}%</span> ` +
      `<span class="lg-dim">VOL</span> <span>${fmtNum(cc.v, cc.v >= 100 ? 0 : 2)}</span> ` +
      `<span class="lg-ema20">EMA20 ${isNum(e20) ? fmtPrice(e20) : '–'}</span> ` +
      `<span class="lg-ema50">EMA50 ${isNum(e50) ? fmtPrice(e50) : '–'}</span>` +
      (sg ? ` <span class="lg-prob">P(long) ${sg.prob_long.toFixed(2)}</span>` : '');
  };

  let xhair = false;
  priceChart.subscribeCrosshairMove((p) => {
    if (destroyed) return;
    if (!p || p.time === undefined || !p.point) {
      if (xhair) {
        try {
          probChart.clearCrosshairPosition();
        } catch (_) {
          /* old lib */
        }
        xhair = false;
      }
      setLegend(null);
      return;
    }
    const c = candles[idxByT.get(p.time)];
    setLegend(c);
    try {
      const sg = sigByT.get(p.time);
      probChart.setCrosshairPosition(sg ? sg.prob_long : 0.5, p.time, probLong);
      xhair = true;
    } catch (_) {
      /* setCrosshairPosition unavailable */
    }
  });
  probChart.subscribeCrosshairMove((p) => {
    if (destroyed) return;
    if (!p || p.time === undefined || !p.point) return;
    const c = candles[idxByT.get(p.time)];
    if (c) {
      setLegend(c);
      try {
        priceChart.setCrosshairPosition(c.c, p.time, candleSeries);
      } catch (_) {
        /* ignore */
      }
    }
  });

  // --- resize ---
  const offRes1 = observeResize(priceEl, (w, h) => w > 0 && h > 0 && priceChart.applyOptions({ width: w, height: h }));
  const offRes2 = observeResize(probEl, (w, h) => w > 0 && h > 0 && probChart.applyOptions({ width: w, height: h }));

  // --- data helpers ---
  const volPoint = (c) => ({ time: c.t, value: c.v, color: c.c >= c.o ? 'rgba(52,240,160,0.30)' : 'rgba(255,77,125,0.30)' });
  const candlePoint = (c) => ({ time: c.t, open: c.o, high: c.h, low: c.l, close: c.c });

  function setPrecision(price) {
    const dp = pricePrecision(price);
    candleSeries.applyOptions({ priceFormat: { type: 'price', precision: dp, minMove: 1 / 10 ** dp } });
    ema20S.applyOptions({ priceFormat: { type: 'price', precision: dp, minMove: 1 / 10 ** dp } });
    ema50S.applyOptions({ priceFormat: { type: 'price', precision: dp, minMove: 1 / 10 ** dp } });
  }

  function probData() {
    return candles.map((c) => {
      const s = sigByT.get(c.t);
      return s ? { time: c.t, value: s.prob_long, color: probColor(s.prob_long) } : { time: c.t };
    });
  }
  function probShortData() {
    return candles.map((c) => {
      const s = sigByT.get(c.t);
      return s ? { time: c.t, value: s.prob_short } : { time: c.t };
    });
  }

  function reindex() {
    idxByT = new Map(candles.map((c, i) => [c.t, i]));
  }

  const api = {
    setData(sym, cs, sigs) {
      symbol = sym;
      candles = cs.map((c) => ({ ...c }));
      reindex();
      sigByT = new Map((sigs || []).map((s) => [s.t, s]));
      ema20 = emaArray(candles, 20);
      ema50 = emaArray(candles, 50);
      if (candles.length) setPrecision(candles[candles.length - 1].c);
      candleSeries.setData(candles.map(candlePoint));
      volSeries.setData(candles.map(volPoint));
      ema20S.setData(candles.map((c, i) => (i >= 19 ? { time: c.t, value: ema20[i] } : { time: c.t })));
      ema50S.setData(candles.map((c, i) => (i >= 49 ? { time: c.t, value: ema50[i] } : { time: c.t })));
      probLong.setData(probData());
      probShort.setData(probShortData());
      lineKey = '';
      const n = candles.length;
      priceChart.timeScale().setVisibleLogicalRange({ from: Math.max(0, n - 130), to: n + 8 });
      setLegend(null);
    },

    updateCandle(c) {
      if (!candles.length) return;
      const last = candles[candles.length - 1];
      let i;
      if (c.t === last.t) {
        candles[candles.length - 1] = { ...c };
        i = candles.length - 1;
      } else if (c.t > last.t) {
        candles.push({ ...c });
        i = candles.length - 1;
        idxByT.set(c.t, i);
        probLong.update({ time: c.t });
        probShort.update({ time: c.t });
      } else return;
      const k20 = 2 / 21;
      const k50 = 2 / 51;
      ema20[i] = i > 0 ? ema20[i - 1] + k20 * (c.c - ema20[i - 1]) : c.c;
      ema50[i] = i > 0 ? ema50[i - 1] + k50 * (c.c - ema50[i - 1]) : c.c;
      candleSeries.update(candlePoint(c));
      volSeries.update(volPoint(c));
      if (i >= 19) ema20S.update({ time: c.t, value: ema20[i] });
      if (i >= 49) ema50S.update({ time: c.t, value: ema50[i] });
      if (!xhair) setLegend(null);
    },

    updateSignal(s) {
      sigByT.set(s.t, s);
      if (!idxByT.has(s.t)) return;
      if (s.t === candles[candles.length - 1].t) {
        probLong.update({ time: s.t, value: s.prob_long, color: probColor(s.prob_long) });
        probShort.update({ time: s.t, value: s.prob_short });
      } else {
        // the final signal of a just-closed bar arrives after the next bar started: update() cannot
        // touch older points, so rewrite the (small) series
        probLong.setData(probData());
        probShort.setData(probShortData());
      }
    },

    setThreshold(v) {
      if (!isNum(v) || v === thrValue) return;
      thrValue = v;
      if (thrLine) probLong.removePriceLine(thrLine);
      thrLine = probLong.createPriceLine({ price: v, color: P.yellow, lineWidth: 1, lineStyle: 2, axisLabelVisible: true, title: 'thr' });
    },

    /** Entry / TP / SL / trailing lines for open positions of the shown symbol. */
    setPositionLines(positions) {
      const key = positions.map((p) => `${p.id}:${p.entry_price}:${p.tp_price}:${p.sl_price}:${p.trail_price}`).join('|');
      if (key === lineKey) return;
      lineKey = key;
      for (const l of lines) candleSeries.removePriceLine(l);
      lines = [];
      for (const p of positions) {
        const tag = `${p.side === 'long' ? 'LONG' : 'SHORT'}`;
        const mk = (price, color, style, title) =>
          isNum(price) && lines.push(candleSeries.createPriceLine({ price, color, lineWidth: 1, lineStyle: style, axisLabelVisible: true, title }));
        mk(p.entry_price, P.cyan, 0, `ENTRY ${tag}`);
        mk(p.tp_price, P.lime, 2, 'TP');
        mk(p.sl_price, P.down, 2, 'SL');
        if (p.trail_active && isNum(p.trail_price)) mk(p.trail_price, P.yellow, 1, 'TRAIL');
      }
    },

    /** Arrow markers for entries (open + past trades of this symbol) and small dots at exits. */
    setMarkers(trades, positions) {
      if (!candles.length) return;
      const first = candles[0].t;
      const last = candles[candles.length - 1].t;
      const ms = [];
      const inRange = (t) => t >= first && t <= last;
      for (const t of trades) {
        const a = snap(t.opened_at);
        const long = t.side === 'long';
        if (inRange(a)) ms.push({ time: a, position: long ? 'belowBar' : 'aboveBar', color: long ? P.cyan : P.magenta, shape: long ? 'arrowUp' : 'arrowDown', text: '' });
        const b = snap(t.closed_at);
        if (inRange(b)) ms.push({ time: b, position: long ? 'aboveBar' : 'belowBar', color: t.pnl >= 0 ? P.up : P.down, shape: 'circle', text: '' });
      }
      for (const p of positions) {
        const a = snap(p.opened_at);
        const long = p.side === 'long';
        const tm = inRange(a) ? a : first;
        ms.push({ time: tm, position: long ? 'belowBar' : 'aboveBar', color: P.cyan, shape: long ? 'arrowUp' : 'arrowDown', text: 'IN' });
      }
      ms.sort((x, y) => x.time - y.time);
      candleSeries.setMarkers(ms.slice(-160));
    },

    resetView() {
      const n = candles.length;
      priceChart.timeScale().setVisibleLogicalRange({ from: Math.max(0, n - 130), to: n + 8 });
    },

    destroy() {
      destroyed = true;
      offRes1();
      offRes2();
      try {
        priceChart.remove();
        probChart.remove();
      } catch (_) {
        /* already gone */
      }
    },
  };
  return api;
}

// ---------------------------------------------------------------------------------------------
/**
 * Simple time-series chart. opts: {kind:'baseline'|'area'|'line', baseValue, color, height, format}
 * data points: {t, v}.
 */
export function createLineChart(el, opts = {}) {
  const P = palette();
  const lw = LW();
  const chart = lw.createChart(el, { ...baseOptions(P), width: el.clientWidth || 400, height: el.clientHeight || 200 });
  const color = opts.color || P.coral;
  const fmt = opts.format || ((v) => fmtNum(v, 2));
  const common = { priceFormat: { type: 'custom', minMove: 0.01, formatter: fmt }, lastValueVisible: true, priceLineVisible: false };
  let series;
  if (opts.kind === 'baseline') {
    series = chart.addBaselineSeries({
      ...common,
      baseValue: { type: 'price', price: opts.baseValue ?? 0 },
      topLineColor: P.up,
      topFillColor1: 'rgba(52,240,160,0.30)',
      topFillColor2: 'rgba(52,240,160,0.02)',
      bottomLineColor: P.down,
      bottomFillColor1: 'rgba(255,77,125,0.02)',
      bottomFillColor2: 'rgba(255,77,125,0.30)',
      lineWidth: 2,
    });
  } else if (opts.kind === 'area') {
    series = chart.addAreaSeries({ ...common, lineColor: color, topColor: `${color}66`, bottomColor: `${color}05`, lineWidth: 2 });
  } else {
    series = chart.addLineSeries({ ...common, color, lineWidth: 2 });
  }
  const off = observeResize(el, (w, h) => w > 0 && h > 0 && chart.applyOptions({ width: w, height: h }));
  let baseLine = null;
  return {
    chart,
    series,
    setBase(v) {
      if (baseLine) series.removePriceLine(baseLine);
      baseLine = isNum(v) ? series.createPriceLine({ price: v, color: 'rgba(255,255,255,0.35)', lineWidth: 1, lineStyle: 2, axisLabelVisible: false, title: opts.baseTitle || 'start' }) : null;
    },
    setData(points, fit = true) {
      const clean = [];
      let lastT = -Infinity;
      for (const p of points) {
        if (!isNum(p.t) || !isNum(p.v) || p.t <= lastT) continue; // strictly ascending, finite
        clean.push({ time: p.t, value: p.v });
        lastT = p.t;
      }
      series.setData(clean);
      if (fit) chart.timeScale().fitContent();
    },
    update(p) {
      if (isNum(p.t) && isNum(p.v)) series.update({ time: p.t, value: p.v });
    },
    destroy() {
      off();
      try {
        chart.remove();
      } catch (_) {
        /* ignore */
      }
    },
  };
}
