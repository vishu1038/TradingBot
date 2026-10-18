// Small HTML-string components shared by views. Every dynamic value goes through esc().
import { esc, fmtUsd, fmtPct, signClass, isNum, clamp, shortSym } from './format.js';
import { coinColor, probColor } from './theme.js';

export function tile(label, value, cls = '', sub = '') {
  return `<div class="tile"><div class="tile-l">${esc(label)}</div><div class="tile-v ${cls}">${value}</div>${sub ? `<div class="tile-s">${sub}</div>` : ''}</div>`;
}

export function kv(label, value, cls = '') {
  return `<div class="kv"><span class="kv-l">${esc(label)}</span><span class="kv-v ${cls}">${value}</span></div>`;
}

export function sidePill(side) {
  const long = side === 'long';
  return `<span class="side ${long ? 'long' : 'short'}">${long ? 'LONG' : 'SHORT'}</span>`;
}

export function exitPill(reason) {
  return `<span class="exit exit-${esc(reason)}">${esc(String(reason).toUpperCase())}</span>`;
}

export function pnlSpan(v, dp = 2) {
  return `<span class="${signClass(v)}">${fmtUsd(v, dp, true)}</span>`;
}

export function pctSpan(v, dp = 2) {
  return `<span class="${signClass(v)}">${fmtPct(v, dp)}</span>`;
}

/** Diverging contribution bars. reasons: [{feature,label,value,contribution}] */
export function reasonBars(reasons) {
  if (!reasons || !reasons.length) return '<div class="empty">no model reasons yet</div>';
  const rs = reasons.slice(0, 9);
  const max = Math.max(0.0001, ...rs.map((r) => Math.abs(r.contribution || 0)));
  return `<div class="why">${rs
    .map((r) => {
      const c = r.contribution || 0;
      const w = clamp((Math.abs(c) / max) * 100, 0, 100);
      const pos = c >= 0;
      return `<div class="why-row" title="${esc(r.feature)} = ${esc(isNum(r.value) ? r.value : '')}">
        <span class="why-l">${esc(r.label || r.feature)}</span>
        <span class="why-bar"><i class="${pos ? 'pos' : 'neg'}" style="width:${w.toFixed(1)}%;${pos ? 'left:50%' : `right:50%`}"></i></span>
        <span class="why-v ${pos ? 'pos' : 'neg'}">${pos ? '+' : ''}${c.toFixed(2)}</span>
      </div>`;
    })
    .join('')}</div>`;
}

/** Progress of price between SL and TP (for the positions list). */
export function slTpBar(p) {
  const lo = Math.min(p.sl_price, p.tp_price);
  const hi = Math.max(p.sl_price, p.tp_price);
  if (!isNum(lo) || !isNum(hi) || hi === lo) return '';
  const pos = clamp(((p.mark_price - lo) / (hi - lo)) * 100, 0, 100);
  const ent = clamp(((p.entry_price - lo) / (hi - lo)) * 100, 0, 100);
  const flip = p.side === 'short'; // for shorts TP is the lower price
  return `<div class="sltp"><i class="sltp-zone ${flip ? 'tp' : 'sl'}" style="left:0;width:${ent}%"></i><i class="sltp-zone ${flip ? 'sl' : 'tp'}" style="left:${ent}%;width:${100 - ent}%"></i><b class="sltp-mark" style="left:${pos}%"></b></div>`;
}

export function symLabel(sym) {
  return `<span class="sym" style="--cc:${coinColor(sym)}"><i class="coin-dot"></i>${esc(shortSym(sym))}</span>`;
}

/** Probability pill on the cool -> hot gradient. */
export function probPill(p) {
  if (!isNum(p)) return '<span class="prob dim">–</span>';
  return `<span class="prob" style="--pc:${probColor(p)}">${p.toFixed(2)}</span>`;
}

/** Drawdown derating multiplier (same curve as backend risk.derate_multiplier). */
export function derateAt(ddPct, killPct) {
  const f = killPct > 0 ? Math.max(0, ddPct) / killPct : 0;
  if (f <= 0.4) return 1;
  if (f <= 0.8) return 1 - ((f - 0.4) / 0.4) * 0.5;
  if (f < 1) return 0.5 * (1 - (f - 0.8) / 0.2);
  return 0;
}

/** Tiny inline SVG line/area chart. values: numbers. */
export function sparkline(values, { w = 160, h = 44, color = '#4fe3ff', fill = true } = {}) {
  const v = values.filter(isNum);
  if (v.length < 2) return `<svg class="spark" viewBox="0 0 ${w} ${h}" width="100%" height="${h}" preserveAspectRatio="none"></svg>`;
  const lo = Math.min(...v);
  const hi = Math.max(...v);
  const span = hi - lo || 1;
  const pts = v.map((x, i) => `${((i / (v.length - 1)) * w).toFixed(1)},${(h - 3 - ((x - lo) / span) * (h - 6)).toFixed(1)}`);
  const line = pts.join(' ');
  return `<svg class="spark" viewBox="0 0 ${w} ${h}" width="100%" height="${h}" preserveAspectRatio="none" aria-hidden="true">
    ${fill ? `<polygon points="0,${h} ${line} ${w},${h}" fill="${color}" opacity="0.16"/>` : ''}
    <polyline points="${line}" fill="none" stroke="${color}" stroke-width="1.8" stroke-linejoin="round" vector-effect="non-scaling-stroke"/>
  </svg>`;
}

/** Mini donut-less "gauge" used by the derate indicator: SVG semicircle. mult 0..1 */
export function derateGauge(mult) {
  const m = clamp(isNum(mult) ? mult : 1, 0, 1);
  const L = 75.4;
  const col = m >= 0.99 ? 'var(--up)' : m >= 0.5 ? 'var(--yellow)' : 'var(--down)';
  return `<svg class="gauge" viewBox="0 0 56 34" width="56" height="34" aria-hidden="true">
    <path d="M4 30 A24 24 0 0 1 52 30" fill="none" stroke="rgba(255,255,255,0.12)" stroke-width="5" stroke-linecap="round"/>
    <path d="M4 30 A24 24 0 0 1 52 30" fill="none" stroke="${col}" stroke-width="5" stroke-linecap="round" stroke-dasharray="${(m * L).toFixed(1)} ${L}" style="filter:drop-shadow(0 0 3px ${col})"/>
    <text x="28" y="29" text-anchor="middle" class="gauge-t" fill="${col}">x${m.toFixed(2)}</text>
  </svg>`;
}

/** Latest-state safety label colours. */
export const SAFETY_CLASS = { NORMAL: 'ok', DEGRADED: 'warn', FLATTEN: 'crit blink', HALTED: 'crit' };

export function safetyOf(status) {
  const s = (status && status.safety) || null;
  return {
    state: (s && s.state) || 'NORMAL',
    reasons: (s && s.reasons) || [],
    derate: s && isNum(s.derate_mult) ? s.derate_mult : 1,
    drift: (s && s.drift) || null,
    faults: (s && s.faults) || [],
    present: !!s,
  };
}

export function faultsTable(faults, since) {
  if (!faults.length) return '<div class="empty">no data faults. all feeds plausible</div>';
  return `<table class="tbl compact"><thead><tr><th>symbol</th><th>kind</th><th class="r">count</th><th class="r">since</th><th>state</th></tr></thead><tbody>${faults
    .map(
      (f) =>
        `<tr><td>${symLabel(f.symbol)}</td><td>${esc(f.kind)}</td><td class="r">${esc(f.count)}</td><td class="r">${since(f.since)}</td><td>${f.count >= 3 ? '<span class="neg">EXCLUDED</span>' : '<span class="warn">debouncing</span>'}</td></tr>`
    )
    .join('')}</tbody></table>`;
}
