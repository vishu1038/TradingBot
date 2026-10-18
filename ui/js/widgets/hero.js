// HERO card: small, rotates every 30 s through 4 cards (PnL + health / open trades / mini chart / bot brain).
// A progress bar shows time to the next card. Rotation pauses on hover/touch; events jump to their card
// (TP/SL -> open trades, new entry -> brain, kill/degraded -> health); swipe or drag to flip.
import { store } from '../state.js';
import {
  esc, fmtUsd, fmtPct, fmtPrice, fmtDuration, signClass, throttle, isNum, winRatePct, shortSym, fmtNum,
} from '../format.js';
import { coinColor, probColor, PROB_GRADIENT } from '../theme.js';
import { symLabel, sidePill, slTpBar, sparkline, safetyOf, derateGauge, probPill, SAFETY_CLASS } from '../components.js';
import { pausedSet } from '../actions.js';

const NAMES = ['PNL + HEALTH', 'OPEN TRADES', 'MINI CHART', 'BOT BRAIN'];
const ROTATE_MS = 30000;

export function createHero() {
  const el = document.createElement('div');
  el.className = 'card hero ac-pink';
  el.innerHTML = `
    <h3>HERO <span class="hero-name" id="h-name"></span><span class="hero-dots" id="h-dots"></span></h3>
    <div class="hero-view" id="h-view"><div class="hero-track" id="h-track">
      ${NAMES.map((n, i) => `<section class="hero-slide" data-i="${i}" aria-label="${esc(n)}"></section>`).join('')}
    </div></div>
    <div class="hero-prog" aria-hidden="true"><i id="h-prog"></i></div>`;
  const $ = (id) => el.querySelector(`#${id}`);
  const track = $('h-track');
  const view = $('h-view');
  const slides = [...el.querySelectorAll('.hero-slide')];
  let idx = 0;
  let elapsed = 0;
  let hold = 0; // number of reasons to pause (hover, touch, drag)
  let touchHold = 0; // ms of grace after a touch
  let dragging = false;
  const offs = [];

  // ------------------------------------------------------------------ slide content
  function slidePnl() {
    const s = store.get('status');
    if (!s) return '<div class="empty">waiting for /api/status…</div>';
    const pnl = isNum(s.equity) && isNum(s.start_equity) ? s.equity - s.start_equity : (s.realized_pnl || 0) + (s.unrealized_pnl || 0);
    const pct = s.start_equity > 0 ? (pnl / s.start_equity) * 100 : 0;
    const sf = safetyOf(s);
    const h = store.get('health');
    const age = h?.engine_heartbeat_age_s;
    const hb = !h ? ['…', 'dim'] : h.unreachable || !h.ok ? ['API DOWN', 'neg'] : !h.engine_alive ? ['ENGINE DEAD', 'neg'] : isNum(age) && age > 5 ? [`ENGINE ${Math.round(age)}s`, 'warn'] : ['ENGINE OK', 'cy'];
    const wr = winRatePct(s.win_rate);
    return `<div class="h-pnl">
        <div class="h-big ${signClass(pnl)}">${fmtUsd(pnl, 2, true)}</div>
        <div class="h-sub"><span class="${signClass(pnl)}">${fmtPct(pct)}</span><span class="dim"> · equity </span>${fmtUsd(s.equity)}</div>
      </div>
      <div class="h-health">
        <span class="chipx ${SAFETY_CLASS[sf.state] || 'warn'}">${esc(sf.state)}</span>
        <span class="chipx ${hb[1]}">${esc(hb[0])}</span>
        <span class="chipx vio">DD ${fmtNum(s.drawdown_pct || 0, 2)}%</span>
        <span class="chipx amb">SIZE x${fmtNum(sf.derate, 2)}</span>
        <span class="chipx sky">${s.max_positions == null ? esc(s.open_positions) : `${esc(s.open_positions)}/${esc(s.max_positions)}`} OPEN</span>
        <span class="chipx pnk">WIN ${wr == null ? '–' : wr.toFixed(0) + '%'}</span>
      </div>`;
  }

  function slideTrades() {
    const ps = store.get('positions');
    if (!ps.length) return '<div class="empty">flat. waiting for a signal above threshold</div>';
    const now = store.serverNow();
    return `<div class="h-trades">${ps
      .slice(0, 6)
      .map(
        (p) => `<div class="h-trade">
          ${symLabel(p.symbol)} ${sidePill(p.side)}
          <span class="h-tbar">${slTpBar(p)}</span>
          <span class="dim h-age">${fmtDuration(now - p.opened_at)}</span>
          <b class="${signClass(p.upnl)}">${fmtUsd(p.upnl, 2, true)}</b></div>`
      )
      .join('')}</div>`;
  }

  function slideMini() {
    const sym = store.get('symbol');
    const cs = store.get('candles')[sym] || [];
    const eq = store.get('equity').slice(-150).map((p) => p.equity);
    const u = store.get('universe').find((x) => x.symbol === sym);
    const tk = store.get('ticks')[sym];
    const price = tk ? tk.price : u?.last_price;
    const last = cs.slice(-72).map((c) => c.c);
    const up = last.length > 1 ? last[last.length - 1] >= last[0] : true;
    return `<div class="h-mini">
        <div class="h-mini-box"><div class="h-mini-l">${symLabel(sym)} <span class="${signClass(u?.change_24h_pct)}">${fmtPct(u?.change_24h_pct)}</span> <b>${fmtPrice(price)}</b></div>
          ${sparkline(last, { color: up ? '#34f0a0' : '#ff4d7d' })}</div>
        <div class="h-mini-box"><div class="h-mini-l"><span class="dim">equity</span> <b>${fmtUsd(store.get('status')?.equity)}</b></div>
          ${sparkline(eq, { color: '#a78bfa' })}</div>
      </div>`;
  }

  function slideBrain() {
    const uni = store.get('universe');
    const brain = store.get('brain');
    const thr = store.get('status')?.model?.threshold_long;
    const paused = pausedSet();
    const held = new Set(store.get('positions').map((p) => p.symbol));
    if (!uni.length) return '<div class="empty">no universe yet</div>';
    const rows = uni
      .map((u) => ({ sym: u.symbol, p: brain[u.symbol]?.prob_long }))
      .sort((a, b) => (b.p ?? -1) - (a.p ?? -1))
      .slice(0, 8);
    return `<div class="h-brain">${rows
      .map(
        (r) => `<div class="h-brow ${paused.has(r.sym) ? 'paused' : ''}">
          <span class="h-bsym" style="--cc:${coinColor(r.sym)}"><i class="coin-dot"></i>${esc(shortSym(r.sym))}${held.has(r.sym) ? ' <em class="held">●</em>' : ''}</span>
          <span class="h-bbar"><i style="width:${isNum(r.p) ? (r.p * 100).toFixed(1) : 0}%;background:${PROB_GRADIENT};background-size:${isNum(r.p) && r.p > 0 ? (100 / r.p).toFixed(0) : 100}% 100%"></i>${isNum(thr) ? `<b style="left:${(thr * 100).toFixed(1)}%"></b>` : ''}</span>
          ${probPill(r.p)}</div>`
      )
      .join('')}</div>`;
  }

  const RENDER = [slidePnl, slideTrades, slideMini, slideBrain];
  function renderSlides() {
    RENDER.forEach((f, i) => {
      slides[i].innerHTML = f();
    });
  }

  function renderChrome() {
    $('h-name').textContent = `· ${NAMES[idx]}`;
    $('h-dots').innerHTML = NAMES.map((n, i) => `<button type="button" class="hdot ${i === idx ? 'on' : ''}" data-go="${i}" aria-label="${esc(n)}"></button>`).join('');
    track.style.transform = `translateX(${-idx * 100}%)`;
  }

  function go(i, { resetTimer = true } = {}) {
    idx = ((i % NAMES.length) + NAMES.length) % NAMES.length;
    if (resetTimer) elapsed = 0;
    renderChrome();
  }
  const jump = (i) => {
    go(i);
    elapsed = 0;
  };

  // ------------------------------------------------------------------ rotation + progress
  const TICK = 100;
  const timer = setInterval(() => {
    if (document.hidden) return;
    if (touchHold > 0) touchHold -= TICK;
    if (hold > 0 || touchHold > 0 || dragging) return;
    elapsed += TICK;
    if (elapsed >= ROTATE_MS) go(idx + 1);
    $('h-prog').style.width = `${Math.min(100, (elapsed / ROTATE_MS) * 100).toFixed(1)}%`;
  }, TICK);

  // pause on hover / touch
  el.addEventListener('mouseenter', () => {
    hold++;
    el.classList.add('paused');
  });
  el.addEventListener('mouseleave', () => {
    hold = Math.max(0, hold - 1);
    if (!hold) el.classList.remove('paused');
  });
  el.addEventListener('touchstart', () => {
    touchHold = 20000;
    el.classList.add('paused');
  }, { passive: true });
  el.addEventListener('touchend', () => {
    touchHold = 8000;
  }, { passive: true });
  el.addEventListener('click', (e) => {
    const d = e.target.closest('[data-go]');
    if (d) jump(Number(d.dataset.go));
  });

  // swipe / drag
  let sx = 0;
  let dx = 0;
  let pid = null;
  view.addEventListener('pointerdown', (e) => {
    if (e.target.closest('button')) return;
    pid = e.pointerId;
    sx = e.clientX;
    dx = 0;
    dragging = true;
  });
  view.addEventListener('pointermove', (e) => {
    if (!dragging || e.pointerId !== pid) return;
    dx = e.clientX - sx;
    if (Math.abs(dx) > 6) {
      try {
        view.setPointerCapture(pid);
      } catch (_) {
        /* ignore */
      }
      track.style.transition = 'none';
      track.style.transform = `translateX(calc(${-idx * 100}% + ${dx}px))`;
    }
  });
  const endDrag = (e) => {
    if (!dragging || (e && e.pointerId !== pid)) return;
    dragging = false;
    track.style.transition = '';
    if (Math.abs(dx) > 48) go(idx + (dx < 0 ? 1 : -1));
    else renderChrome();
    dx = 0;
  };
  view.addEventListener('pointerup', endDrag);
  view.addEventListener('pointercancel', endDrag);
  // programmatic flip for [ and ] shortcuts
  el.flip = (d) => jump(idx + d);

  // ------------------------------------------------------------------ events -> card
  let lastSafe = 'NORMAL';
  offs.push(store.on('trade_closed', () => jump(1)));
  offs.push(store.on('position_opened', () => jump(3)));
  offs.push(store.on('change:status', (s) => {
    const st = safetyOf(s).state;
    if (st !== lastSafe) {
      if (st !== 'NORMAL') jump(0);
      lastSafe = st;
    }
  }));
  offs.push(store.on('kill', () => jump(0)));
  offs.push(store.on('hero_flip', (d) => jump(idx + d)));

  const rs = throttle(renderSlides, 600);
  for (const t of ['change:status', 'change:positions', 'position_updated', 'brain', 'change:universe', 'change:health', 'change:symbol', 'candles_loaded', 'change:pausedLocal']) offs.push(store.on(t, rs));
  offs.push(store.on('equity_point', rs));
  renderSlides();
  renderChrome();

  return {
    el,
    flip: (d) => jump(idx + d),
    destroy() {
      clearInterval(timer);
      rs.cancel();
      offs.forEach((o) => o());
    },
  };
}
