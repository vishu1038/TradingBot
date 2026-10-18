// SETTINGS view: form bound to GET/PUT /api/settings (every field in the contract) with validation.
import { store } from '../state.js';
import { api } from '../api.js';
import * as sound from '../sound.js';
import * as fx from '../fx.js';
import { esc, isNum, fmtUsd } from '../format.js';
import { derateAt } from '../components.js';
import { ctl } from '../actions.js';
import { confirmDialog } from '../dialog.js';

// type: num | int | bool | select. `when` marks which exit_mode a field applies to (dimmed otherwise).
const SECTIONS = [
  {
    title: 'POSITION SIZING',
    fields: [
      { k: 'slice_margin_usd', label: 'slice margin (USD)', type: 'num', min: 1, max: 10000, step: 1, hint: 'isolated margin per position; notional = margin x leverage' },
      { k: 'leverage', label: 'leverage', type: 'int', min: 1, max: 50, step: 1 },
      { k: 'max_positions', label: 'max positions', type: 'int', min: 1, max: 20, step: 1, hint: 'capped by floor(equity / slice margin)' },
    ],
  },
  {
    title: 'EXITS',
    fields: [
      { k: 'exit_mode', label: 'exit mode', type: 'select', options: ['fixed', 'atr'], hint: 'fixed = USD targets, atr = ATR multiples' },
      { k: 'tp_usd', label: 'take profit (USD)', type: 'num', min: 0.01, max: 10000, step: 0.5, when: 'fixed' },
      { k: 'sl_usd', label: 'stop loss (USD)', type: 'num', min: 0.01, max: 10000, step: 0.5, hint: 'fixed: SL loss; atr: risk used to size the position' },
      { k: 'tp_atr', label: 'take profit (x ATR14)', type: 'num', min: 0.1, max: 20, step: 0.1, when: 'atr' },
      { k: 'sl_atr', label: 'stop loss (x ATR14)', type: 'num', min: 0.1, max: 20, step: 0.1, when: 'atr' },
      { k: 'trail.enabled', label: 'trailing stop', type: 'bool' },
      { k: 'trail.activate_r', label: 'trail activates at (R)', type: 'num', min: 0.1, max: 10, step: 0.1, hint: 'favourable move in multiples of initial risk' },
      { k: 'trail.distance_atr', label: 'trail distance (x ATR)', type: 'num', min: 0.1, max: 10, step: 0.1 },
    ],
  },
  {
    title: 'ENTRIES',
    fields: [
      { k: 'short_enabled', label: 'allow shorts', type: 'bool' },
      { k: 'prob_threshold_long', label: 'long threshold', type: 'num', min: 0.5, max: 0.99, step: 0.01, hint: 'trade if prob_long >= this' },
      { k: 'prob_threshold_short', label: 'short threshold', type: 'num', min: 0.5, max: 0.99, step: 0.01, hint: 'stricter than long; needs shorts enabled' },
      { k: 'corr_cap.max_corr', label: 'correlation cap', type: 'num', min: 0, max: 1, step: 0.05, hint: '24h 5m-return correlation limit' },
      { k: 'corr_cap.max_correlated', label: 'max correlated positions', type: 'int', min: 1, max: 20, step: 1 },
    ],
  },
  {
    title: 'UNIVERSE',
    fields: [
      { k: 'universe_size', label: 'universe size', type: 'int', min: 1, max: 30, step: 1 },
      { k: 'universe_refresh_s', label: 'refresh interval (s)', type: 'int', min: 60, max: 86400, step: 60 },
    ],
  },
  {
    title: 'RISK & FEES',
    fields: [
      { k: 'kill_dd_pct', label: 'kill drawdown (%)', type: 'num', min: 1, max: 100, step: 1, hint: 'also the reference for size derating (starts at 40% of this)' },
      { k: 'fee_taker', label: 'taker fee (fraction)', type: 'num', min: 0, max: 0.01, step: 0.0001, hint: '0.0005 = 0.05% per side' },
    ],
  },
];

const getPath = (o, path) => path.split('.').reduce((a, k) => (a == null ? a : a[k]), o);
function setPath(o, path, v) {
  const ks = path.split('.');
  let cur = o;
  for (const k of ks.slice(0, -1)) cur = cur[k] ??= {};
  cur[ks[ks.length - 1]] = v;
}
const clone = (o) => JSON.parse(JSON.stringify(o));

export function validateField(f, v) {
  if (f.type === 'bool') return typeof v === 'boolean' ? null : 'must be on/off';
  if (f.type === 'select') return f.options.includes(v) ? null : `must be one of ${f.options.join(', ')}`;
  if (!isNum(v)) return 'must be a number';
  if (f.type === 'int' && !Number.isInteger(v)) return 'must be a whole number';
  if (v < f.min || v > f.max) return `must be between ${f.min} and ${f.max}`;
  return null;
}

export function mount(root) {
  const ALL = SECTIONS.flatMap((s) => s.fields);
  // the sound toggle lives in this browser (localStorage); show its effective value as the baseline
  const localize = (st) => (st ? { ...clone(st), sound: sound.isEnabled() } : st);
  let server = localize(store.get('settings'));
  let draft = server ? clone(server) : null;
  const errors = {};
  let banner = '';

  root.innerHTML = `<div class="page settings">
    <form class="card settings-form ac-coral span2" id="s-form" novalidate></form>
    <div class="card ac-magenta" id="fx-card"></div>
    <div class="card ac-sky" id="alerts-card"></div>
    <div class="card ac-red span2" id="live-card"></div>
  </div>`;
  const form = root.querySelector('#s-form');
  const extra = mountExtras(root);

  const changedPaths = () => (draft && server ? ALL.map((f) => f.k).filter((k) => JSON.stringify(getPath(draft, k)) !== JSON.stringify(getPath(server, k))) : []);

  function inputHtml(f) {
    const v = getPath(draft, f.k);
    const id = `f-${f.k.replace(/\./g, '-')}`;
    if (f.type === 'bool') return `<label class="switch"><input type="checkbox" id="${id}" data-k="${f.k}" ${v ? 'checked' : ''}><span class="slider"></span><span class="sw-t">${v ? 'ON' : 'OFF'}</span></label>`;
    if (f.type === 'select') return `<select id="${id}" data-k="${f.k}">${f.options.map((o) => `<option value="${esc(o)}" ${o === v ? 'selected' : ''}>${esc(o)}</option>`).join('')}</select>`;
    return `<input type="number" id="${id}" data-k="${f.k}" value="${esc(v)}" min="${f.min}" max="${f.max}" step="${f.step}" inputmode="decimal">`;
  }

  function render() {
    if (!draft) {
      form.innerHTML = '<h3>SETTINGS</h3><div class="empty">waiting for /api/settings…</div>';
      return;
    }
    const dirty = changedPaths();
    const bad = Object.values(errors).some(Boolean);
    const mode = draft.exit_mode;
    form.innerHTML = `
      <h3>BOT SETTINGS <span class="why-sub">saved to the engine via PUT /api/settings (partial merge)</span></h3>
      ${banner ? `<div class="banner neg">${esc(banner)}</div>` : ''}
      <div class="settings-grid">
        ${SECTIONS.map(
          (s) => `<fieldset><legend>${esc(s.title)}</legend>${s.fields
            .map((f) => {
              const inactive = f.when && f.when !== mode;
              return `<div class="field ${inactive ? 'inactive' : ''} ${errors[f.k] ? 'invalid' : ''} ${dirty.includes(f.k) ? 'dirty' : ''}">
                <label for="f-${f.k.replace(/\./g, '-')}">${esc(f.label)}${inactive ? ` <small class="dim">(only ${esc(f.when)} mode)</small>` : ''}</label>
                ${inputHtml(f)}
                ${errors[f.k] ? `<div class="err">${esc(errors[f.k])}</div>` : f.hint ? `<div class="hint">${esc(f.hint)}</div>` : ''}
              </div>`;
            })
            .join('')}</fieldset>`
        ).join('')}
      </div>
      <div class="settings-actions">
        <button type="submit" class="btn btn-primary" id="s-save" ${dirty.length && !bad ? '' : 'disabled'}>SAVE ${dirty.length ? `(${dirty.length})` : ''}</button>
        <button type="button" class="btn" id="s-reset" ${dirty.length ? '' : 'disabled'}>REVERT</button>
        <span class="dim">${bad ? 'fix the highlighted fields' : dirty.length ? 'unsaved changes' : 'in sync with the engine'}</span>
      </div>`;
  }

  // update in place on input so focus is never lost; full render on structural changes
  form.addEventListener('input', (e) => {
    const k = e.target.dataset?.k;
    if (!k || !draft) return;
    const f = ALL.find((x) => x.k === k);
    let v;
    if (f.type === 'bool') v = e.target.checked;
    else if (f.type === 'select') v = e.target.value;
    else v = e.target.value === '' ? NaN : Number(e.target.value);
    setPath(draft, k, v);
    errors[k] = validateField(f, v);
    if (f.type === 'bool' && k === 'sound') sound.setEnabled(v);
    if (f.type === 'bool' || f.type === 'select') return render();
    // number: patch only the surrounding field + buttons
    const field = e.target.closest('.field');
    field.classList.toggle('invalid', !!errors[k]);
    field.classList.toggle('dirty', JSON.stringify(getPath(draft, k)) !== JSON.stringify(getPath(server, k)));
    let msg = field.querySelector('.err, .hint');
    if (errors[k]) {
      if (!msg || !msg.classList.contains('err')) {
        msg?.remove();
        field.insertAdjacentHTML('beforeend', `<div class="err">${esc(errors[k])}</div>`);
      } else msg.textContent = errors[k];
    } else if (msg && msg.classList.contains('err')) {
      msg.remove();
      if (f.hint) field.insertAdjacentHTML('beforeend', `<div class="hint">${esc(f.hint)}</div>`);
    }
    const dirty = changedPaths();
    const bad = Object.values(errors).some(Boolean);
    const save = form.querySelector('#s-save');
    save.disabled = !dirty.length || bad;
    save.textContent = `SAVE ${dirty.length ? `(${dirty.length})` : ''}`;
    form.querySelector('#s-reset').disabled = !dirty.length;
  });

  form.addEventListener('click', (e) => {
    if (e.target.id === 's-reset') {
      draft = clone(server);
      for (const k of Object.keys(errors)) delete errors[k];
      banner = '';
      render();
    }
  });

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    if (!draft) return;
    for (const f of ALL) errors[f.k] = validateField(f, getPath(draft, f.k));
    if (Object.values(errors).some(Boolean)) return render();
    const patch = {};
    for (const k of changedPaths()) setPath(patch, k, getPath(draft, k));
    if (!Object.keys(patch).length) return;
    const btn = form.querySelector('#s-save');
    btn.disabled = true;
    btn.textContent = 'SAVING…';
    try {
      const saved = await api.saveSettings(patch);
      const fresh = saved && typeof saved === 'object' && 'leverage' in saved ? saved : await api.settings();
      store.set('settings', fresh);
      server = localize(fresh);
      draft = clone(server);
      banner = '';
      fx.toast('settings saved', 'win');
    } catch (err) {
      banner = `save failed: ${err.message}`;
      fx.toast(banner, 'loss', 5000);
    }
    render();
  });

  const off = store.on('change:settings', (s) => {
    // another tab/engine changed settings and we have no unsaved edits -> follow
    if (!s) return;
    if (!draft || !changedPaths().length) {
      server = localize(s);
      draft = clone(server);
      render();
    } else server = localize(s);
  });
  render();
  return () => {
    off();
    extra();
  };
}

// --------------------------------------------------------------------------------------------
// Effects slider + alerts info + the separate LIVE-mode card
function mountExtras(root) {
  const fxCard = root.querySelector('#fx-card');
  const alertsCard = root.querySelector('#alerts-card');
  const liveCard = root.querySelector('#live-card');
  const offs = [];

  const MARKS = ['calm', 'loud but readable', 'full degen'];
  function renderFx() {
    const lv = fx.getLevel();
    fxCard.innerHTML = `
      <h3>EFFECTS <span class="why-sub">calm ↔ degen</span></h3>
      <div class="fx-now">${esc(fx.modeLabel())}${fx.reducedMotion() ? ' <span class="dim">(forced calm by your system reduced-motion setting)</span>' : ''}</div>
      <input type="range" id="fx-range" min="0" max="100" step="1" value="${lv}" aria-label="Effects level" class="fx-range">
      <div class="fx-marks">${MARKS.map((m) => `<span>${m}</span>`).join('')}</div>
      <ul class="fx-list">
        <li><b>calm</b> nothing animates, no flashes</li>
        <li><b>loud but readable</b> (default) screen flash + glitch on TP, SL and kill only</li>
        <li><b>degen</b> + idle logo glitch, scanlines, ASCII spinners, typed log lines, tick flashes</li>
      </ul>
      <div class="field fx-sound"><label for="fx-sound">sound on fills</label><label class="switch"><input type="checkbox" id="fx-sound" ${sound.isEnabled() ? 'checked' : ''}><span class="slider"></span><span class="sw-t">${sound.isEnabled() ? 'ON' : 'OFF'}</span></label></div>`;
    const r = fxCard.querySelector('#fx-range');
    r.addEventListener('input', () => {
      fx.setLevel(Number(r.value));
      fxCard.querySelector('.fx-now').firstChild.textContent = fx.modeLabel();
    });
    r.addEventListener('change', () => {
      if (fx.mode() !== 'calm') fx.flash('open');
    });
    fxCard.querySelector('#fx-sound').addEventListener('change', (e) => {
      sound.setEnabled(e.target.checked);
      fxCard.querySelector('.sw-t').textContent = e.target.checked ? 'ON' : 'OFF';
      api.saveSettings({ sound: e.target.checked }).catch(() => {}); // best effort: mirror to the server setting
    });
  }

  alertsCard.innerHTML = `
    <h3>ALERTS <span class="why-sub">read-only</span></h3>
    <div class="kv-grid">
      <div class="kv"><span class="kv-l">channel</span><span class="kv-v">Google Chat incoming webhook</span></div>
      <div class="kv"><span class="kv-l">configured via</span><span class="kv-v">env GOOGLE_CHAT_WEBHOOK_URL (.env)</span></div>
      <div class="kv"><span class="kv-l">rate limit</span><span class="kv-v">~1 message / s, background queue</span></div>
    </div>
    <div class="hint">The webhook URL contains secrets, so it is never sent to the browser and cannot be edited here.</div>`;

  // ---- LIVE mode
  const typed = { a: '', b: '', c: false };
  function liveState() {
    const s = store.get('status') || {};
    const start = Math.max(1, Math.round(s.start_equity || 100));
    return { s, start, armed: s.mode === 'live' || !!s.live_armed, wantA: `LIVE $${start}`, wantB: 'KILL 25%' };
  }
  function renderLive() {
    const { s, start, armed, wantA, wantB } = liveState();
    const kill = 25;
    const rows = [10, 20, 25].map((dd) => `<tr><td>${dd}% drawdown</td><td class="r"><b>x${derateAt(dd, kill).toFixed(1)}</b> size</td></tr>`).join('');
    const okA = typed.a === wantA;
    const okB = typed.b.toUpperCase() === wantB;
    const can = okA && okB && typed.c && !armed;
    liveCard.innerHTML = `
      <h3>LIVE MODE <span class="why-sub">real money. separate from the settings above</span></h3>
      <div class="live-grid">
        <div>
          <div class="live-state ${armed ? 'armed' : ''}">${armed ? '<span class="deco">●</span> LIVE ARMED' : 'DISARMED'} <span class="dim">· current mode ${esc(String(s.mode || '?').toUpperCase())}</span></div>
          <div class="live-sum">
            <div class="kv"><span class="kv-l">kill switch</span><span class="kv-v neg">−${kill}% from peak</span></div>
            <div class="kv"><span class="kv-l">derating</span><span class="kv-v">full until 10%, x0.5 at 20%, 0 at 25%</span></div>
            <div class="kv"><span class="kv-l">first live trades</span><span class="kv-v">20 trades at size x0.5</span></div>
            <div class="kv"><span class="kv-l">rollout</span><span class="kv-v">paper_local → binance_demo → live</span></div>
            <div class="kv"><span class="kv-l">backend gate</span><span class="kv-v">env ALLOW_LIVE_TRADING=true + config flag</span></div>
          </div>
          <table class="tbl compact live-derate"><tbody>${rows}</tbody></table>
        </div>
        <div class="live-confirm">
          <label class="lc-l" for="lc-a">1 · type <code>${esc(wantA)}</code> (your account size)</label>
          <input id="lc-a" type="text" class="lc-in ${okA ? 'ok' : ''}" autocomplete="off" spellcheck="false" value="${esc(typed.a)}" ${armed ? 'disabled' : ''}>
          <label class="lc-l" for="lc-b">2 · type <code>${esc(wantB)}</code> (the live kill limit)</label>
          <input id="lc-b" type="text" class="lc-in ${okB ? 'ok' : ''}" autocomplete="off" spellcheck="false" value="${esc(typed.b)}" ${armed ? 'disabled' : ''}>
          <label class="lc-chk"><input id="lc-c" type="checkbox" ${typed.c ? 'checked' : ''} ${armed ? 'disabled' : ''}> I understand real money can be lost and the bot trades without asking.</label>
          <div class="lc-actions">
            <button type="button" class="pillbtn bad" id="lc-arm" ${can ? '' : 'disabled'}>ARM LIVE</button>
            <button type="button" class="pillbtn" id="lc-disarm" ${armed ? '' : 'disabled'}>DISARM</button>
          </div>
          <div class="hint">Arming sends control action <code>arm_live</code>; the backend still refuses unless its own live gates are open.</div>
        </div>
      </div>`;
    const q = (id) => liveCard.querySelector(id);
    const refresh = () => {
      const ra = typed.a === wantA;
      const rb = typed.b.toUpperCase() === wantB;
      q('#lc-a').classList.toggle('ok', ra);
      q('#lc-b').classList.toggle('ok', rb);
      q('#lc-arm').disabled = !(ra && rb && typed.c && !armed);
    };
    q('#lc-a').addEventListener('input', (e) => { typed.a = e.target.value; refresh(); });
    q('#lc-b').addEventListener('input', (e) => { typed.b = e.target.value; refresh(); });
    q('#lc-c').addEventListener('change', (e) => { typed.c = e.target.checked; refresh(); });
    q('#lc-arm').addEventListener('click', async () => {
      const ok = await confirmDialog({ title: 'ARM LIVE TRADING', message: `Real money, ${fmtUsd(start, 0)} account. Kill switch at -25%. Arm now?`, confirmText: 'ARM LIVE', danger: true });
      if (ok && (await ctl('arm_live', {}, 'arm refused'))) {
        typed.a = typed.b = '';
        typed.c = false;
        fx.flash('kill');
        renderLive();
      }
    });
    q('#lc-disarm').addEventListener('click', async () => {
      const ok = await confirmDialog({ title: 'DISARM LIVE', message: 'Stop live trading and go back to the safe mode? Open positions are not closed.', confirmText: 'DISARM' });
      if (ok && (await ctl('disarm_live', {}, 'disarm failed'))) renderLive();
    });
  }
  let lastArmed = null;
  offs.push(store.on('change:status', () => {
    const a = liveState().armed;
    if (a !== lastArmed) {
      lastArmed = a;
      renderLive();
    }
  }));
  renderFx();
  renderLive();
  lastArmed = liveState().armed;
  return () => offs.forEach((o) => o());
}
