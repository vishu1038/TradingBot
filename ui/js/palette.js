// `/` command palette: /kill /history /lab /coin SOL /pause WIF ... Keyboard-first, also a button in the top bar.
import { store } from './state.js';
import { esc, shortSym } from './format.js';
import { coinColor } from './theme.js';
import * as fx from './fx.js';
import * as sound from './sound.js';
import * as A from './actions.js';

const PAGES = { live: 'live', home: 'live', main: 'live', settings: 'settings', markets: 'markets', history: 'trades', trades: 'trades', bots: 'bots', equity: 'bots', lab: 'lab' };

/** name, argument hint, description, handler(arg) */
const COMMANDS = [
  ['kill', '', 'KILL SWITCH: close everything and halt', () => A.killAll()],
  ['start', '', 'enable new entries', () => A.ctl('start')],
  ['stop', '', 'stop new entries (open trades keep running)', () => A.ctl('stop')],
  ['reset', '', 'reset the kill switch', () => A.resetKill()],
  ['live', '', 'page 1: main', () => A.goPage('live')],
  ['settings', '', 'page 2: bot settings', () => A.goPage('settings')],
  ['markets', '', 'markets: universe, heatmap, data faults', () => A.goPage('markets')],
  ['history', '', 'trades history', () => A.goPage('trades')],
  ['bots', '', 'bots / equity / shadow models', () => A.goPage('bots')],
  ['lab', '', 'lab reports', () => A.goPage('lab')],
  ['coin', 'SOL', 'show a coin on the main chart', (a) => !A.selectCoin(a) && fx.toast(`no coin matching "${a}"`, 'loss')],
  ['pause', '[COIN]', 'pause a coin (no new entries)', (a) => withCoin(a, (s) => !A.pausedSet().has(s) && A.togglePause(s))],
  ['unpause', '[COIN]', 'resume a paused coin', (a) => withCoin(a, (s) => A.pausedSet().has(s) && A.togglePause(s))],
  ['close', '[COIN]', "close a coin's open position", (a) => withCoin(a, (s) => A.closeSymbol(s))],
  ['closeall', '', 'close every open position', () => A.closeAll()],
  ['refresh', '', 'force a universe re-rank', () => A.refreshUniverse()],
  ['sound', 'on|off', 'fill sounds', (a) => sound.setEnabled(!/^(off|0|no)$/i.test(a || 'on'))],
  ['fx', 'calm|loud|degen|0-100', 'effects level', (a) => setFx(a)],
  ['help', '', 'keyboard shortcuts', () => store.emit('help', null)],
];

function withCoin(arg, fn) {
  const sym = arg ? A.selectCoin(arg) : store.get('symbol');
  if (!sym) return fx.toast(`no coin matching "${arg}"`, 'loss');
  return fn(sym);
}

function setFx(a) {
  const named = { calm: 10, loud: 50, degen: 90 };
  const v = a in named ? named[a] : parseInt(a, 10);
  if (Number.isFinite(v)) {
    fx.setLevel(v);
    fx.toast(`effects: ${fx.modeLabel()}`, 'info');
  } else fx.toast('usage: /fx calm|loud|degen|0-100', 'info');
}

export function mountPalette(host) {
  let open = false;
  let sel = 0;
  let items = [];
  host.innerHTML = `
    <div class="pal-backdrop" hidden><div class="pal card ac-coral" role="dialog" aria-label="Command palette">
      <div class="pal-in"><span class="pal-slash deco">/</span><input id="pal-input" type="text" autocomplete="off" autocapitalize="off" spellcheck="false" placeholder="kill, coin SOL, pause WIF, history, lab, fx degen …" aria-label="Command"></div>
      <div class="pal-list" id="pal-list" role="listbox"></div>
      <div class="pal-foot dim">↑↓ choose · Enter run · Esc close</div>
    </div></div>`;
  const back = host.querySelector('.pal-backdrop');
  const input = host.querySelector('#pal-input');
  const list = host.querySelector('#pal-list');

  function compute() {
    const raw = input.value.replace(/^\//, '');
    const [c = '', ...rest] = raw.split(/\s+/);
    const arg = rest.join(' ').trim();
    const cmd = c.toLowerCase();
    const exact = COMMANDS.find((x) => x[0] === cmd);
    if (exact && (rest.length || raw.endsWith(' '))) {
      const out = [{ label: `/${exact[0]} ${arg}`.trim(), hint: exact[2], run: () => exact[3](arg) }];
      if (['coin', 'pause', 'unpause', 'close'].includes(exact[0])) {
        const q = arg.toUpperCase();
        for (const s of A.barSymbols().filter((x) => !q || shortSym(x).startsWith(q) || x.includes(q)).slice(0, 8)) {
          out.push({ label: `/${exact[0]} ${shortSym(s)}`, hint: s, color: coinColor(s), run: () => exact[3](shortSym(s)) });
        }
      }
      return out;
    }
    return COMMANDS.filter((x) => !cmd || x[0].startsWith(cmd) || x[0].includes(cmd))
      .concat(Object.keys(PAGES).includes(cmd) && !COMMANDS.some((x) => x[0] === cmd) ? [[cmd, '', `go to ${PAGES[cmd]}`, () => A.goPage(PAGES[cmd])]] : [])
      .map((x) => ({ label: `/${x[0]}${x[1] ? ' ' + x[1] : ''}`, hint: x[2], run: () => x[3](''), needsArg: !!x[1] && !x[1].startsWith('[') && x[0] !== 'sound' && x[0] !== 'fx', name: x[0] }));
  }

  function render() {
    items = compute();
    sel = Math.min(sel, Math.max(0, items.length - 1));
    list.innerHTML = items.length
      ? items.map((it, i) => `<div class="pal-item ${i === sel ? 'on' : ''}" role="option" data-i="${i}" ${it.color ? `style="--cc:${it.color}"` : ''}><b>${it.color ? '<i class="coin-dot"></i>' : ''}${esc(it.label)}</b><span class="dim">${esc(it.hint)}</span></div>`).join('')
      : '<div class="empty">no such command. try /help</div>';
  }

  function show(prefill = '/') {
    open = true;
    back.hidden = false;
    input.value = prefill;
    sel = 0;
    render();
    input.focus();
    input.setSelectionRange(input.value.length, input.value.length);
  }
  function hide() {
    open = false;
    back.hidden = true;
    input.blur();
  }
  async function run(i) {
    const it = items[i];
    if (!it) return;
    if (it.needsArg) {
      input.value = `${it.label.split(' ')[0]} `;
      sel = 0;
      render();
      return;
    }
    hide();
    try {
      await it.run();
    } catch (e) {
      fx.toast(`command failed: ${e.message}`, 'loss');
    }
  }

  input.addEventListener('input', () => {
    sel = 0;
    render();
  });
  input.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
      e.preventDefault();
      hide();
    } else if (e.key === 'ArrowDown') {
      e.preventDefault();
      sel = (sel + 1) % Math.max(1, items.length);
      render();
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      sel = (sel - 1 + items.length) % Math.max(1, items.length);
      render();
    } else if (e.key === 'Tab') {
      e.preventDefault();
      const it = items[sel];
      if (it) {
        input.value = `${it.label} `;
        render();
      }
    } else if (e.key === 'Enter') {
      e.preventDefault();
      run(sel);
    }
  });
  list.addEventListener('mousemove', (e) => {
    const r = e.target.closest('[data-i]');
    if (r && Number(r.dataset.i) !== sel) {
      sel = Number(r.dataset.i);
      list.querySelectorAll('.pal-item').forEach((n, i) => n.classList.toggle('on', i === sel));
    }
  });
  list.addEventListener('click', (e) => {
    const r = e.target.closest('[data-i]');
    if (r) run(Number(r.dataset.i));
  });
  back.addEventListener('mousedown', (e) => {
    if (e.target === back) hide();
  });
  store.on('palette', (v) => show(v || '/'));
  return { show, hide, isOpen: () => open };
}
