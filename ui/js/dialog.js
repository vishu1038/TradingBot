// Confirm dialog (own modal, not window.confirm, so it matches the terminal look).
import { esc } from './format.js';

export function confirmDialog({ title = 'CONFIRM', message = '', confirmText = 'OK', cancelText = 'CANCEL', danger = false }) {
  return new Promise((resolve) => {
    const host = document.getElementById('modal-host');
    if (!host) return resolve(window.confirm(message));
    const prevFocus = document.activeElement;
    host.innerHTML = `
      <div class="modal-backdrop" role="presentation">
        <div class="modal ${danger ? 'modal-danger' : ''}" role="alertdialog" aria-modal="true" aria-labelledby="modal-title">
          <div class="modal-title" id="modal-title">${danger ? '!! ' : '&gt; '}${esc(title)}</div>
          <div class="modal-body">${esc(message)}</div>
          <div class="modal-actions">
            <button type="button" class="btn" data-act="cancel">${esc(cancelText)}</button>
            <button type="button" class="btn ${danger ? 'btn-kill' : 'btn-primary'}" data-act="ok">${esc(confirmText)}</button>
          </div>
        </div>
      </div>`;
    host.hidden = false;
    const done = (v) => {
      document.removeEventListener('keydown', onKey, true);
      host.hidden = true;
      host.innerHTML = '';
      if (prevFocus && prevFocus.focus) prevFocus.focus();
      resolve(v);
    };
    const onKey = (e) => {
      if (e.key === 'Escape') {
        e.stopPropagation();
        done(false);
      }
    };
    document.addEventListener('keydown', onKey, true);
    host.querySelector('[data-act="cancel"]').addEventListener('click', () => done(false));
    host.querySelector('[data-act="ok"]').addEventListener('click', () => done(true));
    host.querySelector('.modal-backdrop').addEventListener('mousedown', (e) => {
      if (e.target.classList.contains('modal-backdrop')) done(false);
    });
    host.querySelector(danger ? '[data-act="cancel"]' : '[data-act="ok"]').focus();
  });
}
