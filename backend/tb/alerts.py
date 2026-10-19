"""Alerts: pluggable notifiers behind a non-blocking, rate-limited dispatcher.

* ``Notifier`` is the interface (a native Android push channel can implement it later).
* ``GoogleChatNotifier`` POSTs ``{"text": ...}`` (plus ``cardsV2`` cards for trade, kill, safety,
  error and daily events) to the incoming-webhook URL in env ``GOOGLE_CHAT_WEBHOOK_URL``.
  Cards carry no images: a webhook cannot upload files and the bot's charts are not public. The URL carries key/token query params: it is never logged and is
  masked in repr and in every error string.
* ``AlertDispatcher`` owns a background thread + queue, so the engine never blocks. It coalesces
  bursts into one message (webhooks allow ~1 msg/s per space), retries 429/5xx with backoff, and
  de-duplicates repeated error alerts.
"""
from __future__ import annotations

import html
import json
import logging
import os
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol, Sequence
from urllib.parse import urlsplit

log = logging.getLogger("tb.alerts")

MAX_TEXT = 3800  # Google Chat text limit is ~4096 chars
MAX_CARDS = 8    # cards coalesced into one webhook message

GREEN, RED, AMBER, BLUE, GREY = "#1e8e3e", "#d93025", "#f29900", "#1a73e8", "#5f6368"


@dataclass
class Alert:
    """One queued alert: ``text`` for text-only channels; ``headline`` + ``card`` for card channels."""
    text: str
    headline: str = ""
    card: dict | None = None


def _esc(v: Any) -> str:
    return html.escape(str(v), quote=False)


def _colored(v: Any, color: str | None) -> str:
    return f'<font color="{color}">{_esc(v)}</font>' if color else _esc(v)


def build_card(title: str, subtitle: str, rows: Sequence[tuple[str, Any, str | None]],
               card_id: str = "tb") -> dict:
    """A cardsV2 entry: header + one section of label/value rows (value optionally coloured)."""
    widgets: list[dict] = [{"decoratedText": {"topLabel": _esc(label), "text": _colored(value, color)}}
                           for label, value, color in rows]
    dash = os.environ.get("TB_DASHBOARD_URL", "").strip()
    if dash.startswith(("http://", "https://")):
        widgets.append({"buttonList": {"buttons": [{"text": "Open dashboard",
                                                    "onClick": {"openLink": {"url": dash}}}]}})
    return {"cardId": card_id, "card": {"header": {"title": _esc(title), "subtitle": _esc(subtitle)},
                                        "sections": [{"widgets": widgets}]}}


def stats_section(rows: Sequence[tuple[str, Any, str | None]], header: str = "🤖 Bot") -> dict:
    """A cardsV2 section of label/value rows (used for the bot-stats footer)."""
    return {"header": _esc(header),
            "widgets": [{"decoratedText": {"topLabel": _esc(label), "text": _colored(value, color)}}
                        for label, value, color in rows]}


def _pct(a: float, b: float) -> str:
    return f"{(b / a - 1) * 100:+.2f}%" if a else ""


class NotifyError(Exception):
    """Delivery failed. ``retryable`` failures are retried with backoff."""

    def __init__(self, msg: str, *, retryable: bool = False, retry_after: float | None = None):
        super().__init__(msg)
        self.retryable = retryable
        self.retry_after = retry_after


class Notifier(Protocol):
    name: str

    def send(self, text: str) -> None:
        """Deliver one message synchronously; raise NotifyError on failure.

        Notifiers with ``supports_cards = True`` are called as ``send(text, cards=[...])``."""


def mask_url(url: str) -> str:
    """Keep scheme://host/<first path segments> only; hide the query string and the tail."""
    if not url:
        return "<unset>"
    try:
        u = urlsplit(url)
        segs = [s for s in u.path.split("/") if s]
        keep = "/".join(segs[:2])
        return f"{u.scheme}://{u.netloc}/{keep}/***"
    except Exception:
        return "<masked-url>"


class GoogleChatNotifier:
    name = "google_chat"
    supports_cards = True

    def __init__(self, url: str, session: Any = None, timeout: float = 10.0):
        self._url = url
        self._timeout = timeout
        self._session = session

    def __repr__(self) -> str:
        return f"GoogleChatNotifier(url={mask_url(self._url)})"

    __str__ = __repr__

    def _scrub(self, text: str) -> str:
        t = str(text)
        if self._url:
            t = t.replace(self._url, mask_url(self._url))
            q = urlsplit(self._url).query
            if q:
                t = t.replace(q, "***")
        return t

    def send(self, text: str, cards: list[dict] | None = None) -> None:
        import requests

        sess = self._session or requests
        payload: dict[str, Any] = {"text": text[:MAX_TEXT]}
        if cards:
            payload["cardsV2"] = cards[:MAX_CARDS]
        body = json.dumps(payload).encode("utf-8")
        try:
            r = sess.post(self._url, data=body, timeout=self._timeout,
                          headers={"Content-Type": "application/json; charset=UTF-8"})
        except Exception as e:  # requests embeds the URL in its messages
            raise NotifyError(f"{type(e).__name__}: {self._scrub(e)}", retryable=True) from None
        code = getattr(r, "status_code", 0)
        if 200 <= code < 300:
            return
        retry_after = None
        try:
            ra = r.headers.get("Retry-After")
            retry_after = float(ra) if ra else None
        except Exception:
            retry_after = None
        retryable = code == 429 or code >= 500
        raise NotifyError(f"google chat webhook HTTP {code}", retryable=retryable, retry_after=retry_after)


def status_rows(st: Mapping[str, Any] | None, label: str = "") -> list[tuple[str, Any, str | None]]:
    """Bot stats footer rows from the engine's stored ``status`` (kv) or ``/api/status``."""
    if not st:
        return []
    eq, start, peak = st.get("equity"), st.get("start_equity"), st.get("peak_equity")
    if eq is None:
        return []
    p = f"{label} · " if label else ""
    rows: list[tuple[str, Any, str | None]] = [(f"{p}Equity", f"{eq:.2f}" + (f" (start {start:.2f})" if start else ""), None)]
    if start:
        pnl = eq - start
        rows.append((f"{p}PnL", f"{pnl:+.2f} ({pnl / start * 100:+.1f} %)", GREEN if pnl >= 0 else RED))
    dd = st.get("drawdown_pct")
    if dd is None and peak:
        dd = max(0.0, (peak - eq) / peak * 100)
    if dd is not None:
        lim = (st.get("kill") or {}).get("dd_limit_pct")
        rows.append((f"{p}Drawdown", f"{dd:.1f} %" + (f" of {lim:g} % kill" if lim else ""),
                     RED if lim and dd >= 0.8 * lim else (AMBER if dd >= 10 else None)))
    op = st.get("open_positions")
    if op is not None:
        risk = st.get("open_risk_pct")
        rows.append((f"{p}Open", f"{op} positions" + (f" · risk {risk:.0f} %" if risk is not None else "")
                     + (f" · uPnL {st['unrealized_pnl']:+.2f}" if st.get("unrealized_pnl") is not None else ""), None))
    if (st.get("kill") or {}).get("tripped"):
        rows.append((f"{p}Kill switch", "TRIPPED", RED))
    return rows


def store_stats(store: Any) -> Callable[[], list[tuple[str, Any, str | None]]]:
    """Stats provider for engine/guard dispatchers: reads the status the engine stores each loop."""
    def provider() -> list[tuple[str, Any, str | None]]:
        try:
            st = store.kv_get("status") or {}
        except Exception:
            return []
        return status_rows(st, (st.get("mode") or "").upper())
    return provider


def notifier_from_env(env: Mapping[str, str] | None = None) -> list[Notifier]:
    from .config import load_secrets

    s = load_secrets(env if env is not None else os.environ)
    return [GoogleChatNotifier(s.chat_webhook)] if s.chat_webhook else []


class AlertDispatcher:
    """Queue + worker thread. ``send`` never blocks and never raises."""

    def __init__(self, notifiers: Sequence[Notifier] | None = None, *, min_interval: float = 1.1,
                 max_attempts: int = 5, backoff_base: float = 1.0, backoff_max: float = 60.0,
                 dedupe_s: float = 600.0, max_queue: int = 500,
                 stats: Callable[[], list[tuple[str, Any, str | None]]] | None = None,
                 digest_s: float | None = None):
        from_env = notifiers is None
        self.notifiers = list(notifiers if notifiers is not None else notifier_from_env())
        self.stats = stats          # footer rows (equity, PnL, drawdown, open) added to every message
        # Trade open/close alerts are batched into one digest card every digest_s (user: fewer Chat
        # messages); kill / safety / errors / daily summary stay immediate. Production (notifiers from
        # .env) reads TB_ALERT_DIGEST_MIN (default 60; 0 = one message per trade); injected notifiers
        # (tests) default to immediate.
        if digest_s is None:
            digest_s = float(os.environ.get("TB_ALERT_DIGEST_MIN", "60")) * 60 if from_env else 0.0
        self.digest_s = max(0.0, digest_s)
        self._dig_lock = threading.Lock()
        self._dig_open: list[tuple] = []
        self._dig_closed: list[tuple] = []
        self._dig_since = time.time()
        self._dig_thread: threading.Thread | None = None
        self.min_interval = min_interval
        self.max_attempts = max_attempts
        self.backoff_base = backoff_base
        self.backoff_max = backoff_max
        self.dedupe_s = dedupe_s
        self._q: queue.Queue[Alert] = queue.Queue(maxsize=max_queue)
        self._seen: dict[str, float] = {}
        self._stop = threading.Event()
        self._idle = threading.Event()
        self._idle.set()
        self._thread: threading.Thread | None = None
        self.sent = 0          # delivered HTTP messages (after coalescing)
        self.dropped = 0
        self.failed = 0

    @property
    def enabled(self) -> bool:
        return bool(self.notifiers)

    # ------------------------------------------------------------------ public API
    def send(self, text: str, *, dedupe_key: str | None = None, headline: str = "",
             card: dict | None = None) -> None:
        if not self.notifiers:
            return
        if dedupe_key:
            now = time.monotonic()
            last = self._seen.get(dedupe_key)
            if last is not None and now - last < self.dedupe_s:
                return
            self._seen[dedupe_key] = now
            if len(self._seen) > 200:
                self._seen = {k: v for k, v in self._seen.items() if now - v < self.dedupe_s}
        self._ensure_thread()
        try:
            self._idle.clear()
            self._q.put_nowait(Alert(text, headline or text, card))
        except queue.Full:
            self.dropped += 1

    # ------------------------------------------------------------------ digest (batched trade alerts)
    def _digest_add(self, kind: str, item: tuple) -> bool:
        """Buffer a trade event when digest mode is on; returns True if buffered."""
        if self.digest_s <= 0 or not self.notifiers:
            return False
        with self._dig_lock:
            (self._dig_open if kind == "open" else self._dig_closed).append(item)
        if not (self._dig_thread and self._dig_thread.is_alive()):
            self._dig_thread = threading.Thread(target=self._digest_loop, name="tb-alerts-digest", daemon=True)
            self._dig_thread.start()
        return True

    def _digest_loop(self) -> None:
        while not self._stop.wait(self.digest_s):
            self.flush_digest()

    def flush_digest(self) -> None:
        """Send the buffered trades as one card (no-op when nothing happened)."""
        with self._dig_lock:
            opened, closed = self._dig_open, self._dig_closed
            self._dig_open, self._dig_closed = [], []
            since, self._dig_since = self._dig_since, time.time()
        if not opened and not closed:
            return
        mode = (closed or opened)[-1][-1]
        net = sum(c[2] for c in closed)
        wins = sum(1 for c in closed if c[2] > 0)
        mins = max(1, round((time.time() - since) / 60))
        rows: list[tuple[str, Any, str | None]] = [
            ("Period", f"last {mins} min", None),
            ("Opened", f"{len(opened)}" + (": " + ", ".join(f"{'▲' if o[1] == 'long' else '▼'}{o[0]}"
                                                          for o in opened[:12]) if opened else ""), None),
            ("Closed", f"{len(closed)} · {wins} won / {len(closed) - wins} lost", None),
            ("Net PnL (closed)", f"{net:+.2f} USDT", GREEN if net >= 0 else RED)]
        for c in sorted(closed, key=lambda c: -abs(c[2]))[:8]:
            rows.append((f"{c[0]} {c[1].upper()} · {c[3]}", f"{c[2]:+.2f}", GREEN if c[2] > 0 else RED))
        head = f"🕐 Trades digest · {len(closed)} closed, net {net:+.2f}"
        text = (f"[{mode}] DIGEST {mins} min: opened {len(opened)}, closed {len(closed)} "
                f"({wins} won), net {net:+.2f}")
        self.send(text, headline=head, card=build_card(head, mode, rows, "digest"))

    def trade_opened(self, symbol: str, side: str, entry: float, qty: float, notional: float,
                     sl: float, tp: float, prob: float | None = None, mode: str = "") -> None:
        if self._digest_add("open", (symbol, side.lower(), entry, notional, mode)):
            return
        p = f" p={prob:.2f}" if prob is not None else ""
        arrow = "▲" if side.lower() == "long" else "▼"
        rows = [("Entry", f"{entry:g}", None), ("Size", f"{qty:g} (${notional:.0f})", None),
                ("Stop", f"{sl:g} ({_pct(entry, sl)})", RED)]
        if tp:
            rows.append(("Target", f"{tp:g} ({_pct(entry, tp)})", GREEN))
        if prob is not None:
            rows.append(("Model prob", f"{prob:.2f}", BLUE))
        self.send(f"[{mode}] OPEN {side.upper()} {symbol} @ {entry:g} qty {qty:g} (${notional:.0f}) "
                  f"SL {sl:g} TP {tp:g}{p}",
                  headline=f"{arrow} OPEN {side.upper()} {symbol} @ {entry:g}",
                  card=build_card(f"{arrow} OPEN {side.upper()} · {symbol}", mode, rows, "open"))

    def trade_closed(self, symbol: str, side: str, entry: float, exit_price: float, pnl: float,
                     fees: float, reason: str, mode: str = "") -> None:
        net = pnl                       # pnl is NET (after fees and funding)
        if self._digest_add("close", (symbol, side.lower(), net, reason, mode)):
            return
        label = {"tp": "✅ TP hit", "sl": "🛑 SL hit", "trail": "🪢 Trailing stop", "kill": "☠️ Kill close",
                 "liquidation": "💥 Liquidated", "manual": "✋ Closed manually"}.get(reason, f"Closed ({reason})")
        rows = [("Side", side.upper(), None),
                ("Entry → exit", f"{entry:g} → {exit_price:g} ({_pct(entry, exit_price)})", None),
                ("Net PnL", f"{net:+.2f} USDT", GREEN if net > 0 else RED), ("Fees", f"{fees:.2f}", GREY)]
        self.send(f"[{mode}] CLOSE {side.upper()} {symbol} {entry:g} -> {exit_price:g} "
                  f"net PnL {net:+.2f} (fees {fees:.2f}) reason={reason}",
                  headline=f"{label} {symbol} {net:+.2f} USDT",
                  card=build_card(f"{label} · {symbol}", mode, rows, "close"))

    def kill(self, reason: str, equity: float, dd_pct: float, mode: str = "") -> None:
        rows = [("Reason", reason, RED), ("Bot equity", f"{equity:.2f}", None), ("Drawdown", f"{dd_pct:.1f}%", RED),
                ("Next step", "positions closed; manual reset_kill required", AMBER)]
        self.send(f"[{mode}] KILL SWITCH TRIPPED: {reason}. equity {equity:.2f}, drawdown {dd_pct:.1f}%",
                  headline=f"☠️ KILL SWITCH TRIPPED ({dd_pct:.1f}% drawdown)",
                  card=build_card("☠️ KILL SWITCH TRIPPED", mode, rows, "kill"))

    def safety(self, state: str, reasons: Sequence[str], mode: str = "") -> None:
        rows = [("State", state, {"NORMAL": GREEN, "DEGRADED": AMBER}.get(state, RED))]
        rows += [("Reason", r, None) for r in list(reasons)[:5]]
        self.send(f"[{mode}] SAFETY {state}: {'; '.join(reasons)}", dedupe_key=f"safety:{state}",
                  headline=f"🚦 Safety {state}", card=build_card(f"🚦 Safety → {state}", mode, rows, "safety"))

    def engine_error(self, where: str, err: str) -> None:
        self.send(f"ENGINE ERROR in {where}: {err}"[:500], dedupe_key=f"err:{where}",
                  headline=f"⚠️ Engine error in {where}",
                  card=build_card("⚠️ Engine error", where, [("Where", where, AMBER), ("Error", err[:400], None)],
                                  "error"))

    def daily_summary(self, *, date: str, trades: int, wins: int, pnl: float, equity: float,
                      mode: str = "") -> None:
        wr = f"{100 * wins / trades:.0f}%" if trades else "n/a"
        rows = [("Trades", trades, None), ("Win rate", wr, None),
                ("PnL", f"{pnl:+.2f} USDT", GREEN if pnl >= 0 else RED), ("Bot equity", f"{equity:.2f}", None)]
        self.send(f"[{mode}] DAILY {date}: {trades} trades, win rate {wr}, PnL {pnl:+.2f}, "
                  f"equity {equity:.2f}",
                  headline=f"📊 Daily {date}: {pnl:+.2f} USDT",
                  card=build_card(f"📊 Daily summary · {date}", mode, rows, "daily"))

    def flush(self, timeout: float = 10.0) -> bool:
        """Wait until the queue has been delivered (tests / shutdown)."""
        return self._idle.wait(timeout)

    def close(self, timeout: float = 5.0) -> None:
        self.flush_digest()             # don't lose buffered trades on shutdown
        self.flush(timeout)
        self._stop.set()
        if self._thread:
            self._thread.join(timeout)

    # ------------------------------------------------------------------ worker
    def _ensure_thread(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="tb-alerts", daemon=True)
        self._thread.start()

    def _drain_batch(self, first: Alert) -> list[Alert]:
        batch = [first]
        size = len(first.text)
        while size < MAX_TEXT - 300 and sum(a.card is not None for a in batch) < MAX_CARDS:
            try:
                nxt = self._q.get_nowait()
            except queue.Empty:
                break
            batch.append(nxt)
            size += len(nxt.text) + 1
        return batch

    def _run(self) -> None:
        last_sent = 0.0
        while not self._stop.is_set():
            try:
                first = self._q.get(timeout=0.2)
            except queue.Empty:
                if self._q.empty():
                    self._idle.set()
                continue
            wait = self.min_interval - (time.monotonic() - last_sent)
            if wait > 0:
                time.sleep(wait)          # while waiting, bursts accumulate and are coalesced
            self._deliver(self._drain_batch(first))
            last_sent = time.monotonic()
            if self._q.empty():
                self._idle.set()

    def _stats_rows(self) -> list[tuple[str, Any, str | None]]:
        if not self.stats:
            return []
        try:
            return list(self.stats() or [])
        except Exception:  # a stats failure must never block an alert
            return []

    def _deliver(self, batch: list[Alert]) -> None:
        text = "\n".join(a.text for a in batch)
        headlines = "\n".join(a.headline or a.text for a in batch)
        cards = [dict(a.card, cardId=f"{a.card['cardId']}-{i}") for i, a in enumerate(batch) if a.card]
        rows = self._stats_rows()
        if rows:
            # Every message ends with the bot's current stats: on the last card, or on a card made for
            # plain-text alerts; text-only channels get one appended line.
            if not cards:
                first = (batch[0].headline or batch[0].text).splitlines()[0][:120]
                body = text if len(text) <= 3000 else text[:3000] + " …"
                cards = [{"cardId": "msg-0", "card": {"header": {"title": _esc(first)},
                          "sections": [{"widgets": [{"textParagraph": {"text": _esc(body)}}]}]}}]
            last = cards[-1]
            last["card"] = dict(last["card"], sections=list(last["card"].get("sections", [])) + [stats_section(rows)])
            text = text + "\n" + " · ".join(f"{label}: {value}" for label, value, _ in rows)
        for n in self.notifiers:
            for attempt in range(1, self.max_attempts + 1):
                try:
                    if cards and getattr(n, "supports_cards", False):
                        n.send(headlines, cards=cards)
                    else:
                        n.send(text)
                    self.sent += 1
                    break
                except NotifyError as e:
                    if not e.retryable or attempt == self.max_attempts:
                        self.failed += 1
                        log.warning("alert delivery via %s failed: %s", n.name, e)
                        break
                    delay = e.retry_after if e.retry_after else min(
                        self.backoff_max, self.backoff_base * (2 ** (attempt - 1)))
                    time.sleep(min(delay, self.backoff_max))
                except Exception as e:  # never kill the worker
                    self.failed += 1
                    log.warning("alert delivery via %s crashed: %s", n.name, type(e).__name__)
                    break
