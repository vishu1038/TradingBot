"""Alerts: pluggable notifiers behind a non-blocking, rate-limited dispatcher.

* ``Notifier`` is the interface (a native Android push channel can implement it later).
* ``GoogleChatNotifier`` POSTs ``{"text": ...}`` to the incoming-webhook URL in env
  ``GOOGLE_CHAT_WEBHOOK_URL``. The URL carries key/token query params: it is never logged and is
  masked in repr and in every error string.
* ``AlertDispatcher`` owns a background thread + queue, so the engine never blocks. It coalesces
  bursts into one message (webhooks allow ~1 msg/s per space), retries 429/5xx with backoff, and
  de-duplicates repeated error alerts.
"""
from __future__ import annotations

import json
import logging
import os
import queue
import threading
import time
from typing import Any, Mapping, Protocol, Sequence
from urllib.parse import urlsplit

log = logging.getLogger("tb.alerts")

MAX_TEXT = 3800  # Google Chat text limit is ~4096 chars


class NotifyError(Exception):
    """Delivery failed. ``retryable`` failures are retried with backoff."""

    def __init__(self, msg: str, *, retryable: bool = False, retry_after: float | None = None):
        super().__init__(msg)
        self.retryable = retryable
        self.retry_after = retry_after


class Notifier(Protocol):
    name: str

    def send(self, text: str) -> None:
        """Deliver one message synchronously; raise NotifyError on failure."""


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

    def send(self, text: str) -> None:
        import requests

        sess = self._session or requests
        body = json.dumps({"text": text[:MAX_TEXT]}).encode("utf-8")
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


def notifier_from_env(env: Mapping[str, str] | None = None) -> list[Notifier]:
    from .config import load_secrets

    s = load_secrets(env if env is not None else os.environ)
    return [GoogleChatNotifier(s.chat_webhook)] if s.chat_webhook else []


class AlertDispatcher:
    """Queue + worker thread. ``send`` never blocks and never raises."""

    def __init__(self, notifiers: Sequence[Notifier] | None = None, *, min_interval: float = 1.1,
                 max_attempts: int = 5, backoff_base: float = 1.0, backoff_max: float = 60.0,
                 dedupe_s: float = 600.0, max_queue: int = 500):
        self.notifiers = list(notifiers if notifiers is not None else notifier_from_env())
        self.min_interval = min_interval
        self.max_attempts = max_attempts
        self.backoff_base = backoff_base
        self.backoff_max = backoff_max
        self.dedupe_s = dedupe_s
        self._q: queue.Queue[str] = queue.Queue(maxsize=max_queue)
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
    def send(self, text: str, *, dedupe_key: str | None = None) -> None:
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
            self._q.put_nowait(text)
        except queue.Full:
            self.dropped += 1

    def trade_opened(self, symbol: str, side: str, entry: float, qty: float, notional: float,
                     sl: float, tp: float, prob: float | None = None, mode: str = "") -> None:
        p = f" p={prob:.2f}" if prob is not None else ""
        self.send(f"[{mode}] OPEN {side.upper()} {symbol} @ {entry:g} qty {qty:g} (${notional:.0f}) "
                  f"SL {sl:g} TP {tp:g}{p}")

    def trade_closed(self, symbol: str, side: str, entry: float, exit_price: float, pnl: float,
                     fees: float, reason: str, mode: str = "") -> None:
        net = pnl - fees
        self.send(f"[{mode}] CLOSE {side.upper()} {symbol} {entry:g} -> {exit_price:g} "
                  f"PnL {pnl:+.2f} (net {net:+.2f}) reason={reason}")

    def kill(self, reason: str, equity: float, dd_pct: float, mode: str = "") -> None:
        self.send(f"[{mode}] KILL SWITCH TRIPPED: {reason}. equity {equity:.2f}, drawdown {dd_pct:.1f}%")

    def engine_error(self, where: str, err: str) -> None:
        self.send(f"ENGINE ERROR in {where}: {err}"[:500], dedupe_key=f"err:{where}")

    def daily_summary(self, *, date: str, trades: int, wins: int, pnl: float, equity: float,
                      mode: str = "") -> None:
        wr = f"{100 * wins / trades:.0f}%" if trades else "n/a"
        self.send(f"[{mode}] DAILY {date}: {trades} trades, win rate {wr}, PnL {pnl:+.2f}, "
                  f"equity {equity:.2f}")

    def flush(self, timeout: float = 10.0) -> bool:
        """Wait until the queue has been delivered (tests / shutdown)."""
        return self._idle.wait(timeout)

    def close(self, timeout: float = 5.0) -> None:
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

    def _drain_batch(self, first: str) -> str:
        parts = [first]
        size = len(first)
        while size < MAX_TEXT - 300:
            try:
                nxt = self._q.get_nowait()
            except queue.Empty:
                break
            parts.append(nxt)
            size += len(nxt) + 1
        return "\n".join(parts)

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
            text = self._drain_batch(first)
            self._deliver(text)
            last_sent = time.monotonic()
            if self._q.empty():
                self._idle.set()

    def _deliver(self, text: str) -> None:
        for n in self.notifiers:
            for attempt in range(1, self.max_attempts + 1):
                try:
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
